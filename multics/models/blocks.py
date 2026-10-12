import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
from .rope import (
    modulate,
    precompute_freqs_cis,
    rope_apply,
)
from .hier_masks import (
    _squeeze_hier_batch,
    _slice_hier_active_group_for_cached_mask,
    _refuse_membership_on_scalar_kernel,
    build_stage2_hier_self_attn_mask,
    build_stage2_hier_flex_block_mask,
    get_temporal_block_causal_mask,
    BlockMask,
    FLEX_ATTENTION_AVAILABLE,
)
from .attention import (
    flash_attention,
    _rolling_cache_append,
    SelfAttention,
    CrossAttention,
)

_HIER_MASK_DEBUG_DUMPED = False


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-06,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.attention_mode = "dense"
        self.attn_impl = "sdpa"
        self.num_hub_tokens = 8
        self.num_groups = 1
        self.num_local_hub_tokens = 0
        self.num_global_hub_tokens = 8
        self.hier_no_local_hub = False
        self.hier_direct_enabled = False
        self.num_frame_per_block = 7
        self.local_attn_size = 24
        self.hub_embed = None

    def _is_sparse_hub(self):
        return True

    def _modulate_hub(self, x_hub, t_mod, has_seq, chunk_dim):
        k_len = self.num_hub_tokens
        batch_size = x_hub.shape[0]
        num_frames = x_hub.shape[1] // k_len
        assert x_hub.shape[1] == num_frames * k_len, (
            f"x_hub 序列长度 {x_hub.shape[1]} 不是 K={k_len} 的整数倍"
        )
        hub = self.norm1(x_hub)
        mod = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, _, _, _ = mod.chunk(6, dim=chunk_dim)
        if has_seq:
            shift_h = shift_msa.expand(-1, -1, k_len, -1).reshape(
                batch_size, num_frames * k_len, self.dim
            )
            scale_h = scale_msa.expand(-1, -1, k_len, -1).reshape(
                batch_size, num_frames * k_len, self.dim
            )
            gate_h = gate_msa.expand(-1, -1, k_len, -1).reshape(
                batch_size, num_frames * k_len, self.dim
            )
        else:
            shift_h = shift_msa.unsqueeze(1).expand(-1, num_frames * k_len, -1)
            scale_h = scale_msa.unsqueeze(1).expand(-1, num_frames * k_len, -1)
            gate_h = gate_msa.unsqueeze(1).expand(-1, num_frames * k_len, -1)
        hub_in = modulate(hub, shift_h, scale_h)
        return (hub_in, gate_h)

    def _build_camera_tokens(
        self,
        camera_pose,
        num_frames: int,
        num_patches: int,
        num_players: int,
        ref_x: torch.Tensor,
        include_hub: bool,
    ):
        """Expand per-player per-frame camera12 to self-attn token order.

        camera_pose accepts [P, F, C] (current B=1 path) or [B, P, F, C].
        Output follows the same player-major order as input_x after
        rearrange("(b p) l d -> b (p l) d", p=P): [B, P*F*L, C],
        optionally followed by zero camera tokens for the hub band [B, F*K, C].
        """
        if camera_pose is None:
            return None
        cam = camera_pose
        if cam.dim() == 3:
            cam = cam.unsqueeze(0)
        elif cam.dim() != 4:
            raise ValueError(
                f"camera_pose must be [P,F,C] or [B,P,F,C], got shape {tuple(camera_pose.shape)}"
            )
        if cam.shape[1] != num_players or cam.shape[2] != num_frames:
            raise ValueError(
                f"camera_pose expects P={num_players}, F={num_frames}, got {tuple(cam.shape[1:3])}"
            )
        if cam.shape[0] != ref_x.shape[0]:
            raise ValueError(
                f"camera_pose batch {cam.shape[0]} must match self-attn batch {ref_x.shape[0]}"
            )
        cam = cam.to(device=ref_x.device, dtype=ref_x.dtype)
        b, p, f, c = cam.shape
        cam_agent = cam[:, :, :, None, :].expand(b, p, f, num_patches, c)
        cam_agent = cam_agent.reshape(b, p * f * num_patches, c)
        if include_hub:
            cam_hub = cam_agent.new_zeros(b, f * self.num_hub_tokens, c)
            cam_agent = torch.cat([cam_agent, cam_hub], dim=1)
        return cam_agent

    def _self_attn_with_hub(
        self,
        input_x,
        hub_in,
        freqs,
        mask=None,
        kv_cache=None,
        grid_sizes=None,
        seq_lens=None,
        current_start=0,
        cache_start=None,
        num_frames=None,
        num_players=None,
        cam_tokens=None,
        group_labels=None,
        active_group=None,
        direct_player=False,
        prebuilt_hier_tiled_layouts=None,
        membership=None,
        global_only=None,
    ):
        seq = torch.cat([input_x, hub_in], dim=1)
        agent_len = input_x.shape[1]
        if kv_cache is None:
            if getattr(self, "attn_impl", "sdpa") == "hier_tiled_ste":
                if membership is not None:
                    _refuse_membership_on_scalar_kernel(attn_impl="hier_tiled_ste")
                if group_labels is None or active_group is None:
                    raise ValueError(
                        "hier_tiled_ste requires explicit group_labels and active_group"
                    )
                if num_frames is None or num_players is None:
                    raise ValueError(
                        "hier_tiled_ste requires num_frames and num_players"
                    )
                num_patches = input_x.shape[1] // (int(num_players) * int(num_frames))
                out = self.self_attn.forward_hier_tiled_ste(
                    seq,
                    freqs,
                    mask,
                    group_labels=group_labels,
                    active_group=active_group,
                    num_players=int(num_players),
                    num_frames=int(num_frames),
                    num_patches=int(num_patches),
                    num_groups=self.num_groups,
                    num_local_hub_tokens=self.num_local_hub_tokens,
                    num_global_hub_tokens=self.num_global_hub_tokens,
                    direct_player=bool(direct_player),
                    tiled_layouts=prebuilt_hier_tiled_layouts,
                    cam_tokens=cam_tokens,
                )
            else:
                out = self.self_attn(seq, freqs, mask, cam_tokens=cam_tokens)
        else:
            out = self.self_attn.causal_forward(
                seq,
                freqs,
                mask=None,
                grid_sizes=grid_sizes,
                seq_lens=seq_lens,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=cache_start,
                num_frames=num_frames,
                num_players=num_players,
                num_hub_tokens=self.num_hub_tokens,
                cam_tokens=cam_tokens,
                group_labels=group_labels,
                active_group=active_group,
                direct_player=direct_player,
                membership=membership,
                global_only=global_only,
            )
        assert out.shape[1] == agent_len + hub_in.shape[1], (
            f"self-attn 全序列长度 {out.shape[1]} != agent {agent_len} + hub {hub_in.shape[1]}"
        )
        return (out[:, :agent_len], out[:, agent_len:])

    @staticmethod
    def _ensure_action_cache(
        parent, name, batch, n_heads, head_dim, tokens_per_frame, frames, device, dtype
    ):
        """Lazily allocate (or re-allocate on shape change) a rolling action KV cache
        stored inside the per-layer kv_cache dict under `name`."""
        size = frames * tokens_per_frame
        c = parent.get(name)
        if (
            c is None
            or c["k"].shape[0] != batch
            or c["k"].shape[1] < size
            or (c["k"].shape[2] != n_heads)
            or (c["k"].shape[3] != head_dim)
        ):
            parent[name] = {
                "k": torch.zeros(
                    [batch, size, n_heads, head_dim], device=device, dtype=dtype
                ),
                "v": torch.zeros(
                    [batch, size, n_heads, head_dim], device=device, dtype=dtype
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
        return parent[name]

    def _action_temporal_cached_attn(
        self, attn, x, freqs_temporal, kv_cache, start_frame, n_block, local_attn
    ):
        """Cross-block KV-cached temporal self-attention for the mouse/action branch.

        Mirrors the training-time block-causal+window mask (`get_temporal_block_causal_mask`)
        but feeds historical frames from a rolling cache instead of the current block only.
        x: [B, f, D] with the sequence axis = frames (1 token per frame)."""
        B, f, D = x.shape
        nh = attn.num_heads
        hd = D // nh
        q = attn.norm_q(attn.q(x))
        k = attn.norm_k(attn.k(x))
        v = attn.v(x)
        roped_q = rope_apply(q, freqs_temporal, nh)
        roped_k = rope_apply(k, freqs_temporal, nh)
        rk = roped_k.view(B, f, nh, hd)
        vv = v.view(B, f, nh, hd)
        retention = local_attn + f if local_attn != -1 else None
        cache = self._ensure_action_cache(
            kv_cache,
            "action_temporal",
            B,
            nh,
            hd,
            tokens_per_frame=1,
            frames=retention if retention is not None else start_frame + f,
            device=x.device,
            dtype=x.dtype,
        )
        attn_start, local_end = _rolling_cache_append(
            cache,
            rk,
            vv,
            current_start=start_frame,
            current_end=start_frame + f,
            local_attn_size=retention if retention is not None else -1,
            sink_size=0,
            tokens_per_frame=1,
        )
        cached_global_start = (
            cache["global_end_index"].item() - cache["local_end_index"].item()
        )
        win = local_end - attn_start
        win_k = cache["k"][:, attn_start:local_end].reshape(B, win, D)
        win_v = cache["v"][:, attn_start:local_end].reshape(B, win, D)
        gq = start_frame + torch.arange(f, device=x.device)
        gk = cached_global_start + torch.arange(attn_start, local_end, device=x.device)
        mask = gk[None, :] // n_block <= gq[:, None] // n_block
        if local_attn != -1:
            mask = mask & (gq[:, None] - gk[None, :] < local_attn)
        out = flash_attention(roped_q, win_k, win_v, num_heads=nh, mask=mask)
        return attn.o(out)

    def _action_keyboard_cached_attn(
        self,
        attn,
        x,
        kbd_ctx,
        kv_cache,
        start_frame,
        n_block,
        local_attn,
        num_patches,
        tokens_per_kbd_frame,
    ):
        """Cross-block KV-cached keyboard cross-attention.

        Video query tokens (current block) attend to cached + current keyboard tokens
        under the same block-causal+window relation used in training."""
        B = x.shape[0]
        D = x.shape[-1]
        nh = attn.num_heads
        hd = D // nh
        f = kbd_ctx.shape[1] // tokens_per_kbd_frame
        ctx = kbd_ctx[:, 257:] if getattr(attn, "has_image_input", False) else kbd_ctx
        q = attn.norm_q(attn.q(x))
        k = attn.norm_k(attn.k(ctx))
        v = attn.v(ctx)
        k4 = k.view(B, f * tokens_per_kbd_frame, nh, hd)
        v4 = v.view(B, f * tokens_per_kbd_frame, nh, hd)
        retention = local_attn + f if local_attn != -1 else None
        cache = self._ensure_action_cache(
            kv_cache,
            "action_keyboard",
            B,
            nh,
            hd,
            tokens_per_frame=tokens_per_kbd_frame,
            frames=retention if retention is not None else start_frame + f,
            device=x.device,
            dtype=x.dtype,
        )
        attn_start, local_end = _rolling_cache_append(
            cache,
            k4,
            v4,
            current_start=start_frame * tokens_per_kbd_frame,
            current_end=(start_frame + f) * tokens_per_kbd_frame,
            local_attn_size=retention if retention is not None else -1,
            sink_size=0,
            tokens_per_frame=tokens_per_kbd_frame,
        )
        cached_global_start = (
            cache["global_end_index"].item() - cache["local_end_index"].item()
        )
        win = local_end - attn_start
        win_k = cache["k"][:, attn_start:local_end].reshape(B, win, D)
        win_v = cache["v"][:, attn_start:local_end].reshape(B, win, D)
        gq = start_frame + torch.arange(f, device=x.device)
        key_tok = cached_global_start + torch.arange(
            attn_start, local_end, device=x.device
        )
        gk = key_tok // tokens_per_kbd_frame
        frame_mask = gk[None, :] // n_block <= gq[:, None] // n_block
        if local_attn != -1:
            frame_mask = frame_mask & (gq[:, None] - gk[None, :] < local_attn)
        mask = frame_mask.repeat_interleave(num_patches, dim=0)
        out = flash_attention(q, win_k, win_v, num_heads=nh, mask=mask)
        return attn.o(out)

    def _get_action_masks(
        self,
        num_frames,
        num_patches,
        num_players,
        loss_type,
        device,
        prebuilt_self_attn_block_mask=None,
    ):
        local_attn = self.local_attn_size
        n_block = self.num_frame_per_block
        group_labels = getattr(self, "_hier_group_labels", None)
        active_group = getattr(self, "_hier_active_group", None)
        membership = getattr(self, "_hier_membership", None)
        global_only = getattr(self, "_hier_global_only", None)
        if group_labels is None and membership is None:
            raise ValueError(
                "hier_hub requires block._hier_group_labels [T,P] or _hier_membership [T,N,P]"
            )
        direct_enabled = bool(getattr(self, "hier_direct_enabled", False))
        if prebuilt_self_attn_block_mask is not None:
            if not isinstance(prebuilt_self_attn_block_mask, BlockMask):
                raise TypeError(
                    "prebuilt_self_attn_block_mask must be a FlexAttention BlockMask"
                )
            block_mask = prebuilt_self_attn_block_mask
        elif (
            getattr(self, "attn_impl", "sdpa") in ("flex", "hier_tiled_ste")
            and FLEX_ATTENTION_AVAILABLE
        ):
            block_mask, _, _ = build_stage2_hier_flex_block_mask(
                num_players,
                num_frames,
                num_patches,
                self.num_groups,
                self.num_local_hub_tokens,
                self.num_global_hub_tokens,
                n_block,
                local_attn,
                group_labels,
                active_group,
                direct_enabled,
                device,
                membership=membership,
                global_only=global_only,
                hier_no_local_hub=bool(getattr(self, "hier_no_local_hub", False)),
            )
        else:
            block_mask = build_stage2_hier_self_attn_mask(
                num_players,
                num_frames,
                num_patches,
                self.num_groups,
                self.num_local_hub_tokens,
                self.num_global_hub_tokens,
                n_block,
                local_attn,
                group_labels,
                active_group,
                direct_enabled,
                device=device,
                membership=membership,
                global_only=global_only,
                hier_no_local_hub=bool(getattr(self, "hier_no_local_hub", False)),
            )
        frame_mask = get_temporal_block_causal_mask(
            num_frames, n_block, local_attn, device=device
        )
        block_mask_mouse = frame_mask
        block_mask_keyboard = repeat(
            frame_mask, "f g -> (f q) (g k)", q=num_patches, k=12
        )
        import os

        DEBUG = os.environ.get("STAGE2_MASK_DEBUG", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        if DEBUG and os.environ.get("STAGE2_MASK_DEBUG_ONCE", "1") not in (
            "0",
            "",
            "false",
            "False",
        ):
            global _HIER_MASK_DEBUG_DUMPED
            if _HIER_MASK_DEBUG_DUMPED:
                DEBUG = False
            else:
                _HIER_MASK_DEBUG_DUMPED = True
        if DEBUG:
            import numpy as np
            import imageio

            _dir = os.environ.get("STAGE2_MASK_DEBUG_DIR", ".")
            os.makedirs(_dir, exist_ok=True)
            if torch.is_tensor(block_mask):
                _self_attn_dense = block_mask
            else:
                _self_attn_dense = build_stage2_hier_self_attn_mask(
                    num_players,
                    num_frames,
                    num_patches,
                    self.num_groups,
                    self.num_local_hub_tokens,
                    self.num_global_hub_tokens,
                    n_block,
                    local_attn,
                    group_labels,
                    active_group,
                    direct_enabled,
                    device=device,
                    membership=membership,
                    global_only=global_only,
                    hier_no_local_hub=bool(getattr(self, "hier_no_local_hub", False)),
                )
            _self_attn_cpu = _self_attn_dense.detach().cpu().bool()

            def _write_mask_image(_name, _m, _size=1024):
                if torch.is_tensor(_m):
                    _arr = _m.detach().cpu().numpy()
                else:
                    _arr = np.asarray(_m)
                if _arr.dtype == np.bool_:
                    _arr = _arr.astype(np.uint8) * 255
                elif _arr.dtype != np.uint8:
                    _arr = np.uint8(255.0 * _arr.astype(np.float32))
                try:
                    import cv2

                    _arr = cv2.resize(
                        _arr, (_size, _size), interpolation=cv2.INTER_NEAREST
                    )
                except Exception:
                    pass
                imageio.imwrite(os.path.join(_dir, "mask_%s.jpg" % _name), _arr)

            _dbg = [
                ("self_attn", _self_attn_cpu),
                ("mouse", block_mask_mouse),
                ("keyboard", block_mask_keyboard),
            ]
            for _name, _m in _dbg:
                _write_mask_image(_name, _m)
            _agent_len = int(num_players) * int(num_frames) * int(num_patches)
            _seq_len = int(_self_attn_cpu.shape[0])
            _hub_len = max(0, _seq_len - _agent_len)
            if _hub_len > 0:
                _crop_env = os.environ.get("STAGE2_MASK_DEBUG_CROP_AGENT_TOKENS")
                if _crop_env:
                    _crop_agent = int(_crop_env)
                else:
                    _crop_agent = min(
                        _agent_len,
                        max(
                            4 * _hub_len,
                            int(num_players) * int(num_patches) * max(1, int(n_block)),
                            512,
                        ),
                        4096,
                    )
                _crop_agent = max(0, min(_agent_len, _crop_agent))
                _crop_start = max(0, _agent_len - _crop_agent)
                _write_mask_image(
                    "self_attn_agent_hub_crop",
                    _self_attn_cpu[_crop_start:_seq_len, _crop_start:_seq_len],
                )
            _units = []
            _bounds = [0]
            _T = int(num_frames)
            _L = int(num_patches)
            _P = int(num_players)
            for _p in range(_P):
                _p_base = _p * _T * _L
                for _t in range(_T):
                    _units.append((_p_base + _t * _L, _p_base + (_t + 1) * _L))
                _bounds.append(len(_units))
            _N = int(self.num_groups)
            _KL = int(self.num_local_hub_tokens)
            _KG = int(self.num_global_hub_tokens)
            _local_start = _agent_len
            for _g in range(_N):
                for _t in range(_T):
                    _s = _local_start + (_t * _N + _g) * _KL
                    _units.append((_s, _s + _KL))
                _bounds.append(len(_units))
            _global_start = _local_start + _T * _N * _KL
            if _KG > 0:
                for _t in range(_T):
                    _s = _global_start + _t * _KG
                    _units.append((_s, _s + _KG))
                _bounds.append(len(_units))
            _M = len(_units)
            _coarse = np.zeros((_M, _M), dtype=np.uint8)
            for _qi, (_qs, _qe) in enumerate(_units):
                if _qe <= _qs:
                    continue
                for _ki, (_ks, _ke) in enumerate(_units):
                    if _ke <= _ks:
                        continue
                    if bool(_self_attn_cpu[_qs:_qe, _ks:_ke].any().item()):
                        _coarse[_qi, _ki] = 255
            for _b in _bounds:
                if 0 <= _b < _M:
                    _coarse[_b, :] = np.maximum(_coarse[_b, :], 127)
                    _coarse[:, _b] = np.maximum(_coarse[:, _b], 127)
            _write_mask_image("self_attn_coarse", _coarse)
        return (block_mask, block_mask_mouse, block_mask_keyboard)

    def forward(
        self,
        x,
        context,
        mouse_BPFD,
        keyboard_BPFD,
        t_mod,
        freqs,
        loss_type="bidirectional",
        x_hub=None,
        grid_hw=None,
        camera_pose: Optional[torch.Tensor] = None,
        prebuilt_self_attn_block_mask=None,
        prebuilt_hier_tiled_layouts=None,
    ):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            hw = x.shape[1] // t_mod.shape[1]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                v.expand(-1, -1, hw, -1).reshape(v.shape[0], -1, v.shape[-1])
                for v in [
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ]
            ]
        num_frames = keyboard_BPFD.shape[1]
        num_players = keyboard_BPFD.shape[0]
        num_patches = x.shape[1] // num_frames
        block_mask, block_mask_mouse, block_mask_keyboard = self._get_action_masks(
            num_frames,
            num_patches,
            num_players,
            "diffusion_forcing",
            x.device,
            prebuilt_self_attn_block_mask=prebuilt_self_attn_block_mask,
        )
        x_in = x
        delta_q_bp, delta_o_bp, dq_agent = (None, None, None)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        input_x = rearrange(input_x, "(b p) l d -> b (p l) d", p=num_players)
        cam_tokens = self._build_camera_tokens(
            camera_pose,
            num_frames,
            num_patches,
            num_players,
            input_x,
            include_hub=self._is_sparse_hub(),
        )
        assert x_hub is not None, "hier_hub requires an external x_hub residual stream"
        t_hub = t_mod[:1] if has_seq and t_mod.shape[0] == num_players else t_mod
        hub_in, hub_gate_msa = self._modulate_hub(x_hub, t_hub, has_seq, chunk_dim)
        self_attn_out, hub_out = self._self_attn_with_hub(
            input_x,
            hub_in,
            freqs,
            mask=block_mask,
            num_frames=num_frames,
            num_players=num_players,
            cam_tokens=cam_tokens,
            group_labels=getattr(self, "_hier_group_labels", None),
            active_group=getattr(self, "_hier_active_group", None),
            direct_player=bool(getattr(self, "hier_direct_enabled", False)),
            prebuilt_hier_tiled_layouts=prebuilt_hier_tiled_layouts,
            membership=getattr(self, "_hier_membership", None),
            global_only=getattr(self, "_hier_global_only", None),
        )
        x_hub = self.gate(x_hub, hub_gate_msa, hub_out)
        assert self_attn_out.shape[1] == num_players * num_frames * num_patches, (
            f"self_attn_out agent len {self_attn_out.shape[1]} != P*F*L={num_players}*{num_frames}*{num_patches}"
        )
        self_attn_out = rearrange(
            self_attn_out, "b (p l) d -> (b p) l d", p=num_players
        )
        x = self.gate(x, gate_msa, self_attn_out)
        if delta_o_bp is not None:
            x = x + delta_o_bp
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        input_x = torch.cat([input_x, mouse_BPFD], dim=-1)
        input_x = self.action_mlp(input_x)
        bp_dim = input_x.shape[0]
        spatial_tokens_per_frame = input_x.shape[1] // num_frames
        input_x = rearrange(input_x, "b (f l) d -> (b l) f d", f=num_frames)
        n_heads = self.self_attn.num_heads
        head_dim = input_x.shape[-1] // n_heads
        freqs_temporal = precompute_freqs_cis(head_dim, end=input_x.shape[1]).to(
            input_x.device
        )
        freqs_temporal = freqs_temporal.unsqueeze(1)
        action_temporal_attn_output = self.action_temporal_attn(
            input_x, freqs_temporal, block_mask_mouse
        )
        action_temporal_attn_output = rearrange(
            action_temporal_attn_output,
            "(b l) f d -> b (f l) d",
            b=bp_dim,
            l=spatial_tokens_per_frame,
        )
        x = x + action_temporal_attn_output
        keyboard_BPFD = rearrange(keyboard_BPFD, "p f l d -> p (f l) d")
        x = x + self.action_cross_attn(
            self.norm3(x), keyboard_BPFD, block_mask_keyboard
        )
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return (x, x_hub)

    @staticmethod
    def get_causal_attn_mask(
        num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size
    ):
        i = torch.arange(num_q_blocks)[:, None]
        j = torch.arange(num_k_blocks)[None, :]
        if sliding_block_size == -1:
            block_mask = j <= i
        else:
            block_mask = (j <= i) & (j >= i - sliding_block_size + 1)
        block_mask = repeat(
            block_mask,
            "q k -> (q q_block) (k k_block)",
            q_block=q_block_size,
            k_block=k_block_size,
        )
        return block_mask

    def causal_forward(
        self,
        x,
        context,
        mouse_BPFD,
        keyboard_BPFD,
        t_mod,
        freqs,
        loss_type="bidirectional",
        grid_sizes=None,
        seq_lens=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        x_hub=None,
        camera_pose: Optional[torch.Tensor] = None,
        group_labels: Optional[torch.Tensor] = None,
        active_group: Optional[torch.Tensor] = None,
        membership: Optional[torch.Tensor] = None,
        global_only: Optional[torch.Tensor] = None,
    ):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        x_in = x
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            hw = x.shape[1] // t_mod.shape[1]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                v.expand(-1, -1, hw, -1).reshape(v.shape[0], -1, v.shape[-1])
                for v in [
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ]
            ]
        num_frames = keyboard_BPFD.shape[1]
        num_players = keyboard_BPFD.shape[0]
        num_patches = x.shape[1] // num_frames
        start_frame = int(current_start) // max(int(num_players) * int(num_patches), 1)
        self._hier_group_labels = group_labels
        self._hier_active_group = _slice_hier_active_group_for_cached_mask(
            active_group, start_frame, num_frames
        )
        if membership is not None:
            mem = _squeeze_hier_batch(membership, 3, "membership")
            go = (
                _squeeze_hier_batch(global_only, 2, "global_only")
                if global_only is not None
                else None
            )
            if mem.shape[0] > num_frames:
                self._hier_membership = mem[start_frame : start_frame + num_frames]
                self._hier_global_only = (
                    None if go is None else go[start_frame : start_frame + num_frames]
                )
                if active_group is None:
                    self._hier_active_group = mem[
                        start_frame : start_frame + num_frames
                    ].any(dim=-1)
            else:
                self._hier_membership = mem
                self._hier_global_only = go
        else:
            self._hier_membership = None
            self._hier_global_only = None
        block_mask, block_mask_mouse, block_mask_keyboard = self._get_action_masks(
            num_frames, num_patches, num_players, "diffusion_forcing", x.device
        )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        input_x = rearrange(input_x, "(b p) l d -> b (p l) d", p=num_players)
        cam_tokens = self._build_camera_tokens(
            camera_pose,
            num_frames,
            num_patches,
            num_players,
            input_x,
            include_hub=self._is_sparse_hub(),
        )
        if grid_sizes is not None:
            bsz = grid_sizes.shape[0] // num_players
            grid_sizes_bp = rearrange(grid_sizes, "(b p) c -> b p c", p=num_players)
            grid_sizes_repeated = torch.cat(
                [
                    grid_sizes_bp[:, 0, :],
                    torch.full(
                        (bsz, 1),
                        num_players,
                        dtype=grid_sizes.dtype,
                        device=grid_sizes.device,
                    ),
                ],
                dim=1,
            )
        else:
            grid_sizes_repeated = None
        if seq_lens is not None:
            seq_lens_repeated = (
                [sl * num_players for sl in seq_lens] if seq_lens is not None else None
            )
        else:
            seq_lens_repeated = None
        delta_o_bp, dq_agent = (None, None)
        assert x_hub is not None, "hier_hub 需要外部传入 x_hub 残差流"
        t_hub = t_mod[:1] if has_seq and t_mod.shape[0] == num_players else t_mod
        hub_in, hub_gate_msa = self._modulate_hub(x_hub, t_hub, has_seq, chunk_dim)
        self_attn_out, hub_out = self._self_attn_with_hub(
            input_x,
            hub_in,
            freqs,
            mask=block_mask if kv_cache is None else None,
            kv_cache=kv_cache,
            grid_sizes=grid_sizes_repeated,
            seq_lens=seq_lens_repeated,
            current_start=current_start,
            cache_start=cache_start,
            num_frames=num_frames,
            num_players=num_players,
            cam_tokens=cam_tokens,
            group_labels=group_labels,
            active_group=active_group,
            direct_player=bool(getattr(self, "hier_direct_enabled", False)),
            membership=membership,
            global_only=global_only,
        )
        x_hub = self.gate(x_hub, hub_gate_msa, hub_out)
        assert self_attn_out.shape[1] == num_players * num_frames * num_patches, (
            f"self_attn_out agent len {self_attn_out.shape[1]} != P*F*L={num_players}*{num_frames}*{num_patches}"
        )
        self_attn_out = rearrange(
            self_attn_out, "b (p l) d -> (b p) l d", p=num_players
        )
        x = self.gate(x, gate_msa, self_attn_out)
        if delta_o_bp is not None:
            x = x + delta_o_bp
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        input_x = torch.cat([input_x, mouse_BPFD], dim=-1)
        input_x = self.action_mlp(input_x)
        bp_dim = input_x.shape[0]
        spatial_tokens_per_frame = input_x.shape[1] // num_frames
        input_x = rearrange(input_x, "b (f l) d -> (b l) f d", f=num_frames)
        n_heads = self.self_attn.num_heads
        head_dim = input_x.shape[-1] // n_heads
        _tpf_allp = num_patches * num_players
        _start_frame_offset = current_start // _tpf_allp if _tpf_allp > 0 else 0
        freqs_temporal = precompute_freqs_cis(
            head_dim, end=_start_frame_offset + input_x.shape[1]
        ).to(input_x.device)
        freqs_temporal = freqs_temporal[
            _start_frame_offset : _start_frame_offset + input_x.shape[1]
        ]
        freqs_temporal = freqs_temporal.unsqueeze(1)
        if kv_cache is not None:
            action_temporal_attn_output = self._action_temporal_cached_attn(
                self.action_temporal_attn,
                input_x,
                freqs_temporal,
                kv_cache,
                _start_frame_offset,
                self.num_frame_per_block,
                self.local_attn_size,
            )
        else:
            action_temporal_attn_output = self.action_temporal_attn(
                input_x, freqs_temporal, block_mask_mouse
            )
        action_temporal_attn_output = rearrange(
            action_temporal_attn_output,
            "(b l) f d -> b (f l) d",
            b=bp_dim,
            l=spatial_tokens_per_frame,
        )
        x = x + action_temporal_attn_output
        keyboard_BPFD = rearrange(keyboard_BPFD, "p f l d -> p (f l) d")
        if kv_cache is not None:
            tokens_per_kbd_frame = keyboard_BPFD.shape[1] // num_frames
            x = x + self._action_keyboard_cached_attn(
                self.action_cross_attn,
                self.norm3(x),
                keyboard_BPFD,
                kv_cache,
                _start_frame_offset,
                self.num_frame_per_block,
                self.local_attn_size,
                num_patches,
                tokens_per_kbd_frame,
            )
        else:
            x = x + self.action_cross_attn(
                self.norm3(x), keyboard_BPFD, block_mask_keyboard
            )
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return (x, x_hub)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device).unsqueeze(1)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            hw = x.shape[1] // t_mod.shape[1]
            shift = (
                shift.unsqueeze(2)
                .expand(-1, -1, hw, -1)
                .reshape(shift.shape[0], -1, shift.shape[-1])
            )
            scale = (
                scale.unsqueeze(2)
                .expand(-1, -1, hw, -1)
                .reshape(scale.shape[0], -1, scale.shape[-1])
            )
            x = self.head(self.norm(x) * (1 + scale) + shift)
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(1)
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x
