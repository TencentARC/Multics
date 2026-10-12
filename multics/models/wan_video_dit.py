import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Tuple, Optional
from einops import rearrange, repeat
from functools import partial
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter
from .compatible_linear import CompatibleLinear
from .rope import (
    causal_rope_apply,
    modulate,
    sinusoidal_embedding_1d,
    precompute_freqs_cis_3d,
    precompute_freqs_cis,
    build_simplex_vertices,
    precompute_freqs_cis_4d_wan_rerope,
    _agent_freqs_cis_from_vertex,
    build_spatial_freqs_4d_single_player,
    build_multiplayer_freqs_4d,
    build_hub_freqs_4d,
    rope_split_sizes_4d,
    rope_apply,
    WAN_REROPE_SPLITS,
)
from .hier_masks import (
    _prepare_hier_group_tensors,
    _squeeze_hier_batch,
    _debug_labels_from_membership,
    _slice_hier_active_group_for_cached_mask,
    _prepare_hier_membership_tensors,
    _refuse_membership_on_scalar_kernel,
    _validate_hier_no_local_topology,
    normalize_hier_local_hub_mode,
    expand_hier_local_hub_seed,
    prepare_hier_local_hub_state_dict_for_load,
    migrate_hier_no_local_hub_state_dict,
    build_stage2_hier_self_attn_mask,
    _build_stage2_hier_self_attn_mask_membership,
    _put_flex_blockmask_cache,
    _flex_kernel_options,
    _get_flex_attention,
    build_stage2_hier_flex_block_mask,
    _build_stage2_hier_flex_block_mask_membership,
    get_temporal_block_causal_mask,
    BlockMask,
    FLEX_ATTENTION_AVAILABLE,
)
from .attention import (
    flash_attention,
    _rolling_cache_append,
    _rolling_bool_cache_append,
    RMSNorm,
    AttentionModule,
    SelfAttention,
    CrossAttention,
)
from .blocks import (
    GateModule,
    DiTBlock,
    MLP,
    Head,
)


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(),
            CompatibleLinear(dim, dim, output_chunk_size=1024)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

    def enable_camera_pose_alignment(self, cam_dim: int = 12, dropout_p: float = 0.1):
        """Enable ReCamMaster-style relative camera12 conditioning on self-attn Q/K."""
        self.use_camera_pose = True
        self.camera_pose_dim = int(cam_dim)
        self.camera_dropout_p = float(dropout_p)
        for block in self.blocks:
            block.self_attn.enable_camera_pose_qk(
                self.camera_pose_dim, dropout_p=self.camera_dropout_p
            )

    def enable_simplex_rope(
        self,
        pool_size: int = 4,
        gamma: float = 1.0,
        splits: Tuple[int, int, int, int] = WAN_REROPE_SPLITS,
    ):
        """Enable Simplex 4D RoPE (ReRoPE splits) and register simplex vertex pool."""
        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        self.use_simplex_rope = True
        self.simplex_pool_size = int(pool_size)
        self.simplex_gamma = float(gamma)
        self.freqs_4d = precompute_freqs_cis_4d_wan_rerope(head_dim, splits=splits)
        d_a = splits[1]
        self.register_buffer(
            "simplex_vertices", build_simplex_vertices(pool_size, d_a), persistent=False
        )

    def set_stage2_local_attn_size(self, local_attn_size: int):
        """Update the temporal attention window without recreating parameters.

        ``enable_stage2`` also creates the model-level hub parameters, so it must
        not be called at a mid-training schedule boundary.  This setter changes
        only integer runtime configuration used to build the attention masks.
        """
        local_attn_size = int(local_attn_size)
        num_frame_per_block = int(getattr(self, "num_frame_per_block", 1))
        if local_attn_size != -1 and local_attn_size < num_frame_per_block:
            raise ValueError(
                f"Requires local_attn_size=-1 or >= num_frame_per_block; got local_attn_size={local_attn_size}, n={num_frame_per_block}"
            )
        self.local_attn_size = local_attn_size
        for block in self.blocks:
            block.local_attn_size = local_attn_size
            block.self_attn.local_attn_size = local_attn_size
            action_attn = getattr(block, "action_temporal_attn", None)
            if action_attn is not None:
                action_attn.local_attn_size = local_attn_size

    def enable_stage2(
        self,
        num_hub_tokens: int = 8,
        local_attn_size: int = 24,
        num_frame_per_block: int = 7,
        attention_mode: str = "hier_hub",
        num_groups: int = 1,
        num_local_hub_tokens: int = 0,
        num_global_hub_tokens: Optional[int] = None,
        hier_direct_schedule: str = "none",
        hier_direct_frac: float = 0.0,
        hier_direct_layers=None,
        attn_impl: str = "sdpa",
        hier_no_local_hub: bool = False,
        hier_local_hub_mode: str = "shared",
    ):
        """Enable hierarchical attention on all DiT blocks."""
        if attention_mode != "hier_hub":
            raise ValueError("Only hier_hub attention is supported")
        self.attention_mode = "hier_hub"
        self.attn_impl = str(attn_impl)
        self.hier_no_local_hub = bool(hier_no_local_hub)
        self.hier_local_hub_mode = normalize_hier_local_hub_mode(hier_local_hub_mode)
        if self.attn_impl not in ("sdpa", "flex", "hier_tiled_ste"):
            raise ValueError(
                f"attn_impl must be  sdpa, flex, or hier_tiled_ste; got {self.attn_impl!r}"
            )
        if self.attn_impl in ("flex", "hier_tiled_ste") and (
            not FLEX_ATTENTION_AVAILABLE
        ):
            raise RuntimeError(
                f"attn_impl={self.attn_impl!r} requires torch FlexAttention; refusing to silently fall back to dense SDPA"
            )
        if self.hier_no_local_hub and self.hier_local_hub_mode != "shared":
            raise ValueError(
                "hier_no_local_hub has no local seed and therefore requires hier_local_hub_mode=shared"
            )
        if self.hier_no_local_hub and self.attn_impl == "hier_tiled_ste":
            raise ValueError(
                "hier_no_local_hub does not support attn_impl=hier_tiled_ste; use sdpa or flex"
            )
        self.num_hub_tokens = int(num_hub_tokens)
        self.local_attn_size = int(local_attn_size)
        self.num_frame_per_block = int(num_frame_per_block)
        self.num_groups = int(num_groups)
        self.num_local_hub_tokens = int(num_local_hub_tokens)
        if num_global_hub_tokens is None:
            num_global_hub_tokens = (
                self.num_hub_tokens - self.num_groups * self.num_local_hub_tokens
            )
        self.num_global_hub_tokens = int(num_global_hub_tokens)
        if (
            self.local_attn_size != -1
            and self.local_attn_size < self.num_frame_per_block
        ):
            raise ValueError(
                f"Requires local_attn_size=-1 or >= num_frame_per_block; got local_attn_size={self.local_attn_size}, n={self.num_frame_per_block}"
            )
        self.local_hub_embed = None
        self.global_hub_embed = None
        if self.hier_no_local_hub:
            valid_hier_shape = (
                self.num_groups > 0
                and self.num_local_hub_tokens == 0
                and (self.num_global_hub_tokens > 0)
            )
        else:
            valid_hier_shape = (
                self.num_groups > 0
                and self.num_local_hub_tokens > 0
                and (self.num_global_hub_tokens >= 0)
            )
        if not valid_hier_shape:
            raise ValueError(
                f"Invalid hier_hub config: N={self.num_groups}, K_L={self.num_local_hub_tokens}, K_G={self.num_global_hub_tokens}, hier_no_local_hub={self.hier_no_local_hub}"
            )
        if (
            self.num_groups * self.num_local_hub_tokens + self.num_global_hub_tokens
            != self.num_hub_tokens
        ):
            raise ValueError(
                f"hier_hub requires N*K_L + K_G == num_hub_tokens; got {self.num_groups}*{self.num_local_hub_tokens}+{self.num_global_hub_tokens} != {self.num_hub_tokens}"
            )
        self.hub_embed = None
        if self.hier_no_local_hub:
            self.local_hub_embed = None
        else:
            if self.hier_local_hub_mode == "shared":
                local_shape = (self.num_local_hub_tokens, self.dim)
            else:
                local_shape = (self.num_groups, self.num_local_hub_tokens, self.dim)
            self.local_hub_embed = nn.Parameter(torch.zeros(*local_shape))
        self.global_hub_embed = nn.Parameter(
            torch.zeros(self.num_global_hub_tokens, self.dim)
        )
        n_layers = len(self.blocks)
        sched = str(hier_direct_schedule or "none")
        if sched not in ("none", "uniform", "explicit"):
            raise ValueError(
                f"hier_direct_schedule must be none/uniform/explicit, got {sched}"
            )
        direct_layers = set()
        if sched == "uniform" and float(hier_direct_frac) > 0:
            k = max(1, int(round(float(hier_direct_frac) * n_layers)))
            k = min(k, n_layers)
            if k == n_layers:
                direct_layers = set(range(n_layers))
            else:
                direct_layers = {
                    int(round(i * (n_layers - 1) / max(k - 1, 1))) for i in range(k)
                }
        elif sched == "explicit":
            if hier_direct_layers is None:
                direct_layers = set()
            elif isinstance(hier_direct_layers, str):
                direct_layers = {
                    int(x) for x in hier_direct_layers.split(",") if x.strip()
                }
            else:
                direct_layers = {int(x) for x in hier_direct_layers}
            bad = [i for i in direct_layers if i < 0 or i >= n_layers]
            if bad:
                raise ValueError(
                    f"hier_direct_layers out of range [0,{n_layers}): {bad}"
                )
        self.hier_direct_layers = sorted(direct_layers)
        print(
            f"[hier_hub] N={self.num_groups} K_L={self.num_local_hub_tokens} K_G={self.num_global_hub_tokens} K_total={self.num_hub_tokens} no_local={self.hier_no_local_hub} local_mode={self.hier_local_hub_mode} direct_layers={self.hier_direct_layers}",
            flush=True,
        )
        for block in self.blocks:
            block.attention_mode = "hier_hub"
            block.attn_impl = self.attn_impl
            block.self_attn.attn_impl = self.attn_impl
            block.num_hub_tokens = self.num_hub_tokens
            block.num_groups = self.num_groups
            block.num_local_hub_tokens = self.num_local_hub_tokens
            block.num_global_hub_tokens = self.num_global_hub_tokens
            block.hier_no_local_hub = self.hier_no_local_hub
            block.num_frame_per_block = self.num_frame_per_block
            block.local_attn_size = self.local_attn_size
            block.hier_direct_enabled = False
            block.hub_embed = None
            block.self_attn.local_attn_size = self.local_attn_size
            block.self_attn.hier_no_local_hub = self.hier_no_local_hub
            action_attn = getattr(block, "action_temporal_attn", None)
            if action_attn is not None:
                action_attn.local_attn_size = self.local_attn_size
            block.self_attn.num_frame_per_block = self.num_frame_per_block
            block.self_attn.sink_size = 0
        for i in self.hier_direct_layers:
            self.blocks[i].hier_direct_enabled = True

    def build_self_attn_freqs_with_hub(
        self,
        f: int,
        h: int,
        w: int,
        device,
        agent_vertex_ids=None,
        start_frame: int = 0,
    ) -> torch.Tensor:
        import os

        _agent_start = start_frame
        if os.environ.get("STAGE2_ROPE_AGENT_GLOBAL", "1") == "0":
            _agent_start = 0
        agent_freqs = self.build_self_attn_freqs(
            f, h, w, device, agent_vertex_ids=agent_vertex_ids, start_frame=_agent_start
        )
        k_len = getattr(self, "num_hub_tokens", 8)
        hub_freqs = build_hub_freqs_4d(
            self.freqs_4d, f, k_len, start_frame=start_frame, device=device
        )
        return torch.cat([agent_freqs, hub_freqs], dim=0)

    def build_self_attn_freqs(
        self,
        f: int,
        h: int,
        w: int,
        device,
        agent_vertex_ids=None,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """Build self-attn RoPE freqs: 4D simplex (p l) block concat or legacy 3D per-player grid.

        start_frame offsets the temporal band to global frame indices (block-causal KV-cache
        inference) so it matches full-sequence training; default 0 = training behaviour.
        """
        if agent_vertex_ids is None:
            raise ValueError("agent_vertex_ids required when use_simplex_rope=True")
        elif isinstance(agent_vertex_ids, torch.Tensor):
            agent_vertex_ids = agent_vertex_ids.detach().cpu().tolist()
        return build_multiplayer_freqs_4d(
            self.freqs_4d,
            f,
            h,
            w,
            agent_vertex_ids,
            self.simplex_vertices,
            gamma=getattr(self, "simplex_gamma", 1.0),
            device=device,
            start_frame=start_frame,
        )

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return (x, grid_size)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        mouse: Optional[torch.Tensor] = None,
        keyboard: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        loss_type="bidirectional",
        agent_vertex_ids: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        context = self.text_embedding(context)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        x, (f, h, w) = self.patchify(x)
        if context.shape[0] == 1 and x.shape[0] != 1:
            context = context.expand(x.shape[0], -1, -1)
        elif context.shape[0] != x.shape[0]:
            raise ValueError(
                f"context batch {context.shape[0]} must be 1 or match latent BP={x.shape[0]}"
            )
        if camera_pose is not None:
            if camera_pose.dim() == 3:
                cam_f = camera_pose.shape[1]
            elif camera_pose.dim() == 4:
                cam_f = camera_pose.shape[2]
            else:
                raise ValueError(
                    f"camera_pose must be [P,F,C] or [B,P,F,C], got shape {tuple(camera_pose.shape)}"
                )
            if cam_f != f:
                raise ValueError(
                    f"camera_pose latent frames {cam_f} must match patchified frames {f}"
                )
        if keyboard is None or mouse is None:
            raise ValueError(
                "WanModel.forward requires keyboard and mouse tensors for multi-player conditioning"
            )
        num_players = int(keyboard.shape[0])
        if int(mouse.shape[0]) != num_players:
            raise ValueError(f"keyboard P={num_players} but mouse P={mouse.shape[0]}")
        if x.shape[0] % num_players != 0:
            raise ValueError(
                f"latent BP={x.shape[0]} must be divisible by action P={num_players}; the dataloader uses B=1, so BP should equal P."
            )
        if agent_vertex_ids is not None and len(agent_vertex_ids) != num_players:
            raise ValueError(
                f"agent_vertex_ids length {len(agent_vertex_ids)} must match action P={num_players}"
            )
        timestep = timestep.to(device=x.device, dtype=x.dtype)
        if timestep.ndim <= 1:
            t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1)).to(x.dtype)
            )
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            t_head = t
        else:
            timestep_bf = timestep.reshape(timestep.shape[0], -1)
            if timestep_bf.shape[1] != f:
                raise ValueError(
                    f"Expected timestep to have {f} frame positions after flatten, got {timestep_bf.shape[1]}."
                )
            t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep_bf.reshape(-1)).to(
                    x.dtype
                )
            ).reshape(timestep_bf.shape[0], f, self.dim)
            t_mod = self.time_projection(t.reshape(-1, self.dim)).reshape(
                timestep_bf.shape[0], f, 6, self.dim
            )
            t_head = t
        if agent_vertex_ids is None:
            num_p = keyboard.shape[0] if keyboard is not None else x.shape[0]
            agent_vertex_ids = list(range(num_p))
        freqs = self.build_self_attn_freqs_with_hub(
            f, h, w, x.device, agent_vertex_ids=agent_vertex_ids
        )
        keyboard = self.keyboard_embedding(keyboard)
        n_heads = self.blocks[0].self_attn.num_heads
        assert keyboard.shape[-1] % n_heads == 0
        head_dim = keyboard.shape[-1] // n_heads
        freqs_keyboard = precompute_freqs_cis(head_dim, end=keyboard.shape[1]).to(
            keyboard.device
        )
        freqs_keyboard = freqs_keyboard.unsqueeze(1)
        keyboard = rope_apply(keyboard, freqs_keyboard, n_heads)
        latent_frame_num = f
        keyboard_BPFD = torch.zeros(
            (keyboard.shape[0], latent_frame_num * 12, keyboard.shape[2]),
            device=keyboard.device,
            dtype=keyboard.dtype,
        )
        mouse_BPFD = torch.zeros(
            (mouse.shape[0], latent_frame_num * 12, mouse.shape[2]),
            device=mouse.device,
            dtype=mouse.dtype,
        )
        window_size = 3
        compression_ratio = 4
        for i in range(latent_frame_num):
            end_frame_id = (i + 1) * 12
            src_start = (i - window_size + 1) * compression_ratio
            src_end = (i + 1) * compression_ratio
            src_start = max(0, src_start)
            src_end = min(keyboard.shape[1], src_end)
            if src_end > src_start:
                src_len = src_end - src_start
                dst_start = end_frame_id - src_len
                keyboard_BPFD[:, dst_start:end_frame_id, :] = keyboard[
                    :, src_start:src_end, :
                ]
                mouse_BPFD[:, dst_start:end_frame_id, :] = mouse[
                    :, src_start:src_end, :
                ]
        mouse_BPFD = rearrange(
            mouse_BPFD,
            "p (f r win) d -> p f (r win d)",
            f=f,
            r=compression_ratio,
            win=window_size,
        )
        mouse_BPFD = mouse_BPFD.unsqueeze(2).repeat(1, 1, h * w, 1)
        mouse_BPFD = rearrange(mouse_BPFD, "p f l d -> p (f l) d")
        keyboard_BPFD = rearrange(
            keyboard_BPFD,
            "p (f r win) d -> p f (r win) d",
            f=f,
            r=compression_ratio,
            win=window_size,
        )

        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        group_labels = kwargs.pop("group_labels", None)
        active_group = kwargs.pop("active_group", None)
        membership = kwargs.pop("membership", None)
        global_only = kwargs.pop("global_only", None)
        x_hub = None
        if membership is not None:
            _refuse_membership_on_scalar_kernel(
                attn_impl=getattr(self, "attn_impl", "sdpa")
            )
            membership, global_only, active = _prepare_hier_membership_tensors(
                membership,
                global_only,
                active_group,
                num_players,
                f,
                self.num_groups,
                x.device,
            )
            if group_labels is None:
                labels = _debug_labels_from_membership(membership)
            else:
                labels = _squeeze_hier_batch(group_labels, 2, "group_labels")
                if labels.shape != (f, num_players):
                    raise ValueError(
                        f"group_labels must be [T={f},P={num_players}], got {tuple(labels.shape)}"
                    )
                labels = labels.to(device=x.device, dtype=torch.long).contiguous()
        else:
            labels, active = _prepare_hier_group_tensors(
                group_labels, active_group, num_players, f, self.num_groups, x.device
            )
            membership = None
            global_only = None
        global_seed = self.global_hub_embed.to(dtype=x.dtype, device=x.device)
        if self.hier_no_local_hub:
            x_local_hub = x.new_zeros(1, 0, self.dim)
        else:
            local_seed = self.local_hub_embed.to(dtype=x.dtype, device=x.device)
            x_local_hub = expand_hier_local_hub_seed(
                local_seed,
                num_frames=f,
                num_groups=self.num_groups,
                num_local_hub_tokens=self.num_local_hub_tokens,
                model_dim=self.dim,
                mode=self.hier_local_hub_mode,
            )
        x_global_hub = (
            global_seed[None, None, :, :]
            .expand(1, f, self.num_global_hub_tokens, self.dim)
            .reshape(1, f * self.num_global_hub_tokens, self.dim)
        )
        x_hub = torch.cat([x_local_hub, x_global_hub], dim=1)
        for block in self.blocks:
            block._hier_group_labels = labels
            block._hier_active_group = active
            block._hier_membership = membership
            block._hier_global_only = global_only
        import os

        reuse_hier_flex_block_mask = os.environ.get(
            "WAN_HIER_REUSE_BLOCKMASK", "1"
        ) not in ("0", "false", "False", "")
        hier_flex_block_masks = {}
        hier_tiled_layouts = {}
        use_flex_train = (
            getattr(self, "attn_impl", "sdpa") in ("flex", "hier_tiled_ste")
            and FLEX_ATTENTION_AVAILABLE
        )
        if use_flex_train:
            num_p = keyboard_BPFD.shape[0]
            direct_flags = {
                bool(getattr(block, "hier_direct_enabled", False))
                for block in self.blocks
            }
            for direct_flag in direct_flags:
                block_mask, _, _ = build_stage2_hier_flex_block_mask(
                    num_p,
                    f,
                    h * w,
                    self.num_groups,
                    self.num_local_hub_tokens,
                    self.num_global_hub_tokens,
                    self.num_frame_per_block,
                    self.local_attn_size,
                    labels,
                    active,
                    direct_flag,
                    x.device,
                    membership=membership,
                    global_only=global_only,
                    hier_no_local_hub=self.hier_no_local_hub,
                )
                if reuse_hier_flex_block_mask:
                    hier_flex_block_masks[direct_flag] = block_mask
                if getattr(self, "attn_impl", "sdpa") == "hier_tiled_ste":
                    from .kernels.hier_tiled_attention import (
                        build_hier_tiled_full_sequence_layouts,
                    )

                    hier_tiled_layouts[direct_flag] = (
                        build_hier_tiled_full_sequence_layouts(
                            device=x.device,
                            num_players=num_p,
                            num_frames=f,
                            num_patches=h * w,
                            num_groups=self.num_groups,
                            num_local_hub_tokens=self.num_local_hub_tokens,
                            num_global_hub_tokens=self.num_global_hub_tokens,
                            num_frame_per_block=self.num_frame_per_block,
                            local_attn_size=self.local_attn_size,
                            group_labels=labels,
                            active_group=active,
                            direct_player=direct_flag,
                        )
                    )
        for block in self.blocks:
            pass
        try:
            for block in self.blocks:
                prebuilt_self_attn_block_mask = None
                prebuilt_hier_tiled_layouts = None
                direct_flag = bool(getattr(block, "hier_direct_enabled", False))
                if hier_flex_block_masks:
                    prebuilt_self_attn_block_mask = hier_flex_block_masks[direct_flag]
                if hier_tiled_layouts:
                    prebuilt_hier_tiled_layouts = hier_tiled_layouts[direct_flag]
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            checkpointed_forward = partial(
                                block,
                                loss_type="diffusion_forcing",
                                grid_hw=(h, w),
                                camera_pose=camera_pose,
                                prebuilt_self_attn_block_mask=prebuilt_self_attn_block_mask,
                                prebuilt_hier_tiled_layouts=prebuilt_hier_tiled_layouts,
                            )
                            x, x_hub = torch.utils.checkpoint.checkpoint(
                                checkpointed_forward,
                                x,
                                context,
                                mouse_BPFD,
                                keyboard_BPFD,
                                t_mod,
                                freqs,
                                x_hub=x_hub,
                                use_reentrant=False,
                            )
                    else:
                        checkpointed_forward = partial(
                            block,
                            loss_type="diffusion_forcing",
                            grid_hw=(h, w),
                            camera_pose=camera_pose,
                            prebuilt_self_attn_block_mask=prebuilt_self_attn_block_mask,
                            prebuilt_hier_tiled_layouts=prebuilt_hier_tiled_layouts,
                        )
                        x, x_hub = torch.utils.checkpoint.checkpoint(
                            checkpointed_forward,
                            x,
                            context,
                            mouse_BPFD,
                            keyboard_BPFD,
                            t_mod,
                            freqs,
                            x_hub=x_hub,
                            use_reentrant=False,
                        )
                else:
                    x, x_hub = block(
                        x,
                        context,
                        mouse_BPFD,
                        keyboard_BPFD,
                        t_mod,
                        freqs,
                        "diffusion_forcing",
                        x_hub=x_hub,
                        grid_hw=(h, w),
                        camera_pose=camera_pose,
                        prebuilt_self_attn_block_mask=prebuilt_self_attn_block_mask,
                        prebuilt_hier_tiled_layouts=prebuilt_hier_tiled_layouts,
                    )
        finally:
            if not (self.training and use_gradient_checkpointing):
                for block in self.blocks:
                    block._hier_group_labels = None
                    block._hier_active_group = None
                    block._hier_membership = None
                    block._hier_global_only = None
        x = self.head(x, t_head)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
            "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
            "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
            "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
            "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
            "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
            "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
            "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
            "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
            "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
            "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
            "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(
                        name_.split(".")[:1]
                        + [name.split(".")[1]]
                        + name_.split(".")[2:]
                    )
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict_) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict_) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
            }
        else:
            config = {}
        return (state_dict_, config)

    def from_civitai(self, state_dict):
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if not name.startswith("vace")
        }
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if name.split(".")[0]
            not in [
                "pose_patch_embedding",
                "face_adapter",
                "face_encoder",
                "motion_encoder",
            ]
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("model."):
                name = name[len("model.") :]
            state_dict_[name] = param
        state_dict = state_dict_
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
                "has_image_pos_emb": True,
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-06,
                "has_ref_conv": True,
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
                "has_ref_conv": True,
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-06,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-06,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif hash_state_dict_keys(state_dict) == "5b013604280dd715f8457c6ed6d6a626":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "2267d489f0ceb9f21836532952852ee5":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 52,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
                "has_ref_conv": True,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "47dbeab5e560db3180adf51dc0232fb1":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-06,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "require_clip_embedding": False,
            }
        else:
            config = {}
        return (state_dict, config)
