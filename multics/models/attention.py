import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .compatible_linear import CompatibleLinear
from .rope import (
    rope_apply,
)
from .hier_masks import (
    _prepare_hier_group_tensors,
    _prepare_hier_membership_tensors,
    _refuse_membership_on_scalar_kernel,
    _flex_kernel_options,
    _get_flex_attention,
    BlockMask,
    FLEX_ATTENTION_AVAILABLE,
)

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
    mask=None,
):
    if mask is not None:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        mask = mask.to(device=q.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def _rolling_cache_append(
    kv_cache: dict,
    roped_key: torch.Tensor,
    value: torch.Tensor,
    current_start: int,
    current_end: int,
    local_attn_size: int,
    sink_size: int,
    tokens_per_frame: int,
):
    """Append or refresh keys/values in a rolling local attention cache."""
    sink_tokens = sink_size * tokens_per_frame
    kv_cache_size = kv_cache["k"].shape[1]
    current_global_end = kv_cache["global_end_index"].item()
    current_local_end = kv_cache["local_end_index"].item()
    cache_progress = max(0, current_end - current_global_end)
    num_new_tokens = min(roped_key.shape[1], cache_progress)
    if num_new_tokens == 0 and current_local_end > 0:
        cached_global_start = current_global_end - current_local_end
        refresh_start = max(current_start, cached_global_start)
        refresh_end = min(current_end, current_global_end)
        if refresh_end > refresh_start:
            refresh_len = refresh_end - refresh_start
            src_start = refresh_start - current_start
            dst_start = refresh_start - cached_global_start
            kv_cache["k"][:, dst_start : dst_start + refresh_len] = roped_key[
                :, src_start : src_start + refresh_len
            ]
            kv_cache["v"][:, dst_start : dst_start + refresh_len] = value[
                :, src_start : src_start + refresh_len
            ]
    new_local_end = current_local_end + num_new_tokens
    if (
        local_attn_size != -1
        and current_end > current_global_end
        and (new_local_end > kv_cache_size)
    ):
        num_evicted_tokens = min(
            new_local_end - kv_cache_size, new_local_end - sink_tokens - 1
        )
        num_evicted_tokens = max(0, num_evicted_tokens)
        if num_evicted_tokens > 0:
            num_rolled_tokens = max(
                0, current_local_end - num_evicted_tokens - sink_tokens
            )
            if num_rolled_tokens > 0:
                source_start = sink_tokens + num_evicted_tokens
                source_end = source_start + num_rolled_tokens
                target_start = sink_tokens
                target_end = target_start + num_rolled_tokens
                kv_cache["k"][:, target_start:target_end] = kv_cache["k"][
                    :, source_start:source_end
                ].clone()
                kv_cache["v"][:, target_start:target_end] = kv_cache["v"][
                    :, source_start:source_end
                ].clone()
            new_local_end = min(
                current_local_end + num_new_tokens - num_evicted_tokens, kv_cache_size
            )
    if num_new_tokens > 0:
        local_start_index = new_local_end - num_new_tokens
        kv_cache["k"][:, local_start_index : local_start_index + num_new_tokens] = (
            roped_key[:, -num_new_tokens:]
        )
        kv_cache["v"][:, local_start_index : local_start_index + num_new_tokens] = (
            value[:, -num_new_tokens:]
        )
    max_attn = (
        kv_cache_size
        if local_attn_size == -1
        else min(local_attn_size * tokens_per_frame, kv_cache_size)
    )
    attn_start = max(0, new_local_end - max_attn)
    if sink_size > 0 and sink_tokens > attn_start:
        attn_start = 0
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(new_local_end)
    return (attn_start, new_local_end)


def _rolling_bool_cache_append(
    kv_cache: dict,
    key: str,
    values: torch.Tensor,
    current_start: int,
    current_end: int,
    local_attn_size: int,
    sink_size: int,
    tokens_per_frame: int,
):
    """Append/refresh a 1D bool auxiliary cache in lockstep with K/V cache."""
    if key not in kv_cache:
        return
    sink_tokens = sink_size * tokens_per_frame
    cache = kv_cache[key]
    cache_size = cache.shape[0]
    current_global_end = kv_cache["global_end_index"].item()
    current_local_end = kv_cache["local_end_index"].item()
    cache_progress = max(0, current_end - current_global_end)
    num_new_tokens = min(values.shape[0], cache_progress)
    if num_new_tokens == 0 and current_local_end > 0:
        cached_global_start = current_global_end - current_local_end
        refresh_start = max(current_start, cached_global_start)
        refresh_end = min(current_end, current_global_end)
        if refresh_end > refresh_start:
            refresh_len = refresh_end - refresh_start
            src_start = refresh_start - current_start
            dst_start = refresh_start - cached_global_start
            cache[dst_start : dst_start + refresh_len] = values[
                src_start : src_start + refresh_len
            ]
    new_local_end = current_local_end + num_new_tokens
    if (
        local_attn_size != -1
        and current_end > current_global_end
        and (new_local_end > cache_size)
    ):
        num_evicted_tokens = min(
            new_local_end - cache_size, new_local_end - sink_tokens - 1
        )
        num_evicted_tokens = max(0, num_evicted_tokens)
        if num_evicted_tokens > 0:
            num_rolled_tokens = max(
                0, current_local_end - num_evicted_tokens - sink_tokens
            )
            if num_rolled_tokens > 0:
                source_start = sink_tokens + num_evicted_tokens
                source_end = source_start + num_rolled_tokens
                target_start = sink_tokens
                target_end = target_start + num_rolled_tokens
                cache[target_start:target_end] = cache[source_start:source_end].clone()
            new_local_end = min(
                current_local_end + num_new_tokens - num_evicted_tokens, cache_size
            )
    if num_new_tokens > 0:
        local_start_index = new_local_end - num_new_tokens
        cache[local_start_index : local_start_index + num_new_tokens] = values[
            -num_new_tokens:
        ]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, mask=mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-06):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)
        self.sink_size = 0
        self.local_attn_size = 6
        self.num_frame_per_block = 7
        self.attn_impl = "sdpa"
        self.max_attention_size = self.local_attn_size * 3072
        self.cam_dim = None
        self.cam_q = None
        self.cam_k = None
        self.cam_dropout_p = 0.0

    def enable_camera_pose_qk(self, cam_dim: int, dropout_p: float = 0.1):
        """Attach zero-init camera adapters to Q/K only.

        camera tokens are added after q/k projection and before RMSNorm+RoPE, so cached K
        already contains camera information in causal inference. V is intentionally left
        unchanged to keep camera conditioning a geometry-alignment bias rather than a content residual path.
        A shared input-dropout mask regularizes Q/K without breaking their camera-token
        correspondence. Bias-free projections keep zero-valued hub camera tokens inert.
        """
        cam_dim = int(cam_dim)
        dropout_p = float(dropout_p)
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"camera dropout_p must be in [0,1), got {dropout_p}")
        if self.cam_q is not None and self.cam_dim == cam_dim:
            self.cam_dropout_p = dropout_p
            return
        device = self.q.weight.device
        dtype = self.q.weight.dtype
        self.cam_dim = cam_dim
        self.cam_dropout_p = dropout_p
        self.cam_q = CompatibleLinear(cam_dim, self.dim, bias=False, input_alignment=16).to(
            device=device, dtype=dtype
        )
        self.cam_k = CompatibleLinear(cam_dim, self.dim, bias=False, input_alignment=16).to(
            device=device, dtype=dtype
        )
        nn.init.zeros_(self.cam_q.weight)
        nn.init.zeros_(self.cam_k.weight)

    def _apply_camera_qk(self, qx, kx, cam_tokens=None):
        if cam_tokens is None:
            return (qx, kx)
        if self.cam_q is None or self.cam_k is None:
            raise RuntimeError(
                "camera_pose was provided but enable_camera_pose_qk() was not called"
            )
        if cam_tokens.shape[:2] != qx.shape[:2]:
            raise ValueError(
                f"camera tokens shape {tuple(cam_tokens.shape[:2])} must match q/k tokens {tuple(qx.shape[:2])}"
            )
        cam_tokens = cam_tokens.to(device=qx.device, dtype=qx.dtype)
        cam_tokens = F.dropout(cam_tokens, p=self.cam_dropout_p, training=self.training)
        return (qx + self.cam_q(cam_tokens), kx + self.cam_k(cam_tokens))

    def _flex_from_projected_qkv(self, q, k, v, mask):
        """FlexAttention on already projected/normed/rotated Q/K/V.

        Keeping this operation separate lets the Tiled128 STE path reuse the
        exact existing Flex Jacobian without projecting Q/K/V or applying RoPE
        twice.  The default Flex forward calls the same helper, so its operation
        order and output remain unchanged.
        """
        if not isinstance(mask, BlockMask):
            raise TypeError(
                "_flex_from_projected_qkv requires a FlexAttention BlockMask"
            )
        b, s = q.shape[:2]
        n, d = (self.num_heads, self.head_dim)
        s_pad = mask.shape[-1]
        pad = s_pad - s

        def to_bhsd(t):
            t = rearrange(t, "b s (n d) -> b n s d", n=n)
            if pad > 0:
                t = torch.cat([t, t.new_zeros(b, n, pad, d)], dim=2)
            return t

        _qb, _kb, _vb = (to_bhsd(q), to_bhsd(k), to_bhsd(v))
        _kopt = _flex_kernel_options()
        if _kopt is not None:
            out = _get_flex_attention()(
                _qb, _kb, _vb, block_mask=mask, kernel_options=_kopt
            )
        else:
            out = _get_flex_attention()(_qb, _kb, _vb, block_mask=mask)
        return rearrange(out[:, :, :s], "b n s d -> b s (n d)")

    def _project_self_attn_qkv(self, x, freqs, cam_tokens=None):
        """Shared Q/K/V projection used by the default and Tiled STE paths."""
        qx = self.q(x)
        kx = self.k(x)
        qx, kx = self._apply_camera_qk(qx, kx, cam_tokens)
        q = self.norm_q(qx)
        k = self.norm_k(kx)
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return (q, k, v)

    def forward_hier_tiled_ste(
        self,
        x,
        freqs,
        mask,
        *,
        group_labels,
        active_group,
        num_players,
        num_frames,
        num_patches,
        num_groups,
        num_local_hub_tokens,
        num_global_hub_tokens,
        direct_player,
        tiled_layouts=None,
        cam_tokens=None,
    ):
        """Tiled128 forward value with the established Flex backward Jacobian."""
        if not FLEX_ATTENTION_AVAILABLE or not isinstance(mask, BlockMask):
            raise RuntimeError(
                "hier_tiled_ste requires an available FlexAttention backend and BlockMask"
            )
        from .kernels.hier_tiled_attention import (
            build_hier_query_active_mask,
            build_hier_tiled_full_sequence_layouts,
            hierarchical_tiled_full_sequence_attention,
        )

        q, k, v = self._project_self_attn_qkv(x, freqs, cam_tokens=cam_tokens)
        y_flex = self._flex_from_projected_qkv(q, k, v, mask)
        active_rows = build_hier_query_active_mask(
            active_group=active_group,
            num_players=num_players,
            num_frames=num_frames,
            num_patches=num_patches,
            num_groups=num_groups,
            num_local_hub_tokens=num_local_hub_tokens,
            num_global_hub_tokens=num_global_hub_tokens,
            device=x.device,
            dtype=y_flex.dtype,
        )
        y_flex_valid = y_flex * active_rows
        if tiled_layouts is None:
            tiled_layouts = build_hier_tiled_full_sequence_layouts(
                device=x.device,
                num_players=num_players,
                num_frames=num_frames,
                num_patches=num_patches,
                num_groups=num_groups,
                num_local_hub_tokens=num_local_hub_tokens,
                num_global_hub_tokens=num_global_hub_tokens,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                group_labels=group_labels,
                active_group=active_group,
                direct_player=direct_player,
            )
        with torch.no_grad():
            b, s = q.shape[:2]
            q4 = q.detach().view(b, s, self.num_heads, self.head_dim)
            k4 = k.detach().view(b, s, self.num_heads, self.head_dim)
            v4 = v.detach().view(b, s, self.num_heads, self.head_dim)
            y_tiled = hierarchical_tiled_full_sequence_attention(
                roped_q=q4, roped_k=k4, value=v4, layouts=tiled_layouts
            )
        y = y_tiled.detach() + (y_flex_valid - y_flex_valid.detach())
        return self.o(y)

    def forward(self, x, freqs, mask=None, cam_tokens=None):
        q, k, v = self._project_self_attn_qkv(x, freqs, cam_tokens=cam_tokens)
        if isinstance(mask, BlockMask):
            out = self._flex_from_projected_qkv(q, k, v, mask)
            return self.o(out)
        x = self.attn(q, k, v, mask)
        return self.o(x)

    @staticmethod
    def _frame_causal_window_mask(q_frames, k_frames, n_block, local_attn):
        """Block-causal + sliding-window mask on frame indices. Key visible to query iff
        same-or-earlier block (kf//n <= qf//n) AND within the window (qf - kf < local).
        Exactly mirrors build_stage2_self_attn_mask's (m_causal & m_window)."""
        qi = q_frames[:, None]
        kj = k_frames[None, :]
        m = kj // n_block <= qi // n_block
        if local_attn != -1:
            m = m & (qi - kj < local_attn)
        return m

    def causal_forward_hier_hub(
        self,
        x,
        freqs,
        kv_cache,
        current_start=0,
        num_frames=None,
        num_players=None,
        num_hub_tokens=None,
        cam_tokens=None,
        group_labels=None,
        active_group=None,
        direct_player=False,
        membership=None,
        global_only=None,
    ):
        """Incremental hierarchical hub self-attn.

        Current block tensor contract:
          x: [B, P*n*L + n*(N*K_L + K_G), D]
          agent cache: P separate timelines, each frame has L tokens.
          local cache: N separate timelines, each frame has K_L tokens.
          global cache: one timeline, each frame has K_G tokens.

        The masks mirror build_stage2_hier_self_attn_mask on a causal block:
          player -> own history + same-group local hub history/current
          local  -> current-group player history + same local slot history/current + global history
          global -> active local hub history/current + global history

        membership mode additionally:
          player -> every owned local slot (Anchor) and global iff global_only[q_t, p]
          global -> player iff global_only[k_t, p]  (key time; needs full-clip tensors)
        """
        b, s, n_heads, head_dim = (*x.shape[:2], self.num_heads, self.head_dim)
        P = int(num_players if num_players is not None else kv_cache["num_players"])
        N = int(kv_cache["num_groups"])
        KL = int(kv_cache["num_local_hub_tokens"])
        KG = int(kv_cache["num_global_hub_tokens"])
        no_local = bool(getattr(self, "hier_no_local_hub", False))
        if bool(kv_cache.get("hier_no_local_hub", False)) != no_local:
            raise ValueError(
                "hier_no_local_hub mismatch between SelfAttention and KV cache"
            )
        n_frames = int(num_frames)
        K_total = int(num_hub_tokens if num_hub_tokens is not None else N * KL + KG)
        if K_total != N * KL + KG:
            raise ValueError(f"hier_hub K_total mismatch: {K_total} != {N}*{KL}+{KG}")
        if no_local and (KL != 0 or KG <= 0):
            raise ValueError(
                f"hier_no_local_hub cached attention requires K_L=0 and K_G>0; got K_L={KL}, K_G={KG}"
            )
        use_membership = membership is not None
        membership_full = None
        global_only_full = None
        if use_membership:
            membership_full, global_only_full, active_full = (
                _prepare_hier_membership_tensors(
                    membership,
                    global_only,
                    active_group,
                    P,
                    n_frames,
                    N,
                    x.device,
                    allow_full_t=True,
                )
            )
            labels = None
            active = active_full
        else:
            labels, active = _prepare_hier_group_tensors(
                group_labels, active_group, P, n_frames, N, x.device
            )
            if (
                n_frames > 1
                and labels.numel()
                and (not torch.equal(labels, labels[:1].expand_as(labels)))
            ):
                raise ValueError(
                    "hier_hub cached inference requires group_labels constant inside each latent block"
                )
        hub_len = n_frames * K_total
        agent_len = s - hub_len
        if agent_len <= 0 or agent_len % (P * n_frames) != 0:
            raise ValueError(
                f"hier_hub causal seq shape invalid: S={s}, hub_len={hub_len}, P={P}, n={n_frames}"
            )
        L = agent_len // (P * n_frames)
        n_block = int(self.num_frame_per_block)
        local_attn = int(self.local_attn_size)
        device = x.device
        if freqs.shape[0] != s:
            raise ValueError(
                "hier_hub causal_forward expects freqs.shape[0] == seq_len"
            )
        qx = self.q(x)
        kx = self.k(x)
        qx, kx = self._apply_camera_qk(qx, kx, cam_tokens)
        q = self.norm_q(qx).view(b, s, n_heads, head_dim)
        k = self.norm_k(kx).view(b, s, n_heads, head_dim)
        v = self.v(x).view(b, s, n_heads, head_dim)
        q_flat = q.reshape(b, s, n_heads * head_dim)
        k_flat = k.reshape(b, s, n_heads * head_dim)
        roped_q = rope_apply(q_flat, freqs, self.num_heads).view(
            b, s, n_heads, head_dim
        )
        roped_k = rope_apply(k_flat, freqs, self.num_heads).view(
            b, s, n_heads, head_dim
        )
        local_len = n_frames * N * KL
        global_len = n_frames * KG
        local_start = agent_len
        global_start = agent_len + local_len
        agent_caches = kv_cache["agent_caches"]
        local_caches = kv_cache["local_hubs"]
        global_cache = kv_cache["global_hub"]
        retention = local_attn + n_frames if local_attn != -1 else -1
        start_frame = current_start // max(P * L, 1)
        end_frame = start_frame + n_frames - 1
        if use_membership:
            T_full = int(membership_full.shape[0])
            if start_frame < 0 or start_frame + n_frames > T_full:
                raise ValueError(
                    f"hier_hub membership T={T_full} does not cover cached block [{start_frame},{start_frame + n_frames})"
                )
            mem_block = membership_full[start_frame : start_frame + n_frames]
            go_block = global_only_full[start_frame : start_frame + n_frames]
            active = active[start_frame : start_frame + n_frames]
            if n_frames > 1:
                if not torch.equal(mem_block, mem_block[:1].expand_as(mem_block)):
                    raise ValueError(
                        "hier_hub cached inference requires membership constant inside each latent block"
                    )
                if not torch.equal(go_block, go_block[:1].expand_as(go_block)):
                    raise ValueError(
                        "hier_hub cached inference requires global_only constant inside each latent block"
                    )
        agent_k_cur = roped_k[:, :agent_len].reshape(
            b, P, n_frames, L, n_heads, head_dim
        )
        agent_v_cur = v[:, :agent_len].reshape(b, P, n_frames, L, n_heads, head_dim)
        local_k_cur = roped_k[:, local_start:global_start].reshape(
            b, n_frames, N, KL, n_heads, head_dim
        )
        local_v_cur = v[:, local_start:global_start].reshape(
            b, n_frames, N, KL, n_heads, head_dim
        )
        if KG > 0:
            global_k_cur = roped_k[:, global_start:].reshape(
                b, n_frames, KG, n_heads, head_dim
            )
            global_v_cur = v[:, global_start:].reshape(
                b, n_frames, KG, n_heads, head_dim
            )
        else:
            global_k_cur = roped_k.new_zeros(b, n_frames, 0, n_heads, head_dim)
            global_v_cur = v.new_zeros(b, n_frames, 0, n_heads, head_dim)
        for p_idx in range(P):
            pk = agent_k_cur[:, p_idx].reshape(b, n_frames * L, n_heads, head_dim)
            pv = agent_v_cur[:, p_idx].reshape(b, n_frames * L, n_heads, head_dim)
            _rolling_cache_append(
                agent_caches[p_idx],
                pk,
                pv,
                start_frame * L,
                (start_frame + n_frames) * L,
                retention,
                0,
                L,
            )
        if not no_local:
            for g_idx in range(N):
                lk = local_k_cur[:, :, g_idx].reshape(
                    b, n_frames * KL, n_heads, head_dim
                )
                lv = local_v_cur[:, :, g_idx].reshape(
                    b, n_frames * KL, n_heads, head_dim
                )
                local_active_cur = active[:, g_idx].repeat_interleave(KL)
                _rolling_bool_cache_append(
                    local_caches[g_idx],
                    "active",
                    local_active_cur,
                    start_frame * KL,
                    (start_frame + n_frames) * KL,
                    retention,
                    0,
                    KL,
                )
                _rolling_cache_append(
                    local_caches[g_idx],
                    lk,
                    lv,
                    start_frame * KL,
                    (start_frame + n_frames) * KL,
                    retention,
                    0,
                    KL,
                )
        if KG > 0:
            _rolling_cache_append(
                global_cache,
                global_k_cur.reshape(b, n_frames * KG, n_heads, head_dim),
                global_v_cur.reshape(b, n_frames * KG, n_heads, head_dim),
                start_frame * KG,
                (start_frame + n_frames) * KG,
                retention,
                0,
                KG,
            )

        def cache_window(cache, tokens_per_frame):
            local_end = cache["local_end_index"].item()
            n_cached = local_end // tokens_per_frame if tokens_per_frame > 0 else 0
            first = end_frame - n_cached + 1
            frames = first + torch.arange(local_end, device=device) // tokens_per_frame
            kk = cache["k"][:, :local_end].reshape(b, local_end, n_heads * head_dim)
            vv = cache["v"][:, :local_end].reshape(b, local_end, n_heads * head_dim)
            return (frames, kk, vv)

        def local_cache_window(cache, tokens_per_frame):
            frames, kk, vv = cache_window(cache, tokens_per_frame)
            active_tokens = cache.get("active")
            if active_tokens is None:
                active_tokens = torch.ones(
                    frames.shape[0], device=device, dtype=torch.bool
                )
            else:
                active_tokens = active_tokens[: frames.shape[0]].to(
                    device=device, dtype=torch.bool
                )
            return (frames, kk, vv, active_tokens)

        agent_frames, agent_keys, agent_vals = ([], [], [])
        for p_idx in range(P):
            af, ak, av = cache_window(agent_caches[p_idx], L)
            agent_frames.append(af)
            agent_keys.append(ak)
            agent_vals.append(av)
        local_frames, local_keys, local_vals = ([], [], [])
        local_active_tokens = []
        if no_local:
            local_all_frames = torch.empty(0, device=device, dtype=torch.long)
            local_all_keys = x.new_zeros(b, 0, n_heads * head_dim)
            local_all_vals = x.new_zeros(b, 0, n_heads * head_dim)
            local_all_active = torch.empty(0, device=device, dtype=torch.bool)
            local_all_groups = torch.empty(0, device=device, dtype=torch.long)
        else:
            for g_idx in range(N):
                lf, lk, lv, la = local_cache_window(local_caches[g_idx], KL)
                local_frames.append(lf)
                local_keys.append(lk)
                local_vals.append(lv)
                local_active_tokens.append(la)
            local_all_frames = torch.cat(local_frames, dim=0)
            local_all_keys = torch.cat(local_keys, dim=1)
            local_all_vals = torch.cat(local_vals, dim=1)
            local_all_active = torch.cat(local_active_tokens, dim=0)
            local_all_groups = torch.cat(
                [torch.full_like(local_frames[g_idx], g_idx) for g_idx in range(N)],
                dim=0,
            )
        if KG > 0:
            global_frames, global_keys, global_vals = cache_window(global_cache, KG)
        else:
            global_frames = torch.empty(0, device=device, dtype=torch.long)
            global_keys = x.new_zeros(b, 0, n_heads * head_dim)
            global_vals = x.new_zeros(b, 0, n_heads * head_dim)
        import os

        hier_infer_kernel = (
            os.environ.get("WAN_HIER_INFER_KERNEL", "sdpa").strip().lower()
        )
        if hier_infer_kernel not in (
            "sdpa",
            "segmented",
            "tiled128",
            "packed_varlen",
            "paged_kv",
        ):
            raise ValueError(
                f"WAN_HIER_INFER_KERNEL must be 'sdpa', 'segmented', 'tiled128', 'packed_varlen', or 'paged_kv', got {hier_infer_kernel!r}"
            )
        if hier_infer_kernel in ("segmented", "tiled128", "packed_varlen", "paged_kv"):
            if no_local and hier_infer_kernel not in ("packed_varlen", "paged_kv"):
                raise NotImplementedError(
                    "hier_no_local_hub cached inference supports sdpa, packed_varlen, or paged_kv; got "
                    + hier_infer_kernel
                )
            if use_membership:
                if hier_infer_kernel not in ("packed_varlen", "paged_kv"):
                    _refuse_membership_on_scalar_kernel(infer_kernel=hier_infer_kernel)
            if hier_infer_kernel == "segmented":
                from .kernels.hier_segmented_attention import (
                    hierarchical_segmented_attention as selected_hier_attention,
                )
            elif hier_infer_kernel == "tiled128":
                from .kernels.hier_tiled_attention import (
                    hierarchical_tiled_cached_attention as selected_hier_attention,
                )
            elif hier_infer_kernel == "packed_varlen":
                from .kernels.hier_packed_attention import (
                    hierarchical_packed_varlen_attention as selected_hier_attention,
                )
            else:
                from .kernels.hier_paged_attention import (
                    hierarchical_paged_kv_attention as selected_hier_attention,
                )
            membership_kernel_kwargs = {}
            if hier_infer_kernel in ("packed_varlen", "paged_kv"):
                membership_kernel_kwargs = {
                    "membership": membership_full if use_membership else None,
                    "global_only": global_only_full if use_membership else None,
                    "hier_no_local_hub": no_local,
                }
            out = selected_hier_attention(
                roped_q=roped_q,
                agent_frames=agent_frames,
                agent_keys=agent_keys,
                agent_vals=agent_vals,
                local_frames=local_frames,
                local_keys=local_keys,
                local_vals=local_vals,
                local_active_tokens=local_active_tokens,
                global_frames=global_frames,
                global_keys=global_keys,
                global_vals=global_vals,
                labels=labels,
                active=active,
                P=P,
                N=N,
                L=L,
                KL=KL,
                KG=KG,
                n_frames=n_frames,
                n_block=n_block,
                local_attn=local_attn,
                direct_player=direct_player,
                **membership_kernel_kwargs,
            )
            return self.o(out.type_as(self.o.weight))
        q_agent_frames = start_frame + torch.arange(n_frames * L, device=device) // L
        q_rel_agent_frames = (q_agent_frames - start_frame).long()
        player_outputs = []
        for p_idx in range(P):
            q0 = p_idx * n_frames * L
            player_q = roped_q[:, q0 : q0 + n_frames * L].reshape(
                b, n_frames * L, n_heads * head_dim
            )
            if use_membership:
                q_mem = membership_full[q_agent_frames, :, p_idx]
                q_go = global_only_full[q_agent_frames, p_idx]
            else:
                q_group = labels[q_rel_agent_frames, p_idx]
                q_mem = None
                q_go = None
            keys = [agent_keys[p_idx]]
            vals = [agent_vals[p_idx]]
            masks = [
                self._frame_causal_window_mask(
                    q_agent_frames, agent_frames[p_idx], n_block, local_attn
                )
            ]
            if bool(direct_player):
                other_keys, other_vals, other_masks = ([], [], [])
                for other_p in range(P):
                    if other_p == p_idx:
                        continue
                    ok = agent_k_cur[:, other_p].reshape(
                        b, n_frames * L, n_heads * head_dim
                    )
                    ov = agent_v_cur[:, other_p].reshape(
                        b, n_frames * L, n_heads * head_dim
                    )
                    other_frames = (
                        start_frame + torch.arange(n_frames * L, device=device) // L
                    )
                    if use_membership:
                        other_mem_at_q = membership_full[q_agent_frames, :, other_p]
                        om = (q_mem & other_mem_at_q).any(dim=-1)[:, None]
                    else:
                        other_group_at_q = labels[q_rel_agent_frames[:, None], other_p]
                        om = other_group_at_q == q_group[:, None]
                    om = om & (
                        other_frames[None, :] // n_block
                        == q_agent_frames[:, None] // n_block
                    )
                    other_keys.append(ok)
                    other_vals.append(ov)
                    other_masks.append(om)
                if other_keys:
                    keys.append(torch.cat(other_keys, dim=1))
                    vals.append(torch.cat(other_vals, dim=1))
                    masks.append(torch.cat(other_masks, dim=1))
            if not no_local:
                local_mask = self._frame_causal_window_mask(
                    q_agent_frames, local_all_frames, n_block, local_attn
                )
                if use_membership:
                    q_in_k_local = q_mem[:, local_all_groups]
                    local_mask = local_mask & local_all_active[None, :] & q_in_k_local
                else:
                    local_mask = (
                        local_mask
                        & local_all_active[None, :]
                        & (local_all_groups[None, :] == q_group[:, None])
                    )
                keys.append(local_all_keys)
                vals.append(local_all_vals)
                masks.append(local_mask)
            if no_local and KG > 0:
                gm = self._frame_causal_window_mask(
                    q_agent_frames, global_frames, n_block, local_attn
                )
                keys.append(global_keys)
                vals.append(global_vals)
                masks.append(gm)
            elif use_membership and KG > 0:
                gm = self._frame_causal_window_mask(
                    q_agent_frames, global_frames, n_block, local_attn
                )
                gm = gm & q_go[:, None]
                keys.append(global_keys)
                vals.append(global_vals)
                masks.append(gm)
            attn_k = torch.cat(keys, dim=1)
            attn_v = torch.cat(vals, dim=1)
            mask = torch.cat(masks, dim=1)
            player_outputs.append(
                flash_attention(
                    player_q, attn_k, attn_v, num_heads=self.num_heads, mask=mask
                )
            )
        if no_local:
            local_out = x.new_zeros(b, 0, n_heads * head_dim)
        else:
            local_out = x.new_zeros(b, n_frames, N, KL, n_heads * head_dim)
            q_local_frames = (
                start_frame + torch.arange(n_frames * KL, device=device) // KL
            )
            q_rel_local_frames = (q_local_frames - start_frame).long()
            for g_idx in range(N):
                if not bool(active[:, g_idx].any().item()):
                    continue
                q_local = (
                    roped_q[:, local_start:global_start]
                    .reshape(b, n_frames, N, KL, n_heads, head_dim)[:, :, g_idx]
                    .reshape(b, n_frames * KL, n_heads * head_dim)
                )
                keys, vals, masks = ([], [], [])
                q_active = active[q_rel_local_frames, g_idx]
                for p_idx in range(P):
                    pm = self._frame_causal_window_mask(
                        q_local_frames, agent_frames[p_idx], n_block, local_attn
                    )
                    if use_membership:
                        in_slot = membership_full[q_local_frames, g_idx, p_idx]
                        pm = pm & q_active[:, None] & in_slot[:, None]
                    else:
                        pm = (
                            pm
                            & q_active[:, None]
                            & (labels[q_rel_local_frames, p_idx][:, None] == g_idx)
                        )
                    keys.append(agent_keys[p_idx])
                    vals.append(agent_vals[p_idx])
                    masks.append(pm)
                lm = self._frame_causal_window_mask(
                    q_local_frames, local_frames[g_idx], n_block, local_attn
                )
                lm = lm & q_active[:, None] & local_active_tokens[g_idx][None, :]
                keys.append(local_keys[g_idx])
                vals.append(local_vals[g_idx])
                masks.append(lm)
                if KG > 0:
                    gm = self._frame_causal_window_mask(
                        q_local_frames, global_frames, n_block, local_attn
                    )
                    gm = gm & q_active[:, None]
                    keys.append(global_keys)
                    vals.append(global_vals)
                    masks.append(gm)
                attn_k = torch.cat(keys, dim=1)
                attn_v = torch.cat(vals, dim=1)
                mask = torch.cat(masks, dim=1)
                out_g = flash_attention(
                    q_local, attn_k, attn_v, num_heads=self.num_heads, mask=mask
                )
                local_out[:, :, g_idx] = out_g.reshape(
                    b, n_frames, KL, n_heads * head_dim
                )
            local_out = local_out.reshape(b, local_len, n_heads * head_dim)
        if KG > 0:
            q_global_frames = (
                start_frame + torch.arange(n_frames * KG, device=device) // KG
            )
            global_q = roped_q[:, global_start:].reshape(
                b, n_frames * KG, n_heads * head_dim
            )
            keys, vals, masks = ([], [], [])
            if not no_local:
                local_mask = self._frame_causal_window_mask(
                    q_global_frames, local_all_frames, n_block, local_attn
                )
                keys.append(local_all_keys)
                vals.append(local_all_vals)
                masks.append(local_mask & local_all_active[None, :])
            gm = self._frame_causal_window_mask(
                q_global_frames, global_frames, n_block, local_attn
            )
            keys.append(global_keys)
            vals.append(global_vals)
            masks.append(gm)
            if no_local:
                for p_idx in range(P):
                    pm = self._frame_causal_window_mask(
                        q_global_frames, agent_frames[p_idx], n_block, local_attn
                    )
                    keys.append(agent_keys[p_idx])
                    vals.append(agent_vals[p_idx])
                    masks.append(pm)
            elif use_membership:
                for p_idx in range(P):
                    kf = agent_frames[p_idx]
                    go_k = global_only_full[kf.clamp(0, T_full - 1), p_idx]
                    pm = self._frame_causal_window_mask(
                        q_global_frames, agent_frames[p_idx], n_block, local_attn
                    )
                    pm = pm & go_k[None, :]
                    keys.append(agent_keys[p_idx])
                    vals.append(agent_vals[p_idx])
                    masks.append(pm)
            global_out = flash_attention(
                global_q,
                torch.cat(keys, dim=1),
                torch.cat(vals, dim=1),
                num_heads=self.num_heads,
                mask=torch.cat(masks, dim=1),
            )
        else:
            global_out = x.new_zeros(b, 0, n_heads * head_dim)
        if os.environ.get("WAN_HIER_CAUSAL_DEBUG") == "1":
            print(
                f"[hier_causal] start={start_frame} n={n_frames} P={P} L={L} N={N} KL={KL} KG={KG} active={active[0].to(torch.int).tolist()}",
                flush=True,
            )
        out = torch.cat(player_outputs + [local_out, global_out], dim=1).type_as(
            self.o.weight
        )
        return self.o(out)

    def causal_forward(
        self,
        x,
        freqs,
        mask=None,
        grid_sizes=None,
        seq_lens=None,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        num_frames=None,
        num_players=None,
        num_hub_tokens=None,
        cam_tokens=None,
        group_labels=None,
        active_group=None,
        direct_player=False,
        membership=None,
        global_only=None,
    ):
        if kv_cache is not None and kv_cache.get("hier_hub", False):
            return self.causal_forward_hier_hub(
                x,
                freqs,
                kv_cache,
                current_start=current_start,
                num_frames=num_frames,
                num_players=num_players,
                num_hub_tokens=num_hub_tokens,
                cam_tokens=cam_tokens,
                group_labels=group_labels,
                active_group=active_group,
                direct_player=direct_player,
                membership=membership,
                global_only=global_only,
            )
        raise ValueError("Hierarchical causal attention requires a hier KV cache")


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-06,
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask=None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v, mask)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y_img = flash_attention(
                q, k_img, v_img, num_heads=self.num_heads, mask=mask
            )
            x = x + y_img
        return self.o(x)
