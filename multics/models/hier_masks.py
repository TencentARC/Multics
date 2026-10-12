_HIER_LOCAL_HUB_MODES = ("shared", "per_group")
import torch
import math
import copy
from typing import Optional

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
        BlockMask,
    )

    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = ()
_FLEX_ATTENTION_COMPILED = None
_FLEX_BLOCKMASK_CACHE = {}


def _prepare_hier_group_tensors(
    group_labels: torch.Tensor,
    active_group: Optional[torch.Tensor],
    num_players: int,
    num_frames: int,
    num_groups: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize hierarchical grouping tensors.

    group_labels: [T,P] or [1,T,P], values in [0,N).
    active_group: optional [T,N] or [1,T,N].
    """
    if group_labels is None:
        raise ValueError("hier_hub requires group_labels [T,P] or [1,T,P]")
    labels = group_labels
    if labels.dim() == 3:
        if labels.shape[0] != 1:
            raise ValueError(
                f"hier_hub expects group_labels batch B=1, got {tuple(labels.shape)}"
            )
        labels = labels[0]
    if labels.shape != (num_frames, num_players):
        raise ValueError(
            f"group_labels must be [T={num_frames},P={num_players}], got {tuple(labels.shape)}"
        )
    labels = labels.to(device=device, dtype=torch.long).contiguous()
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= num_groups
    ):
        raise ValueError(
            f"group_labels values must be in [0,{num_groups}), got min={int(labels.min().item())} max={int(labels.max().item())}"
        )
    if active_group is None:
        active = torch.zeros(num_frames, num_groups, device=device, dtype=torch.bool)
        active.scatter_(1, labels, True)
    else:
        active = active_group
        if active.dim() == 3:
            if active.shape[0] != 1:
                raise ValueError(
                    f"hier_hub expects active_group batch B=1, got {tuple(active.shape)}"
                )
            active = active[0]
        if active.shape != (num_frames, num_groups):
            raise ValueError(
                f"active_group must be [T={num_frames},N={num_groups}], got {tuple(active.shape)}"
            )
        active = active.to(device=device, dtype=torch.bool).contiguous()
    return (labels, active)


def _squeeze_hier_batch(
    tensor: torch.Tensor, expected_tail_ndim: int, name: str
) -> torch.Tensor:
    """Drop an optional leading B=1 dim.

    expected_tail_ndim is the rank without batch:
      labels / global_only / active_group: 2  -> [T, P] or [T, N]
      membership: 3                          -> [T, N, P]
    """
    if tensor.dim() == expected_tail_ndim + 1:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} batch B must be 1, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.dim() != expected_tail_ndim:
        raise ValueError(
            f"{name} must have rank {expected_tail_ndim} or {expected_tail_ndim + 1} with B=1, got {tuple(tensor.shape)}"
        )
    return tensor


def _debug_labels_from_membership(membership: torch.Tensor) -> torch.Tensor:
    """Argmax group id for debug only. membership [T,N,P] -> labels [T,P], -1 if no local group."""
    any_g = membership.any(dim=1)
    labels = membership.to(dtype=torch.long).argmax(dim=1)
    return torch.where(any_g, labels, torch.full_like(labels, -1))


def _slice_hier_active_group_for_cached_mask(
    active_group: Optional[torch.Tensor], start_frame: int, num_frames: int
) -> Optional[torch.Tensor]:
    """Return the current-block active slots used by the cached mask builder.

    Cached membership inference keeps the full ``[T, N]`` tensor for causal
    self-attention, which needs absolute key-frame metadata.  The auxiliary
    full-sequence mask builder inside a DiT block only covers the current
    ``[n, N]`` block, so exposing the full tensor there is a shape error.
    """
    if active_group is None:
        return None
    active = _squeeze_hier_batch(active_group, 2, "active_group")
    start = int(start_frame)
    n = int(num_frames)
    if start < 0 or n <= 0:
        raise ValueError(f"invalid cached active_group slice: start={start}, n={n}")
    if active.shape[0] == n:
        return active.contiguous()
    if start + n > active.shape[0]:
        raise ValueError(
            f"active_group T={active.shape[0]} does not cover cached block [{start},{start + n})"
        )
    return active[start : start + n].contiguous()


def _prepare_hier_membership_tensors(
    membership: torch.Tensor,
    global_only: Optional[torch.Tensor],
    active_group: Optional[torch.Tensor],
    num_players: int,
    num_frames: Optional[int],
    num_groups: int,
    device,
    *,
    allow_full_t: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize flexible-grouping tensors.

    Returns:
      membership:  [T, N, P] bool   membership[t, g, p] ↔ player p in slot g at t
      global_only: [T, P]    bool   has frames, in no local group, only talks to global
      active:      [T, N]    bool   slot g has any member at t  (membership.any(player))

    T equals num_frames unless allow_full_t (cached inference passes the full clip
    so key-time global_only can be read on earlier player frames).
    """
    if membership is None:
        raise ValueError(
            "hier_group_format=membership requires membership [T,N,P] or [1,T,N,P]"
        )
    mem = _squeeze_hier_batch(membership, 3, "membership")
    T, N, P = (int(mem.shape[0]), int(mem.shape[1]), int(mem.shape[2]))
    if P != num_players:
        raise ValueError(
            f"membership P={P} must equal num_players={num_players}, got {tuple(mem.shape)}"
        )
    if N != num_groups:
        raise ValueError(
            f"membership N={N} must equal num_groups={num_groups}, got {tuple(mem.shape)}"
        )
    if num_frames is not None:
        if allow_full_t:
            if T < int(num_frames):
                raise ValueError(
                    f"membership T={T} shorter than current block n={int(num_frames)}"
                )
        elif T != int(num_frames):
            raise ValueError(
                f"membership must be [T={int(num_frames)},N={num_groups},P={num_players}], got {tuple(mem.shape)}"
            )
    mem = mem.to(device=device, dtype=torch.bool).contiguous()
    if global_only is None:
        raise ValueError(
            "hier_group_format=membership requires global_only [T,P] or [1,T,P]"
        )
    go = _squeeze_hier_batch(global_only, 2, "global_only")
    if go.shape != (T, P):
        raise ValueError(f"global_only must be [T={T},P={P}], got {tuple(go.shape)}")
    go = go.to(device=device, dtype=torch.bool).contiguous()
    if active_group is None:
        active = mem.any(dim=-1)
    else:
        active = _squeeze_hier_batch(active_group, 2, "active_group")
        if allow_full_t and active.shape != (T, N):
            active = mem.any(dim=-1).contiguous()
        elif active.shape != (T, N):
            raise ValueError(
                f"active_group must be [T={T},N={N}], got {tuple(active.shape)}"
            )
        else:
            active = active.to(device=device, dtype=torch.bool).contiguous()
    return (mem, go, active)


def _refuse_membership_on_scalar_kernel(
    *, attn_impl: Optional[str] = None, infer_kernel: Optional[str] = None
):
    """Phase-1 fail-fast: tiled/segmented kernels still index scalar labels[t][p]."""
    if attn_impl == "hier_tiled_ste":
        raise ValueError(
            "hier_group_format=membership is not supported with attn_impl=hier_tiled_ste; use --attn_impl flex or sdpa until the kernel grows a membership lookup"
        )
    if infer_kernel in ("segmented", "tiled128"):
        raise ValueError(
            f"hier_group_format=membership is not supported with WAN_HIER_INFER_KERNEL={infer_kernel}; use sdpa (or full_sequence flex)"
        )


def _validate_hier_no_local_topology(no_local: bool, KL: int, KG: int) -> None:
    if bool(no_local) and (int(KL) != 0 or int(KG) <= 0):
        raise ValueError(
            f"hier_no_local_hub requires K_L=0 and K_G>0; got K_L={KL}, K_G={KG}"
        )


def normalize_hier_local_hub_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in _HIER_LOCAL_HUB_MODES:
        raise ValueError(
            f"hier_local_hub_mode must be shared or per_group; got {mode!r}"
        )
    return mode


def expand_hier_local_hub_seed(
    local_seed: torch.Tensor,
    *,
    num_frames: int,
    num_groups: int,
    num_local_hub_tokens: int,
    model_dim: int,
    mode: str,
) -> torch.Tensor:
    """Expand a local-hub seed into frame-major ``[1,F*N*K_L,D]`` tokens.

    The shared branch intentionally retains the original indexing/expand
    expression so the default path keeps exactly the previous semantics.
    """
    mode = normalize_hier_local_hub_mode(mode)
    num_frames = int(num_frames)
    num_groups = int(num_groups)
    num_local_hub_tokens = int(num_local_hub_tokens)
    model_dim = int(model_dim)
    if mode == "shared":
        expected = (num_local_hub_tokens, model_dim)
        if tuple(local_seed.shape) != expected:
            raise ValueError(
                f"shared local_hub_embed must have shape {expected}, got {tuple(local_seed.shape)}"
            )
        expanded = local_seed[None, None, None, :, :].expand(
            1, num_frames, num_groups, num_local_hub_tokens, model_dim
        )
    else:
        expected = (num_groups, num_local_hub_tokens, model_dim)
        if tuple(local_seed.shape) != expected:
            raise ValueError(
                f"per_group local_hub_embed must have shape {expected}, got {tuple(local_seed.shape)}"
            )
        expanded = local_seed[None, None, :, :, :].expand(
            1, num_frames, num_groups, num_local_hub_tokens, model_dim
        )
    return expanded.reshape(
        1, num_frames * num_groups * num_local_hub_tokens, model_dim
    )


def prepare_hier_local_hub_state_dict_for_load(
    state_dict: dict,
    *,
    target_mode: str,
    target_num_groups: int,
    target_num_local_hub_tokens: int,
    model_dim: int,
):
    """Validate/migrate ``local_hub_embed`` for a target sharing mode.

    A legacy/shared seed may be copied into every per-group slot.  The reverse
    direction is deliberately rejected because silently reducing independently
    trained group parameters would change the checkpoint's learned function.
    Checkpoints without a local seed pass through for the existing base-model
    warm-start path, whose missing-key handling initializes the seed from zero.
    """
    target_mode = normalize_hier_local_hub_mode(target_mode)
    migrated = copy.copy(state_dict)
    local_seed = migrated.get("local_hub_embed")
    if local_seed is None:
        return (migrated, None)
    if not torch.is_tensor(local_seed):
        raise RuntimeError(
            f"local_hub_embed in checkpoint must be a tensor; got {type(local_seed).__name__}"
        )
    shared_shape = (int(target_num_local_hub_tokens), int(model_dim))
    per_group_shape = (
        int(target_num_groups),
        int(target_num_local_hub_tokens),
        int(model_dim),
    )
    source_shape = tuple(local_seed.shape)
    expected = shared_shape if target_mode == "shared" else per_group_shape
    if source_shape == expected:
        return (migrated, None)
    if target_mode == "per_group" and source_shape == shared_shape:
        migrated["local_hub_embed"] = (
            local_seed.unsqueeze(0).expand(per_group_shape).clone()
        )
        return (
            migrated,
            {
                "source_mode": "shared",
                "source_shape": source_shape,
                "target_mode": target_mode,
                "target_shape": per_group_shape,
            },
        )
    if target_mode == "shared" and source_shape == per_group_shape:
        raise RuntimeError(
            f"Cannot load a per_group local_hub_embed into shared mode: checkpoint shape={source_shape}, target shape={shared_shape}. Use hier_local_hub_mode=per_group; automatic group reduction is intentionally disabled."
        )
    raise RuntimeError(
        f"Incompatible local_hub_embed shape: checkpoint={source_shape}, target mode={target_mode}, expected={expected}; the only supported migration is shared {shared_shape} -> per_group {per_group_shape}."
    )


def migrate_hier_no_local_hub_state_dict(
    state_dict: dict, *, target_num_global_hub_tokens: int, model_dim: int
):
    """Return a training-only warm-start state for the no-local topology.

    The old local seed is preferred because it was the player-facing hub.  A
    state already saved in the new topology passes through unchanged.  This
    helper intentionally does not migrate optimizer state.
    """
    migrated = dict(state_dict)
    target_shape = (int(target_num_global_hub_tokens), int(model_dim))
    old_local = migrated.pop("local_hub_embed", None)
    old_global = migrated.get("global_hub_embed")
    if old_local is None:
        if torch.is_tensor(old_global) and tuple(old_global.shape) == target_shape:
            return (migrated, None)
        raise RuntimeError(
            f"No-local training warm-start requires either an old local_hub_embed with at least {target_shape[0]} slots or an already migrated global_hub_embed of shape {target_shape}; got global={(None if old_global is None else tuple(old_global.shape))}"
        )
    if not torch.is_tensor(old_local) or old_local.dim() != 2:
        raise RuntimeError(
            "local_hub_embed must be a rank-2 tensor for no-local migration"
        )
    if old_local.shape[1] != target_shape[1] or old_local.shape[0] < target_shape[0]:
        raise RuntimeError(
            f"Cannot migrate local_hub_embed to no-local global seed: source={tuple(old_local.shape)}, target={target_shape}"
        )
    migrated["global_hub_embed"] = old_local[: target_shape[0]].clone()
    report = {
        "source": "local_hub_embed",
        "source_shape": tuple(old_local.shape),
        "replaced_global_shape": None
        if old_global is None
        else tuple(old_global.shape),
        "target_shape": target_shape,
    }
    return (migrated, report)


def build_stage2_hier_self_attn_mask(
    num_players: int,
    num_frames: int,
    num_patches: int,
    num_groups: int,
    num_local_hub_tokens: int,
    num_global_hub_tokens: int,
    num_frame_per_block: int,
    local_attn_size: int,
    group_labels: torch.Tensor,
    active_group: Optional[torch.Tensor] = None,
    direct_player: bool = False,
    device=None,
    membership: Optional[torch.Tensor] = None,
    global_only: Optional[torch.Tensor] = None,
    hier_no_local_hub: bool = False,
) -> torch.Tensor:
    """Hierarchical attention mask for [P*T*L || T*N*K_L || T*K_G].

    Base hierarchical edges use the same block-causal local temporal window as
    centralized sparse_hub. Cross-player direct edges remain optional and
    same-block only. Local hub slots never attend other local hub slots, but each
    active local slot can attend its own temporal history within the same window.

    If membership is provided, use the flexible cover (global-only + Anchor).
    Otherwise keep the original partition formula (labels in [0,N), no player↔global).
    """
    if membership is not None:
        return _build_stage2_hier_self_attn_mask_membership(
            num_players,
            num_frames,
            num_patches,
            num_groups,
            num_local_hub_tokens,
            num_global_hub_tokens,
            num_frame_per_block,
            local_attn_size,
            membership,
            global_only,
            active_group,
            direct_player,
            device,
            hier_no_local_hub,
        )
    P, T, L = (int(num_players), int(num_frames), int(num_patches))
    N, KL, KG = (int(num_groups), int(num_local_hub_tokens), int(num_global_hub_tokens))
    _validate_hier_no_local_topology(hier_no_local_hub, KL, KG)
    if local_attn_size != -1 and local_attn_size < num_frame_per_block:
        raise ValueError(
            f"hier_hub requires local_attn_size=-1 or >= num_frame_per_block; got local_attn_size={local_attn_size}, n={num_frame_per_block}"
        )
    labels, active = _prepare_hier_group_tensors(
        group_labels, active_group, P, T, N, device
    )
    agent_len = P * T * L
    local_len = T * N * KL
    global_len = T * KG
    seq_len = agent_len + local_len + global_len
    idx = torch.arange(seq_len, device=device)
    local_start = agent_len
    global_start = agent_len + local_len
    is_player = idx < local_start
    is_local = (idx >= local_start) & (idx < global_start)
    is_global = idx >= global_start
    t_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    p_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    g_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    a = idx[is_player]
    p_id[is_player] = a // (T * L)
    t_id[is_player] = a % (T * L) // L
    if KL > 0:
        loc = idx[is_local] - local_start
        t_id[is_local] = loc // (N * KL)
        g_id[is_local] = loc % (N * KL) // KL
    glob = idx[is_global] - global_start
    if KG > 0:
        t_id[is_global] = glob // KG
    qt = t_id[:, None]
    kt = t_id[None, :]
    qp = p_id[:, None]
    kp = p_id[None, :]
    qg = g_id[:, None]
    kg_id = g_id[None, :]
    q_player = is_player[:, None]
    k_player = is_player[None, :]
    q_local = is_local[:, None]
    k_local = is_local[None, :]
    q_global = is_global[:, None]
    k_global = is_global[None, :]
    q_block = qt // num_frame_per_block
    k_block = kt // num_frame_per_block
    m_window = k_block <= q_block
    if local_attn_size != -1:
        m_window = m_window & (qt - kt < local_attn_size)
    m_same_block = q_block == k_block
    q_player_group = labels[t_id, p_id][:, None]
    k_player_group_at_qt = labels[t_id[:, None], p_id[None, :]]
    active_q_local = active[t_id, g_id][:, None]
    active_k_local = active[t_id, g_id][None, :]
    same_player_hist = q_player & k_player & (qp == kp) & m_window
    if bool(direct_player):
        direct_same_group = (
            q_player
            & k_player
            & (qp != kp)
            & (q_player_group == k_player_group_at_qt)
            & m_same_block
        )
    else:
        direct_same_group = torch.zeros_like(same_player_hist)
    player_to_local = (
        q_player & k_local & active_k_local & (kg_id == q_player_group) & m_window
    )
    local_to_player = (
        q_local & active_q_local & k_player & (k_player_group_at_qt == qg) & m_window
    )
    local_to_global = q_local & active_q_local & k_global & m_window
    local_internal = (
        q_local & active_q_local & k_local & active_k_local & (qg == kg_id) & m_window
    )
    inactive_local_self = (
        q_local & ~active_q_local & k_local & (idx[:, None] == idx[None, :])
    )
    global_to_local = q_global & k_local & active_k_local & m_window
    global_internal = q_global & k_global & m_window
    if bool(hier_no_local_hub):
        player_to_global = q_player & k_global & m_window
        global_to_player = q_global & k_player & m_window
    else:
        player_to_global = torch.zeros_like(same_player_hist)
        global_to_player = torch.zeros_like(same_player_hist)
    return (
        same_player_hist
        | direct_same_group
        | player_to_local
        | local_to_player
        | local_to_global
        | local_internal
        | inactive_local_self
        | global_to_local
        | global_internal
        | player_to_global
        | global_to_player
    )


def _build_stage2_hier_self_attn_mask_membership(
    num_players: int,
    num_frames: int,
    num_patches: int,
    num_groups: int,
    num_local_hub_tokens: int,
    num_global_hub_tokens: int,
    num_frame_per_block: int,
    local_attn_size: int,
    membership: torch.Tensor,
    global_only: Optional[torch.Tensor],
    active_group: Optional[torch.Tensor],
    direct_player: bool,
    device,
    hier_no_local_hub: bool = False,
) -> torch.Tensor:
    """Dense [S,S] oracle for hier_group_format=membership.

    Token layout (same as labels path): [P*T*L | T*N*K_L | T*K_G].

    New / changed edges vs the partition mask:
      player → local g:   membership[q_t, g, p]  (Anchor hits every owned slot)
      player ↔ player:    share ≥1 group at query time; still same-block + direct
      player → global:    only if global_only[q_t, p]
      global → player:    only if global_only[k_t, p]
      empty local slots:  no local↔global / player↔local; inactive query = self-edge
    """
    P, T, L = (int(num_players), int(num_frames), int(num_patches))
    N, KL, KG = (int(num_groups), int(num_local_hub_tokens), int(num_global_hub_tokens))
    _validate_hier_no_local_topology(hier_no_local_hub, KL, KG)
    if local_attn_size != -1 and local_attn_size < num_frame_per_block:
        raise ValueError(
            f"hier_hub requires local_attn_size=-1 or >= num_frame_per_block; got local_attn_size={local_attn_size}, n={num_frame_per_block}"
        )
    if N > 62:
        raise ValueError(f"membership bitset path supports N<=62, got N={N}")
    mem, go, active = _prepare_hier_membership_tensors(
        membership, global_only, active_group, P, T, N, device
    )
    agent_len = P * T * L
    local_len = T * N * KL
    global_len = T * KG
    seq_len = agent_len + local_len + global_len
    idx = torch.arange(seq_len, device=device)
    local_start = agent_len
    global_start = agent_len + local_len
    is_player = idx < local_start
    is_local = (idx >= local_start) & (idx < global_start)
    is_global = idx >= global_start
    t_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    p_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    g_id = torch.zeros(seq_len, dtype=torch.long, device=device)
    a = idx[is_player]
    p_id[is_player] = a // (T * L)
    t_id[is_player] = a % (T * L) // L
    if KL > 0:
        loc = idx[is_local] - local_start
        t_id[is_local] = loc // (N * KL)
        g_id[is_local] = loc % (N * KL) // KL
    glob = idx[is_global] - global_start
    if KG > 0:
        t_id[is_global] = glob // KG
    qt = t_id[:, None]
    kt = t_id[None, :]
    qp = p_id[:, None]
    kp = p_id[None, :]
    qg = g_id[:, None]
    kg_id = g_id[None, :]
    q_player = is_player[:, None]
    k_player = is_player[None, :]
    q_local = is_local[:, None]
    k_local = is_local[None, :]
    q_global = is_global[:, None]
    k_global = is_global[None, :]
    q_block = qt // num_frame_per_block
    k_block = kt // num_frame_per_block
    m_window = k_block <= q_block
    if local_attn_size != -1:
        m_window = m_window & (qt - kt < local_attn_size)
    m_same_block = q_block == k_block
    g_pow = 1 << torch.arange(N, device=device, dtype=torch.long)
    player_group_bits = (mem.to(dtype=torch.int64) * g_pow[None, :, None]).sum(dim=1)
    q_bits = player_group_bits[t_id, p_id]
    k_bits_at_qt = player_group_bits[t_id[:, None], p_id[None, :]]
    same_group = q_bits[:, None] & k_bits_at_qt != 0
    active_q_local = active[t_id, g_id][:, None]
    active_k_local = active[t_id, g_id][None, :]
    q_go = go[t_id, p_id][:, None]
    k_go = go[t_id, p_id][None, :]
    same_player_hist = q_player & k_player & (qp == kp) & m_window
    if bool(direct_player):
        direct_same_group = q_player & k_player & (qp != kp) & same_group & m_same_block
    else:
        direct_same_group = torch.zeros_like(same_player_hist)
    player_to_local = (
        q_player
        & k_local
        & active_k_local
        & (q_bits[:, None] & 1 << kg_id != 0)
        & m_window
    )
    local_to_player = (
        q_local & active_q_local & k_player & (k_bits_at_qt & 1 << qg != 0) & m_window
    )
    local_to_global = q_local & active_q_local & k_global & m_window
    local_internal = (
        q_local & active_q_local & k_local & active_k_local & (qg == kg_id) & m_window
    )
    inactive_local_self = (
        q_local & ~active_q_local & k_local & (idx[:, None] == idx[None, :])
    )
    global_to_local = q_global & k_local & active_k_local & m_window
    global_internal = q_global & k_global & m_window
    if bool(hier_no_local_hub):
        player_to_global = q_player & k_global & m_window
        global_to_player = q_global & k_player & m_window
    else:
        player_to_global = q_player & k_global & q_go & m_window
        global_to_player = q_global & k_player & k_go & m_window
    return (
        same_player_hist
        | direct_same_group
        | player_to_local
        | local_to_player
        | local_to_global
        | local_internal
        | inactive_local_self
        | global_to_local
        | global_internal
        | player_to_global
        | global_to_player
    )


def _put_flex_blockmask_cache(key, value):
    """Cache BlockMasks with bounded retention for per-sample hierarchical masks."""
    _FLEX_BLOCKMASK_CACHE[key] = value
    if not (isinstance(key, tuple) and len(key) > 0 and (key[0] == "hier")):
        return value
    import os

    max_hier = int(os.environ.get("WAN_HIER_BLOCKMASK_CACHE_SIZE", "2"))
    if max_hier < 0:
        return value
    if max_hier == 0:
        _FLEX_BLOCKMASK_CACHE.pop(key, None)
        return value
    hier_keys = [
        k
        for k in _FLEX_BLOCKMASK_CACHE
        if isinstance(k, tuple) and len(k) > 0 and (k[0] == "hier")
    ]
    while len(hier_keys) > max_hier:
        old_key = hier_keys.pop(0)
        if old_key != key:
            _FLEX_BLOCKMASK_CACHE.pop(old_key, None)
    return value


def _flex_kernel_options():
    """Forward kernel options for flex_attention.

    WAN_FLEX_KOPT:
      - unset or "0": return None and let autotune choose.
      - "1": fix BLOCK_M=BLOCK_N=64 so autotune does not try other candidates.
      - "M,N": use those block sizes.
    num_warps and num_stages stay with the template; putting them here
    conflicts with maybe_append_choice.
    """
    import os

    v = os.environ.get("WAN_FLEX_KOPT", "0")
    if v in ("0", ""):
        return None
    if v == "1":
        return {"BLOCK_M": 64, "BLOCK_N": 64}
    m, n = (int(x) for x in v.split(","))
    return {"BLOCK_M": m, "BLOCK_N": n}


def _get_flex_attention():
    """Lazily compile flex_attention once (module-level singleton).

    The default compile mode is max-autotune-no-cudagraphs. The plain default
    mode can raise NoValidChoicesError in the Flex backward on this Torch build.

    WAN_FLEX_MODE:
      - unset or "max-autotune-no-cudagraphs": compile with that mode.
      - "none" or "eager": call flex_attention without torch.compile.
      - any other string: pass it through as the torch.compile mode.
    """
    global _FLEX_ATTENTION_COMPILED
    if _FLEX_ATTENTION_COMPILED is None:
        import os

        mode = os.environ.get("WAN_FLEX_MODE", "max-autotune-no-cudagraphs")
        if mode in ("none", "eager"):
            _FLEX_ATTENTION_COMPILED = _flex_attention
        elif mode in ("default", "default-aten", ""):
            import torch._inductor.config as _ind

            _bk = getattr(_ind, "max_autotune_gemm_backends", "ATEN,TRITON")
            if "ATEN" not in _bk.upper():
                _ind.max_autotune_gemm_backends = "ATEN," + _bk
            _FLEX_ATTENTION_COMPILED = torch.compile(_flex_attention, dynamic=False)
        else:
            _FLEX_ATTENTION_COMPILED = torch.compile(
                _flex_attention, dynamic=False, mode=mode
            )
    return _FLEX_ATTENTION_COMPILED


def build_stage2_hier_flex_block_mask(
    num_players: int,
    num_frames: int,
    num_patches: int,
    num_groups: int,
    num_local_hub_tokens: int,
    num_global_hub_tokens: int,
    num_frame_per_block: int,
    local_attn_size: int,
    group_labels: torch.Tensor,
    active_group: Optional[torch.Tensor],
    direct_player: bool,
    device,
    membership: Optional[torch.Tensor] = None,
    global_only: Optional[torch.Tensor] = None,
    hier_no_local_hub: bool = False,
):
    """Flex BlockMask for build_stage2_hier_self_attn_mask semantics.

    The grouping tensors vary per sample, so the cache key includes their values.
    labels mode keys stay (labels, active). membership mode keys are
    (membership, global_only, active) so Anchor / global-only changes bust the cache.
    """
    if membership is not None:
        return _build_stage2_hier_flex_block_mask_membership(
            num_players,
            num_frames,
            num_patches,
            num_groups,
            num_local_hub_tokens,
            num_global_hub_tokens,
            num_frame_per_block,
            local_attn_size,
            membership,
            global_only,
            active_group,
            direct_player,
            device,
            hier_no_local_hub,
        )
    P, T, L = (int(num_players), int(num_frames), int(num_patches))
    N, KL, KG = (int(num_groups), int(num_local_hub_tokens), int(num_global_hub_tokens))
    _validate_hier_no_local_topology(hier_no_local_hub, KL, KG)
    if local_attn_size != -1 and local_attn_size < num_frame_per_block:
        raise ValueError(
            f"hier_hub requires local_attn_size=-1 or >= num_frame_per_block; got local_attn_size={local_attn_size}, n={num_frame_per_block}"
        )
    labels, active = _prepare_hier_group_tensors(
        group_labels, active_group, P, T, N, device
    )
    agent_len = P * T * L
    local_len = T * N * KL
    global_len = T * KG
    S = agent_len + local_len + global_len
    S_pad = math.ceil(S / 128) * 128
    labels_cpu = labels.detach().cpu()
    active_cpu = active.detach().cpu()
    labels_key = tuple(labels_cpu.reshape(-1).tolist())
    active_key = tuple(active_cpu.reshape(-1).tolist())
    key = (
        "hier",
        P,
        T,
        L,
        N,
        KL,
        KG,
        num_frame_per_block,
        local_attn_size,
        bool(direct_player),
        bool(hier_no_local_hub),
        labels_key,
        active_key,
        S_pad,
        str(device),
    )
    cached = _FLEX_BLOCKMASK_CACHE.get(key)
    if cached is not None:
        return cached
    local_start = agent_len
    global_start = agent_len + local_len
    idx = torch.arange(S_pad, device=device)
    valid = idx < S
    is_player = valid & (idx < local_start)
    is_local = valid & (idx >= local_start) & (idx < global_start)
    is_global = valid & (idx >= global_start)
    t_id = torch.zeros(S_pad, dtype=torch.long, device=device)
    p_id = torch.zeros(S_pad, dtype=torch.long, device=device)
    g_id = torch.zeros(S_pad, dtype=torch.long, device=device)
    a = idx[is_player]
    p_id[is_player] = a // (T * L)
    t_id[is_player] = a % (T * L) // L
    if KL > 0:
        loc = idx[is_local] - local_start
        t_id[is_local] = loc // (N * KL)
        g_id[is_local] = loc % (N * KL) // KL
    glob = idx[is_global] - global_start
    if KG > 0:
        t_id[is_global] = glob // KG
    if P > 62:
        raise ValueError(f"hier_hub flex bitset path supports P<=62, got P={P}")
    group_player_bits = []
    for _t in range(T):
        for _g in range(N):
            _bits = 0
            for _p in range(P):
                if int(labels_cpu[_t, _p].item()) == _g:
                    _bits |= 1 << _p
            group_player_bits.append(_bits)
    group_player_bits = torch.tensor(
        group_player_bits, dtype=torch.long, device=device
    ).contiguous()
    player_group_by_token = torch.zeros(S_pad, dtype=torch.long, device=device)
    if is_player.any():
        player_group_by_token[is_player] = labels[t_id[is_player], p_id[is_player]]
    local_active_by_token = torch.zeros(S_pad, dtype=torch.bool, device=device)
    if is_local.any():
        local_active_by_token[is_local] = active[t_id[is_local], g_id[is_local]]
    n_block = int(num_frame_per_block)
    local_attn = int(local_attn_size)
    direct = bool(direct_player)
    no_local = bool(hier_no_local_hub)

    def mask_mod(b, hd, q, kv):
        q_t = t_id[q]
        k_t = t_id[kv]
        q_block = q_t // n_block
        k_block = k_t // n_block
        m_window = k_block <= q_block
        if local_attn != -1:
            m_window = m_window & (k_t > q_t - local_attn)
        same_block = q_block == k_block
        q_player_group = player_group_by_token[q]
        q_player_group_bits = group_player_bits[q_t * N + q_player_group]
        q_local_group_bits = group_player_bits[q_t * N + g_id[q]]
        k_player_bit = 1 << p_id[kv]
        k_player_in_q_player_group = q_player_group_bits & k_player_bit != 0
        k_player_in_q_local_group = q_local_group_bits & k_player_bit != 0
        q_local_active = local_active_by_token[q]
        k_local_active = local_active_by_token[kv]
        same_player_hist = (
            is_player[q] & is_player[kv] & (p_id[q] == p_id[kv]) & m_window
        )
        if direct:
            direct_same_group = (
                is_player[q]
                & is_player[kv]
                & (p_id[q] != p_id[kv])
                & k_player_in_q_player_group
                & same_block
            )
        else:
            direct_same_group = same_player_hist & False
        player_to_local = (
            is_player[q]
            & is_local[kv]
            & k_local_active
            & (g_id[kv] == q_player_group)
            & m_window
        )
        local_to_player = (
            is_local[q]
            & q_local_active
            & is_player[kv]
            & k_player_in_q_local_group
            & m_window
        )
        local_to_global = is_local[q] & q_local_active & is_global[kv] & m_window
        local_internal = (
            is_local[q]
            & q_local_active
            & is_local[kv]
            & k_local_active
            & (g_id[q] == g_id[kv])
            & m_window
        )
        inactive_local_self = is_local[q] & ~q_local_active & is_local[kv] & (q == kv)
        global_to_local = is_global[q] & is_local[kv] & k_local_active & m_window
        global_internal = is_global[q] & is_global[kv] & m_window
        if no_local:
            player_to_global = is_player[q] & is_global[kv] & m_window
            global_to_player = is_global[q] & is_player[kv] & m_window
        else:
            player_to_global = same_player_hist & False
            global_to_player = same_player_hist & False
        m = (
            same_player_hist
            | direct_same_group
            | player_to_local
            | local_to_player
            | local_to_global
            | local_internal
            | inactive_local_self
            | global_to_local
            | global_internal
            | player_to_global
            | global_to_player
        )
        return valid[q] & valid[kv] & m

    import os

    compile_mask = os.environ.get("WAN_HIER_BLOCKMASK_COMPILE", "1") not in ("0", "")
    block_mask = _create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=S_pad,
        KV_LEN=S_pad,
        device=device,
        _compile=compile_mask,
    )
    return _put_flex_blockmask_cache(key, (block_mask, S, S_pad))


def _build_stage2_hier_flex_block_mask_membership(
    num_players: int,
    num_frames: int,
    num_patches: int,
    num_groups: int,
    num_local_hub_tokens: int,
    num_global_hub_tokens: int,
    num_frame_per_block: int,
    local_attn_size: int,
    membership: torch.Tensor,
    global_only: Optional[torch.Tensor],
    active_group: Optional[torch.Tensor],
    direct_player: bool,
    device,
    hier_no_local_hub: bool = False,
):
    """Flex BlockMask aligned with _build_stage2_hier_self_attn_mask_membership.

    Avoids q+kv compound gathers inside mask_mod (those trip CUDA IMA on
    torch/triton 2.6/3.2). All data-dependent lookups are pre-gathered onto
    per-token 1D tables; mask_mod only indexes those tables.
    """
    P, T, L = (int(num_players), int(num_frames), int(num_patches))
    N, KL, KG = (int(num_groups), int(num_local_hub_tokens), int(num_global_hub_tokens))
    _validate_hier_no_local_topology(hier_no_local_hub, KL, KG)
    if local_attn_size != -1 and local_attn_size < num_frame_per_block:
        raise ValueError(
            f"hier_hub requires local_attn_size=-1 or >= num_frame_per_block; got local_attn_size={local_attn_size}, n={num_frame_per_block}"
        )
    if P > 62:
        raise ValueError(f"hier_hub flex bitset path supports P<=62, got P={P}")
    if N > 62:
        raise ValueError(f"hier_hub flex bitset path supports N<=62, got N={N}")
    mem, go, active = _prepare_hier_membership_tensors(
        membership, global_only, active_group, P, T, N, device
    )
    agent_len = P * T * L
    local_len = T * N * KL
    global_len = T * KG
    S = agent_len + local_len + global_len
    S_pad = math.ceil(S / 128) * 128
    mem_cpu = mem.detach().cpu()
    go_cpu = go.detach().cpu()
    active_cpu = active.detach().cpu()
    mem_key = tuple(mem_cpu.reshape(-1).tolist())
    go_key = tuple(go_cpu.reshape(-1).tolist())
    active_key = tuple(active_cpu.reshape(-1).tolist())
    key = (
        "hier",
        "membership",
        P,
        T,
        L,
        N,
        KL,
        KG,
        num_frame_per_block,
        local_attn_size,
        bool(direct_player),
        bool(hier_no_local_hub),
        mem_key,
        go_key,
        active_key,
        S_pad,
        str(device),
    )
    cached = _FLEX_BLOCKMASK_CACHE.get(key)
    if cached is not None:
        return cached
    local_start = agent_len
    global_start = agent_len + local_len
    idx = torch.arange(S_pad, device=device)
    valid = idx < S
    is_player = valid & (idx < local_start)
    is_local = valid & (idx >= local_start) & (idx < global_start)
    is_global = valid & (idx >= global_start)
    t_id = torch.zeros(S_pad, dtype=torch.long, device=device)
    p_id = torch.zeros(S_pad, dtype=torch.long, device=device)
    g_id = torch.zeros(S_pad, dtype=torch.long, device=device)
    a = idx[is_player]
    p_id[is_player] = a // (T * L)
    t_id[is_player] = a % (T * L) // L
    if KL > 0:
        loc = idx[is_local] - local_start
        t_id[is_local] = loc // (N * KL)
        g_id[is_local] = loc % (N * KL) // KL
    glob = idx[is_global] - global_start
    if KG > 0:
        t_id[is_global] = glob // KG
    group_player_bits = []
    for _t in range(T):
        for _g in range(N):
            _bits = 0
            for _p in range(P):
                if bool(mem_cpu[_t, _g, _p].item()):
                    _bits |= 1 << _p
            group_player_bits.append(_bits)
    group_player_bits = torch.tensor(
        group_player_bits, dtype=torch.long, device=device
    ).contiguous()
    g_pow = 1 << torch.arange(N, device=device, dtype=torch.long)
    player_group_bits = (mem.to(dtype=torch.int64) * g_pow[None, :, None]).sum(dim=1)
    union_player_bits = torch.zeros(T, P, dtype=torch.long, device=device)
    group_bits_tn = group_player_bits.view(T, N)
    for _g in range(N):
        union_player_bits = (
            union_player_bits
            | mem[:, _g, :].to(dtype=torch.long) * group_bits_tn[:, _g : _g + 1]
        )
    player_group_bits_by_token = torch.zeros(S_pad, dtype=torch.long, device=device)
    union_bits_by_token = torch.zeros(S_pad, dtype=torch.long, device=device)
    go_by_token = torch.zeros(S_pad, dtype=torch.bool, device=device)
    if is_player.any():
        player_group_bits_by_token[is_player] = player_group_bits[
            t_id[is_player], p_id[is_player]
        ]
        union_bits_by_token[is_player] = union_player_bits[
            t_id[is_player], p_id[is_player]
        ]
        go_by_token[is_player] = go[t_id[is_player], p_id[is_player]]
    local_active_by_token = torch.zeros(S_pad, dtype=torch.bool, device=device)
    if is_local.any():
        local_active_by_token[is_local] = active[t_id[is_local], g_id[is_local]]
    n_block = int(num_frame_per_block)
    local_attn = int(local_attn_size)
    direct = bool(direct_player)
    no_local = bool(hier_no_local_hub)

    def mask_mod(b, hd, q, kv):
        q_t = t_id[q]
        k_t = t_id[kv]
        q_block = q_t // n_block
        k_block = k_t // n_block
        m_window = k_block <= q_block
        if local_attn != -1:
            m_window = m_window & (k_t > q_t - local_attn)
        same_block = q_block == k_block
        q_local_group_bits = group_player_bits[q_t * N + g_id[q]]
        k_player_bit = 1 << p_id[kv]
        k_player_in_q_local_group = q_local_group_bits & k_player_bit != 0
        q_local_active = local_active_by_token[q]
        k_local_active = local_active_by_token[kv]
        q_in_k_local = player_group_bits_by_token[q] & 1 << g_id[kv] != 0
        k_shares_q_groups = union_bits_by_token[q] & k_player_bit != 0
        same_player_hist = (
            is_player[q] & is_player[kv] & (p_id[q] == p_id[kv]) & m_window
        )
        if direct:
            direct_same_group = (
                is_player[q]
                & is_player[kv]
                & (p_id[q] != p_id[kv])
                & k_shares_q_groups
                & same_block
            )
        else:
            direct_same_group = same_player_hist & False
        player_to_local = (
            is_player[q] & is_local[kv] & k_local_active & q_in_k_local & m_window
        )
        local_to_player = (
            is_local[q]
            & q_local_active
            & is_player[kv]
            & k_player_in_q_local_group
            & m_window
        )
        local_to_global = is_local[q] & q_local_active & is_global[kv] & m_window
        local_internal = (
            is_local[q]
            & q_local_active
            & is_local[kv]
            & k_local_active
            & (g_id[q] == g_id[kv])
            & m_window
        )
        inactive_local_self = is_local[q] & ~q_local_active & is_local[kv] & (q == kv)
        global_to_local = is_global[q] & is_local[kv] & k_local_active & m_window
        global_internal = is_global[q] & is_global[kv] & m_window
        if no_local:
            player_to_global = is_player[q] & is_global[kv] & m_window
            global_to_player = is_global[q] & is_player[kv] & m_window
        else:
            player_to_global = is_player[q] & is_global[kv] & go_by_token[q] & m_window
            global_to_player = is_global[q] & is_player[kv] & go_by_token[kv] & m_window
        m = (
            same_player_hist
            | direct_same_group
            | player_to_local
            | local_to_player
            | local_to_global
            | local_internal
            | inactive_local_self
            | global_to_local
            | global_internal
            | player_to_global
            | global_to_player
        )
        return valid[q] & valid[kv] & m

    import os

    compile_mask = os.environ.get("WAN_HIER_BLOCKMASK_COMPILE", "1") not in ("0", "")
    block_mask = _create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=S_pad,
        KV_LEN=S_pad,
        device=device,
        _compile=compile_mask,
    )
    return _put_flex_blockmask_cache(key, (block_mask, S, S_pad))


def get_temporal_block_causal_mask(
    num_frames: int, num_frame_per_block: int, local_attn_size: int, device=None
) -> torch.Tensor:
    """Per-frame block-causal mask for action branches."""
    t = torch.arange(num_frames, device=device)
    ti = t[:, None]
    tj = t[None, :]
    bi = ti // num_frame_per_block
    bj = tj // num_frame_per_block
    m_causal = bj <= bi
    if local_attn_size == -1:
        m_window = torch.ones_like(m_causal)
    else:
        m_window = ti - tj < local_attn_size
    return m_causal & m_window
