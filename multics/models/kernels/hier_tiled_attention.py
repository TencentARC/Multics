"""Layout-preserving Tiled128 attention for hierarchical Stage-2 training.

The CUDA kernel is forward-only.  Training combines its forward value with the
existing FlexAttention Jacobian using an STE in ``SelfAttention``.  Metadata is
built from the full training sequence, but reproduces the rolling-cache slot
order used by hierarchical inference for every temporal block:

* player query: own cache, current-block direct players, all local caches;
* local query: all player caches, its own local cache, global cache;
* global query: all local caches, global cache.

Each dense slot list is split into 128-token chunks.  Chunks with no logical
edge are removed; retained chunks keep both their original slot order and a
per-slot validity bit.  This is intentionally different from compact segmented
attention, which packs only valid keys and therefore has a different floating
point reduction order.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from typing import Sequence

import torch
import triton
import triton.language as tl


BLOCK_M = 16
BLOCK_N = 128
_LAYOUT_CACHE: OrderedDict[tuple, tuple["HierTiledBlockLayout", ...]] = OrderedDict()


@dataclass(frozen=True)
class HierTiledBlockLayout:
    """Device metadata for one temporal-block launch."""

    prog_qbase: torch.Tensor
    prog_nq: torch.Tensor
    prog_seg: torch.Tensor
    kv_index: torch.Tensor
    kv_valid: torch.Tensor
    kv_count: torch.Tensor
    logical_slots: int
    physical_slots: int
    dense_slots: int


@triton.jit
def _tiled128_fwd(
    Q,
    K,
    V,
    O,
    PROG_QBASE,
    PROG_NQ,
    PROG_SEG,
    KV_INDEX,
    KV_VALID,
    KV_COUNT,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_oh: tl.constexpr,
    scale: tl.constexpr,
    MAX_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M_: tl.constexpr,
    BLOCK_N_: tl.constexpr,
):
    """One online softmax over every retained tile of a query segment."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    qbase = tl.load(PROG_QBASE + pid_m)
    nq = tl.load(PROG_NQ + pid_m)
    seg = tl.load(PROG_SEG + pid_m)
    nk = tl.load(KV_COUNT + seg)

    offs_m = tl.arange(0, BLOCK_M_)
    offs_n = tl.arange(0, BLOCK_N_)
    offs_d = tl.arange(0, HEAD_DIM)
    qmask = offs_m < nq
    q = tl.load(
        Q + (qbase + offs_m[:, None]) * stride_qs + pid_h * stride_qh + offs_d[None, :],
        mask=qmask[:, None],
        other=0.0,
    )

    running_max = tl.full((BLOCK_M_,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M_,), tl.float32)
    acc = tl.zeros((BLOCK_M_, HEAD_DIM), tl.float32)

    for k0 in tl.range(0, MAX_K, BLOCK_N_):
        slot = k0 + offs_n
        present = slot < nk
        kidx = tl.load(KV_INDEX + seg * MAX_K + slot, mask=present, other=0)
        logical = tl.load(KV_VALID + seg * MAX_K + slot, mask=present, other=0).to(
            tl.int1
        )
        kval = tl.load(
            K + kidx[:, None] * stride_ks + pid_h * stride_kh + offs_d[None, :],
            mask=present[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(kval)) * scale
        valid = qmask[:, None] & present[None, :] & logical[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        # Inactive local-query segments can have nk=0.  Mask alpha explicitly
        # so the all-invalid -inf - -inf case never propagates a NaN.
        block_has_value = block_max != -float("inf")
        alpha = tl.where(
            block_has_value,
            tl.exp2((running_max - new_max) * 1.4426950408889634),
            1.0,
        )
        probs = tl.where(
            valid,
            tl.exp2((scores - new_max[:, None]) * 1.4426950408889634),
            0.0,
        )
        vval = tl.load(
            V + kidx[:, None] * stride_vs + pid_h * stride_vh + offs_d[None, :],
            mask=present[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.bfloat16), vval)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = tl.where(block_has_value, new_max, running_max)

    has_output = qmask & (running_sum > 0)
    safe_sum = tl.where(has_output, running_sum, 1.0)
    out = tl.where(has_output[:, None], acc / safe_sum[:, None], 0.0)
    tl.store(
        O + (qbase + offs_m[:, None]) * stride_os + pid_h * stride_oh + offs_d[None, :],
        out,
        mask=qmask[:, None],
    )


def _temporal(qf: int, kf: int, n_block: int, local_attn: int) -> bool:
    allowed = (kf // n_block) <= (qf // n_block)
    if local_attn != -1:
        allowed = allowed and (qf - kf) < local_attn
    return allowed


def _as_cpu_rows(
    tensor: torch.Tensor, dtype: torch.dtype
) -> tuple[tuple[int | bool, ...], ...]:
    cpu = tensor.detach().to(device="cpu", dtype=dtype)
    if dtype == torch.bool:
        return tuple(tuple(bool(x) for x in row) for row in cpu.tolist())
    return tuple(tuple(int(x) for x in row) for row in cpu.tolist())


def _cache_get(key: tuple):
    value = _LAYOUT_CACHE.get(key)
    if value is not None:
        _LAYOUT_CACHE.move_to_end(key)
    return value


def _cache_put(key: tuple, value: tuple[HierTiledBlockLayout, ...]) -> None:
    max_entries = max(int(os.environ.get("WAN_HIER_TILED_LAYOUT_CACHE_SIZE", "2")), 0)
    if max_entries == 0:
        return
    _LAYOUT_CACHE[key] = value
    _LAYOUT_CACHE.move_to_end(key)
    while len(_LAYOUT_CACHE) > max_entries:
        _LAYOUT_CACHE.popitem(last=False)


def _pack_layout(
    *,
    device: torch.device,
    dense_segments: Sequence[Sequence[tuple[int, bool]]],
    query_specs: Sequence[tuple[int, int]],
) -> HierTiledBlockLayout:
    if len(dense_segments) != len(query_specs):
        raise ValueError("dense segment/query-spec length mismatch")

    selected_segments: list[list[tuple[int, bool]]] = []
    logical_slots = physical_slots = dense_slots = 0
    for dense_seq in dense_segments:
        dense = list(dense_seq)
        selected: list[tuple[int, bool]] = []
        dense_slots += len(dense)
        logical_slots += sum(int(valid) for _, valid in dense)
        for start in range(0, len(dense), BLOCK_N):
            chunk = dense[start : start + BLOCK_N]
            if not any(valid for _, valid in chunk):
                continue
            chunk.extend([(0, False)] * (BLOCK_N - len(chunk)))
            selected.extend(chunk)
        physical_slots += len(selected)
        selected_segments.append(selected)

    max_k = max(max((len(x) for x in selected_segments), default=1), 1)
    max_k = math.ceil(max_k / BLOCK_N) * BLOCK_N
    kv_index = torch.zeros((len(selected_segments), max_k), dtype=torch.int32)
    kv_valid = torch.zeros((len(selected_segments), max_k), dtype=torch.uint8)
    kv_count = torch.zeros((len(selected_segments),), dtype=torch.int32)
    prog_qbase: list[int] = []
    prog_nq: list[int] = []
    prog_seg: list[int] = []
    for sid, ((qstart, qlen), selected) in enumerate(
        zip(query_specs, selected_segments)
    ):
        if selected:
            kv_index[sid, : len(selected)] = torch.tensor(
                [index for index, _ in selected], dtype=torch.int32
            )
            kv_valid[sid, : len(selected)] = torch.tensor(
                [valid for _, valid in selected], dtype=torch.uint8
            )
        kv_count[sid] = len(selected)
        for offset in range(0, qlen, BLOCK_M):
            prog_qbase.append(qstart + offset)
            prog_nq.append(min(BLOCK_M, qlen - offset))
            prog_seg.append(sid)

    return HierTiledBlockLayout(
        prog_qbase=torch.tensor(prog_qbase, dtype=torch.int32, device=device),
        prog_nq=torch.tensor(prog_nq, dtype=torch.int32, device=device),
        prog_seg=torch.tensor(prog_seg, dtype=torch.int32, device=device),
        kv_index=kv_index.to(device=device),
        kv_valid=kv_valid.to(device=device),
        kv_count=kv_count.to(device=device),
        logical_slots=logical_slots,
        physical_slots=physical_slots,
        dense_slots=dense_slots,
    )


def build_hier_tiled_full_sequence_layouts(
    *,
    device,
    num_players: int,
    num_frames: int,
    num_patches: int,
    num_groups: int,
    num_local_hub_tokens: int,
    num_global_hub_tokens: int,
    num_frame_per_block: int,
    local_attn_size: int,
    group_labels: torch.Tensor,
    active_group: torch.Tensor,
    direct_player: bool,
) -> tuple[HierTiledBlockLayout, ...]:
    """Build full-sequence metadata equivalent to cached Tiled128 launches."""
    P, T, L = int(num_players), int(num_frames), int(num_patches)
    N = int(num_groups)
    KL, KG = int(num_local_hub_tokens), int(num_global_hub_tokens)
    n, window = int(num_frame_per_block), int(local_attn_size)
    if T % n != 0:
        raise ValueError(f"Tiled128 requires T % n == 0, got T={T}, n={n}")
    if window != -1 and window < n:
        raise ValueError(
            f"Tiled128 requires local_attn_size=-1 or >=n, got {window}, n={n}"
        )
    if tuple(group_labels.shape) != (T, P) or tuple(active_group.shape) != (T, N):
        raise ValueError(
            "Tiled128 grouping shape mismatch: "
            f"labels={tuple(group_labels.shape)} expected {(T, P)}, "
            f"active={tuple(active_group.shape)} expected {(T, N)}"
        )

    labels = _as_cpu_rows(group_labels, torch.long)
    active = _as_cpu_rows(active_group, torch.bool)
    for start in range(0, T, n):
        end = min(start + n, T)
        first = labels[start]
        if any(labels[t] != first for t in range(start + 1, end)):
            raise ValueError(
                "Tiled128 requires group labels constant inside each temporal block"
            )

    key = (
        str(torch.device(device)),
        P,
        T,
        L,
        N,
        KL,
        KG,
        n,
        window,
        labels,
        active,
        bool(direct_player),
        BLOCK_M,
        BLOCK_N,
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached

    agent_len = P * T * L
    local_start = agent_len
    local_len = T * N * KL
    global_start = local_start + local_len
    retention = T if window == -1 else window + n

    def agent_base(player: int, frame: int) -> int:
        return (player * T + frame) * L

    def local_base(group: int, frame: int) -> int:
        # Full-sequence local hubs are time-major [T,N,KL].
        return local_start + (frame * N + group) * KL

    def global_base(frame: int) -> int:
        return global_start + frame * KG

    def append_frame(
        dense: list[tuple[int, bool]], base: int, width: int, allowed: bool
    ) -> None:
        dense.extend((base + token, bool(allowed)) for token in range(width))

    layouts: list[HierTiledBlockLayout] = []
    for block_start in range(0, T, n):
        block_end = block_start + n
        retained_start = max(0, block_end - retention)
        retained_frames = range(retained_start, block_end)
        current_frames = range(block_start, block_end)
        dense_segments: list[list[tuple[int, bool]]] = []
        query_specs: list[tuple[int, int]] = []

        # Player rows preserve cached SDPA concatenation: own retained cache,
        # every other current-block stream, then every retained local stream.
        for player in range(P):
            for qf in current_frames:
                dense: list[tuple[int, bool]] = []
                for kf in retained_frames:
                    append_frame(
                        dense, agent_base(player, kf), L, _temporal(qf, kf, n, window)
                    )
                if direct_player:
                    for other in range(P):
                        if other == player:
                            continue
                        same_group = labels[qf][other] == labels[qf][player]
                        for kf in current_frames:
                            append_frame(
                                dense,
                                agent_base(other, kf),
                                L,
                                same_group and (kf // n == qf // n),
                            )
                query_group = labels[qf][player]
                for group in range(N):
                    for kf in retained_frames:
                        allowed = (
                            group == query_group
                            and active[kf][group]
                            and _temporal(qf, kf, n, window)
                        )
                        append_frame(dense, local_base(group, kf), KL, allowed)
                dense_segments.append(dense)
                query_specs.append((agent_base(player, qf), L))

        # Local rows: every retained agent stream, own local stream, globals.
        for qf in current_frames:
            for group in range(N):
                dense = []
                query_active = active[qf][group]
                for player in range(P):
                    player_allowed = query_active and labels[qf][player] == group
                    for kf in retained_frames:
                        append_frame(
                            dense,
                            agent_base(player, kf),
                            L,
                            player_allowed and _temporal(qf, kf, n, window),
                        )
                for kf in retained_frames:
                    append_frame(
                        dense,
                        local_base(group, kf),
                        KL,
                        query_active
                        and active[kf][group]
                        and _temporal(qf, kf, n, window),
                    )
                for kf in retained_frames:
                    append_frame(
                        dense,
                        global_base(kf),
                        KG,
                        query_active and _temporal(qf, kf, n, window),
                    )
                dense_segments.append(dense)
                query_specs.append((local_base(group, qf), KL))

        # Global rows: every retained local stream, then retained globals.
        for qf in current_frames:
            dense = []
            for group in range(N):
                for kf in retained_frames:
                    append_frame(
                        dense,
                        local_base(group, kf),
                        KL,
                        active[kf][group] and _temporal(qf, kf, n, window),
                    )
            for kf in retained_frames:
                append_frame(dense, global_base(kf), KG, _temporal(qf, kf, n, window))
            dense_segments.append(dense)
            query_specs.append((global_base(qf), KG))

        layouts.append(
            _pack_layout(
                device=torch.device(device),
                dense_segments=dense_segments,
                query_specs=query_specs,
            )
        )

    result = tuple(layouts)
    _cache_put(key, result)
    return result


def build_hier_query_active_mask(
    *,
    active_group: torch.Tensor,
    num_players: int,
    num_frames: int,
    num_patches: int,
    num_groups: int,
    num_local_hub_tokens: int,
    num_global_hub_tokens: int,
    device,
    dtype,
) -> torch.Tensor:
    """Return [1,S,1] mask; only inactive local-hub query rows are zero."""
    P, T, L = int(num_players), int(num_frames), int(num_patches)
    N, KL, KG = int(num_groups), int(num_local_hub_tokens), int(num_global_hub_tokens)
    if tuple(active_group.shape) != (T, N):
        raise ValueError(
            f"active_group must have shape {(T, N)}, got {tuple(active_group.shape)}"
        )
    agent = torch.ones(P * T * L, device=device, dtype=dtype)
    local = (
        active_group.to(device=device, dtype=dtype)
        .reshape(T, N, 1)
        .expand(T, N, KL)
        .reshape(-1)
    )
    global_rows = torch.ones(T * KG, device=device, dtype=dtype)
    return torch.cat((agent, local, global_rows), dim=0).view(1, -1, 1)


def _frame_list(frame_tokens: torch.Tensor, width: int) -> tuple[int, ...]:
    if width <= 0:
        return ()
    return tuple(int(x) for x in frame_tokens.reshape(-1, width)[:, 0].tolist())


def _active_frames(active_tokens: torch.Tensor, width: int) -> tuple[bool, ...]:
    if width <= 0:
        return ()
    return tuple(bool(x) for x in active_tokens.reshape(-1, width).any(dim=1).tolist())


def build_hier_tiled_cached_layout(
    *,
    device,
    P: int,
    N: int,
    L: int,
    KL: int,
    KG: int,
    n_frames: int,
    n_block: int,
    local_attn: int,
    agent_frame_lists: Sequence[Sequence[int]],
    local_frame_lists: Sequence[Sequence[int]],
    global_frame_list: Sequence[int],
    local_active_frames: Sequence[Sequence[bool]],
    labels: Sequence[Sequence[int]],
    q_active: Sequence[Sequence[bool]],
    direct_player: bool,
) -> HierTiledBlockLayout:
    """Build metadata from the live cached-inference frame lists."""
    if not agent_frame_lists or any(
        len(x) != len(agent_frame_lists[0]) for x in agent_frame_lists
    ):
        raise ValueError("Tiled128 requires equal agent cache lengths")
    if not local_frame_lists or any(
        len(x) != len(local_frame_lists[0]) for x in local_frame_lists
    ):
        raise ValueError("Tiled128 requires equal local cache lengths")
    agent_frames = tuple(tuple(int(x) for x in row) for row in agent_frame_lists)
    local_frames = tuple(tuple(int(x) for x in row) for row in local_frame_lists)
    global_frames = tuple(int(x) for x in global_frame_list)
    local_active = tuple(tuple(bool(x) for x in row) for row in local_active_frames)
    labels = tuple(tuple(int(x) for x in row) for row in labels)
    q_active = tuple(tuple(bool(x) for x in row) for row in q_active)
    key = (
        "cached",
        str(torch.device(device)),
        P,
        N,
        L,
        KL,
        KG,
        n_frames,
        n_block,
        local_attn,
        agent_frames,
        local_frames,
        global_frames,
        local_active,
        labels,
        q_active,
        bool(direct_player),
        BLOCK_M,
        BLOCK_N,
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached[0]

    fa, fl = len(agent_frames[0]), len(local_frames[0])
    agent_pool_start = 0
    local_pool_start = P * fa * L
    global_pool_start = local_pool_start + N * fl * KL
    query_frames = global_frames[-n_frames:] if KG > 0 else agent_frames[0][-n_frames:]
    if len(query_frames) != n_frames:
        raise ValueError("Tiled128 cannot identify the current cached query frames")

    def agent_base(player: int, frame_index: int) -> int:
        return agent_pool_start + (player * fa + frame_index) * L

    def local_base(group: int, frame_index: int) -> int:
        return local_pool_start + (group * fl + frame_index) * KL

    def global_base(frame_index: int) -> int:
        return global_pool_start + frame_index * KG

    def append_frame(
        dense: list[tuple[int, bool]], base: int, width: int, allowed: bool
    ) -> None:
        dense.extend((base + token, bool(allowed)) for token in range(width))

    current_agent_indices = []
    for player in range(P):
        by_frame = {frame: index for index, frame in enumerate(agent_frames[player])}
        current_agent_indices.append(tuple(by_frame[frame] for frame in query_frames))

    dense_segments: list[list[tuple[int, bool]]] = []
    query_specs: list[tuple[int, int]] = []
    for player in range(P):
        for qrel, qf in enumerate(query_frames):
            dense: list[tuple[int, bool]] = []
            for fi, kf in enumerate(agent_frames[player]):
                append_frame(
                    dense,
                    agent_base(player, fi),
                    L,
                    _temporal(qf, kf, n_block, local_attn),
                )
            if direct_player:
                for other in range(P):
                    if other == player:
                        continue
                    same_group = labels[qrel][other] == labels[qrel][player]
                    for key_rel, kf in enumerate(query_frames):
                        append_frame(
                            dense,
                            agent_base(other, current_agent_indices[other][key_rel]),
                            L,
                            same_group and (kf // n_block == qf // n_block),
                        )
            query_group = labels[qrel][player]
            for group in range(N):
                for fi, kf in enumerate(local_frames[group]):
                    append_frame(
                        dense,
                        local_base(group, fi),
                        KL,
                        group == query_group
                        and local_active[group][fi]
                        and _temporal(qf, kf, n_block, local_attn),
                    )
            dense_segments.append(dense)
            query_specs.append((player * n_frames * L + qrel * L, L))

    agent_q_len = P * n_frames * L
    local_q_len = n_frames * N * KL
    for qrel, qf in enumerate(query_frames):
        for group in range(N):
            dense = []
            query_active = q_active[qrel][group]
            for player in range(P):
                player_allowed = query_active and labels[qrel][player] == group
                for fi, kf in enumerate(agent_frames[player]):
                    append_frame(
                        dense,
                        agent_base(player, fi),
                        L,
                        player_allowed and _temporal(qf, kf, n_block, local_attn),
                    )
            for fi, kf in enumerate(local_frames[group]):
                append_frame(
                    dense,
                    local_base(group, fi),
                    KL,
                    query_active
                    and local_active[group][fi]
                    and _temporal(qf, kf, n_block, local_attn),
                )
            for fi, kf in enumerate(global_frames):
                append_frame(
                    dense,
                    global_base(fi),
                    KG,
                    query_active and _temporal(qf, kf, n_block, local_attn),
                )
            dense_segments.append(dense)
            query_specs.append((agent_q_len + qrel * N * KL + group * KL, KL))

    for qrel, qf in enumerate(query_frames):
        dense = []
        for group in range(N):
            for fi, kf in enumerate(local_frames[group]):
                append_frame(
                    dense,
                    local_base(group, fi),
                    KL,
                    local_active[group][fi] and _temporal(qf, kf, n_block, local_attn),
                )
        for fi, kf in enumerate(global_frames):
            append_frame(
                dense, global_base(fi), KG, _temporal(qf, kf, n_block, local_attn)
            )
        dense_segments.append(dense)
        query_specs.append((agent_q_len + local_q_len + qrel * KG, KG))

    layout = _pack_layout(
        device=torch.device(device),
        dense_segments=dense_segments,
        query_specs=query_specs,
    )
    _cache_put(key, (layout,))
    return layout


def hierarchical_tiled_cached_attention(
    *,
    roped_q,
    agent_frames,
    agent_keys,
    agent_vals,
    local_frames,
    local_keys,
    local_vals,
    local_active_tokens,
    global_frames,
    global_keys,
    global_vals,
    labels,
    active,
    P,
    N,
    L,
    KL,
    KG,
    n_frames,
    n_block,
    local_attn,
    direct_player,
):
    """Cached-inference Tiled128 using the same kernel as STE training."""
    b, seq_len, heads, head_dim = roped_q.shape
    if b != 1 or head_dim != 128 or roped_q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "cached Tiled128 supports B=1, BF16, head_dim=128; "
            f"got B={b}, dtype={roped_q.dtype}, head_dim={head_dim}"
        )
    if not roped_q.is_cuda:
        raise NotImplementedError("cached Tiled128 requires CUDA")
    agent_frame_lists = tuple(_frame_list(x, L) for x in agent_frames)
    local_frame_lists = tuple(_frame_list(x, KL) for x in local_frames)
    global_frame_list = _frame_list(global_frames, KG) if KG > 0 else ()
    local_active_frames = tuple(_active_frames(x, KL) for x in local_active_tokens)
    labels_cpu = _as_cpu_rows(labels, torch.long)
    active_cpu = _as_cpu_rows(active, torch.bool)
    layout = build_hier_tiled_cached_layout(
        device=roped_q.device,
        P=P,
        N=N,
        L=L,
        KL=KL,
        KG=KG,
        n_frames=n_frames,
        n_block=n_block,
        local_attn=local_attn,
        agent_frame_lists=agent_frame_lists,
        local_frame_lists=local_frame_lists,
        global_frame_list=global_frame_list,
        local_active_frames=local_active_frames,
        labels=labels_cpu,
        q_active=active_cpu,
        direct_player=direct_player,
    )
    k = (
        torch.cat(agent_keys + local_keys + [global_keys], dim=1)
        .reshape(b, -1, heads, head_dim)
        .contiguous()
    )
    v = (
        torch.cat(agent_vals + local_vals + [global_vals], dim=1)
        .reshape(b, -1, heads, head_dim)
        .contiguous()
    )
    return hierarchical_tiled_full_sequence_attention(
        roped_q=roped_q,
        roped_k=k,
        value=v,
        layouts=(layout,),
    )


def hierarchical_tiled_full_sequence_attention(
    *,
    roped_q: torch.Tensor,
    roped_k: torch.Tensor,
    value: torch.Tensor,
    layouts: Sequence[HierTiledBlockLayout],
) -> torch.Tensor:
    """Run blockwise forward-only Tiled128 on projected full-sequence Q/K/V."""
    if roped_k.shape != value.shape or roped_q.ndim != 4 or roped_k.ndim != 4:
        raise ValueError(
            f"Tiled128 expects Q [B,Sq,H,D] and matching K/V [B,Sk,H,D], "
            f"got {roped_q.shape}, {roped_k.shape}, {value.shape}"
        )
    batch, seq_len, heads, head_dim = roped_q.shape
    if roped_k.shape[0] != batch or roped_k.shape[2:] != (heads, head_dim):
        raise ValueError(
            f"Tiled128 Q/K batch/head shapes must match, got {roped_q.shape}, {roped_k.shape}"
        )
    if batch != 1 or head_dim != 128 or roped_q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "Tiled128 training currently supports B=1, BF16, head_dim=128; "
            f"got B={batch}, dtype={roped_q.dtype}, head_dim={head_dim}"
        )
    if not roped_q.is_cuda:
        raise NotImplementedError("Tiled128 training requires CUDA")
    if roped_k.dtype != roped_q.dtype or value.dtype != roped_q.dtype:
        raise ValueError("Tiled128 Q/K/V dtypes must match")

    q = roped_q.contiguous()
    k = roped_k.contiguous()
    v = value.contiguous()
    out = torch.zeros_like(q)
    for layout in layouts:
        if layout.prog_qbase.numel() == 0:
            continue
        _tiled128_fwd[(layout.prog_qbase.numel(), heads)](
            q,
            k,
            v,
            out,
            layout.prog_qbase,
            layout.prog_nq,
            layout.prog_seg,
            layout.kv_index,
            layout.kv_valid,
            layout.kv_count,
            q.stride(1),
            q.stride(2),
            k.stride(1),
            k.stride(2),
            v.stride(1),
            v.stride(2),
            out.stride(1),
            out.stride(2),
            1.0 / math.sqrt(head_dim),
            MAX_K=layout.kv_index.shape[1],
            HEAD_DIM=head_dim,
            BLOCK_M_=BLOCK_M,
            BLOCK_N_=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
    return out.reshape(batch, seq_len, heads * head_dim)


def clear_hier_tiled_layout_cache() -> None:
    """Testing/debug helper; normal inference uses bounded automatic eviction."""
    _LAYOUT_CACHE.clear()
