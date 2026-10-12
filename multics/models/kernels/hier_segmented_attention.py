"""Forward-only segmented attention for cached hierarchical inference.

The existing cached implementation launches masked SDPA separately for every
player, local hub, and global hub query stream.  This module preserves the same
logical key set for each query segment, but visits all of those keys in one
Triton launch and one online softmax.  It is deliberately inference-only: there
is no backward kernel and the caller imports it only after the explicit
``WAN_HIER_INFER_KERNEL=segmented`` opt-in.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import os

import torch
import triton
import triton.language as tl


# Layout metadata contains device tensors, so bound its lifetime.  Dynamic
# grouping can otherwise create an unbounded number of label-dependent entries.
_META_CACHE = OrderedDict()


@triton.jit
def _segmented_attn_fwd(
    Q,
    K,
    V,
    O,
    PROG_QBASE,
    PROG_NQ,
    PROG_SEG,
    KV_INDEX,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Attend one query tile to the exact KV index list of its segment."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    qbase = tl.load(PROG_QBASE + pid_m)
    nq = tl.load(PROG_NQ + pid_m)
    seg = tl.load(PROG_SEG + pid_m)
    nk = tl.load(KV_COUNT + seg)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    qmask = offs_m < nq
    q = tl.load(
        Q + (qbase + offs_m[:, None]) * stride_qs + pid_h * stride_qh + offs_d[None, :],
        mask=qmask[:, None],
        other=0.0,
    )

    # FlashAttention-style online softmax.  own/direct/local/global keys all
    # update the same m/l/acc state; they are not normalized independently.
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for k0 in tl.range(0, MAX_K, BLOCK_N):
        slot = k0 + offs_n
        kmask = slot < nk
        kidx = tl.load(KV_INDEX + seg * MAX_K + slot, mask=kmask, other=0)
        kval = tl.load(
            K + kidx[:, None] * stride_ks + pid_h * stride_kh + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(kval)) * scale
        valid = qmask[:, None] & kmask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp2((running_max - new_max) * 1.4426950408889634)
        probs = tl.exp2((scores - new_max[:, None]) * 1.4426950408889634)
        probs = tl.where(valid, probs, 0.0)
        vval = tl.load(
            V + kidx[:, None] * stride_vs + pid_h * stride_vh + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        # The BF16 tensor-core P@V matches the dtype used by the current model.
        # FP32 m/l/acc keep the online normalization numerically stable.
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.bfloat16), vval)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = new_max

    # The current cached SDPA path writes zero for an inactive local-hub query.
    # Such a segment has nk=0, and is handled explicitly instead of evaluating
    # an all-masked softmax (which could produce NaN on another backend).
    out = tl.where(
        (qmask & (running_sum > 0))[:, None], acc / running_sum[:, None], 0.0
    )
    tl.store(
        O + (qbase + offs_m[:, None]) * stride_os + pid_h * stride_oh + offs_d[None, :],
        out,
        mask=qmask[:, None],
    )


def _temporal(qf, kf, n_block, local_attn):
    """Mirror ``_frame_causal_window_mask`` for scalar frame indices."""
    allowed = (kf // n_block) <= (qf // n_block)
    if local_attn != -1:
        allowed = allowed and (qf - kf) < local_attn
    return allowed


def _frame_list(frames, width):
    return tuple(int(x) for x in frames.reshape(-1, width)[:, 0].tolist())


def _active_frames(active_tokens, width):
    return tuple(bool(x) for x in active_tokens.reshape(-1, width).any(dim=1).tolist())


def _range_tokens(start, width):
    return list(range(start, start + width))


def _cache_get(key):
    result = _META_CACHE.get(key)
    if result is not None:
        _META_CACHE.move_to_end(key)
    return result


def _cache_put(key, result):
    max_entries = max(
        int(os.environ.get("WAN_HIER_SEGMENTED_META_CACHE_SIZE", "64")), 0
    )
    if max_entries == 0:
        return
    _META_CACHE[key] = result
    _META_CACHE.move_to_end(key)
    while len(_META_CACHE) > max_entries:
        _META_CACHE.popitem(last=False)


def _build_metadata(
    *,
    device,
    P,
    N,
    L,
    KL,
    KG,
    n_frames,
    n_block,
    local_attn,
    agent_frame_lists,
    local_frame_lists,
    global_frame_list,
    local_active_frames,
    labels,
    q_active,
    direct_player,
    block_m,
    membership=None,
    global_only=None,
    hier_no_local_hub=False,
):
    """Build exact cached-SDPA key indices for each homogeneous query segment.

    Shared K/V order is agent-major, local-group-major, then global.  Query order
    stays exactly as ``causal_forward_hier_hub``: player-major agents, time-major
    local hubs, then global hubs.
    """
    if not agent_frame_lists or any(
        len(x) != len(agent_frame_lists[0]) for x in agent_frame_lists
    ):
        raise ValueError(
            "segmented hierarchical attention requires equal agent cache lengths"
        )
    no_local = bool(hier_no_local_hub)
    if no_local:
        if KL != 0 or local_frame_lists or local_active_frames:
            raise ValueError(
                "hier_no_local_hub metadata requires K_L=0 and empty local caches"
            )
    elif not local_frame_lists or any(
        len(x) != len(local_frame_lists[0]) for x in local_frame_lists
    ):
        raise ValueError(
            "segmented hierarchical attention requires equal local cache lengths"
        )
    fa = len(agent_frame_lists[0])
    fl = 0 if no_local else len(local_frame_lists[0])
    use_membership = membership is not None
    membership = (
        None
        if membership is None
        else tuple(
            tuple(tuple(bool(v) for v in player_row) for player_row in group_rows)
            for group_rows in membership
        )
    )
    global_only = (
        None
        if global_only is None
        else tuple(tuple(bool(v) for v in row) for row in global_only)
    )
    if use_membership and global_only is None:
        raise ValueError("membership metadata requires global_only")
    key = (
        str(device),
        P,
        N,
        L,
        KL,
        KG,
        n_frames,
        n_block,
        local_attn,
        tuple(agent_frame_lists),
        tuple(local_frame_lists),
        tuple(global_frame_list),
        tuple(local_active_frames),
        None if labels is None else tuple(labels),
        tuple(q_active),
        bool(direct_player),
        bool(no_local),
        block_m,
        membership,
        global_only,
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached

    agent_base = 0
    local_base = P * fa * L
    global_base = local_base + N * fl * KL

    def agent_tokens(p, fi):
        return _range_tokens(agent_base + (p * fa + fi) * L, L)

    def local_tokens(g, fi):
        return _range_tokens(local_base + (g * fl + fi) * KL, KL)

    def global_tokens(fi):
        return _range_tokens(global_base + fi * KG, KG)

    # The newest n_frames entries are the block just appended by the caller.
    q_frames = tuple(global_frame_list[-n_frames:]) if KG > 0 else ()
    if len(q_frames) != n_frames:
        q_frames = tuple(agent_frame_lists[0][-n_frames:])
    if len(q_frames) != n_frames:
        raise ValueError(
            "segmented hierarchical attention cannot identify the current query frames"
        )

    agent_q_len = P * n_frames * L
    local_q_len = n_frames * N * KL
    segments = []

    # Player rows. Membership mode preserves the cached SDPA key set: direct
    # players share any query-time slot, anchors read every owned local slot,
    # and global-only queries additionally read the global timeline.
    for p in range(P):
        for qrel, qf in enumerate(q_frames):
            selected = []
            for fi, kf in enumerate(agent_frame_lists[p]):
                if _temporal(qf, kf, n_block, local_attn):
                    selected += agent_tokens(p, fi)
            if direct_player:
                for other in range(P):
                    if other == p:
                        continue
                    if use_membership:
                        shares_slot = any(
                            membership[qf][g][p] and membership[qf][g][other]
                            for g in range(N)
                        )
                    else:
                        shares_slot = labels[qrel][other] == labels[qrel][p]
                    if not shares_slot:
                        continue
                    for fi, kf in enumerate(agent_frame_lists[other]):
                        if kf in q_frames and kf // n_block == qf // n_block:
                            selected += agent_tokens(other, fi)
            if not no_local:
                owned_groups = (
                    [g for g in range(N) if membership[qf][g][p]]
                    if use_membership
                    else [labels[qrel][p]]
                )
                for group in owned_groups:
                    for fi, kf in enumerate(local_frame_lists[group]):
                        if local_active_frames[group][fi] and _temporal(
                            qf, kf, n_block, local_attn
                        ):
                            selected += local_tokens(group, fi)
            if KG > 0 and (no_local or (use_membership and global_only[qf][p])):
                for fi, kf in enumerate(global_frame_list):
                    if _temporal(qf, kf, n_block, local_attn):
                        selected += global_tokens(fi)
            segments.append((p * n_frames * L + qrel * L, L, selected))

    # Local rows: players currently assigned to this group + the same local-hub
    # timeline + global timeline.  Inactive rows intentionally have no keys and
    # therefore produce zero, matching the existing cached implementation.
    if not no_local:
        for qrel, qf in enumerate(q_frames):
            for group in range(N):
                selected = []
                if q_active[qrel][group]:
                    for p in range(P):
                        in_group = (
                            membership[qf][group][p]
                            if use_membership
                            else labels[qrel][p] == group
                        )
                        if in_group:
                            for fi, kf in enumerate(agent_frame_lists[p]):
                                if _temporal(qf, kf, n_block, local_attn):
                                    selected += agent_tokens(p, fi)
                    for fi, kf in enumerate(local_frame_lists[group]):
                        if local_active_frames[group][fi] and _temporal(
                            qf, kf, n_block, local_attn
                        ):
                            selected += local_tokens(group, fi)
                    for fi, kf in enumerate(global_frame_list):
                        if _temporal(qf, kf, n_block, local_attn):
                            selected += global_tokens(fi)
                qstart = agent_q_len + qrel * N * KL + group * KL
                segments.append((qstart, KL, selected))

    # Global rows: all active local-hub timelines + the global timeline.
    for qrel, qf in enumerate(q_frames):
        selected = []
        if not no_local:
            for group in range(N):
                for fi, kf in enumerate(local_frame_lists[group]):
                    if local_active_frames[group][fi] and _temporal(
                        qf, kf, n_block, local_attn
                    ):
                        selected += local_tokens(group, fi)
        for fi, kf in enumerate(global_frame_list):
            if _temporal(qf, kf, n_block, local_attn):
                selected += global_tokens(fi)
        if no_local:
            for p in range(P):
                for fi, kf in enumerate(agent_frame_lists[p]):
                    if _temporal(qf, kf, n_block, local_attn):
                        selected += agent_tokens(p, fi)
        elif use_membership:
            for p in range(P):
                for fi, kf in enumerate(agent_frame_lists[p]):
                    if global_only[kf][p] and _temporal(qf, kf, n_block, local_attn):
                        selected += agent_tokens(p, fi)
        segments.append((agent_q_len + local_q_len + qrel * KG, KG, selected))

    max_k = max(max((len(segment[2]) for segment in segments), default=1), 1)
    max_k = math.ceil(max_k / 64) * 64
    kv_index = torch.zeros((len(segments), max_k), dtype=torch.int32)
    kv_count = torch.zeros((len(segments),), dtype=torch.int32)
    prog_qbase, prog_nq, prog_seg = [], [], []
    logical_pairs = 0
    for sid, (qstart, qlen, selected) in enumerate(segments):
        if selected:
            kv_index[sid, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        kv_count[sid] = len(selected)
        logical_pairs += qlen * len(selected)
        for offset in range(0, qlen, block_m):
            prog_qbase.append(qstart + offset)
            prog_nq.append(min(block_m, qlen - offset))
            prog_seg.append(sid)

    result = (
        torch.tensor(prog_qbase, dtype=torch.int32, device=device),
        torch.tensor(prog_nq, dtype=torch.int32, device=device),
        torch.tensor(prog_seg, dtype=torch.int32, device=device),
        kv_index.to(device),
        kv_count.to(device),
        logical_pairs,
    )
    _cache_put(key, result)
    return result


def hierarchical_segmented_attention(
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
    """Run the supported cached hierarchical forward in one Triton launch."""
    b, s, heads, head_dim = roped_q.shape
    if b != 1 or head_dim != 128 or roped_q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "segmented hierarchical inference currently supports B=1, BF16, head_dim=128; "
            f"got B={b}, dtype={roped_q.dtype}, head_dim={head_dim}"
        )
    if not roped_q.is_cuda:
        raise NotImplementedError("segmented hierarchical inference requires CUDA")

    agent_frame_lists = tuple(_frame_list(x, L) for x in agent_frames)
    local_frame_lists = tuple(_frame_list(x, KL) for x in local_frames)
    global_frame_list = _frame_list(global_frames, KG) if KG > 0 else ()
    local_active_frames = tuple(_active_frames(x, KL) for x in local_active_tokens)
    labels_cpu = tuple(tuple(int(v) for v in row) for row in labels.tolist())
    active_cpu = tuple(tuple(bool(v) for v in row) for row in active.tolist())
    block_m = 16
    prog_qbase, prog_nq, prog_seg, kv_index, kv_count, _ = _build_metadata(
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
        direct_player=bool(direct_player),
        block_m=block_m,
    )

    # One physical K/V pool lets every program consume arbitrary logical
    # segments without launching P+N+1 independent masked SDPA kernels.
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
    q = roped_q.contiguous()
    out = torch.empty_like(q)
    grid = (prog_qbase.numel(), heads)
    _segmented_attn_fwd[grid](
        q,
        k,
        v,
        out,
        prog_qbase,
        prog_nq,
        prog_seg,
        kv_index,
        kv_count,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        out.stride(1),
        out.stride(2),
        1.0 / math.sqrt(head_dim),
        MAX_K=kv_index.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    return out.reshape(b, s, heads * head_dim)
