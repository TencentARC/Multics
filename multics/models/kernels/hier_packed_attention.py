"""Packed varlen FlashAttention for cached hierarchical inference.

This module changes only the execution of the exact key lists used by cached
hierarchical SDPA. It is deliberately inference-only and is imported only when
``WAN_HIER_INFER_KERNEL=packed_varlen`` is selected.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import os

import torch

from .hier_segmented_attention import _active_frames, _build_metadata, _frame_list


_PLAN_CACHE = OrderedDict()


def _cache_plan(key, value):
    max_entries = max(int(os.environ.get("WAN_HIER_PACKED_PLAN_CACHE_SIZE", "64")), 0)
    if max_entries == 0:
        return
    _PLAN_CACHE[key] = value
    _PLAN_CACHE.move_to_end(key)
    while len(_PLAN_CACHE) > max_entries:
        _PLAN_CACHE.popitem(last=False)


def _get_varlen_plan(*, kv_index, kv_count, qbase, qlen, seg_ids, query_tokens):
    # _build_metadata returns stable device tensors on a metadata-cache hit.
    # Keep references to those tensors in the cached value as well, preventing
    # an allocator data-pointer reuse from aliasing a stale plan.
    key = (
        str(kv_index.device),
        kv_index.data_ptr(),
        kv_count.data_ptr(),
        qbase.data_ptr(),
        qlen.data_ptr(),
        seg_ids.data_ptr(),
        int(query_tokens),
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached

    qb = [int(x) for x in qbase.cpu().tolist()]
    qn = [int(x) for x in qlen.cpu().tolist()]
    si = [int(x) for x in seg_ids.cpu().tolist()]
    nk = [int(x) for x in kv_count.cpu().tolist()]
    if si != list(range(len(si))):
        raise AssertionError(
            "packed metadata must contain exactly one program per segment"
        )
    coverage = [
        row for base, length in zip(qb, qn) for row in range(base, base + length)
    ]
    if coverage != list(range(query_tokens)):
        raise AssertionError(
            "packed query segments must partition the query sequence exactly"
        )

    active = [sid for sid in si if qn[sid] > 0 and nk[sid] > 0]
    q_parts = [
        torch.arange(qb[sid], qb[sid] + qn[sid], device=kv_index.device)
        for sid in active
    ]
    k_parts = [kv_index[sid, : nk[sid]].long() for sid in active]
    q_flat = (
        torch.cat(q_parts)
        if q_parts
        else torch.empty(0, device=kv_index.device, dtype=torch.long)
    )
    k_flat = (
        torch.cat(k_parts)
        if k_parts
        else torch.empty(0, device=kv_index.device, dtype=torch.long)
    )
    min_k = int(k_flat.min().item()) if k_flat.numel() else 0
    max_k_index = int(k_flat.max().item()) if k_flat.numel() else -1
    cu_q = [0]
    cu_k = [0]
    for sid in active:
        cu_q.append(cu_q[-1] + qn[sid])
        cu_k.append(cu_k[-1] + nk[sid])
    result = {
        "q_index": q_flat,
        "k_index": k_flat,
        "cu_q": torch.tensor(cu_q, device=kv_index.device, dtype=torch.int32),
        "cu_k": torch.tensor(cu_k, device=kv_index.device, dtype=torch.int32),
        "max_q": max((qn[sid] for sid in active), default=0),
        "max_k": max((nk[sid] for sid in active), default=0),
        "min_k_index": min_k,
        "max_k_index": max_k_index,
        # See pointer-reuse note above.
        "metadata_sources": (kv_index, kv_count, qbase, qlen, seg_ids),
    }
    _cache_plan(key, result)
    return result


def hierarchical_packed_varlen_attention(
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
    membership=None,
    global_only=None,
    hier_no_local_hub=False,
):
    """Run exact cached hierarchical membership edges in one varlen kernel."""
    b, s, heads, head_dim = roped_q.shape
    if b != 1 or head_dim != 128 or roped_q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "packed-varlen hierarchical inference requires B=1, BF16, head_dim=128; "
            f"got B={b}, dtype={roped_q.dtype}, head_dim={head_dim}"
        )
    if not roped_q.is_cuda:
        raise NotImplementedError("packed-varlen hierarchical inference requires CUDA")
    if roped_q.requires_grad:
        raise RuntimeError("packed-varlen hierarchical attention is inference-only")
    if not hasattr(torch.ops.aten, "_flash_attention_forward"):
        raise RuntimeError(
            "this PyTorch build has no native varlen FlashAttention operator; "
            "use WAN_HIER_INFER_KERNEL=sdpa"
        )
    if membership is not None and global_only is None:
        raise ValueError("membership requires global_only")

    agent_frame_lists = tuple(_frame_list(x, L) for x in agent_frames)
    no_local = bool(hier_no_local_hub)
    local_frame_lists = (
        () if no_local else tuple(_frame_list(x, KL) for x in local_frames)
    )
    global_frame_list = _frame_list(global_frames, KG) if KG > 0 else ()
    local_active_frames = (
        () if no_local else tuple(_active_frames(x, KL) for x in local_active_tokens)
    )
    labels_cpu = (
        None
        if labels is None
        else tuple(tuple(int(v) for v in row) for row in labels.tolist())
    )
    active_cpu = tuple(tuple(bool(v) for v in row) for row in active.tolist())
    membership_cpu = None
    global_only_cpu = None
    if membership is not None:
        membership_cpu = membership.detach().to(device="cpu", dtype=torch.bool).tolist()
        global_only_cpu = (
            global_only.detach().to(device="cpu", dtype=torch.bool).tolist()
        )

    # One metadata program per logical segment, independent of Triton tiling.
    block_m = max(L, KL, KG, 1)
    qbase, qlen, seg_ids, kv_index, kv_count, _ = _build_metadata(
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
        membership=membership_cpu,
        global_only=global_only_cpu,
        hier_no_local_hub=no_local,
    )
    plan = _get_varlen_plan(
        kv_index=kv_index,
        kv_count=kv_count,
        qbase=qbase,
        qlen=qlen,
        seg_ids=seg_ids,
        query_tokens=s,
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
    q = roped_q.contiguous()
    q_index = plan["q_index"]
    k_index = plan["k_index"]
    out = torch.zeros_like(q)
    if q_index.numel() == 0:
        return out.reshape(b, s, heads * head_dim)
    if plan["min_k_index"] < 0 or plan["max_k_index"] >= k.shape[1]:
        raise AssertionError("packed KV index is outside the physical cache pool")

    q_packed = q[0].index_select(0, q_index)
    k_packed = k[0].index_select(0, k_index)
    v_packed = v[0].index_select(0, k_index)
    packed_out = torch.ops.aten._flash_attention_forward(
        q_packed,
        k_packed,
        v_packed,
        plan["cu_q"],
        plan["cu_k"],
        plan["max_q"],
        plan["max_k"],
        0.0,
        False,
        False,
        scale=1.0 / math.sqrt(head_dim),
    )[0]
    out[0].index_copy_(0, q_index, packed_out)
    return out.reshape(b, s, heads * head_dim)
