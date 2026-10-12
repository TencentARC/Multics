"""Exact Paged-KV attention for cached hierarchical inference.

This backend preserves the logical membership/global-only edges produced by
``hier_segmented_attention._build_metadata``.  It stores physical K/V values
once and routes homogeneous query segments through an int32 FlashInfer page
table.  The module is inference-only and FlashInfer is imported lazily, so the
default SDPA path and training do not acquire an optional runtime dependency.
"""

from __future__ import annotations

import math

import torch

from .hier_packed_attention import _get_varlen_plan
from .hier_segmented_attention import _active_frames, _build_metadata, _frame_list


_FLASHINFER_VERSION = "0.6.18"
_WORKSPACE_BYTES = 128 * 1024 * 1024
_STATE = {}


def _device_key(device: torch.device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return device.type, int(index)


def _load_flashinfer_wrapper():
    try:
        import flashinfer
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    except ImportError as exc:
        raise RuntimeError(
            "WAN_HIER_INFER_KERNEL=paged_kv requires "
            f"flashinfer-python=={_FLASHINFER_VERSION}; run "
            "`uv sync --extra paged-kv --frozen` or use sdpa"
        ) from exc
    version = getattr(flashinfer, "__version__", None)
    if version != _FLASHINFER_VERSION:
        raise RuntimeError(
            "WAN_HIER_INFER_KERNEL=paged_kv was validated with "
            f"flashinfer-python=={_FLASHINFER_VERSION}, got {version!r}"
        )
    return BatchPrefillWithPagedKVCacheWrapper


def _paged_state(device: torch.device):
    key = _device_key(device)
    state = _STATE.get(key)
    if state is None:
        wrapper_cls = _load_flashinfer_wrapper()
        workspace = torch.empty(
            _WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        state = {
            "workspace": workspace,
            "wrapper": wrapper_cls(
                workspace,
                kv_layout="NHD",
                backend="fa2",
            ),
            "plan_key": None,
        }
        _STATE[key] = state
    return state


def clear_hier_paged_state():
    """Release cached wrappers/workspaces; intended for tests and teardown."""
    _STATE.clear()


def hierarchical_paged_kv_attention(
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
    """Run exact cached hierarchical edges with FlashInfer Paged-KV.

    Tensor contract:
      roped_q: [1, S, H, 128] BF16
      physical K/V: [1, K_physical, H, 128] BF16 after one concat
      q_packed: [Q_packed, H, 128]
      K/V pages (NHD, page_size=1): [K_physical, 1, H, 128]
    """
    b, s, heads, head_dim = roped_q.shape
    if b != 1 or head_dim != 128 or roped_q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "Paged-KV hierarchical inference requires B=1, BF16, head_dim=128; "
            f"got B={b}, dtype={roped_q.dtype}, head_dim={head_dim}"
        )
    if not roped_q.is_cuda:
        raise NotImplementedError("Paged-KV hierarchical inference requires CUDA")
    if roped_q.requires_grad:
        raise RuntimeError("Paged-KV hierarchical attention is inference-only")
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
        block_m=max(L, KL, KG, 1),
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
    if k.dtype != roped_q.dtype or v.dtype != roped_q.dtype:
        raise TypeError(
            "Paged-KV Q/K/V dtype mismatch: "
            f"q={roped_q.dtype}, k={k.dtype}, v={v.dtype}"
        )

    q = roped_q.contiguous()
    q_index = plan["q_index"]
    out = torch.zeros_like(q)
    if q_index.numel() == 0:
        return out.reshape(b, s, heads * head_dim)
    if plan["min_k_index"] < 0 or plan["max_k_index"] >= k.shape[1]:
        raise AssertionError(
            f"Paged-KV page index [{plan['min_k_index']}, "
            f"{plan['max_k_index']}] outside physical pool [0, {k.shape[1]})"
        )

    state = _paged_state(roped_q.device)
    source_plan_key = (
        plan["q_index"].data_ptr(),
        plan["q_index"].numel(),
        plan["k_index"].data_ptr(),
        plan["k_index"].numel(),
        plan["cu_q"].data_ptr(),
        plan["cu_q"].numel(),
        plan["cu_k"].data_ptr(),
        plan["cu_k"].numel(),
        int(k.shape[1]),
        heads,
        head_dim,
        roped_q.dtype,
    )
    if state["plan_key"] != source_plan_key:
        if int(plan["cu_q"][-1].item()) != q_index.numel():
            raise AssertionError("Paged-KV cu_q does not match q_index")
        if int(plan["cu_k"][-1].item()) != plan["k_index"].numel():
            raise AssertionError("Paged-KV cu_k does not match k_index")
        if plan["cu_q"].numel() != plan["cu_k"].numel():
            raise AssertionError("Paged-KV query and KV segment counts differ")

        page_indices = plan["k_index"].to(dtype=torch.int32)
        last_page_len = torch.ones(
            plan["cu_q"].numel() - 1,
            dtype=torch.int32,
            device=roped_q.device,
        )
        state["wrapper"].plan(
            qo_indptr=plan["cu_q"],
            paged_kv_indptr=plan["cu_k"],
            paged_kv_indices=page_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=heads,
            num_kv_heads=heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=1,
            causal=False,
            pos_encoding_mode="NONE",
            sm_scale=1.0 / math.sqrt(head_dim),
            q_data_type=roped_q.dtype,
            kv_data_type=roped_q.dtype,
            o_data_type=roped_q.dtype,
        )
        state.update(
            plan_key=source_plan_key,
            source_plan=plan,
            page_indices=page_indices,
            last_page_len=last_page_len,
        )

    q_packed = q[0].index_select(0, q_index).contiguous()
    packed_out = state["wrapper"].run(
        q_packed,
        (k[0].unsqueeze(1), v[0].unsqueeze(1)),
    )
    out[0].index_copy_(0, q_index, packed_out)
    return out.reshape(b, s, heads * head_dim)
