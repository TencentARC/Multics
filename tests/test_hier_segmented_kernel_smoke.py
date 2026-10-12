"""Small CUDA numerical smoke test for the hierarchical segmented kernel."""

import math

import pytest
import torch
import torch.nn.functional as F

from multics.models.kernels.hier_segmented_attention import _segmented_attn_fwd


def test_segmented_kernel_matches_sdpa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, query_len, kv_len, heads, head_dim = 1, 14, 24, 4, 128
    q = torch.randn(
        batch, query_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(batch, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)

    # The final empty segment models an inactive cached local-hub query.
    segments = [
        (0, 8, list(range(0, 10)) + list(range(16, 21))),
        (8, 4, list(range(3, 14))),
        (12, 2, []),
    ]
    block_m, max_k = 16, 64
    kv_index = torch.zeros(len(segments), max_k, dtype=torch.int32, device=device)
    kv_count = torch.tensor(
        [len(item[2]) for item in segments], dtype=torch.int32, device=device
    )
    prog_qbase = torch.tensor(
        [item[0] for item in segments], dtype=torch.int32, device=device
    )
    prog_nq = torch.tensor(
        [item[1] for item in segments], dtype=torch.int32, device=device
    )
    prog_seg = torch.arange(len(segments), dtype=torch.int32, device=device)
    for sid, (_, _, selected) in enumerate(segments):
        if selected:
            kv_index[sid, : len(selected)] = torch.tensor(
                selected, dtype=torch.int32, device=device
            )

    actual = torch.empty_like(q)
    _segmented_attn_fwd[(len(segments), heads)](
        q,
        k,
        v,
        actual,
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
        actual.stride(1),
        actual.stride(2),
        1.0 / math.sqrt(head_dim),
        MAX_K=max_k,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )

    expected = torch.zeros_like(q)
    for qstart, qlen, selected in segments:
        if not selected:
            continue
        q_part = q[:, qstart : qstart + qlen].transpose(1, 2)
        k_part = k[:, selected].transpose(1, 2)
        v_part = v[:, selected].transpose(1, 2)
        expected[:, qstart : qstart + qlen] = F.scaled_dot_product_attention(
            q_part, k_part, v_part
        ).transpose(1, 2)

    diff = (actual.float() - expected.float()).flatten()
    rel_l2 = diff.norm() / expected.float().flatten().norm().clamp_min(1e-12)
    max_abs = diff.abs().max()
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, 12:]) == 0
    assert float(rel_l2) < 0.005, float(rel_l2)
    assert float(max_abs) < 0.05, float(max_abs)
    print(
        f"segmented CUDA smoke: rel_l2={float(rel_l2):.6g} max_abs={float(max_abs):.6g}"
    )


if __name__ == "__main__":
    test_segmented_kernel_matches_sdpa()
