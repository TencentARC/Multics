import copy
import math
import os

import pytest
import torch

from multics.models.kernels.hier_tiled_attention import (
    build_hier_query_active_mask,
    build_hier_tiled_full_sequence_layouts,
    hierarchical_tiled_full_sequence_attention,
)
from multics.models.wan_video_dit import (
    FLEX_ATTENTION_AVAILABLE,
    SelfAttention,
    build_stage2_hier_flex_block_mask,
    build_stage2_hier_self_attn_mask,
    flash_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not FLEX_ATTENTION_AVAILABLE, reason="FlexAttention required")
def test_tiled128_ste_forward_value_and_flex_surrogate_gradient():
    os.environ["WAN_FLEX_KOPT"] = "0"
    os.environ["WAN_FLEX_MODE"] = "default"
    torch.manual_seed(20260813)
    device = torch.device("cuda")
    P, T, L = 2, 3, 17
    N, KL, KG = 3, 2, 2
    n, window = 3, 7
    heads, head_dim = 1, 128
    dim = heads * head_dim
    seq_len = P * T * L + T * (N * KL + KG)
    labels = torch.tensor([[0, 1]] * T, dtype=torch.long, device=device)
    active = torch.tensor([[True, True, False]] * T, dtype=torch.bool, device=device)
    freqs = torch.ones(seq_len, 1, head_dim // 2, dtype=torch.complex128, device=device)
    block_mask, _, _ = build_stage2_hier_flex_block_mask(
        P, T, L, N, KL, KG, n, window, labels, active, True, device
    )
    layouts = build_hier_tiled_full_sequence_layouts(
        device=device,
        num_players=P,
        num_frames=T,
        num_patches=L,
        num_groups=N,
        num_local_hub_tokens=KL,
        num_global_hub_tokens=KG,
        num_frame_per_block=n,
        local_attn_size=window,
        group_labels=labels,
        active_group=active,
        direct_player=True,
    )
    layer = SelfAttention(dim=dim, num_heads=heads).to(
        device=device, dtype=torch.bfloat16
    )
    layer.num_frame_per_block = n
    layer.local_attn_size = window
    reference = copy.deepcopy(layer)
    x = torch.randn(
        1, seq_len, dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    x_reference = x.detach().clone().requires_grad_(True)

    actual = layer.forward_hier_tiled_ste(
        x,
        freqs,
        block_mask,
        group_labels=labels,
        active_group=active,
        num_players=P,
        num_frames=T,
        num_patches=L,
        num_groups=N,
        num_local_hub_tokens=KL,
        num_global_hub_tokens=KG,
        direct_player=True,
        tiled_layouts=layouts,
    )
    with torch.no_grad():
        q, k, v = layer._project_self_attn_qkv(x.detach(), freqs)
        y_tiled = hierarchical_tiled_full_sequence_attention(
            roped_q=q.view(1, seq_len, heads, head_dim),
            roped_k=k.view(1, seq_len, heads, head_dim),
            value=v.view(1, seq_len, heads, head_dim),
            layouts=layouts,
        )
        expected = layer.o(y_tiled)
    assert torch.equal(actual, expected)

    # Tiled and dense SDPA express the same active-row logical edges; only the
    # reduction order differs. Inactive local rows are explicitly zero in both.
    with torch.no_grad():
        dense_mask = build_stage2_hier_self_attn_mask(
            P, T, L, N, KL, KG, n, window, labels, active, True, device
        )
        y_dense = flash_attention(q, k, v, heads, mask=dense_mask)
        row_mask = build_hier_query_active_mask(
            active_group=active,
            num_players=P,
            num_frames=T,
            num_patches=L,
            num_groups=N,
            num_local_hub_tokens=KL,
            num_global_hub_tokens=KG,
            device=device,
            dtype=y_dense.dtype,
        )
        y_dense = y_dense * row_mask
        rel = (
            y_tiled.float() - y_dense.float()
        ).norm() / y_dense.float().norm().clamp_min(1e-12)
    assert float(rel) < 0.01

    actual.float().sum().backward()
    q_ref, k_ref, v_ref = reference._project_self_attn_qkv(x_reference, freqs)
    y_flex_ref = reference._flex_from_projected_qkv(q_ref, k_ref, v_ref, block_mask)
    row_mask_ref = build_hier_query_active_mask(
        active_group=active,
        num_players=P,
        num_frames=T,
        num_patches=L,
        num_groups=N,
        num_local_hub_tokens=KL,
        num_global_hub_tokens=KG,
        device=device,
        dtype=y_flex_ref.dtype,
    )
    reference.o(y_flex_ref * row_mask_ref).float().sum().backward()

    # With a linear loss, the pre-output-projection gradient is independent of
    # the forward activation. Thus every upstream STE gradient must equal Flex.
    assert torch.equal(x.grad, x_reference.grad)
    for name in ("q.weight", "k.weight", "v.weight", "norm_q.weight", "norm_k.weight"):
        actual_grad = dict(layer.named_parameters())[name].grad
        reference_grad = dict(reference.named_parameters())[name].grad
        assert torch.equal(actual_grad, reference_grad), name
    # Output-projection weights intentionally train on the real tiled activation.
    expected_o_grad = y_tiled.reshape(-1, dim).float().sum(dim=0).to(torch.bfloat16)
    assert torch.equal(layer.o.weight.grad[0], expected_o_grad)
