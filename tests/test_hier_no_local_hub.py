"""Correctness gates for the opt-in player-direct-global ablation."""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multics.models.wan_video_dit import (
    SelfAttention,
    build_stage2_hier_flex_block_mask,
    build_stage2_hier_self_attn_mask,
    migrate_hier_no_local_hub_state_dict,
)


def _grouping(T=6, P=4, N=2, device="cpu"):
    labels = torch.tensor([[0, 0, 1, 1]] * T, device=device)
    membership = torch.zeros(T, N, P, dtype=torch.bool, device=device)
    membership[:, 0, :2] = True
    membership[:, 1, 2:] = True
    active = membership.any(dim=-1)
    global_only = torch.zeros(T, P, dtype=torch.bool, device=device)
    global_only[::2, 1] = True
    return labels, membership, active, global_only


def _dense_no_local(*, membership, global_only, direct=True, device="cpu"):
    P, T, L, N, KL, KG = 4, 6, 3, 2, 0, 5
    labels, _, active, _ = _grouping(T, P, N, device)
    return build_stage2_hier_self_attn_mask(
        P,
        T,
        L,
        N,
        KL,
        KG,
        3,
        6,
        labels,
        active,
        direct,
        device,
        membership=membership,
        global_only=global_only,
        hier_no_local_hub=True,
    )


def test_no_local_dense_semantics_and_global_only_invariance():
    P, T, L, N, KL, KG = 4, 6, 3, 2, 0, 5
    labels, membership, _, global_only = _grouping(T, P, N)
    mask = _dense_no_local(membership=membership, global_only=global_only)
    changed_go = ~global_only
    assert torch.equal(
        mask,
        _dense_no_local(membership=membership, global_only=changed_go),
    )
    assert mask.shape == (P * T * L + T * KG,) * 2
    assert mask.any(dim=-1).all()

    agent = P * T * L
    player = lambda p, t, patch=0: p * T * L + t * L + patch
    global_token = lambda t, slot=0: agent + t * KG + slot
    assert mask[player(0, 4), global_token(1)]
    assert mask[global_token(4), player(3, 1)]
    # Direct is same group and same diffusion block only.
    assert mask[player(0, 1), player(1, 2)]
    assert not mask[player(0, 1), player(2, 2)]
    assert not mask[player(0, 1), player(1, 4)]

    labels_mask = build_stage2_hier_self_attn_mask(
        P,
        T,
        L,
        N,
        KL,
        KG,
        3,
        6,
        labels,
        membership.any(dim=-1),
        True,
        "cpu",
        hier_no_local_hub=True,
    )
    assert torch.equal(mask, labels_mask)


def test_no_local_checkpoint_migration_is_deterministic():
    local = torch.arange(7 * 8, dtype=torch.float32).reshape(7, 8)
    old_global = torch.randn(6, 8)
    source = {
        "local_hub_embed": local,
        "global_hub_embed": old_global,
        "other": torch.ones(1),
    }
    migrated, report = migrate_hier_no_local_hub_state_dict(
        source, target_num_global_hub_tokens=5, model_dim=8
    )
    assert "local_hub_embed" not in migrated
    assert torch.equal(migrated["global_hub_embed"], local[:5])
    assert report["source"] == "local_hub_embed"
    assert "local_hub_embed" in source
    reloaded, second_report = migrate_hier_no_local_hub_state_dict(
        migrated, target_num_global_hub_tokens=5, model_dim=8
    )
    assert second_report is None
    assert torch.equal(reloaded["global_hub_embed"], local[:5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_local_flex_forward_backward_matches_dense():
    # Use the same compile mode as the hierarchical training launcher.  Torch
    # 2.8 has no max-autotune backward candidate for this small synthetic
    # shape, while the default mode includes the valid ATen fallback and
    # exercises the exact same BlockMask semantics.
    os.environ["WAN_FLEX_MODE"] = "default"
    os.environ["WAN_FLEX_KOPT"] = "0"
    os.environ["WAN_HIER_BLOCKMASK_COMPILE"] = "0"
    torch.manual_seed(904)
    device = torch.device("cuda")
    # Keep S_pad above the degenerate 128-token shape, for which torch 2.8's
    # max-autotune Flex backward has no valid candidate.
    P, T, L, N, KL, KG = 4, 6, 16, 2, 0, 5
    labels, membership, active, global_only = _grouping(T, P, N, device)
    dense = build_stage2_hier_self_attn_mask(
        P,
        T,
        L,
        N,
        KL,
        KG,
        3,
        6,
        labels,
        active,
        True,
        device,
        membership=membership,
        global_only=global_only,
        hier_no_local_hub=True,
    )
    block_mask, S, S_pad = build_stage2_hier_flex_block_mask(
        P,
        T,
        L,
        N,
        KL,
        KG,
        3,
        6,
        labels,
        active,
        True,
        device,
        membership=membership,
        global_only=global_only,
        hier_no_local_hub=True,
    )
    from torch.nn.attention.flex_attention import create_mask as flex_create_mask

    materialized = flex_create_mask(
        block_mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=S_pad,
        KV_LEN=S_pad,
        device=device,
    )[0, 0, :S, :S]
    assert torch.equal(materialized, dense)
    dim, heads = 256, 2
    dense_layer = SelfAttention(dim, heads).to(device=device, dtype=torch.bfloat16)
    flex_layer = copy.deepcopy(dense_layer)
    dense_layer.train()
    flex_layer.train()
    x_dense = torch.randn(
        1, S, dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    x_flex = x_dense.detach().clone().requires_grad_(True)
    freqs = torch.ones(S, 1, 64, device=device, dtype=torch.complex64)
    out_dense = dense_layer(x_dense, freqs, dense)
    out_flex = flex_layer(x_flex, freqs, block_mask)
    weight = torch.randn_like(out_dense)
    (out_dense.float() * weight.float()).mean().backward()
    (out_flex.float() * weight.float()).mean().backward()
    rel = (
        out_dense.float() - out_flex.float()
    ).norm() / out_dense.float().norm().clamp_min(1e-12)
    grad_rel = (
        x_dense.grad.float() - x_flex.grad.float()
    ).norm() / x_dense.grad.float().norm().clamp_min(1e-12)
    assert torch.isfinite(out_flex).all() and torch.isfinite(x_flex.grad).all()
    assert x_flex.grad[:, P * T * L :].abs().sum() > 0
    rel_value = float(rel.detach())
    grad_rel_value = float(grad_rel.detach())
    assert rel_value < 0.008
    assert grad_rel_value < 0.012
    print(
        f"no-local Flex PASS output_rel_l2={rel_value:.7f} "
        f"grad_rel_l2={grad_rel_value:.7f}"
    )


def _stream(capacity, heads, head_dim, device):
    return {
        "k": torch.zeros(
            1, capacity, heads, head_dim, device=device, dtype=torch.bfloat16
        ),
        "v": torch.zeros(
            1, capacity, heads, head_dim, device=device, dtype=torch.bfloat16
        ),
        "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
        "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
    }


def _no_local_cache(P, N, L, KG, retention, heads, head_dim, device):
    return {
        "hier_hub": True,
        "sparse_hub": True,
        "hier_no_local_hub": True,
        "num_players": P,
        "num_groups": N,
        "num_local_hub_tokens": 0,
        "num_global_hub_tokens": KG,
        "agent_caches": [
            _stream(retention * L, heads, head_dim, device) for _ in range(P)
        ],
        "local_hubs": [],
        "global_hub": _stream(retention * KG, heads, head_dim, device),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["packed_varlen", "paged_kv"])
def test_no_local_cached_backend_matches_multiblock_sdpa(backend):
    if backend == "paged_kv" and importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer not installed")
    torch.manual_seed(905)
    device = torch.device("cuda")
    P, N, L, KG = 4, 2, 8, 5
    n_frames = n_block = 2
    window, total_frames = 4, 8
    heads, head_dim, dim = 2, 128, 256
    seq_len = P * n_frames * L + n_frames * KG
    layer = SelfAttention(dim, heads).to(device=device, dtype=torch.bfloat16).eval()
    layer.num_frame_per_block = n_block
    layer.local_attn_size = window
    layer.hier_no_local_hub = True

    membership = torch.zeros(total_frames, N, P, dtype=torch.bool, device=device)
    for block in range(total_frames // n_frames):
        row = (0, 0, 1, 1) if block % 2 == 0 else (0, 1, 0, 1)
        for player, group in enumerate(row):
            membership[block * n_frames : (block + 1) * n_frames, group, player] = True
    global_only = torch.zeros(total_frames, P, dtype=torch.bool, device=device)
    global_only[:, 3] = True
    active = membership.any(dim=-1)
    inputs = [
        torch.randn(1, seq_len, dim, device=device, dtype=torch.bfloat16)
        for _ in range(total_frames // n_frames)
    ]
    freqs = torch.ones(seq_len, 1, 64, device=device, dtype=torch.complex64)

    outputs = {}
    old_backend = os.environ.get("WAN_HIER_INFER_KERNEL")
    try:
        for name in ("sdpa", backend):
            cache = _no_local_cache(
                P, N, L, KG, window + n_frames, heads, head_dim, device
            )
            os.environ["WAN_HIER_INFER_KERNEL"] = name
            stream_outputs = []
            with torch.inference_mode():
                for block, x in enumerate(inputs):
                    stream_outputs.append(
                        layer.causal_forward_hier_hub(
                            x,
                            freqs,
                            cache,
                            current_start=block * n_frames * P * L,
                            num_frames=n_frames,
                            num_players=P,
                            num_hub_tokens=KG,
                            membership=membership,
                            global_only=global_only,
                            active_group=active,
                            direct_player=True,
                        )
                    )
            outputs[name] = stream_outputs
    finally:
        if old_backend is None:
            os.environ.pop("WAN_HIER_INFER_KERNEL", None)
        else:
            os.environ["WAN_HIER_INFER_KERNEL"] = old_backend
        if backend == "paged_kv":
            from multics.models.kernels.hier_paged_attention import (
                clear_hier_paged_state,
            )

            clear_hier_paged_state()

    worst = 0.0
    for expected, actual in zip(outputs["sdpa"], outputs[backend]):
        rel = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-12)
        worst = max(worst, float(rel))
        assert torch.isfinite(actual).all()
        assert float(rel) < 0.007
    print(f"no-local {backend} multiblock PASS worst_rel_l2={worst:.7f}")
