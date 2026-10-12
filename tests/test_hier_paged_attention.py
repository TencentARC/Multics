"""GPU integration checks for the opt-in exact Paged-KV backend."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multics.models.wan_video_dit import SelfAttention


def _stream(capacity, heads, head_dim, device, dtype):
    return {
        "k": torch.zeros(1, capacity, heads, head_dim, device=device, dtype=dtype),
        "v": torch.zeros(1, capacity, heads, head_dim, device=device, dtype=dtype),
        "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
        "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
    }


def _cache(P, N, L, KL, KG, n_frames, window, heads, head_dim, device, dtype):
    retention = window + n_frames
    local = [_stream(retention * KL, heads, head_dim, device, dtype) for _ in range(N)]
    for cache in local:
        cache["active"] = torch.zeros(retention * KL, device=device, dtype=torch.bool)
    return {
        "hier_hub": True,
        "sparse_hub": True,
        "num_players": P,
        "num_groups": N,
        "num_local_hub_tokens": KL,
        "num_global_hub_tokens": KG,
        "agent_caches": [
            _stream(retention * L, heads, head_dim, device, dtype) for _ in range(P)
        ],
        "local_hubs": local,
        "global_hub": _stream(retention * max(KG, 1), heads, head_dim, device, dtype),
    }


def test_paged_kv_matches_membership_sdpa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer is not installed")

    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    P, N, L, KL, KG = 4, 2, 8, 2, 2
    n_frames = n_block = 2
    window = 4
    heads, head_dim = 2, 128
    dim = heads * head_dim
    seq_len = P * n_frames * L + n_frames * (N * KL + KG)

    layer = (
        SelfAttention(dim=dim, num_heads=heads).to(device=device, dtype=dtype).eval()
    )
    layer.num_frame_per_block = n_block
    layer.local_attn_size = window
    base_cache = _cache(
        P, N, L, KL, KG, n_frames, window, heads, head_dim, device, dtype
    )
    sdpa_cache = copy.deepcopy(base_cache)
    paged_cache = copy.deepcopy(base_cache)

    membership = torch.zeros(n_frames, N, P, device=device, dtype=torch.bool)
    membership[:, 0, 0:2] = True
    membership[:, 1, 2] = True
    global_only = torch.zeros(n_frames, P, device=device, dtype=torch.bool)
    global_only[:, 3] = True
    active = membership.any(dim=-1)
    x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
    freqs = torch.ones(seq_len, 1, head_dim // 2, device=device, dtype=torch.complex64)

    previous = os.environ.get("WAN_HIER_INFER_KERNEL")
    try:
        with torch.inference_mode():
            os.environ["WAN_HIER_INFER_KERNEL"] = "sdpa"
            reference = layer.causal_forward_hier_hub(
                x,
                freqs,
                sdpa_cache,
                current_start=0,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                membership=membership,
                global_only=global_only,
                active_group=active,
                direct_player=True,
            )
            os.environ["WAN_HIER_INFER_KERNEL"] = "paged_kv"
            output = layer.causal_forward_hier_hub(
                x,
                freqs,
                paged_cache,
                current_start=0,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                membership=membership,
                global_only=global_only,
                active_group=active,
                direct_player=True,
            )
            # Same topology/cache position must reuse the existing plan and be
            # deterministic when the current block is refreshed in-place.
            output_repeat = layer.causal_forward_hier_hub(
                x,
                freqs,
                paged_cache,
                current_start=0,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                membership=membership,
                global_only=global_only,
                active_group=active,
                direct_player=True,
            )

            # Exercise topology-dependent re-plan, then A -> B -> A. The last
            # A uses a fresh physical cache but must recover the first result.
            membership_b = torch.zeros_like(membership)
            membership_b[:, 0, (0, 3)] = True
            membership_b[:, 1, (1, 2)] = True
            global_only_b = torch.zeros_like(global_only)
            cache_b = copy.deepcopy(base_cache)
            output_b = layer.causal_forward_hier_hub(
                x,
                freqs,
                cache_b,
                current_start=0,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                membership=membership_b,
                global_only=global_only_b,
                active_group=membership_b.any(dim=-1),
                direct_player=True,
            )
            cache_a_again = copy.deepcopy(base_cache)
            output_a_again = layer.causal_forward_hier_hub(
                x,
                freqs,
                cache_a_again,
                current_start=0,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                membership=membership,
                global_only=global_only,
                active_group=active,
                direct_player=True,
            )
    finally:
        if previous is None:
            os.environ.pop("WAN_HIER_INFER_KERNEL", None)
        else:
            os.environ["WAN_HIER_INFER_KERNEL"] = previous
        from multics.models.kernels.hier_paged_attention import (
            clear_hier_paged_state,
        )

        clear_hier_paged_state()

    delta = output.float() - reference.float()
    relative_l2 = delta.norm() / reference.float().norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(output_b).all()
    assert torch.equal(output, output_repeat)
    assert torch.equal(output, output_a_again)
    assert float(relative_l2) <= 0.007
    assert float(cosine) >= 0.99998
    print(
        "Paged-KV GPU PASS "
        f"relative_l2={float(relative_l2):.7f} "
        f"cosine={float(cosine):.7f} A-repeat/A-B-A=bitwise"
    )


if __name__ == "__main__":
    test_paged_kv_matches_membership_sdpa()
    print("ok")
