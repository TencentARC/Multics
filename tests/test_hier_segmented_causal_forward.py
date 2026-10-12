"""Integrated cached hierarchical SDPA/segmented forward comparison on CUDA."""

import os

import pytest
import torch

from multics.models.wan_video_dit import SelfAttention


def _stream_cache(batch, capacity, heads, head_dim, device):
    return {
        "k": torch.zeros(
            batch, capacity, heads, head_dim, dtype=torch.bfloat16, device=device
        ),
        "v": torch.zeros(
            batch, capacity, heads, head_dim, dtype=torch.bfloat16, device=device
        ),
        "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
        "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
    }


def _hier_cache(P, N, L, KL, KG, retention, heads, head_dim, device):
    agent = [_stream_cache(1, retention * L, heads, head_dim, device) for _ in range(P)]
    local = [
        _stream_cache(1, retention * KL, heads, head_dim, device) for _ in range(N)
    ]
    for cache in local:
        cache["active"] = torch.zeros(retention * KL, dtype=torch.bool, device=device)
    return {
        "hier_hub": True,
        "sparse_hub": True,
        "num_players": P,
        "num_groups": N,
        "num_local_hub_tokens": KL,
        "num_global_hub_tokens": KG,
        "agent_caches": agent,
        "local_hubs": local,
        "global_hub": _stream_cache(1, retention * max(KG, 1), heads, head_dim, device),
    }


def _cache_tensors(cache):
    streams = cache["agent_caches"] + cache["local_hubs"] + [cache["global_hub"]]
    for stream in streams:
        yield stream["k"]
        yield stream["v"]
        yield stream["global_end_index"]
        yield stream["local_end_index"]
        if "active" in stream:
            yield stream["active"]


@pytest.mark.parametrize("backend", ["segmented", "tiled128"])
def test_causal_forward_sparse_kernel_matches_sdpa(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(11)
    device = torch.device("cuda")
    P, N, L, KL, KG = 4, 2, 40, 3, 2
    n_frames, n_block, local_attn = 2, 2, 4
    heads, head_dim, dim = 4, 128, 512
    retention = local_attn + n_frames
    seq_len = P * n_frames * L + n_frames * (N * KL + KG)

    layer = (
        SelfAttention(dim=dim, num_heads=heads)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    layer.num_frame_per_block = n_block
    layer.local_attn_size = local_attn
    cache_sdpa = _hier_cache(P, N, L, KL, KG, retention, heads, head_dim, device)
    cache_sparse = _hier_cache(P, N, L, KL, KG, retention, heads, head_dim, device)
    freqs = torch.ones(seq_len, 1, head_dim // 2, dtype=torch.complex128, device=device)

    worst_rel_l2 = 0.0
    with torch.inference_mode():
        # Four blocks cover initial append, a multi-KV-tile steady state, dynamic
        # grouping between blocks, and rolling-cache eviction.
        for block in range(4):
            x = torch.randn(1, seq_len, dim, dtype=torch.bfloat16, device=device)
            labels_row = (0, 0, 1, 1) if block % 2 == 0 else (0, 1, 0, 1)
            labels = torch.tensor(
                [labels_row] * n_frames, dtype=torch.long, device=device
            )
            active = torch.ones(n_frames, N, dtype=torch.bool, device=device)
            current_start = block * n_frames * P * L

            os.environ["WAN_HIER_INFER_KERNEL"] = "sdpa"
            expected = layer.causal_forward_hier_hub(
                x,
                freqs,
                cache_sdpa,
                current_start=current_start,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                group_labels=labels,
                active_group=active,
                direct_player=True,
            )
            os.environ["WAN_HIER_INFER_KERNEL"] = backend
            actual = layer.causal_forward_hier_hub(
                x,
                freqs,
                cache_sparse,
                current_start=current_start,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                group_labels=labels,
                active_group=active,
                direct_player=True,
            )

            diff = (actual.float() - expected.float()).flatten()
            rel_l2 = float(
                diff.norm() / expected.float().flatten().norm().clamp_min(1e-12)
            )
            worst_rel_l2 = max(worst_rel_l2, rel_l2)
            assert torch.isfinite(actual).all()
            assert rel_l2 < 0.005, (block, rel_l2)
            for left, right in zip(
                _cache_tensors(cache_sdpa), _cache_tensors(cache_sparse)
            ):
                assert torch.equal(left, right), block

    os.environ["WAN_HIER_INFER_KERNEL"] = "sdpa"
    print(
        f"integrated {backend} causal forward: worst_rel_l2={worst_rel_l2:.6g}, cache exact"
    )


if __name__ == "__main__":
    test_causal_forward_sparse_kernel_matches_sdpa("segmented")
    test_causal_forward_sparse_kernel_matches_sdpa("tiled128")
