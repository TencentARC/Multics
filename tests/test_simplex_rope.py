"""Unit tests for Simplex 4D RoPE utilities."""

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multics.models.wan_video_dit import (  # noqa: E402
    WAN_REROPE_SPLITS,
    build_multiplayer_freqs_4d,
    build_simplex_vertices,
    build_spatial_freqs_4d_single_player,
    precompute_freqs_cis_4d_wan_rerope,
)


def test_simplex_vertices_equidistance():
    m, d_a = 4, WAN_REROPE_SPLITS[1]
    verts = build_simplex_vertices(m, d_a)
    assert verts.shape == (m, d_a // 2)
    norms = verts.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    expected = 2 * m / (m - 1)
    for i in range(m):
        for j in range(i + 1, m):
            dist_sq = (verts[i] - verts[j]).pow(2).sum().item()
            assert abs(dist_sq - expected) < 1e-4, (i, j, dist_sq, expected)


def test_4d_freqs_shape_and_agent_constancy():
    head_dim = 128
    freqs_bands = precompute_freqs_cis_4d_wan_rerope(head_dim)
    verts = build_simplex_vertices(4, freqs_bands["d_a"])
    f, h, w = 3, 4, 5
    freqs_p0 = build_spatial_freqs_4d_single_player(freqs_bands, f, h, w, verts, 0)
    freqs_p1 = build_spatial_freqs_4d_single_player(freqs_bands, f, h, w, verts, 1)
    assert freqs_p0.shape == (f * h * w, 1, head_dim // 2)
    n_a = freqs_bands["d_a"] // 2
    n_t = WAN_REROPE_SPLITS[0] // 2
    agent_start = n_t
    assert torch.allclose(
        freqs_p0[:, 0, agent_start],
        freqs_p0[0, 0, agent_start].expand_as(freqs_p0[:, 0, agent_start]),
    )
    assert not torch.allclose(
        freqs_p0[0, 0, agent_start : agent_start + n_a],
        freqs_p1[0, 0, agent_start : agent_start + n_a],
    )
    multi = build_multiplayer_freqs_4d(freqs_bands, f, h, w, [0, 1], verts)
    assert multi.shape[0] == 2 * f * h * w


def test_wan_model_forward_smoke():
    from multics.models.wan_video_dit import WanModel

    dim = 512
    dit = WanModel(
        dim=dim,
        in_dim=48,
        ffn_dim=1024,
        out_dim=48,
        text_dim=64,
        freq_dim=64,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
    )
    dit.enable_simplex_rope(pool_size=4, gamma=1.0)
    b, c, f, h, w = 2, 48, 3, 4, 4
    x = torch.randn(b, c, f, h, w)
    context = torch.randn(1, 8, 64)
    mouse = torch.randn(2, 11, 2)
    keyboard = torch.randn(2, 11, 23)
    dit.keyboard_embedding = torch.nn.Linear(23, dim)
    block = dit.blocks[0]
    block.action_mlp = torch.nn.Sequential(
        torch.nn.LayerNorm(dim + 24),
        torch.nn.Linear(dim + 24, dim),
    )
    block.action_temporal_attn = copy.deepcopy(block.self_attn)
    block.action_cross_attn = copy.deepcopy(block.cross_attn)
    dit.enable_stage2(
        num_hub_tokens=4,
        local_attn_size=3,
        num_frame_per_block=3,
        attention_mode="hier_hub",
        num_groups=1,
        num_local_hub_tokens=2,
        num_global_hub_tokens=2,
        attn_impl="sdpa",
    )
    out = dit(
        x,
        timestep=torch.full((b, f), 500.0),
        context=context,
        mouse=mouse,
        keyboard=keyboard,
        agent_vertex_ids=[0, 1],
        group_labels=torch.zeros(1, 3, 2, dtype=torch.long),
        active_group=torch.ones(1, 3, 1, dtype=torch.bool),
    )
    assert out.shape == x.shape


if __name__ == "__main__":
    test_simplex_vertices_equidistance()
    test_4d_freqs_shape_and_agent_constancy()
    test_wan_model_forward_smoke()
    print("All simplex RoPE tests passed.")
