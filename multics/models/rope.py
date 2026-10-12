import torch
from typing import Tuple
from einops import rearrange

WAN_REROPE_SPLITS = (12, 32, 42, 42)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, freqs_splits=None):
    n, c = (x.size(2), x.size(3) // 2)
    s = x.size(1)
    if freqs_splits is not None:
        freqs = freqs.split(list(freqs_splits), dim=1)
    else:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    if grid_sizes.shape[1] == 3:
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
            )
            freqs_i = torch.cat(
                [
                    freqs[0][start_frame : start_frame + f]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
    elif grid_sizes.shape[1] == 4:
        for i, (f, h, w, p) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w * p
            if len(freqs) == 4:
                thw_parts = [
                    freqs[0][start_frame : start_frame + f]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    freqs[2][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[3][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ]
                base_freqs = torch.cat(thw_parts, dim=-1).reshape(f * h * w, -1)
                agent_band = freqs[1].view(1, -1).expand(f * h * w, -1)
                base_freqs = torch.cat(
                    [
                        base_freqs[:, : freqs[0].shape[-1]],
                        agent_band,
                        base_freqs[:, freqs[0].shape[-1] :],
                    ],
                    dim=-1,
                )
                freqs_i = base_freqs.repeat(p, 1)
            else:
                base_freqs = torch.cat(
                    [
                        freqs[0][start_frame : start_frame + f]
                        .view(f, 1, 1, -1)
                        .expand(f, h, w, -1),
                        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                ).reshape(f * h * w, c)
                freqs_i = base_freqs.repeat_interleave(p, dim=0)
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
            )
            x_i = torch.view_as_real(x_i * freqs_i.unsqueeze(1)).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
    return torch.stack(output).type_as(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return (f_freqs_cis, h_freqs_cis, w_freqs_cis)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim)
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def build_simplex_vertices(pool_size: int, d_agent: int) -> torch.Tensor:
    """Regular simplex vertices in agent angle space, shape [pool_size, d_agent // 2]."""
    m = int(pool_size)
    half = d_agent // 2
    if m < 2:
        raise ValueError("pool_size must be >= 2")
    if m > half + 1:
        raise ValueError(
            f"pool_size {m} exceeds limit d_agent//2+1={half + 1} for d_agent={d_agent}"
        )
    e = torch.eye(m, dtype=torch.float64)
    centered = e - 1.0 / m
    norm = centered.norm(dim=1, keepdim=True).clamp(min=1e-12)
    s = centered / norm
    out = torch.zeros(m, half, dtype=torch.float32)
    out[:, :m] = s.to(torch.float32)
    return out


def precompute_freqs_cis_4d_wan_rerope(
    head_dim: int = 128,
    end: int = 1024,
    theta: float = 10000.0,
    splits: Tuple[int, int, int, int] = WAN_REROPE_SPLITS,
):
    """ReRoPE: agent band from low-freq temporal; spatial h/w unchanged vs Wan 3D."""
    d_t, d_a, d_h, d_w = splits
    assert d_t + d_a + d_h + d_w == head_dim, splits
    d_t_old = head_dim - 2 * (head_dim // 3)
    freqs_t_old = precompute_freqs_cis(d_t_old, end, theta)
    n_t_high = d_t // 2
    freqs_t = freqs_t_old[:, -n_t_high:]
    freqs_h = precompute_freqs_cis(d_h, end, theta)
    freqs_w = precompute_freqs_cis(d_w, end, theta)
    return {"t": freqs_t, "h": freqs_h, "w": freqs_w, "splits": splits, "d_a": d_a}


def _agent_freqs_cis_from_vertex(
    simplex_vertices: torch.Tensor, vertex_id: int, gamma: float = 1.0
) -> torch.Tensor:
    """Complex agent-band multipliers, shape [d_a//2]."""
    angles = simplex_vertices[vertex_id].to(torch.float64) * float(gamma)
    return torch.polar(torch.ones_like(angles), angles)


def build_spatial_freqs_4d_single_player(
    freqs_bands: dict,
    f: int,
    h: int,
    w: int,
    simplex_vertices: torch.Tensor,
    vertex_id: int,
    gamma: float = 1.0,
    device=None,
    start_frame: int = 0,
) -> torch.Tensor:
    """Per-player (f,h,w) grid freqs with simplex agent band, shape [f*h*w, 1, head_dim//2].

    start_frame shifts the temporal band to GLOBAL frame indices so block-causal KV-cache
    inference (one block at a time) matches full-sequence training: block k's video tokens get
    temporal positions [start_frame, start_frame+f) instead of resetting to [0, f).
    """
    d_t, d_a, d_h, d_w = freqs_bands["splits"]
    n_t, n_h, n_w = (d_t // 2, d_h // 2, d_w // 2)
    ft = freqs_bands["t"][start_frame : start_frame + f].to(device)
    fh = freqs_bands["h"][:h].to(device)
    fw = freqs_bands["w"][:w].to(device)
    t_grid = ft.view(f, 1, 1, n_t).expand(f, h, w, n_t)
    h_grid = fh.view(1, h, 1, n_h).expand(f, h, w, n_h)
    w_grid = fw.view(1, 1, w, n_w).expand(f, h, w, n_w)
    spatial_thw = torch.cat([t_grid, h_grid, w_grid], dim=-1).reshape(
        f * h * w, n_t + n_h + n_w
    )
    a_freqs = _agent_freqs_cis_from_vertex(simplex_vertices, vertex_id, gamma).to(
        device
    )
    a_grid = a_freqs.view(1, -1).expand(f * h * w, -1)
    full = torch.cat([spatial_thw[:, :n_t], a_grid, spatial_thw[:, n_t:]], dim=-1)
    return full.unsqueeze(1)


def build_multiplayer_freqs_4d(
    freqs_bands: dict,
    f: int,
    h: int,
    w: int,
    agent_vertex_ids,
    simplex_vertices: torch.Tensor,
    gamma: float = 1.0,
    device=None,
    start_frame: int = 0,
) -> torch.Tensor:
    """Block-concat (p l) freqs for all players, shape [p*f*h*w, 1, head_dim//2]."""
    if device is None:
        device = simplex_vertices.device
    parts = []
    for vid in agent_vertex_ids:
        parts.append(
            build_spatial_freqs_4d_single_player(
                freqs_bands,
                f,
                h,
                w,
                simplex_vertices,
                int(vid),
                gamma,
                device,
                start_frame=start_frame,
            )
        )
    return torch.cat(parts, dim=0)


def build_hub_freqs_4d(
    freqs_bands: dict, f: int, num_hub_tokens: int, start_frame: int = 0, device=None
) -> torch.Tensor:
    """Hub RoPE: temporal phase from frame index; agent/h/w bands are identity."""
    d_t, d_a, d_h, d_w = freqs_bands["splits"]
    n_t, n_a, n_h, n_w = (d_t // 2, d_a // 2, d_h // 2, d_w // 2)
    if device is None:
        device = freqs_bands["t"].device
    ft = freqs_bands["t"][start_frame : start_frame + f].to(device)
    t_grid = ft.repeat_interleave(num_hub_tokens, dim=0)
    identity = torch.ones(
        f * num_hub_tokens, n_a + n_h + n_w, device=device, dtype=torch.complex64
    )
    full = torch.cat([t_grid, identity], dim=-1)
    return full.unsqueeze(1)


def rope_split_sizes_4d(
    head_dim: int, splits: Tuple[int, int, int, int] = WAN_REROPE_SPLITS
):
    """Complex dim counts per (t, a, h, w) band."""
    d_t, d_a, d_h, d_w = splits
    return [d_t // 2, d_a // 2, d_h // 2, d_w // 2]


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)
