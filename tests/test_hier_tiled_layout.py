import torch

from multics.models.kernels.hier_tiled_attention import (
    build_hier_query_active_mask,
    build_hier_tiled_cached_layout,
    build_hier_tiled_full_sequence_layouts,
    clear_hier_tiled_layout_cache,
)
from multics.models.wan_video_dit import build_stage2_hier_self_attn_mask


def _layout_edges(layout):
    """Expand packed metadata into query-row -> logical key-index sets."""
    kv_index = layout.kv_index.cpu()
    kv_valid = layout.kv_valid.cpu().bool()
    kv_count = layout.kv_count.cpu()
    qbase = layout.prog_qbase.cpu()
    nq = layout.prog_nq.cpu()
    prog_seg = layout.prog_seg.cpu()
    rows = {}
    for base, length, segment in zip(qbase.tolist(), nq.tolist(), prog_seg.tolist()):
        count = int(kv_count[segment])
        keys = set(kv_index[segment, :count][kv_valid[segment, :count]].tolist())
        for query in range(base, base + length):
            assert query not in rows
            rows[query] = keys
    return rows


def test_full_sequence_tiled_layout_matches_dense_hierarchical_edges():
    clear_hier_tiled_layout_cache()
    P, T, L = 4, 9, 7
    N, KL, KG = 3, 2, 3
    n, window = 3, 4
    labels = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    active = torch.zeros(T, N, dtype=torch.bool)
    active.scatter_(1, labels, True)

    agent_len = P * T * L
    local_start = agent_len
    global_start = local_start + T * N * KL
    for direct in (False, True):
        layouts = build_hier_tiled_full_sequence_layouts(
            device=torch.device("cpu"),
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
            direct_player=direct,
        )
        dense = build_stage2_hier_self_attn_mask(
            P, T, L, N, KL, KG, n, window, labels, active, direct
        )
        all_rows = {}
        for layout in layouts:
            all_rows.update(_layout_edges(layout))
            assert layout.physical_slots % 128 == 0
            assert (
                layout.logical_slots
                <= layout.physical_slots
                <= layout.dense_slots + 128 * dense.shape[0]
            )

        assert len(all_rows) == dense.shape[0]
        for query in range(dense.shape[0]):
            if local_start <= query < global_start:
                rel = query - local_start
                frame = rel // (N * KL)
                group = (rel % (N * KL)) // KL
                if not bool(active[frame, group]):
                    # Flex has a numerical-safety self edge, then STE masks this
                    # row to zero. Tiled represents it directly as an empty row.
                    assert all_rows[query] == set()
                    continue
            expected = set(dense[query].nonzero(as_tuple=False).flatten().tolist())
            assert all_rows[query] == expected


def test_hier_query_active_mask_uses_full_sequence_token_order():
    P, T, L, N, KL, KG = 2, 3, 4, 3, 2, 1
    active = torch.tensor(
        [[True, True, False], [True, False, True], [False, True, True]]
    )
    mask = build_hier_query_active_mask(
        active_group=active,
        num_players=P,
        num_frames=T,
        num_patches=L,
        num_groups=N,
        num_local_hub_tokens=KL,
        num_global_hub_tokens=KG,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).reshape(-1)
    agent_len = P * T * L
    local_len = T * N * KL
    assert torch.equal(mask[:agent_len], torch.ones(agent_len))
    expected_local = active[:, :, None].expand(T, N, KL).reshape(-1).float()
    assert torch.equal(mask[agent_len : agent_len + local_len], expected_local)
    assert torch.equal(mask[agent_len + local_len :], torch.ones(T * KG))


def test_full_sequence_layout_preserves_cached_tiled128_slot_order():
    P, T, L = 4, 9, 5
    N, KL, KG, n, window = 2, 2, 3, 3, 4
    labels = torch.tensor(
        [[0, 0, 1, 1]] * 3 + [[0, 1, 0, 1]] * 3 + [[1, 1, 0, 0]] * 3,
        dtype=torch.long,
    )
    active = torch.ones(T, N, dtype=torch.bool)
    full_layouts = build_hier_tiled_full_sequence_layouts(
        device=torch.device("cpu"),
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

    agent_len = P * T * L
    local_start = agent_len
    global_start = local_start + T * N * KL
    retention = window + n
    for block_index, block_start in enumerate(range(0, T, n)):
        block_end = block_start + n
        retained = tuple(range(max(0, block_end - retention), block_end))
        fa = len(retained)
        cached = build_hier_tiled_cached_layout(
            device=torch.device("cpu"),
            P=P,
            N=N,
            L=L,
            KL=KL,
            KG=KG,
            n_frames=n,
            n_block=n,
            local_attn=window,
            agent_frame_lists=(retained,) * P,
            local_frame_lists=(retained,) * N,
            global_frame_list=retained,
            local_active_frames=tuple(
                tuple(bool(active[frame, group]) for frame in retained)
                for group in range(N)
            ),
            labels=tuple(
                tuple(int(x) for x in row)
                for row in labels[block_start:block_end].tolist()
            ),
            q_active=tuple(
                tuple(bool(x) for x in row)
                for row in active[block_start:block_end].tolist()
            ),
            direct_player=True,
        )
        full = full_layouts[block_index]
        assert torch.equal(full.kv_count, cached.kv_count)
        assert torch.equal(full.kv_valid, cached.kv_valid)

        def cached_to_full(index):
            index = int(index)
            agent_pool = P * fa * L
            local_pool = N * fa * KL
            if index < agent_pool:
                player, rem = divmod(index, fa * L)
                frame_index, token = divmod(rem, L)
                return (player * T + retained[frame_index]) * L + token
            if index < agent_pool + local_pool:
                rel = index - agent_pool
                group, rem = divmod(rel, fa * KL)
                frame_index, token = divmod(rem, KL)
                return local_start + (retained[frame_index] * N + group) * KL + token
            rel = index - agent_pool - local_pool
            frame_index, token = divmod(rel, KG)
            return global_start + retained[frame_index] * KG + token

        for segment in range(full.kv_count.numel()):
            count = int(full.kv_count[segment])
            valid = full.kv_valid[segment, :count].bool()
            full_indices = full.kv_index[segment, :count][valid].tolist()
            cached_indices = cached.kv_index[segment, :count][valid].tolist()
            assert [cached_to_full(index) for index in cached_indices] == full_indices
