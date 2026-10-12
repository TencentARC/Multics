"""Logical-edge tests for the cached hierarchical segmented metadata.

This test does not launch Triton.  It independently materializes the current
multi-SDPA rules and checks that every segmented query stream selects the same
physical K/V indices for several player counts, direct modes, and inactive
groups.  Run directly or through pytest.
"""

import itertools

import torch

from multics.models.kernels.hier_segmented_attention import _build_metadata


def _temporal(qf, kf, n_block, local_attn):
    return (kf // n_block) <= (qf // n_block) and (
        local_attn == -1 or (qf - kf) < local_attn
    )


def _oracle_segments(
    *,
    P,
    N,
    L,
    KL,
    KG,
    n_frames,
    n_block,
    local_attn,
    agent_frames,
    local_frames,
    global_frames,
    local_active,
    labels,
    active,
    direct,
):
    """Independent token-by-token form of causal_forward_hier_hub masks."""
    q_frames = global_frames[-n_frames:] if KG else agent_frames[0][-n_frames:]
    fa, fl = len(agent_frames[0]), len(local_frames[0])
    agent_base = 0
    local_base = P * fa * L
    global_base = local_base + N * fl * KL

    def agent_ids(player, frame_idx):
        start = agent_base + (player * fa + frame_idx) * L
        return range(start, start + L)

    def local_ids(group, frame_idx):
        start = local_base + (group * fl + frame_idx) * KL
        return range(start, start + KL)

    def global_ids(frame_idx):
        start = global_base + frame_idx * KG
        return range(start, start + KG)

    expected = []
    for player, (qrel, qf) in itertools.product(range(P), enumerate(q_frames)):
        selected = []
        for fi, kf in enumerate(agent_frames[player]):
            if _temporal(qf, kf, n_block, local_attn):
                selected.extend(agent_ids(player, fi))
        if direct:
            for other in range(P):
                if other == player or labels[qrel][other] != labels[qrel][player]:
                    continue
                for fi, kf in enumerate(agent_frames[other]):
                    if kf in q_frames and kf // n_block == qf // n_block:
                        selected.extend(agent_ids(other, fi))
        group = labels[qrel][player]
        for fi, kf in enumerate(local_frames[group]):
            if local_active[group][fi] and _temporal(qf, kf, n_block, local_attn):
                selected.extend(local_ids(group, fi))
        expected.append(selected)

    for qrel, qf in enumerate(q_frames):
        for group in range(N):
            selected = []
            if active[qrel][group]:
                for player in range(P):
                    if labels[qrel][player] != group:
                        continue
                    for fi, kf in enumerate(agent_frames[player]):
                        if _temporal(qf, kf, n_block, local_attn):
                            selected.extend(agent_ids(player, fi))
                for fi, kf in enumerate(local_frames[group]):
                    if local_active[group][fi] and _temporal(
                        qf, kf, n_block, local_attn
                    ):
                        selected.extend(local_ids(group, fi))
                for fi, kf in enumerate(global_frames):
                    if _temporal(qf, kf, n_block, local_attn):
                        selected.extend(global_ids(fi))
            expected.append(selected)

    for qf in q_frames:
        selected = []
        for group in range(N):
            for fi, kf in enumerate(local_frames[group]):
                if local_active[group][fi] and _temporal(qf, kf, n_block, local_attn):
                    selected.extend(local_ids(group, fi))
        for fi, kf in enumerate(global_frames):
            if _temporal(qf, kf, n_block, local_attn):
                selected.extend(global_ids(fi))
        expected.append(selected)
    return expected


def test_segmented_metadata_matches_cached_sdpa_edges():
    for P in (2, 4, 6, 8):
        for direct in (False, True):
            # Leave the last group inactive to exercise the cached baseline's
            # all-masked/zero local-query behavior.
            N = max(2, P // 2 + 1)
            L, KL, KG = 3, 2, 2
            n_frames, n_block, local_attn = 2, 2, 4
            frames = tuple(range(1, 7))
            agent_frames = tuple(frames for _ in range(P))
            local_frames = tuple(frames for _ in range(N))
            global_frames = frames
            labels_row = tuple(player % (N - 1) for player in range(P))
            labels = (labels_row,) * n_frames
            active_row = tuple(group in labels_row for group in range(N))
            active = (active_row,) * n_frames
            local_active = tuple(
                tuple(
                    (frame + group) % 3 != 0 and active_row[group] for frame in frames
                )
                for group in range(N)
            )

            metadata = _build_metadata(
                device="cpu",
                P=P,
                N=N,
                L=L,
                KL=KL,
                KG=KG,
                n_frames=n_frames,
                n_block=n_block,
                local_attn=local_attn,
                agent_frame_lists=agent_frames,
                local_frame_lists=local_frames,
                global_frame_list=global_frames,
                local_active_frames=local_active,
                labels=labels,
                q_active=active,
                direct_player=direct,
                block_m=16,
            )
            kv_index, kv_count = metadata[3].cpu(), metadata[4].cpu()
            actual = [
                kv_index[i, : int(kv_count[i])].tolist() for i in range(len(kv_count))
            ]
            expected = _oracle_segments(
                P=P,
                N=N,
                L=L,
                KL=KL,
                KG=KG,
                n_frames=n_frames,
                n_block=n_block,
                local_attn=local_attn,
                agent_frames=agent_frames,
                local_frames=local_frames,
                global_frames=global_frames,
                local_active=local_active,
                labels=labels,
                active=active,
                direct=direct,
            )
            assert actual == expected, (P, direct)


if __name__ == "__main__":
    test_segmented_metadata_matches_cached_sdpa_edges()
    print("segmented metadata matches cached SDPA edges")
