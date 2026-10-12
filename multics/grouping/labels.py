"""Stable group slots and the compatible 24-tensor label schema."""

from __future__ import annotations
import numpy as np
from .partition import cohesion, flexible_partition
from .relevance import relevance_periods, threshold_graph
from .budget import resolve
from .state import eligible_players

LATENT_STRIDE = 4


def latent_frame_count(num_frames: int, stride: int = LATENT_STRIDE) -> int:
    return (num_frames - 1) // stride + 1


def track_labels(seq: list[list[list[int]]], n_slots: int) -> list[list[int]]:
    """Match slots by intersection, Jaccard, then stable group/slot indices."""
    out: list[list[int]] = []
    prev: list[tuple[int, set[int]]] = []  # 上一周期的 (编号, 成员集合)
    for groups in seq:
        sets = [set(g) for g in groups]
        cand = []
        for gi, s in enumerate(sets):
            for ci, ps in prev:
                inter = len(s & ps)
                if inter:  # 交集大的先配；gi/ci 兜底定序，保证可复现
                    cand.append((-inter, -inter / len(s | ps), gi, ci))
        cand.sort()

        lab = [-1] * len(groups)
        taken: set[int] = set()
        for _, _, gi, ci in cand:
            if lab[gi] < 0 and ci not in taken:
                lab[gi] = ci
                taken.add(ci)
        free = [c for c in range(n_slots) if c not in taken]
        for gi in range(len(groups)):
            if lab[gi] < 0:
                lab[gi] = free.pop(0) if free else gi % n_slots
                taken.add(lab[gi])
        out.append(lab)
        prev = [(lab[gi], sets[gi]) for gi in range(len(groups))]
    return out


def flex_label_arrays(
    groupings,
    periods,
    keep: np.ndarray,
    *,
    n_players: int,
    max_groups: int,
    start_frame: int,
    num_frames: int,
    threshold: float,
    budget,
    stride: int = LATENT_STRIDE,
) -> dict[str, np.ndarray]:
    """把逐周期的 `FlexGrouping` 摊成定长 tensor 字典。

    列序 = CSV 的 `video_path1..P`（world 按 `row.slots` 加载时，world 下标 == 列下标）。
    组号经 `track_labels` 跨周期跟踪，槽位落在 `0 .. G-1`（G = max_groups）。
    """
    P, G, K = n_players, max_groups, len(groupings)
    if K == 0:
        raise ValueError("没有更新周期，写不出标签")
    labs = track_labels([g.groups for g in groupings], n_slots=max(G, 1))

    p_start = np.zeros(K, np.int32)
    p_frames = np.zeros(K, np.int32)
    p_mem = np.zeros((K, G, P), bool)
    p_go = np.zeros((K, P), bool)
    p_anchor = np.zeros((K, P), bool)
    p_score = np.full((K, G), -1.0, np.float32)
    p_size = np.zeros((K, G), np.int8)
    p_coh = np.full((K, G), -1.0, np.float32)
    p_pair = np.zeros(K, np.int32)
    p_cost = np.full(K, -1.0, np.float32)

    eligible = np.zeros(P, bool)
    eligible[np.asarray(keep, int)] = True

    for k, (gr, per, sid) in enumerate(zip(groupings, periods, labs)):
        p_start[k] = int(per.start_frame - start_frame)
        p_frames[k] = int(per.num_frames)
        W = threshold_graph(per.smooth, threshold)
        for gi, g in enumerate(gr.groups):
            s = int(sid[gi]) % G
            p_size[k, s] = len(g)
            p_score[k, s] = gr.scores[gi] if gi < len(gr.scores) else 0.0
            p_coh[k, s] = cohesion(W, g) if len(g) > 1 else 0.0
            for j in g:
                p_mem[k, s, int(keep[j])] = True
        for j in gr.global_only:
            p_go[k, int(keep[j])] = True
        for j in gr.shared:
            p_anchor[k, int(keep[j])] = True
        p_pair[k] = gr.pair_used
        # gr.groups 的下标是 keep 内部下标；未进 keep 的 player 在实际
        # membership/global_only 里两者都是 False，不能误算成 global-only。
        p_cost[k] = budget.topology_cost(gr.groups, n_players=len(keep))
    has = p_mem.any(axis=1)
    p_lab = np.where(has, p_mem.argmax(axis=1).astype(np.int8), np.int8(-1)).astype(
        np.int8
    )

    raw = np.minimum(
        np.arange(latent_frame_count(num_frames, stride)) * stride,
        max(num_frames - 1, 0),
    )
    idx = np.clip(np.searchsorted(p_start, raw, side="right") - 1, 0, K - 1)

    n_local = has.sum(axis=1).astype(np.float32)
    n_go = p_go.sum(axis=1).astype(np.float32)
    n_anchor = p_anchor.sum(axis=1).astype(np.float32)
    denom = np.float32(max(P, 1))

    membership_latent = p_mem[idx]
    global_only_latent = p_go[idx]
    clip_cost = budget.topology_cost_timeline(membership_latent, global_only_latent)
    if clip_cost > budget.cost_limit:
        raise RuntimeError(
            f"整段 dynamic membership 通信超预算：{clip_cost:,.0f} > "
            f"{budget.cost_limit:,.0f}"
        )
    return {
        "membership": membership_latent,
        "global_only": global_only_latent,
        "eligible": eligible,
        "labels": p_lab[idx],
        "latent_period": idx.astype(np.int16),
        "period_membership": p_mem,
        "period_global_only": p_go,
        "period_anchor": p_anchor,
        "period_labels": p_lab,
        "period_start": p_start,
        "period_frames": p_frames,
        "period_scores": p_score,
        "period_sizes": p_size,
        "period_cohesion": p_coh,
        "period_pair_used": p_pair,
        "period_total_cost": p_cost,
        "clip_total_cost": np.array(clip_cost, dtype=np.float64),
        "cost_limit": np.array(budget.cost_limit, dtype=np.float64),
        "coverage": n_local / denom,
        "orphan_ratio": n_go / denom,
        "anchor_ratio": n_anchor / denom,
        "pair_budget": np.array(budget.pair_budget, dtype=np.int32),
        "max_groups": np.array(G, dtype=np.int32),
        "max_size": np.array(budget.max_size, dtype=np.int32),
    }


def generate_labels(state):
    """Generate pose-only labels without model state or random-number draws."""
    state.validate()
    if state.clip_num_frames != 165:
        raise ValueError(
            "Automatic grouping currently requires 165 RGB frames per CSV row"
        )
    budget = resolve(state.num_players)
    span = state.frame_window(state.clip_start_frame, state.clip_num_frames)
    keep = np.flatnonzero(eligible_players(state, span))
    if len(keep) < 2:
        raise ValueError("Fewer than two eligible players in grouping clip")
    periods = relevance_periods(state, span, state.clip_start_frame, keep)
    groups = [flexible_partition(p.smooth, budget=budget) for p in periods]
    return flex_label_arrays(
        groups,
        periods,
        keep,
        n_players=state.num_players,
        max_groups=budget.groups,
        start_frame=state.clip_start_frame,
        num_frames=state.clip_num_frames,
        threshold=0.3,
        budget=budget,
    )
