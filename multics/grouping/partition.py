"""Bounded clique cover with anchors and idle-hub recovery."""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from collections.abc import Sequence
import numpy as np
from .budget import Budget
from .relevance import threshold_graph

MAX_MEMBERSHIPS = 2
MAX_PAIRWISE_GROUP_OVERLAP = 1
OVERLAP_PENALTY = 0.15
BEAM_SIZE = 300
MAX_MAXIMAL_CLIQUES = 32


@dataclass
class FlexGrouping:
    """One target period's Local groups and Global-only players."""

    groups: list[list[int]] = field(default_factory=list)
    global_only: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def pair_used(self) -> int:
        return 2 * len(pairs_of(self.groups))

    @property
    def shared(self) -> list[int]:
        counts: dict[int, int] = {}
        for group in self.groups:
            for player in group:
                counts[player] = counts.get(player, 0) + 1
        return sorted(player for player, count in counts.items() if count > 1)


def pairs_of(groups: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    """Unique undirected player pairs made direct by this topology."""
    return {
        pair for group in groups for pair in itertools.combinations(sorted(group), 2)
    }


def group_mass(scores: np.ndarray, group: Sequence[int]) -> float:
    return float(sum(scores[i, j] for i, j in itertools.combinations(sorted(group), 2)))


def group_score(scores: np.ndarray, group: Sequence[int]) -> float:
    return group_mass(scores, group) / len(group) if len(group) >= 2 else 0.0


def cohesion(scores: np.ndarray, group: Sequence[int]) -> float:
    pairs = list(itertools.combinations(sorted(group), 2))
    if not pairs:
        return 0.0
    return float(sum(scores[i, j] for i, j in pairs) / len(pairs))


def clique_candidates(
    graph: np.ndarray, max_size: int, max_maximal: int = MAX_MAXIMAL_CLIQUES
) -> list[list[int]]:
    """Bounded maximal cliques plus all threshold-edge fallback pairs."""
    player_count = len(graph)
    maximal: list[tuple[int, ...]] = []
    edges: list[tuple[int, ...]] = []
    for size in range(2, min(max_size, player_count) + 1):
        for group in itertools.combinations(range(player_count), size):
            if not all(graph[i, j] > 0 for i, j in itertools.combinations(group, 2)):
                continue
            if size == 2:
                edges.append(group)
            members = set(group)
            extendable = size < max_size and any(
                player not in members
                and all(graph[player, member] > 0 for member in group)
                for player in range(player_count)
            )
            if not extendable:
                maximal.append(group)

    if len(maximal) > max_maximal:
        ranked = sorted(
            maximal,
            key=lambda group: (
                group_mass(graph, group),
                len(group),
                tuple(-x for x in group),
            ),
            reverse=True,
        )
        retained = ranked[:max_maximal]
        for player in range(player_count):
            if any(player in group for group in maximal) and not any(
                player in group for group in retained
            ):
                retained.append(next(group for group in ranked if player in group))
        maximal = retained

    result: list[list[int]] = []
    for candidate in maximal + edges:
        group = list(candidate)
        if group not in result:
            result.append(group)
    return result


def _membership_counts(
    groups: Sequence[Sequence[int]], player_count: int
) -> tuple[int, ...]:
    return tuple(
        sum(player in group for group in groups) for player in range(player_count)
    )


def topology_constraints_hold(
    groups: Sequence[Sequence[int]],
    *,
    player_count: int,
    max_groups: int,
    max_size: int,
    pair_budget: int,
    max_memberships: int = MAX_MEMBERSHIPS,
    max_pairwise_overlap: int = MAX_PAIRWISE_GROUP_OVERLAP,
    budget: Budget | None = None,
) -> bool:
    """Full structural and communication feasibility predicate."""
    if len(groups) > max_groups:
        return False
    if any(len(group) < 2 or len(group) > max_size for group in groups):
        return False
    if max(_membership_counts(groups, player_count), default=0) > max_memberships:
        return False
    if any(
        len(set(left) & set(right)) > max_pairwise_overlap
        for left, right in itertools.combinations(groups, 2)
    ):
        return False
    if 2 * len(pairs_of(groups)) > pair_budget:
        return False
    return budget is None or budget.affordable_groups(groups, n_players=player_count)


def _selection_key(
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    budget: Budget,
    player_count: int,
) -> tuple:
    edges = pairs_of(groups)
    covered = {player for group in groups for player in group}
    edge_mass = float(sum(scores[i, j] for i, j in edges))
    extra_memberships = sum(len(group) for group in groups) - len(covered)
    adjusted = edge_mass - OVERLAP_PENALTY * extra_memberships
    cost = (
        budget.topology_cost(groups, n_players=player_count)
        if budget is not None
        else 2 * len(edges)
    )
    return (len(covered), adjusted, edge_mass, -extra_memberships, -cost)


def select_semantic_groups(
    graph: np.ndarray,
    candidates: Sequence[Sequence[int]],
    *,
    max_groups: int,
    max_size: int,
    pair_budget: int,
    budget: Budget | None,
    beam_size: int = BEAM_SIZE,
) -> list[list[int]]:
    """Bounded deterministic search over feasible overlapping clique covers."""
    player_count = len(graph)
    pool = sorted(
        (sorted(group) for group in candidates),
        key=lambda group: (
            len(group),
            group_mass(graph, group),
            tuple(-x for x in group),
        ),
        reverse=True,
    )
    candidate_pairs = [pairs_of([group]) for group in pool]
    # state = chosen candidate indices, groups, unique edges
    beam: list[tuple[tuple[int, ...], list[list[int]], frozenset]] = [
        ((), [], frozenset())
    ]
    feasible = []
    for _ in range(max_groups):
        expanded = []
        for indices, groups, edges in beam:
            start = indices[-1] + 1 if indices else 0
            for index in range(start, len(pool)):
                # Opening a Hub that contributes no new direct edge is wasteful.
                if candidate_pairs[index] <= edges:
                    continue
                trial = groups + [pool[index]]
                if not topology_constraints_hold(
                    trial,
                    player_count=player_count,
                    max_groups=max_groups,
                    max_size=max_size,
                    pair_budget=pair_budget,
                    budget=budget,
                ):
                    continue
                expanded.append(
                    (
                        indices + (index,),
                        trial,
                        edges | candidate_pairs[index],
                    )
                )
        if not expanded:
            break
        expanded.sort(
            key=lambda state: _selection_key(
                graph, state[1], budget=budget, player_count=player_count
            ),
            reverse=True,
        )
        beam = expanded[:beam_size]
        feasible.extend(beam)
    if not feasible:
        return []
    return max(
        feasible,
        key=lambda state: _selection_key(
            graph, state[1], budget=budget, player_count=player_count
        ),
    )[1]


def recover_new_style(
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    max_groups: int,
    max_size: int,
    pair_budget: int,
    budget: Budget | None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Allocate idle Hub slots without a score floor, under hard constraints.

    Uncovered players are spread over all remaining Hub slots.  If this yields
    a singleton, an already covered player is reused as an Anchor.  Candidate
    groups are ranked by total raw smoothed relevance, then weakest edge, with
    no minimum score.  Every accepted candidate must also satisfy the
    membership/overlap constraints, the N/K/pair limits, and the exact
    centralized communication cap.
    """
    result = [sorted(group) for group in groups]
    filled: list[list[int]] = []
    player_count = len(scores)
    if max_size < 2:
        return result, filled

    while len(result) < max_groups:
        counts = _membership_counts(result, player_count)
        uncovered = [player for player, count in enumerate(counts) if count == 0]
        if not uncovered:
            break

        slots_left = max_groups - len(result)
        take = min(max_size, max(1, -(-len(uncovered) // slots_left)))
        take = min(take, len(uncovered))
        candidates: list[list[int]] = []
        for subset in itertools.combinations(uncovered, take):
            candidate = list(subset)
            if len(candidate) == 1:
                anchors = [
                    player
                    for player, count in enumerate(counts)
                    if 0 < count < MAX_MEMBERSHIPS
                ]
                candidates.extend(sorted(candidate + [anchor]) for anchor in anchors)
            else:
                candidates.append(sorted(candidate))

        def rank(group: Sequence[int]) -> tuple[float, float, tuple[int, ...]]:
            edges = [
                float(scores[left, right])
                for left, right in itertools.combinations(group, 2)
            ]
            return (
                sum(edges),
                min(edges) if edges else 0.0,
                tuple(-player for player in group),
            )

        candidates.sort(key=rank, reverse=True)
        chosen = next(
            (
                candidate
                for candidate in candidates
                if topology_constraints_hold(
                    result + [candidate],
                    player_count=player_count,
                    max_groups=max_groups,
                    max_size=max_size,
                    pair_budget=pair_budget,
                    budget=budget,
                )
            ),
            None,
        )
        if chosen is None:
            break
        result.append(chosen)
        filled.append(chosen)
    return result, filled


def flexible_partition(scores: np.ndarray, *, budget: Budget) -> FlexGrouping:
    """Coverage-first semantic groups, followed by no-floor idle-hub recovery."""
    player_count = len(scores)
    if scores.shape != (player_count, player_count):
        raise ValueError(f"scores must be square, got {scores.shape}")
    if budget.max_size < 2 or budget.pair_budget < 2:
        return FlexGrouping(global_only=list(range(player_count)))
    constraints = dict(
        max_groups=budget.groups,
        max_size=budget.max_size,
        pair_budget=budget.pair_budget,
        budget=budget,
    )
    graph = threshold_graph(scores, 0.30)
    candidates = clique_candidates(graph, budget.max_size)
    semantic = select_semantic_groups(graph, candidates, **constraints)
    groups, filled = recover_new_style(scores, semantic, **constraints)
    covered = {player for group in groups for player in group}
    group_scores = [
        (-1.0 if group in filled else 1.0) * group_score(scores, group)
        for group in groups
    ]
    result = FlexGrouping(
        groups=groups,
        global_only=sorted(set(range(player_count)) - covered),
        scores=group_scores,
    )
    if not topology_constraints_hold(
        result.groups, player_count=player_count, **constraints
    ):
        raise RuntimeError(f"Internal error: infeasible grouping {result.groups}")
    return result
