"""Communication costs for the fixed label-generation profile."""

from __future__ import annotations
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

TOKENS_PER_PLAYER = 220
LATENT_FRAMES = 42
FRAMES_PER_BLOCK = 3
LOCAL_ATTN_SIZE = 7
LOCAL_HUB_TOKENS = 5
GLOBAL_HUB_TOKENS = 6
DIRECT_FRAC = 1.0
CENTRAL_HUB_TOKENS = {4: 109, 6: 111, 8: 111, 10: 195}


def temporal_window_pairs(
    num_frames: int, frames_per_block: int, local_attn_size: int
) -> int:
    """与 WAN ``m_window`` 一致的有向 `(query_frame, key_frame)` 数 ``A_w``。"""
    if num_frames <= 0 or frames_per_block <= 0:
        raise ValueError("num_frames 与 frames_per_block 必须为正整数")
    if local_attn_size != -1 and local_attn_size < frames_per_block:
        raise ValueError("local_attn_size 必须是 -1 或 >= frames_per_block")
    return sum(
        1
        for q in range(num_frames)
        for k in range(num_frames)
        if k // frames_per_block <= q // frames_per_block
        and (local_attn_size == -1 or q - k < local_attn_size)
    )


def same_block_pairs(num_frames: int, frames_per_block: int) -> int:
    """same-block direct attention 的有向帧对数 ``B_w``。"""
    if num_frames <= 0 or frames_per_block <= 0:
        raise ValueError("num_frames 与 frames_per_block 必须为正整数")
    return sum(
        min(frames_per_block, num_frames - start) ** 2
        for start in range(0, num_frames, frames_per_block)
    )


def _max_size_for(pair_budget: int) -> int:
    """满足 `p(p−1) ≤ pair_budget` 的最大 p。

    走整数路径：`p(p−1) ≤ B` ⟺ `(2p−1)² ≤ 4B+1`。闭式解 `⌊1/2+sqrt(1/4+B)⌋` 在
    `B=6`（K 恰好 3）这种边界上完全依赖 `sqrt(6.25)` 不掉精度，没必要赌。
    """
    return (math.isqrt(4 * max(pair_budget, 0) + 1) + 1) // 2


@dataclass(frozen=True)
class Budget:
    """一个 `(P, N)` 配置下的通信预算。成本是整个 latent clip 的 mask edge 数。"""

    players: int
    groups: int
    tokens: int = TOKENS_PER_PLAYER
    num_frames: int = LATENT_FRAMES
    frames_per_block: int = FRAMES_PER_BLOCK
    local_attn_size: int = LOCAL_ATTN_SIZE
    local_hub: int = LOCAL_HUB_TOKENS
    global_hub: int = GLOBAL_HUB_TOKENS
    central_hub: int = 0
    direct_frac: float = DIRECT_FRAC

    def __post_init__(self) -> None:
        for name in (
            "players",
            "groups",
            "tokens",
            "num_frames",
            "frames_per_block",
            "local_hub",
            "global_hub",
            "central_hub",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正，给的是 {getattr(self, name)!r}")
        # 同时做 local_attn_size 合法性检查。
        temporal_window_pairs(
            self.num_frames, self.frames_per_block, self.local_attn_size
        )
        if not 0.0 <= float(self.direct_frac) <= 1.0:
            raise ValueError(f"direct_frac 必须在 [0,1]，给的是 {self.direct_frac}")

    @property
    def a_w(self) -> int:
        """Hub/history 边的有向帧对数。"""
        return temporal_window_pairs(
            self.num_frames, self.frames_per_block, self.local_attn_size
        )

    @property
    def b_w(self) -> int:
        """same-block direct-player 边的有向帧对数。"""
        return same_block_pairs(self.num_frames, self.frames_per_block)

    @property
    def c_central(self) -> float:
        return self.a_w * (
            2.0 * self.players * self.tokens * self.central_hub + self.central_hub**2
        )

    @property
    def c_hier_base(self) -> float:
        """分层架构不开任何组内直连时的成本。"""
        return self.a_w * (
            2.0 * self.players * self.tokens * self.local_hub
            + self.groups * self.local_hub**2
            + 2.0 * self.groups * self.local_hub * self.global_hub
            + self.global_hub**2
        )

    @property
    def delta_c(self) -> float:
        return self.c_central - self.c_hier_base

    @property
    def budget_cost(self) -> float:
        """可用于组内直连的成本预算，即 centralized 通信盈余 ΔC。"""
        return self.delta_c

    @property
    def cost_limit(self) -> float:
        """最终允许的总成本，始终等于 centralized communication。"""
        return self.c_central

    @property
    def pair_budget(self) -> int:
        """Total directed pair units available under the sum-capacity rule."""
        if self.budget_cost <= 0 or self.direct_frac <= 0:
            return 0
        b = int(
            math.floor(
                self.budget_cost / (self.direct_frac * self.b_w * self.tokens**2)
            )
        )
        return b

    @property
    def sum_max_size(self) -> int:
        """`sum` 口径下的 K：满足 `p(p−1) ≤ B` 的最大 p，只当兜底硬上限。"""
        if self.budget_cost <= 0 or self.direct_frac <= 0:
            return 1
        b = int(
            math.floor(
                self.budget_cost / (self.direct_frac * self.b_w * self.tokens**2)
            )
        )
        return max(1, min(_max_size_for(b), self.players))

    @property
    def max_size(self) -> int:
        """Maximum group size permitted by the shared pair budget."""
        return self.sum_max_size

    @staticmethod
    def direct_pair_units(groups: Sequence[Sequence[int]]) -> int:
        """实际双向 player-player edge units；同一 pair 跨组重复只算一次。"""
        pairs = {
            tuple(sorted((int(i), int(j))))
            for group in groups
            for i, j in itertools.combinations(sorted(set(group)), 2)
        }
        return 2 * len(pairs)

    def direct_cost_groups(self, groups: Sequence[Sequence[int]]) -> float:
        return (
            self.direct_frac
            * self.b_w
            * self.tokens**2
            * self.direct_pair_units(groups)
        )

    def hub_cost(self, active_groups: int) -> float:
        """实际 Hub↔Hub mask 成本。

        active Local Hub 有 ``K_L² + 2K_LK_G``；inactive slot 不连 Global，只有每个
        Local Hub query 的 self edge，所以是 ``K_L``，不是 0 也不是 ``K_L²``。
        """
        active = max(0, min(int(active_groups), self.groups))
        inactive = self.groups - active
        return (
            self.a_w
            * (
                active * self.local_hub**2
                + 2.0 * active * self.local_hub * self.global_hub
                + self.global_hub**2
            )
            # inactive Local query 只在同帧对自己，不走历史窗口。
            + self.num_frames * inactive * self.local_hub
        )

    def topology_cost(
        self,
        groups: Sequence[Sequence[int]],
        n_players: int | None = None,
    ) -> float:
        """按 WAN membership mask 计算「该分组整段不变」时的 clip 通信量。

        ``n_players`` 默认取预算里的 P。未被任何 Local group 覆盖的玩家自动视为
        global-only；Anchor 的重复 Local attachment 由 ``sum(len(g))`` 自然计入。
        """
        gs = [tuple(sorted(set(int(i) for i in g))) for g in groups if len(set(g)) >= 2]
        n = self.players if n_players is None else int(n_players)
        covered = {i for g in gs for i in g}
        if any(i < 0 or i >= n for i in covered):
            raise ValueError(f"group 玩家下标越界：n_players={n}, groups={groups}")
        n_global = n - len(covered)
        attach = self.a_w * (
            2.0 * self.tokens * self.local_hub * sum(len(g) for g in gs)
            + 2.0 * self.tokens * self.global_hub * n_global
        )
        return attach + self.hub_cost(len(gs)) + self.direct_cost_groups(gs)

    def topology_cost_timeline(self, membership, global_only) -> float:
        """逐帧复现 WAN membership mask，精确计整段 dynamic topology 的 logical edges。

        ``membership`` 为 ``[T,N,P]`` bool，``global_only`` 为 ``[T,P]`` bool。
        不计 same-player history，因为它在 flat/hier 两边完全相同。
        """
        if len(membership) != self.num_frames or len(global_only) != self.num_frames:
            raise ValueError(
                f"timeline T 必须等于预算 T={self.num_frames}；得到 "
                f"membership={len(membership)}, global_only={len(global_only)}"
            )
        if len(membership) == 0:
            return 0.0
        if len(membership[0]) != self.groups:
            raise ValueError(
                f"timeline N={len(membership[0])} != budget.groups={self.groups}"
            )
        p_count = len(global_only[0])
        if any(len(go) != p_count for go in global_only):
            raise ValueError("global_only 的 P 维不一致")
        if any(
            len(frame) != self.groups or any(len(group) != p_count for group in frame)
            for frame in membership
        ):
            raise ValueError("membership 必须是规则的 [T,N,P]")

        T, n, w = self.num_frames, self.frames_per_block, self.local_attn_size
        mem = [[[bool(x) for x in group] for group in frame] for frame in membership]
        go = [[bool(x) for x in frame] for frame in global_only]
        active = [[any(mem[t][g]) for g in range(self.groups)] for t in range(T)]

        edges = 0.0
        for q in range(T):
            # inactive Local query 的 self-edge 只有 q==k，不乘历史窗口。
            edges += self.local_hub * sum(not x for x in active[q])
            for k in range(T):
                in_window = (k // n <= q // n) and (w == -1 or q - k < w)
                if not in_window:
                    continue

                # global internal；Local internal / Local<->Global。
                edges += self.global_hub**2
                for g in range(self.groups):
                    aq, ak = active[q][g], active[k][g]
                    if aq and ak:
                        edges += self.local_hub**2
                    if aq:
                        edges += self.local_hub * self.global_hub
                    if ak:
                        edges += self.global_hub * self.local_hub

                    # player->Local 用 query-time membership + key-time active；
                    # Local->player 用 query-time membership + query-time active。
                    members_q = sum(mem[q][g])
                    if ak:
                        edges += self.tokens * self.local_hub * members_q
                    if aq:
                        edges += self.local_hub * self.tokens * members_q

                # global-only 的两个方向分别取 query/key time。
                edges += self.tokens * self.global_hub * sum(go[q])
                edges += self.global_hub * self.tokens * sum(go[k])

                if k // n == q // n and self.direct_frac > 0:
                    pair_units_q = 0
                    for i in range(p_count):
                        for j in range(p_count):
                            if i != j and any(
                                mem[q][g][i] and mem[q][g][j]
                                for g in range(self.groups)
                            ):
                                pair_units_q += 1
                    edges += self.direct_frac * self.tokens**2 * pair_units_q
        return edges

    def feasible_groups(self, groups: Sequence[Sequence[int]]) -> bool:
        """用 unique direct pairs 做快筛，并检查组数/容量。"""
        gs = [tuple(sorted(set(int(i) for i in g))) for g in groups]
        if len(gs) > self.groups or any(
            len(g) > self.max_size or len(g) < 2 for g in gs
        ):
            return False
        return self.direct_pair_units(gs) <= self.pair_budget

    def affordable_groups(
        self,
        groups: Sequence[Sequence[int]],
        n_players: int | None = None,
    ) -> bool:
        """按实际 membership 拓扑做最终通信量校验。"""
        return (
            self.feasible_groups(groups)
            and self.topology_cost(groups, n_players=n_players) <= self.cost_limit
        )


def resolve(players: int) -> Budget:
    """M is derived; unsupported baseline budgets must not be guessed."""
    if type(players) is not int or players not in CENTRAL_HUB_TOKENS:
        raise ValueError(f"Automatic grouping supports 4/6/8/10 players, got {players}")
    return Budget(
        players=players,
        groups=(players - 1).bit_length(),
        central_hub=CENTRAL_HUB_TOKENS[players],
    )
