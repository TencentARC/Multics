"""Structural validation for existing and generated grouping tensors."""

import torch
from .budget import resolve
from .partition import topology_constraints_hold

LABEL_FIELDS = frozenset(
    """
membership global_only eligible labels latent_period period_membership
period_global_only period_anchor period_labels period_start period_frames
period_scores period_sizes period_cohesion period_pair_used period_total_cost
clip_total_cost cost_limit coverage orphan_ratio anchor_ratio pair_budget
max_groups max_size
""".split()
)
CORE_FIELDS = frozenset(("membership", "global_only", "eligible", "labels"))


def validate_generated_labels(payload, num_players):
    """Validate the complete generated schema, including temporal consistency."""
    budget = resolve(num_players)
    t, g, p, periods = 42, budget.groups, num_players, 4
    schema = {
        "membership": ((t, g, p), torch.bool),
        "global_only": ((t, p), torch.bool),
        "eligible": ((p,), torch.bool),
        "labels": ((t, p), torch.int8),
        "latent_period": ((t,), torch.int16),
        "period_membership": ((periods, g, p), torch.bool),
        "period_global_only": ((periods, p), torch.bool),
        "period_anchor": ((periods, p), torch.bool),
        "period_labels": ((periods, p), torch.int8),
        "period_start": ((periods,), torch.int32),
        "period_frames": ((periods,), torch.int32),
        "period_scores": ((periods, g), torch.float32),
        "period_sizes": ((periods, g), torch.int8),
        "period_cohesion": ((periods, g), torch.float32),
        "period_pair_used": ((periods,), torch.int32),
        "period_total_cost": ((periods,), torch.float32),
        "clip_total_cost": ((), torch.float64),
        "cost_limit": ((), torch.float64),
        "coverage": ((periods,), torch.float32),
        "orphan_ratio": ((periods,), torch.float32),
        "anchor_ratio": ((periods,), torch.float32),
        "pair_budget": ((), torch.int32),
        "max_groups": ((), torch.int32),
        "max_size": ((), torch.int32),
    }
    if not isinstance(payload, dict) or set(payload) != set(schema):
        raise ValueError(
            "Generated grouping must contain exactly the 24 expected tensor fields"
        )
    for name, (shape, dtype) in schema.items():
        value = payload[name]
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != shape
            or value.dtype != dtype
        ):
            raise ValueError(
                f"Invalid generated grouping field {name}: expected {shape}/{dtype}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite generated grouping field {name}")
    for name, expected in (
        ("max_groups", g),
        ("max_size", budget.max_size),
        ("pair_budget", budget.pair_budget),
        ("cost_limit", budget.cost_limit),
    ):
        if payload[name].item() != expected:
            raise ValueError(f"Generated grouping {name} disagrees with the label-generation profile")
    if not payload["eligible"].all():
        raise ValueError(
            "All selected players must be eligible for the training/inference clip"
        )
    if payload["period_start"].tolist() != [0, 48, 96, 144] or payload[
        "period_frames"
    ].tolist() != [48, 48, 48, 21]:
        raise ValueError("Invalid generated grouping periods")
    idx = torch.arange(t) // 12
    if not torch.equal(payload["latent_period"].long(), idx):
        raise ValueError("Invalid latent-to-period mapping")
    mem = payload["period_membership"]
    counts = mem.sum(1)
    has = mem.any(1)
    labels = torch.where(has, mem.long().argmax(1), -1).to(torch.int8)
    checks = {
        "membership": mem[idx],
        "global_only": (~has)[idx],
        "period_global_only": ~has,
        "labels": labels[idx],
        "period_labels": labels,
        "period_anchor": counts > 1,
        "period_sizes": mem.sum(2).to(torch.int8),
        "coverage": has.sum(1).float() / p,
        "orphan_ratio": (~has).sum(1).float() / p,
        "anchor_ratio": (counts > 1).sum(1).float() / p,
    }
    for name, expected in checks.items():
        if not torch.equal(payload[name], expected):
            raise ValueError(f"Inconsistent generated grouping field {name}")
    for i, frame in enumerate(mem):
        groups = [torch.where(group)[0].tolist() for group in frame if group.any()]
        if not topology_constraints_hold(
            groups,
            player_count=p,
            max_groups=g,
            max_size=budget.max_size,
            pair_budget=budget.pair_budget,
            budget=budget,
        ):
            raise ValueError(f"Invalid grouping topology at period {i}")
        if payload["period_pair_used"][i] != budget.direct_pair_units(groups):
            raise ValueError("Incorrect grouping pair accounting")
        expected_cost = torch.tensor(budget.topology_cost(groups), dtype=torch.float32)
        if not torch.equal(payload["period_total_cost"][i], expected_cost):
            raise ValueError("Incorrect grouping period cost")
    cost = budget.topology_cost_timeline(
        payload["membership"].numpy(), payload["global_only"].numpy()
    )
    if payload["clip_total_cost"].item() != cost or cost > budget.cost_limit:
        raise ValueError("Incorrect or over-budget grouping timeline cost")


def validate_existing_labels(payload):
    """Preserve legacy membership files while rejecting contradictory tensors."""
    if not isinstance(payload, dict):
        raise ValueError("Grouping labels must be a tensor dictionary")
    if "membership" not in payload:
        # Legacy labels-only files have their own loader checks.
        return
    for name in ("membership", "global_only", "eligible"):
        if (
            name not in payload
            or not torch.is_tensor(payload[name])
            or payload[name].dtype != torch.bool
        ):
            raise ValueError(f"Membership grouping requires a boolean tensor {name}")
    mem, go, eligible = (payload[k] for k in ("membership", "global_only", "eligible"))
    if (
        mem.ndim != 3
        or go.shape != (mem.shape[0], mem.shape[2])
        or eligible.shape != (mem.shape[2],)
    ):
        raise ValueError("Inconsistent membership/global_only/eligible shapes")
    has = mem.any(1)
    if not torch.equal(go, eligible.unsqueeze(0) & ~has) or (has & ~eligible).any():
        raise ValueError("Contradictory membership/global_only/eligible tensors")
    if "labels" in payload:
        labels = payload["labels"]
        expected = torch.where(has, mem.long().argmax(1), -1)
        if (
            not torch.is_tensor(labels)
            or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or not torch.equal(labels, expected)
        ):
            raise ValueError("Grouping labels disagree with membership")
    if set(payload) & (LABEL_FIELDS - CORE_FIELDS):
        if set(payload) != LABEL_FIELDS:
            raise ValueError("Partial grouping payload: all 24 fields are required")
        if mem.shape[2] in (4, 6, 8, 10) and mem.shape[0] == 42:
            validate_generated_labels(payload, int(mem.shape[2]))
        else:
            # Other precomputed profiles remain readable; do not invent their budget.
            for name, value in payload.items():
                if not torch.is_tensor(value) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid existing grouping field {name}")
            if "clip_total_cost" not in payload or "cost_limit" not in payload:
                raise ValueError(
                    "Existing labels require clip_total_cost and cost_limit"
                )
            if (
                payload["clip_total_cost"].numel() != 1
                or payload["cost_limit"].numel() != 1
            ):
                raise ValueError("Existing label costs must be scalar")
            if payload["clip_total_cost"].item() > payload["cost_limit"].item():
                raise ValueError("Existing grouping exceeds its communication budget")
