"""Pose-only relevance over 48-frame history with topology hysteresis."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .state import update_slices

HALF_LIFE = 4.0
DEFAULT_ALPHA = 0.6
VFOV = 73.74
REFERENCE_DEPTH = 900.0
MUTUAL_FULL_DISTANCE = 900.0
MUTUAL_ZERO_DISTANCE = 1200.0
ACTION_FULL_DISTANCE = 1000.0
ACTION_ZERO_DISTANCE = 1400.0
FAST_HISTORY_FRAMES = 12
SLOW_HISTORY_FRAMES = 48
UPDATE_FRAMES = 48
FAST_HALF_LIFE = 4.0
SLOW_HALF_LIFE = 12.0
FAST_SCENE_WEIGHT = 0.75
SLOW_SCENE_WEIGHT = 0.25


def view_dirs(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """(T,P) 的 yaw/pitch -> (T,P,3) 单位视线向量。Source 里 pitch 正值朝下。"""
    y, p = np.radians(yaw), np.radians(pitch)
    return np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), -np.sin(p)], -1)


@dataclass
class Period:
    """Relevance and provenance for one target grouping interval."""

    start_frame: int
    num_frames: int
    span: slice
    raw: np.ndarray
    smooth: np.ndarray
    co_alive: np.ndarray
    stride: int = 1
    parts: dict[str, np.ndarray] = field(default_factory=dict)


def recent_weights(n: int, half_life: float = HALF_LIFE) -> np.ndarray:
    """Normalized recency weights, newest history frame receiving most mass."""
    if n <= 0:
        return np.zeros(0, np.float64)
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    weights = 2.0 ** ((np.arange(n, dtype=np.float64) - (n - 1)) / half_life)
    return weights / weights.sum()


def cosine_distance_gate(
    distance: np.ndarray | float, full: float, zero: float
) -> np.ndarray:
    """One below ``full``, cosine taper, and exactly zero at/above ``zero``."""
    if not 0 <= full < zero:
        raise ValueError(f"distance gate requires 0 <= full < zero, got {full}, {zero}")
    distance = np.asarray(distance, dtype=np.float64)
    x = np.clip((distance - full) / (zero - full), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _camera_quality(
    eye: np.ndarray,
    dirs: np.ndarray,
    active: np.ndarray,
    *,
    vfov: float = VFOV,
    reference_depth: float = REFERENCE_DEPTH,
) -> np.ndarray:
    """Pose-only ``Q[t, observer, target]``; an in-frame candidate, not LOS."""
    right = np.cross(dirs, np.array([0.0, 0.0, 1.0]))
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-9)
    up = np.cross(right, dirs)

    rel = eye[:, None, :, :] - eye[:, :, None, :]
    depth = np.einsum("tijc,tic->tij", rel, dirs)
    side = np.einsum("tijc,tic->tij", rel, right)
    vert = np.einsum("tijc,tic->tij", rel, up)
    safe_depth = np.maximum(depth, 1.0)
    tan_v = np.tan(np.radians(vfov / 2.0))
    tan_h = tan_v * 16.0 / 9.0
    nx = side / (safe_depth * tan_h)
    ny = vert / (safe_depth * tan_v)
    inside = (
        (depth > 1.0)
        & (np.abs(nx) <= 1.0)
        & (np.abs(ny) <= 1.0)
        & active[:, :, None]
        & active[:, None, :]
    )
    centre = np.clip(1.0 - np.hypot(nx, ny) / np.sqrt(2.0), 0.0, 1.0)
    apparent_size = np.clip(reference_depth / safe_depth, 0.0, 1.0)
    quality = np.where(inside, centre * apparent_size, 0.0)
    diagonal = np.arange(quality.shape[1])
    quality[:, diagonal, diagonal] = 0.0
    return quality


def _frame_rows(world, start_frame: int, num_frames: int) -> np.ndarray:
    rows = []
    for frame in range(start_frame, start_frame + num_frames):
        tick = world.frame_to_tick(frame)
        rows.append(min(int(np.searchsorted(world.ticks, tick)), len(world.ticks) - 1))
    return np.asarray(rows, dtype=np.int64)


def history_components(
    world,
    start_frame: int,
    num_frames: int,
    *,
    prox_len: float = 700.0,
    aim_range: float = 1200.0,
    aim_len: float = 900.0,
    mutual_full_distance: float = MUTUAL_FULL_DISTANCE,
    mutual_zero_distance: float = MUTUAL_ZERO_DISTANCE,
    action_full_distance: float = ACTION_FULL_DISTANCE,
    action_zero_distance: float = ACTION_ZERO_DISTANCE,
    half_life: float = HALF_LIFE,
) -> dict[str, np.ndarray]:
    """Compute all causal components over one history window."""
    player_count = world.num_players
    zero = np.zeros((player_count, player_count), np.float64)
    if num_frames <= 0 or len(world.ticks) == 0:
        return {
            name: zero.copy()
            for name in (
                "prox",
                "aim",
                "scene",
                "mutual",
                "interaction",
                "raw",
                "co_alive",
                "interaction_valid",
            )
        }
    if prox_len <= 0 or aim_len <= 0 or aim_range <= 0:
        raise ValueError("prox_len, aim_range and aim_len must be positive")

    rows = _frame_rows(world, start_frame, num_frames)
    eye = world.eye_pos()[rows].astype(np.float64)
    yaw = world.yaw[rows]
    pitch = world.pitch[rows]
    active = world.alive[rows] & world.present[rows]
    both = active[:, :, None] & active[:, None, :]
    dirs = view_dirs(yaw, pitch)
    recency = recent_weights(len(rows), half_life=half_life)

    delta = eye[:, None, :, :] - eye[:, :, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    prox_frame = np.where(both, np.exp(-distance / prox_len), 0.0)

    endpoint = eye + aim_range * dirs
    endpoint_delta = endpoint[:, None, :, :] - endpoint[:, :, None, :]
    endpoint_distance = np.linalg.norm(endpoint_delta, axis=-1)
    aim_frame = np.where(both, np.exp(-endpoint_distance / aim_len), 0.0)

    def recent(value):
        return np.tensordot(recency, value, axes=(0, 0))

    prox_recent = recent(prox_frame)
    aim_recent = recent(aim_frame)
    scene = np.sqrt(np.clip(prox_recent, 0.0, 1.0) * np.clip(aim_recent, 0.0, 1.0))

    quality = _camera_quality(eye, dirs, active)
    mutual_gate = (
        cosine_distance_gate(distance, mutual_full_distance, mutual_zero_distance)
        * both
    )
    action_gate = (
        cosine_distance_gate(distance, action_full_distance, action_zero_distance)
        * both
    )
    mutual = recent(np.sqrt(quality * quality.transpose(0, 2, 1)) * mutual_gate)

    # Pose-only removes firing scores, but NOT the wider validity gate below.
    interaction = mutual
    raw = np.maximum(scene, interaction)
    co_alive = both.mean(axis=0)
    interaction_valid = recent(np.maximum(mutual_gate, action_gate)) > 1e-8

    matrices = {
        "prox": prox_recent,
        "aim": aim_recent,
        "scene": scene,
        "mutual": mutual,
        "interaction": interaction,
        "raw": raw,
        "co_alive": co_alive,
        "interaction_valid": interaction_valid.astype(np.float64),
    }
    for value in matrices.values():
        np.fill_diagonal(value, 0.0)
    matrices["raw"] = np.clip((raw + raw.T) / 2.0, 0.0, 1.0)
    return matrices


def threshold_graph(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Prune pair edges only; never remove a player by component-average score."""
    graph = np.where(scores >= threshold, scores, 0.0)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0.0)
    return graph


def relevance_periods(
    world, span: slice, start_frame: int, keep: np.ndarray
) -> list[Period]:
    """Predict each 48-frame interval from history including its observed boundary."""
    alpha = DEFAULT_ALPHA
    update_every = UPDATE_FRAMES
    sub = np.ix_(keep, keep)
    periods: list[Period] = []
    previous_scene = None
    previous_interaction = None
    previous_informative = False
    for target_frame, target_span in update_slices(
        world, span, start_frame, update_every
    ):
        target_frame = int(target_frame)
        fast_start = max(0, target_frame - FAST_HISTORY_FRAMES)
        slow_start = max(0, target_frame - SLOW_HISTORY_FRAMES)
        # history_components uses a half-open integer range.  Adding one makes
        # the observed boundary frame t part of both fast and slow windows:
        # [t-12,t] and [t-48,t], respectively.
        fast_all = history_components(
            world, fast_start, target_frame - fast_start + 1, half_life=FAST_HALF_LIFE
        )
        slow_all = history_components(
            world, slow_start, target_frame - slow_start + 1, half_life=SLOW_HALF_LIFE
        )
        fast = {name: value[sub] for name, value in fast_all.items()}
        slow = {name: value[sub] for name, value in slow_all.items()}

        scene_fast = fast["scene"]
        scene_slow = slow["scene"]
        scene = FAST_SCENE_WEIGHT * scene_fast + SLOW_SCENE_WEIGHT * scene_slow
        interaction = fast["interaction"]
        scene_smooth = (
            scene
            if previous_scene is None or not previous_informative
            else alpha * scene + (1.0 - alpha) * previous_scene
        )
        interaction_smooth = (
            interaction
            if previous_interaction is None or not previous_informative
            else alpha * interaction + (1.0 - alpha) * previous_interaction
        )

        # Only current fast-window distance validity may carry interaction EMA.
        interaction_smooth = np.where(
            fast["interaction_valid"] > 0, interaction_smooth, 0.0
        )
        smooth = np.maximum(scene_smooth, interaction_smooth)
        smooth = np.where(fast["co_alive"] > 0, smooth, 0.0)
        raw = np.maximum(scene, interaction)
        for value in (
            raw,
            smooth,
            scene,
            scene_fast,
            scene_slow,
            interaction,
            scene_smooth,
            interaction_smooth,
        ):
            np.fill_diagonal(value, 0.0)

        parts = {
            **fast,
            "scene_fast": scene_fast,
            "scene_slow": scene_slow,
            "scene": scene,
            "interaction_fast": interaction,
            "interaction": interaction,
            "raw": raw,
            "scene_smooth": scene_smooth,
            "interaction_smooth": interaction_smooth,
            "slow_prox": slow["prox"],
            "slow_aim": slow["aim"],
        }
        periods.append(
            Period(
                start_frame=target_frame,
                num_frames=int(round((target_span.stop - target_span.start) / 2.0)),
                span=target_span,
                raw=np.clip((raw + raw.T) / 2.0, 0.0, 1.0),
                smooth=np.clip((smooth + smooth.T) / 2.0, 0.0, 1.0),
                co_alive=fast["co_alive"],
                stride=1,
                parts=parts,
            )
        )
        previous_scene = scene_smooth
        previous_interaction = interaction_smooth
        previous_informative = np.any(fast["co_alive"] > 0)
    return periods
