"""Lossless tick states; model camera embeddings cannot replace this input."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import re

import numpy as np


@dataclass
class GroupingState:
    ticks: np.ndarray
    pos: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    alive: np.ndarray
    present: np.ndarray
    player_slots: np.ndarray
    round_tick_start: int
    round_tick_end: int  # inclusive, before cropping
    match_id: int
    map_name: str
    round_no: int
    clip_start_frame: int
    clip_num_frames: int

    @property
    def num_players(self):
        return len(self.player_slots)

    def frame_to_tick(self, frame):
        return int(self.round_tick_start + round(frame * 64.0 / 32.0))

    def frame_window(self, start_frame, num_frames):
        return slice(
            int(np.searchsorted(self.ticks, self.frame_to_tick(start_frame))),
            int(
                np.searchsorted(
                    self.ticks, self.frame_to_tick(start_frame + num_frames)
                )
            ),
        )

    def eye_pos(self):
        # Addition in float32, THEN float64 conversion in relevance is deliberate.
        eye = self.pos.copy()
        eye[..., 2] += 64.0
        return eye

    def validate(self):
        t, p = len(self.ticks), self.num_players
        if not t or p < 2 or p > 10:
            raise ValueError("Grouping state requires ticks and 2–10 players")
        shapes = {
            "ticks": ((t,), np.int64),
            "pos": ((t, p, 3), np.float32),
            "yaw": ((t, p), np.float32),
            "pitch": ((t, p), np.float32),
            "alive": ((t, p), np.bool_),
            "present": ((t, p), np.bool_),
            "player_slots": ((p,), np.int64),
        }
        for name, (shape, dtype) in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"Invalid {name}: expected {shape}/{np.dtype(dtype)}")
        if not np.all(np.diff(self.ticks) == 1):
            raise ValueError(
                "Grouping tick grid must be continuous (retain missing ticks)"
            )
        if len(set(self.player_slots.tolist())) != p or np.any(
            (self.player_slots < 0) | (self.player_slots >= 10)
        ):
            raise ValueError(
                "Grouping player_slots must be distinct physical slots in [0,9]"
            )
        if self.clip_start_frame < 0 or self.clip_num_frames <= 0:
            raise ValueError("Invalid grouping clip range")
        lo = self.frame_to_tick(max(0, self.clip_start_frame - 48))
        hi = self.frame_to_tick(self.clip_start_frame + self.clip_num_frames) - 1
        if (
            self.ticks[0] > lo
            or self.ticks[-1] < hi
            or self.round_tick_start > self.ticks[0]
            or self.round_tick_end < self.ticks[-1]
        ):
            raise ValueError(
                "Grouping state does not cover the clip and its 48-frame history"
            )
        active = self.alive & self.present
        for name in ("pos", "yaw", "pitch"):
            if not np.isfinite(getattr(self, name)[active]).all():
                raise ValueError(f"Non-finite {name} for an active player")

    def save(self, path):
        self.validate()
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        np.savez_compressed(path, format_version=np.int32(1), **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            expected = {f.name for f in fields(cls)}
            if set(data.files) != expected | {"format_version"}:
                raise ValueError(f"Invalid grouping state fields: {path}")
            if data["format_version"].shape != () or data["format_version"].item() != 1:
                raise ValueError(f"Unsupported grouping state version: {path}")
            values = {name: data[name].copy() for name in expected}
        for name in (
            "round_tick_start",
            "round_tick_end",
            "match_id",
            "round_no",
            "clip_start_frame",
            "clip_num_frames",
        ):
            value = values[name]
            if value.shape != () or value.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer scalar: {path}")
            values[name] = int(value)
        if values["map_name"].shape != () or values["map_name"].dtype.kind != "U":
            raise ValueError(f"map_name must be a string scalar: {path}")
        values["map_name"] = str(values["map_name"].item())
        result = cls(**values)
        result.validate()
        return result


def row_identity(row, num_players):
    """CSV order is distinct from physical player slots, especially for P6."""
    pattern = r"match_id=(\d+)/map_name=([^/]+)/round=(\d+)/player=(\d+)(?:/|$)"
    matches = [
        re.search(pattern, str(row[f"video_path{i}"]).replace("\\", "/"))
        for i in range(1, num_players + 1)
    ]
    if any(m is None for m in matches):
        raise ValueError(
            "Grouping requires video_path1..N containing round/player identity"
        )
    keys = {(int(m[1]), m[2], int(m[3])) for m in matches}
    if len(keys) != 1:
        raise ValueError("All grouping players must belong to the same round")
    slots = [int(m[4]) for m in matches]
    if len(set(slots)) != num_players:
        raise ValueError("Duplicate physical players in grouping row")

    def integer(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be an integer")
        result = int(value)
        if float(value) != result:
            raise ValueError(f"{name} must be an integer")
        return result

    start = integer(row["start_frame"], "start_frame")
    lengths = [
        integer(row[f"video{i}_frame"], f"video{i}_frame")
        for i in range(1, num_players + 1)
    ]
    if start < 0 or min(lengths) <= 0:
        raise ValueError("Invalid grouping frame range")
    return (*next(iter(keys)), slots, start, min(lengths))


def state_from_parquet(rounds_root, row, num_players):
    """Read only required columns, align full-round states, then crop losslessly."""
    match, map_name, round_no, slots, start, count = row_identity(row, num_players)
    folder = (
        Path(rounds_root)
        / f"match_id={match}"
        / f"map_name={map_name}"
        / f"round={round_no:02d}"
    )
    return state_from_tick_files(
        [folder / f"player={s:02d}" / "ticks.parquet" for s in slots],
        (match, map_name, round_no, slots, start, count),
    )


def state_from_tick_files(paths, identity):
    """Build a state from one tick file per player, in CSV player order.

    ``identity`` is ``(match_id, map_name, round_no, player_slots, start_frame,
    num_frames)``, as returned by :func:`row_identity`.
    """
    import pandas as pd

    match, map_name, round_no, slots, start, count = identity
    num_players = len(paths)
    columns = ["tick", "x", "y", "z", "yaw", "pitch", "is_alive"]
    frames = [
        pd.read_parquet(path, columns=columns).sort_values("tick", kind="stable")
        for path in paths
    ]
    if any(df.empty for df in frames):
        raise ValueError("Empty player tick file")
    tick0 = min(int(df.tick.iloc[0]) for df in frames)
    tick1 = max(int(df.tick.iloc[-1]) for df in frames)
    ticks = np.arange(tick0, tick1 + 1, dtype=np.int64)
    shape = (len(ticks), num_players)
    pos = np.full((*shape, 3), np.nan, np.float32)
    yaw, pitch = (np.full(shape, np.nan, np.float32) for _ in range(2))
    alive, present = (np.zeros(shape, bool) for _ in range(2))
    for k, df in enumerate(frames):
        idx = df.tick.to_numpy(np.int64) - tick0
        for j, name in enumerate(("x", "y", "z")):
            pos[idx, k, j] = df[name].to_numpy(np.float32)
        yaw[idx, k] = df.yaw.to_numpy(np.float32)
        pitch[idx, k] = df.pitch.to_numpy(np.float32)
        alive[idx, k] = df.is_alive.to_numpy(bool)
        present[idx, k] = True
        last = np.flatnonzero(present[:, k])[-1]
        pos[last + 1 :, k] = pos[last, k]
        yaw[last + 1 :, k] = yaw[last, k]
        pitch[last + 1 :, k] = pitch[last, k]
    crop = slice(max(0, start - 48) * 2, (start + count) * 2)
    result = GroupingState(
        ticks[crop],
        pos[crop],
        yaw[crop],
        pitch[crop],
        alive[crop],
        present[crop],
        np.asarray(slots, np.int64),
        tick0,
        tick1,
        match,
        map_name,
        round_no,
        start,
        count,
    )
    result.validate()
    return result


def eligible_players(world, sl):
    present = world.present[sl]
    grid = present.any(1) if len(present) else np.zeros(0, bool)
    if not grid.any():
        return np.zeros(world.num_players, bool)
    return present[grid].mean(0) >= 1.0


def update_slices(world, sl, f0, every):
    n = int(round((sl.stop - sl.start) / 2.0))
    if every <= 0 or every >= n:
        return [(f0, sl)]
    out = []
    for k in range(0, n, every):
        ssl = world.frame_window(f0 + k, min(every, n - k))
        if ssl.stop > ssl.start:
            out.append((f0 + k, ssl))
    return out
