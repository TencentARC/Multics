"""Prepare missing labels before data workers start; never overwrite input labels."""

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import torch

from .labels import generate_labels
from .state import GroupingState, row_identity
from .validation import validate_generated_labels, validate_existing_labels

# Included only in the cache filename when labels are generated because
# group_path is missing. Bundled groups.pt files are loaded by path.
# Change this string if the generated label numbers change.
ALGORITHM_VERSION = "grouping"
DEFAULT_CACHE = "outputs/grouping_cache"


def _optional_path(base_path, value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else Path(base_path) / path


@contextmanager
def _cache_lock(path):
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_save(payload, path):
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_group_path(
    row,
    base_path,
    *,
    num_players,
    cache_dir=DEFAULT_CACHE,
    group_path_column="group_path",
):
    """Return an existing label path, or deterministically generate a missing one."""
    existing = _optional_path(base_path, row.get(group_path_column))
    if existing is not None and existing.exists():
        if not existing.is_file():
            raise ValueError(f"Grouping path is not a file: {existing}")
        validate_existing_labels(
            torch.load(existing, map_location="cpu", weights_only=True)
        )
        return str(existing)
    state_path = _optional_path(base_path, row.get("grouping_state_path"))
    if state_path is None or not state_path.is_file():
        raise FileNotFoundError(
            f"Missing grouping labels ({existing}) and grouping_state_path ({state_path}). "
            "Supply the lossless tick state including 48 frames of history; "
            "camera_pose_path files do not contain enough information."
        )
    raw = state_path.read_bytes()
    state = GroupingState.load(io.BytesIO(raw))
    identity = row_identity(row, num_players)
    actual = (
        state.match_id,
        state.map_name,
        state.round_no,
        state.player_slots.tolist(),
        state.clip_start_frame,
        state.clip_num_frames,
    )
    if actual != identity:
        raise ValueError(
            f"Grouping state does not match CSV round/player order/clip: {state_path}"
        )
    key = hashlib.sha256(
        json.dumps(
            {
                "algorithm": ALGORITHM_VERSION,
                "state_sha256": hashlib.sha256(raw).hexdigest(),
                "row": identity,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (key + ".pt")
    with _cache_lock(cache / (key + ".lock")):
        if output.exists():
            payload = torch.load(output, map_location="cpu", weights_only=True)
            validate_generated_labels(payload, num_players)
        else:
            payload = {
                k: torch.from_numpy(v) for k, v in generate_labels(state).items()
            }
            validate_generated_labels(payload, num_players)
            _atomic_save(payload, output)
    return str(output)


def prepare_dataset_groups(dataset, sample_indices, cache_dir, model_profile):
    """Prepare only selected samples (including every constituent CSV row)."""
    if sample_indices is None:
        sample_indices = range(dataset.n_rows)
    sample_indices = tuple(sorted(set(sample_indices)))
    dataset.grouping_sample_indices = sample_indices
    rows = set()
    for index in sample_indices:
        if not 0 <= index < dataset.n_rows:
            raise ValueError(f"Grouping sample index out of range: {index}")
        start = dataset.sample_start_rows[index]
        rows.update(range(start, start + dataset.consecutive_rows))
    paths = [None] * len(dataset.metadata)
    dataset.group_paths = paths
    for index in sorted(rows):
        row = dataset.metadata.iloc[index]
        existing = _optional_path(dataset.base_path, row.get(dataset.group_path_column))
        if existing is None or not existing.exists():
            if (
                dataset.hier_group_format != "membership"
                or dataset.hier_group_mode != "csv"
                or dataset.group_label_player_ids
                != list(range(1, dataset.num_players + 1))
            ):
                raise ValueError(
                    "Automatic grouping requires CSV membership in compact player order"
                )
            expected = (5, 6, 1.0, 3)
            if tuple(model_profile) != expected:
                raise ValueError(
                    f"Automatic grouping requires local/global hubs, direct fraction, block = {expected}"
                )
            if dataset.num_groups != (dataset.num_players - 1).bit_length():
                raise ValueError(
                    "Automatic grouping requires ceil(log2(num_players)) group slots"
                )
        try:
            paths[index] = ensure_group_path(
                row,
                dataset.base_path,
                num_players=dataset.num_players,
                cache_dir=cache_dir,
                group_path_column=dataset.group_path_column,
            )
            rgb_frames = min(
                int(row.get(f"video{i}_frame", dataset.target_rgb_frames))
                for i in range(1, dataset.num_players + 1)
            )
            # Reuse the loader's canonical checks, before __getitem__'s retry loop.
            dataset._load_group_labels(index, (rgb_frames - 1) // 4 + 1)
        except Exception as error:
            raise ValueError(
                f"Grouping preparation failed at CSV row {index} "
                f"(case {row.get('case_id', index)}): {error}"
            ) from error
    dataset.prepared_group_rows = frozenset(rows)
