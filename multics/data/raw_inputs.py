"""Build model inputs from raw per-player data at inference time.

Raw input CSV, one row per sample and one numbered column per player:

    image_path{i}   first frame of the clip (PNG/JPEG)
    ticks_path{i}   64 Hz tick log (``ticks.parquet``) ending with the clip
    caption{i}      optional text prompt
    case_id         optional output number

Optional columns for longer tick logs, such as whole rounds:

    start_frame     first clip frame on the tick log's 32 fps timeline;
                    defaults to the last ``num_frames`` frames of the log
    num_frames      clip length; defaults to ``target_rgb_frames``
    source{i}       ``match_id=M/map_name=X/round=R/player=S`` identity

The tick log provides the keyboard/mouse actions, the absolute camera poses,
and the player trajectories used for grouping, which also read the 48 frames
before the clip when the log contains them. The first frame is encoded with
the Wan2.2 VAE as the clip's first latent frame. The resulting files use the
same formats as the precomputed example inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from multics.grouping.state import row_identity, state_from_tick_files

VIDEO_FPS = 32.0
TICK_RATE = 64.0
TICKS_PER_FRAME = 2
HEIGHT = 352
WIDTH = 640
KEYBOARD_DIM = 23
LATENT_STRIDE = 4
POS_SCALE_M = 32.0
WORLD_UNIT_TO_METRE = 0.0254
STANDING_EYE_HEIGHT = 64.0
CROUCH_EYE_HEIGHT = 46.0
MOUSE_DEGREES_PER_UNIT = 15.0

# Tick action ids -> keyboard dimensions.
ACTION_TO_KEYBOARD = {
    5: 11,  # forward
    3: 12,  # back
    8: 13,  # move left
    9: 14,  # move right
    6: 15,  # jump
    4: 16,  # duck
    12: 17,  # walk
    1: 19,  # attack
    13: 20,  # use
    2: 21,  # secondary attack (scope)
    10: 22,  # reload
}
WEAPON_SWITCH_DIM = 18
WEAPON_CLASS_TO_KEYBOARD = {
    "taser": 0,
    "rifle": 2,
    "sniper": 3,
    "smg": 4,
    "shotgun": 5,
    "machine_gun": 6,
    "pistol": 7,
    "knife": 8,
    "grenade": 9,
    "objective": 10,
}
WEAPON_NAME_TO_CLASS = {
    name: cls
    for cls, names in {
        "rifle": "ak47 aug famas galilar m4a1 m4a1_silencer m4a1_silencer_off sg556",
        "sniper": "awp g3sg1 scar20 ssg08",
        "smg": "bizon mac10 mp5sd mp7 mp9 p90 ump45",
        "shotgun": "mag7 nova sawedoff xm1014",
        "machine_gun": "m249 negev",
        "pistol": (
            "cz75a deagle elite fiveseven glock hkp2000 p250 revolver tec9 "
            "usp_silencer usp_silencer_off uspsilenceroff"
        ),
        "knife": (
            "bayonet knife knife_butterfly knife_karambit knife_m9_bayonet "
            "knife_stiletto m9bayonet"
        ),
        "grenade": (
            "decoy flashbang hegrenade highexplosivegrenade incgrenade inferno "
            "molotov smokegrenade"
        ),
        "objective": "c4 c4explosive planted_c4",
        "taser": "taser",
    }.items()
    for name in names.split()
}
# Weapon value of a player without a weapon; "nan" is stored as a string.
NO_WEAPON = {None, "", "nan"}
# A single missing interior tick may be interpolated; larger gaps are errors.
MAX_INTERPOLATION_GAP_TICKS = 2


def tick_actions(path):
    """Return per-frame ``keyboard[N,23]`` and ``mouse[N,2]`` in degrees.

    Each 32 fps frame combines two consecutive 64 Hz tick rows. ``mouse`` is
    ``[delta_yaw, delta_pitch]`` summed over both ticks. A weapon switch sets
    dimension 18 and the dimension of the new weapon class; ticks without a
    weapon (after death) never count as a switch.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(
        path, columns=["active_ids", "delta_pitch", "delta_yaw", "input_weapon"]
    )
    n_frames = table.num_rows // TICKS_PER_FRAME
    keyboard = np.zeros((n_frames, KEYBOARD_DIM), dtype=np.float32)
    mouse = np.zeros((n_frames, 2), dtype=np.float32)
    active_ids = table.column("active_ids").to_pylist()
    delta_pitch = (
        table.column("delta_pitch").to_numpy(zero_copy_only=False).astype(np.float32)
    )
    delta_yaw = (
        table.column("delta_yaw").to_numpy(zero_copy_only=False).astype(np.float32)
    )
    weapons = table.column("input_weapon").to_pylist()
    previous = None
    for frame in range(n_frames):
        first = frame * TICKS_PER_FRAME
        for tick in (first, first + 1):
            for action in active_ids[tick] or ():
                dim = ACTION_TO_KEYBOARD.get(int(action))
                if dim is not None:
                    keyboard[frame, dim] = 1.0
            mouse[frame, 0] += float(delta_yaw[tick])
            mouse[frame, 1] += float(delta_pitch[tick])
            weapon = weapons[tick]
            if weapon in NO_WEAPON:
                continue
            if previous is not None and weapon != previous:
                keyboard[frame, WEAPON_SWITCH_DIM] = 1.0
                weapon_class = WEAPON_NAME_TO_CLASS.get(str(weapon))
                if weapon_class is not None:
                    keyboard[frame, WEAPON_CLASS_TO_KEYBOARD[weapon_class]] = 1.0
            previous = weapon
    return keyboard, mouse


def clip_actions(path, start_frame, num_frames):
    """Keyboard/mouse tensors for one clip, mouse scaled to model units."""
    keyboard, mouse = tick_actions(path)
    if start_frame + num_frames > keyboard.shape[0]:
        raise ValueError(
            f"{path}: {keyboard.shape[0]} action frames do not cover "
            f"[{start_frame}, {start_frame + num_frames})"
        )
    window = slice(start_frame, start_frame + num_frames)
    scaled = mouse[window].copy() * (1.0 / MOUSE_DEGREES_PER_UNIT)
    return {
        "keyboard": torch.from_numpy(keyboard[window].copy()).float(),
        "mouse": torch.from_numpy(scaled.astype(np.float32)).float(),
    }


def _is_duck(active):
    if active is None:
        return False
    if isinstance(active, str):
        return active == "duck" or "duck" in active.split(",")
    try:
        return "duck" in active
    except TypeError:
        return False


def _eye_poses(path, start_frame, num_frames):
    """Eye pose ``[T,5]`` (metres, degrees), ticks, and interpolation mask."""
    import pyarrow.parquet as pq

    columns = ["tick", "x", "y", "z", "yaw", "pitch", "active"]
    table = pq.read_table(path, columns=columns)
    data = {}
    for name in columns:
        column = table[name].combine_chunks()
        data[name] = (
            column.to_pylist()
            if name == "active"
            else column.to_numpy(zero_copy_only=False)
        )
    ticks = np.asarray(data["tick"], dtype=np.int64)
    if ticks.size == 0:
        raise ValueError(f"Empty tick log: {path}")
    if np.any(np.diff(ticks) <= 0):
        raise ValueError(f"Ticks are not strictly increasing: {path}")
    frames = start_frame + np.arange(num_frames, dtype=np.int64)
    target = ticks[0] + TICKS_PER_FRAME * frames
    loc = np.searchsorted(ticks, target)
    valid = np.zeros(len(target), dtype=bool)
    inside = loc < len(ticks)
    valid[inside] = ticks[loc[inside]] == target[inside]
    interpolated = ~valid
    left = loc.copy()
    right = loc.copy()
    alpha = np.zeros(len(target), dtype=np.float64)
    for index in np.flatnonzero(interpolated):
        j = loc[index]
        if not (
            0 < j < len(ticks)
            and ticks[j] - ticks[j - 1] == MAX_INTERPOLATION_GAP_TICKS
            and target[index] - ticks[j - 1] == 1
            and ticks[j] - target[index] == 1
        ):
            raise ValueError(
                f"{path}: frame {int(frames[index])} needs tick {int(target[index])}; "
                "only one missing interior tick can be interpolated"
            )
        left[index], right[index], alpha[index] = j - 1, j, 0.5

    def eye(indices):
        xyz = np.stack(
            [np.asarray(data[k], dtype=np.float64)[indices] for k in ("x", "y", "z")],
            axis=-1,
        )
        xyz[:, 2] += np.asarray(
            [
                CROUCH_EYE_HEIGHT if _is_duck(data["active"][int(i)]) else
                STANDING_EYE_HEIGHT
                for i in indices
            ],
            dtype=np.float64,
        )
        return xyz

    xyz = (1.0 - alpha[:, None]) * eye(left) + alpha[:, None] * eye(right)
    yaw_values = np.asarray(data["yaw"], dtype=np.float64)
    yaw_delta = (yaw_values[right] - yaw_values[left] + 180.0) % 360.0 - 180.0
    yaw = yaw_values[left] + alpha * yaw_delta
    pitch_values = np.asarray(data["pitch"], dtype=np.float64)
    pitch = (1.0 - alpha) * pitch_values[left] + alpha * pitch_values[right]
    pose = np.concatenate([xyz, yaw[:, None], pitch[:, None]], axis=-1)
    pose[:, :3] *= WORLD_UNIT_TO_METRE
    return pose, target, interpolated


def _camera_to_world(pose):
    """``[T,5]`` eye poses -> ``[T,4,4]`` c2w with right/up/back local axes.

    Yaw 0 points along +X and increases toward +Y; positive pitch looks down.
    """
    yaw = np.deg2rad(pose[..., 3])
    pitch = np.deg2rad(pose[..., 4])
    forward = np.stack(
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
        axis=-1,
    )
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(min=1e-12)
    world_up = np.zeros_like(forward)
    world_up[..., 2] = 1.0
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(min=1e-12)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(min=1e-12)
    out = np.broadcast_to(np.eye(4, dtype=np.float64), (*pose.shape[:-1], 4, 4)).copy()
    out[..., :3, 0] = right
    out[..., :3, 1] = up
    out[..., :3, 2] = -forward
    out[..., :3, 3] = pose[..., :3]
    return out


def clip_camera(path, start_frame, num_frames, player_key):
    """Absolute camera poses at the latent frames of one clip."""
    pose, ticks, interpolated = _eye_poses(path, start_frame, num_frames)
    latent = np.arange(0, num_frames, LATENT_STRIDE, dtype=np.int64)
    payload = {
        "c2w_abs": _camera_to_world(pose[latent]).astype(np.float64),
        "rgb_ticks": ticks,
        "latent_ticks": ticks[latent],
        "latent_rgb_indices": latent,
        "interpolated_rgb_mask": interpolated,
        "interpolated_latent_mask": interpolated[latent],
        "rgb_source_frame_indices": start_frame + np.arange(num_frames, dtype=np.int64),
        "player_key": np.asarray(player_key, dtype=np.int64),
        "start_frame": np.asarray(start_frame, dtype=np.int64),
        "num_rgb_frames": np.asarray(num_frames, dtype=np.int64),
        "latent_stride": np.asarray(LATENT_STRIDE, dtype=np.int64),
        "video_fps": np.asarray(VIDEO_FPS, dtype=np.float32),
        "tick_rate": np.asarray(TICK_RATE, dtype=np.float32),
        "world_unit_to_metre": np.asarray(WORLD_UNIT_TO_METRE, dtype=np.float32),
        "pos_scale": np.asarray(POS_SCALE_M, dtype=np.float32),
        "format_version": np.asarray(3, dtype=np.int32),
    }
    return {
        key: torch.from_numpy(np.array(value, copy=True, order="C"))
        for key, value in payload.items()
    }


def load_first_frame(path, height=HEIGHT, width=WIDTH):
    """RGB uint8 ``[H,W,3]``; other sizes are scaled to cover and center-cropped."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if (w, h) != (width, height):
        scale = max(width / w, height / h)
        new_w, new_h = round(w * scale), round(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        left = int(round((new_w - width) / 2.0))
        top = int(round((new_h - height) / 2.0))
        image = image.crop((left, top, left + width, top + height))
    return np.asarray(image)


def encode_first_frame(pipe, frame):
    """VAE latent ``[48,1,h,w]`` of one RGB uint8 frame, on the CPU."""
    video = torch.from_numpy(frame[None]).float().div_(127.5).sub_(1.0)
    video = video.permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    video = video.to(dtype=pipe.torch_dtype, device=pipe.device)
    with torch.no_grad():
        latent = pipe.encode_video(
            video, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)
        )
    return latent[0].to("cpu")


def is_raw_metadata(table):
    return "image_path1" in table.columns


def _value(row, column, default=None):
    value = row.get(column)
    if value is None or pd.isna(value) or not str(value).strip():
        return default
    return value


def _resolve(base, value):
    path = Path(str(value))
    return path if path.is_absolute() else Path(base) / path


def _clip_start_from_log_end(paths, num_frames):
    import pyarrow.parquet as pq

    lengths = {pq.ParquetFile(path).metadata.num_rows // TICKS_PER_FRAME for path in paths}
    if len(lengths) != 1:
        raise ValueError(
            f"Tick logs of one row have different lengths {sorted(lengths)}; "
            "set start_frame to choose the clip"
        )
    start = lengths.pop() - num_frames
    if start < 0:
        raise ValueError(f"Tick logs are shorter than {num_frames} frames: {paths}")
    return start


def prepare_raw_inputs(table, dataset_path, rows, num_players, num_frames, output_dir):
    """Write actions, cameras, and grouping states for ``rows``.

    Returns the path of a CSV in the precomputed-input format. Its first-frame
    latents are written later by :func:`encode_raw_first_frames`, once the VAE
    is loaded.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Isolate even concurrent runs of the same setting and case numbers.
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir))
    rows = set(rows)
    out_rows = []
    for index, series in table.iterrows():
        if index not in rows:
            # Preserve source row positions without opening unselected inputs.
            out_rows.append({})
            continue
        row = series.to_dict()
        case = _value(row, "case_id")
        folder = output_dir / (f"{int(case):02d}" if case is not None else f"row{index}")
        count = int(_value(row, "num_frames", num_frames))
        start = _value(row, "start_frame")
        if start is None:
            start = _clip_start_from_log_end(
                [
                    _resolve(dataset_path, row[f"ticks_path{i}"])
                    for i in range(1, num_players + 1)
                ],
                count,
            )
        start = int(start)
        out = {"start_frame": start}
        if case is not None:
            out["case_id"] = int(case)
        for i in range(1, num_players + 1):
            source = _value(
                row,
                f"source{i}",
                f"match_id=0/map_name=custom/round=00/player={i - 1:02d}",
            )
            out[f"video_path{i}"] = str(source)
            out[f"video{i}_frame"] = count
            out[f"caption{i}"] = _value(row, f"caption{i}", "")
            out[f"image_path{i}"] = str(_resolve(dataset_path, row[f"image_path{i}"]))
            out[f"ticks_path{i}"] = str(_resolve(dataset_path, row[f"ticks_path{i}"]))
        identity = row_identity(out, num_players)
        player_dirs = [folder / f"player={slot:02d}" for slot in identity[3]]
        for i, player_dir in enumerate(player_dirs, start=1):
            out[f"vae_path{i}"] = str(player_dir / "first_frame.latent.pt")
            out[f"action_path{i}"] = str(player_dir / "action.pt")
            out[f"camera_pose_path{i}"] = str(player_dir / "camera.pt")
        out["grouping_state_path"] = str(folder / "grouping_state.npz")
        out_rows.append(out)
        ticks = [out[f"ticks_path{i}"] for i in range(1, num_players + 1)]
        for i, (player_dir, path) in enumerate(zip(player_dirs, ticks), start=1):
            player_dir.mkdir(parents=True, exist_ok=True)
            torch.save(clip_actions(path, start, count), out[f"action_path{i}"])
            torch.save(
                clip_camera(path, start, count, identity[3][i - 1]),
                out[f"camera_pose_path{i}"],
            )
        state_from_tick_files(ticks, identity).save(out["grouping_state_path"])
    csv_path = output_dir / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    pd.DataFrame(out_rows).to_csv(temporary, index=False)
    os.replace(temporary, csv_path)
    return str(csv_path)


def encode_raw_first_frames(pipe, metadata_path, rows, num_players):
    """Encode the first frame of every player in ``rows`` of a prepared CSV."""
    table = pd.read_csv(metadata_path)
    pipe.load_models_to_device(["vae"])
    for index in sorted(set(rows)):
        row = table.iloc[index]
        for i in range(1, num_players + 1):
            frame = load_first_frame(row[f"image_path{i}"])
            torch.save(encode_first_frame(pipe, frame), row[f"vae_path{i}"])
