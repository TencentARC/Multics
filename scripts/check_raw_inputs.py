"""Prepare inputs from a raw-input CSV and compare them with precomputed files.

Every action, camera, grouping-state, group-label, and caption field must be
exactly equal, except the fields that only record where the clip came from:
its source match/round, the players' slots in that round, the round's tick
range, and the clip's frame numbers in the source video. The bundled tick logs
start 48 frames before each clip, so these differ from the precomputed files,
which were cut from whole rounds.
With ``--vae``, the first-frame latents are also encoded on the
GPU and compared with the first frame of the precomputed latents. The latent
difference is reported but not required to be zero: the bundled latents were
encoded on an H200, and other GPUs can round some bf16 values differently.
"""

import argparse
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from multics.data.raw_inputs import (
    encode_first_frame,
    load_first_frame,
    prepare_raw_inputs,
)
from multics.grouping.labels import generate_labels
from multics.grouping.state import GroupingState


def _equal_tensors(expected, actual):
    return (
        expected.dtype == actual.dtype
        and expected.shape == actual.shape
        and torch.equal(expected, actual)
    )


SOURCE_STATE_FIELDS = {
    "match_id",
    "map_name",
    "round_no",
    "round_tick_start",
    "round_tick_end",
    "clip_start_frame",
    "player_slots",
}
SOURCE_CAMERA_FIELDS = {"start_frame", "rgb_source_frame_indices", "player_key"}


def _equal_cameras(expected, actual):
    if set(expected) != set(actual):
        return False
    offsets = [
        payload["rgb_source_frame_indices"] - payload["start_frame"]
        for payload in (expected, actual)
    ]
    return torch.equal(*offsets) and all(
        _equal_tensors(expected[k], actual[k])
        for k in expected
        if k not in SOURCE_CAMERA_FIELDS
    )


def _equal_states(expected, actual):
    for field in fields(GroupingState):
        if field.name in SOURCE_STATE_FIELDS:
            continue
        a = np.asarray(getattr(expected, field.name))
        b = np.asarray(getattr(actual, field.name))
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            return False
    return True


def _load_vae_pipe(path):
    from multics.models.model_manager import ModelManager
    from multics.pipelines.wan_video_Nplayers import WanVideoNPlayersPipeline

    manager = ModelManager()
    manager.load_model(str(path), device="cpu", torch_dtype=torch.bfloat16)
    pipe = WanVideoNPlayersPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.vae = manager.fetch_model("wan_video_vae").to("cuda")
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path", type=Path, default=Path("asset/examples/4players")
    )
    parser.add_argument("--raw_csv", default="inputs.csv")
    parser.add_argument("--reference_csv", default="metadata.csv")
    parser.add_argument(
        "--num_players", type=int, help="Defaults to the image_path columns of the raw CSV"
    )
    parser.add_argument("--num_frames", type=int, default=165)
    parser.add_argument(
        "--vae", type=Path, help="Wan2.2_VAE.pth; also compare first-frame latents"
    )
    args = parser.parse_args()
    base = args.dataset_path.resolve()
    raw = pd.read_csv(base / args.raw_csv)
    if args.num_players is None:
        args.num_players = sum(c.startswith("image_path") for c in raw.columns)
    reference = pd.read_csv(base / args.reference_csv).set_index("case_id")
    pipe = _load_vae_pipe(args.vae) if args.vae else None
    failures = []
    latent_equal, latent_count, latent_max = 0, 0, 0.0
    with tempfile.TemporaryDirectory() as folder:
        prepared = pd.read_csv(
            prepare_raw_inputs(
                raw,
                base,
                range(len(raw)),
                args.num_players,
                args.num_frames,
                folder,
            )
        )
        for _, row in prepared.iterrows():
            case = int(row["case_id"])
            ref = reference.loc[case]
            results = {}
            for i in range(1, args.num_players + 1):
                expected = torch.load(base / ref[f"action_path{i}"], weights_only=True)
                actual = torch.load(row[f"action_path{i}"], weights_only=True)
                results[f"action{i}"] = set(expected) == set(actual) and all(
                    _equal_tensors(expected[k], actual[k]) for k in expected
                )
                expected = torch.load(
                    base / ref[f"camera_pose_path{i}"], weights_only=True
                )
                actual = torch.load(row[f"camera_pose_path{i}"], weights_only=True)
                results[f"camera{i}"] = _equal_cameras(expected, actual)
                results[f"caption{i}"] = str(row[f"caption{i}"]) == str(
                    ref[f"caption{i}"]
                )
                if pipe is not None:
                    expected = torch.load(
                        base / ref[f"vae_path{i}"], weights_only=True
                    )[:, :1]
                    actual = encode_first_frame(
                        pipe, load_first_frame(row[f"image_path{i}"])
                    )
                    if expected.dtype != actual.dtype or expected.shape != actual.shape:
                        results[f"latent{i}"] = False
                        continue
                    difference = (actual.float() - expected.float()).abs().max().item()
                    latent_count += 1
                    latent_equal += difference == 0
                    latent_max = max(latent_max, difference)
            state = GroupingState.load(row["grouping_state_path"])
            results["grouping_state"] = _equal_states(
                GroupingState.load(base / ref["grouping_state_path"]), state
            )
            expected = torch.load(base / ref["group_path"], weights_only=True)
            actual = {k: torch.from_numpy(v) for k, v in generate_labels(state).items()}
            results["groups"] = set(expected) == set(actual) and all(
                _equal_tensors(expected[k], actual[k]) for k in expected
            )
            bad = [name for name, ok in results.items() if not ok]
            if bad:
                failures.append(case)
                print(
                    f"Example {args.num_players}_{case:02d}: differs in {', '.join(bad)}"
                )
            else:
                print(
                    f"Example {args.num_players}_{case:02d}: "
                    f"{len(results)} checks exactly equal"
                )
    if latent_count:
        print(
            f"First-frame latents: {latent_equal}/{latent_count} bitwise equal, "
            f"max abs difference {latent_max:.3e}"
        )
    if failures:
        sys.exit(f"Prepared inputs differ for examples {failures}")


if __name__ == "__main__":
    main()
