from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from multics.visualization import (
    ActionCameraOverview,
    load_tracks,
    overview_path,
    save_action_camera,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "asset/examples"


@pytest.mark.parametrize("players", [4, 6, 8, 10])
def test_bundled_examples_action_alignment_and_layout(players):
    base = EXAMPLES / f"{players}players"
    references = (
        pd.read_csv(base / "metadata.csv") if (base / "metadata.csv").exists() else None
    )
    for index, row in pd.read_csv(base / "inputs.csv").iterrows():
        tracks = load_tracks(row, base, players, 165, 165)
        for p, track in enumerate(tracks):
            if references is not None:
                reference = torch.load(
                    base / references.iloc[index][f"action_path{p + 1}"],
                    weights_only=True,
                )
                np.testing.assert_array_equal(
                    track["keyboard"], reference["keyboard"].numpy()
                )
                np.testing.assert_allclose(
                    track["mouse"] / 15, reference["mouse"].numpy(), atol=1e-6
                )
            import pyarrow.parquet as pq

            data = pq.read_table(
                base / row[f"ticks_path{p + 1}"], columns=["tick", "x", "y", "yaw"]
            ).to_pydict()
            tick_indices = np.searchsorted(data["tick"], track["ticks"])
            xy = np.stack([data["x"], data["y"]], axis=-1)[tick_indices] * 0.0254
            np.testing.assert_array_equal(track["pos"], xy)
            np.testing.assert_array_equal(
                track["yaw"], np.asarray(data["yaw"])[tick_indices]
            )
        overview = ActionCameraOverview(tracks, 20)
        frame = overview.frame([Image.new("RGB", (640, 352))] * players, 164)
        assert frame.size == (players // 2 * 720 + 480, 1080)
        # Equal scale; positive world Y is up, positive X is right.
        center = np.asarray(overview.project(overview.center))
        np.testing.assert_allclose(
            np.asarray(overview.project(overview.center + [1, 0])) - center,
            [overview.scale, 0],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(overview.project(overview.center + [0, 1])) - center,
            [0, -overview.scale],
            atol=1e-10,
        )


def test_short_render_keeps_clip_origin_and_atomic_output(tmp_path):
    base = EXAMPLES / "4players"
    row = pd.read_csv(base / "inputs.csv").iloc[1]
    full = load_tracks(row, base, 4, 165, 165)
    short = load_tracks(row, base, 4, 3, 165)
    for a, b in zip(full, short):
        np.testing.assert_array_equal(a["pos"][:3], b["pos"])
        np.testing.assert_array_equal(a["keyboard"][:3], b["keyboard"])
    for track in short:
        track["pos"][:] = 0  # Stationary players must not produce zero-scale maps.
    assert np.isfinite(ActionCameraOverview(short, 20).scale)
    videos = [[Image.new("RGB", (64, 32), (p * 50, 0, 0))] * 3 for p in range(4)]
    path = overview_path(tmp_path, 2)
    save_action_camera(videos, row, base, path, 20, 165)
    with imageio.get_reader(str(path)) as reader:
        assert reader.count_frames() == 3
        assert reader.get_meta_data()["fps"] == 20
        assert reader.get_data(2).shape == (1080, 1920, 3)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="lengths"):
        save_action_camera([videos[0][:-1], *videos[1:]], row, base, path, 20, 165)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".overview-*"))


def test_resume_backfills_selected_case_and_respects_rank_and_dry_run(
    tmp_path, monkeypatch
):
    from multics.inference import _backfill_action_camera

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        num_players=4,
        grid_fps=20,
        target_rgb_frames=165,
        causal_num_frames=165,
        dataset_path=str(EXAMPLES / "4players"),
    )
    metadata = pd.read_csv(EXAMPLES / "4players/inputs.csv")
    for p in range(1, 5):
        (tmp_path / f"video2_player{p:02d}.mp4").touch()
    monkeypatch.delenv("INFER_SHARD_DRY", raising=False)
    with patch("multics.inference.save_action_camera") as render:
        done = _backfill_action_camera(
            args, metadata, [1, 2, 3, 4, 5], {0, 1}, {1}, 1, 2
        )
        assert done == {0, 1}
        assert render.call_count == 1
        assert render.call_args.args[3] == overview_path(tmp_path, 2)
        assert int(render.call_args.args[1]["case_id"]) == 2
        render.reset_mock()
        _backfill_action_camera(args, metadata, [1, 2], {1}, {1}, 0, 2)
        assert not render.called
        monkeypatch.setenv("INFER_SHARD_DRY", "1")
        _backfill_action_camera(args, metadata, [1, 2], {1}, {1}, 1, 2)
        assert not render.called
    # Missing independent player videos must cause regeneration.
    assert _backfill_action_camera(args, metadata, [1], {0}, {0}, 0, 1) == set()
