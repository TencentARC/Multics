from types import SimpleNamespace
from unittest.mock import Mock, patch

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from PIL import Image

from multics.data.video import generated_player_grid, save_video


@pytest.mark.parametrize("players", [2, 4, 6, 8, 10])
def test_generated_grid_keeps_every_player_and_frame(players):
    rng = np.random.default_rng(0)
    videos = [
        [
            Image.fromarray(rng.integers(0, 256, (16, 32, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        for _ in range(players)
    ]
    originals = [[np.array(frame) for frame in frames] for frames in videos]
    grid = generated_player_grid(videos)
    assert len(grid) == 3
    for time, frame in enumerate(grid):
        assert frame.size == (players * 32 + players - 1, 16)
        pixels = np.array(frame)
        for player in range(players):
            start = player * 33
            np.testing.assert_array_equal(
                pixels[:, start : start + 32], originals[player][time]
            )
            np.testing.assert_array_equal(videos[player][time], originals[player][time])
            if player < players - 1:
                assert not pixels[:, start + 32].any()


def test_generated_grid_rejects_missing_frames():
    frame = Image.new("RGB", (32, 16))
    with pytest.raises(ValueError, match="equal lengths"):
        generated_player_grid([[frame], [frame, frame]])


def test_encoded_grid_has_only_generated_row(tmp_path):
    videos = [[Image.new("RGB", (640, 352), (p * 60, 0, 0))] * 3 for p in range(4)]
    output = tmp_path / "generated.mp4"
    save_video(generated_player_grid(videos), output, fps=20)
    with imageio.get_reader(output) as reader:
        assert reader.count_frames() == 3
        assert reader.get_meta_data()["fps"] == 20
        assert reader.get_data(0).shape[0] == 352


@pytest.mark.parametrize("players", [4, 6, 8, 10])
def test_training_preview_uses_latents_without_reference_decode(tmp_path, players):
    from multics.training import LightningModelForTrain

    videos = [[Image.new("RGB", (640, 352), (p * 20, 0, 0))] for p in range(players)]
    pipe = SimpleNamespace(
        torch_dtype=torch.float32,
        vae=SimpleNamespace(upsampling_factor=16),
        causal_inference=Mock(return_value=videos),
        scheduler=SimpleNamespace(set_timesteps=Mock()),
        decode_video=Mock(
            side_effect=AssertionError("Reference clips must not be decoded")
        ),
    )
    model = SimpleNamespace(
        global_step=5,
        pipe=pipe,
        device=torch.device("cpu"),
        use_camera_pose=False,
        val_causal_num_frames=1,
        val_num_inference_steps=1,
        val_agent_vertex_ids=None,
        tiler_kwargs={"tiled": True, "tile_size": (34, 34), "tile_stride": (18, 16)},
        val_output_dir=str(tmp_path),
        val_grid_fps=20,
    )
    latents = torch.randn(1, players, 48, 2, 22, 40)
    before = latents.clone()
    batch = {
        "latents": latents,
        "keyboard": torch.zeros(1, players, 5, 23),
        "mouse": torch.zeros(1, players, 5, 2),
        "group_labels": torch.zeros(1, 2, players, dtype=torch.long),
        "active_group": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    with patch("multics.data.video.save_video") as save:
        LightningModelForTrain._run_val_sample(model, batch, 0)
    inputs = pipe.causal_inference.call_args.kwargs
    assert torch.equal(inputs["video_latents"], before.squeeze(0))
    assert torch.equal(latents, before)
    assert (inputs["height"], inputs["width"]) == (352, 640)
    pipe.decode_video.assert_not_called()
    pipe.scheduler.set_timesteps.assert_called_once_with(1000, training=True)
    grid = save.call_args.args[0]
    assert grid[0].size == (players * 640 + players - 1, 352)
