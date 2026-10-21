import imageio, os
import numpy as np
from PIL import Image
from tqdm import tqdm
import subprocess
import shutil


def generated_player_grid(player_videos, border=1):
    """Arrange generated player videos in one row without changing their frames."""
    if not player_videos or not player_videos[0]:
        raise ValueError("Generated player videos must not be empty")
    num_frames = len(player_videos[0])
    width, height = player_videos[0][0].size
    for frames in player_videos:
        if len(frames) != num_frames:
            raise ValueError("Generated player videos must have equal lengths")
        if any(frame.size != (width, height) for frame in frames):
            raise ValueError("Generated player frames must have equal dimensions")
    grid_width = len(player_videos) * width + (len(player_videos) - 1) * border
    grid = []
    for index in range(num_frames):
        canvas = Image.new("RGB", (grid_width, height), (0, 0, 0))
        for player, frames in enumerate(player_videos):
            canvas.paste(frames[index], (player * (width + border), 0))
        grid.append(canvas)
    return grid


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(
        save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
    )
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()
