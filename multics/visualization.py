"""Generated views with synchronized input actions and a shared XY schematic."""

import math
import tempfile
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

COLORS = (
    "#52ccff",
    "#ffb454",
    "#b69aff",
    "#62dfa9",
    "#ff7788",
    "#e5dc62",
    "#f29dd8",
    "#7699ff",
    "#c1e095",
    "#c6b5a6",
)
BG, PANEL, LINE, INK, MUTED = (
    "#090b0d",
    "#111519",
    "#30363d",
    "#edf0f3",
    "#929ba4",
)


def overview_path(output_dir, output_id):
    return Path(output_dir) / f"video{output_id}" / "generated_action_camera.mp4"


def load_tracks(row, dataset_path, num_players, num_frames, target_rgb_frames):
    """Use exactly the raw-input clip origin, action decoder and eye poses.

    Positions are in metres; yaw/pitch and mouse deltas are in degrees.
    No ground-truth video or future RGB frames are read.
    """
    import pyarrow.parquet as pq

    from multics.data.raw_inputs import (
        _clip_start_from_log_end,
        _eye_poses,
        _resolve,
        _value,
        tick_actions,
    )

    paths = [
        _resolve(dataset_path, row[f"ticks_path{i + 1}"]) for i in range(num_players)
    ]
    count = int(_value(row, "num_frames", target_rgb_frames))
    start = _value(row, "start_frame")
    start = _clip_start_from_log_end(paths, count) if start is None else int(start)
    if start < 0 or not 0 < num_frames <= count:
        raise ValueError(
            f"Invalid visualization window: {start=}, {num_frames=}, {count=}"
        )
    tracks = []
    for path in paths:
        keyboard, mouse = tick_actions(path)
        if start + num_frames > len(keyboard):
            raise ValueError(f"Action log is too short: {path}")
        pose, ticks, _ = _eye_poses(path, start, num_frames)
        data = pq.read_table(path, columns=["tick", "input_weapon"]).to_pydict()
        # Show the weapon at the end of the same two-tick action interval.
        end = np.searchsorted(data["tick"], ticks + 1, side="right") - 1
        tracks.append(
            {
                "pos": pose[:, :2],
                "yaw": pose[:, 3],
                "pitch": pose[:, 4],
                "ticks": ticks,
                "keyboard": keyboard[start : start + num_frames],
                "mouse": mouse[start : start + num_frames],
                "weapon": [data["input_weapon"][i] for i in end],
            }
        )
    if any(not np.array_equal(t["ticks"], tracks[0]["ticks"]) for t in tracks):
        raise ValueError("Player tick timelines are not synchronized")
    if any(
        not np.isfinite(t[key]).all()
        for t in tracks
        for key in ("pos", "yaw", "pitch", "keyboard", "mouse")
    ):
        raise ValueError("Non-finite visualization inputs")
    return tracks


@lru_cache(maxsize=32)
def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        name,
        f"/usr/share/fonts/dejavu/{name}",
        f"/usr/share/fonts/truetype/dejavu/{name}",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _text(draw, xy, text, size=16, fill=INK, bold=False, anchor=None):
    draw.text(xy, text, font=_font(size, bold), fill=fill, anchor=anchor)


def _key(draw, box, label, active, color, size=12):
    draw.rounded_rectangle(
        box,
        radius=3,
        fill=color if active else "#171c21",
        outline=color if active else LINE,
    )
    _text(
        draw,
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        label,
        size,
        BG if active else MUTED,
        anchor="mm",
    )


class ActionCameraOverview:
    """Two rows of generated views; fixed player colors and fixed map bounds."""

    def __init__(self, tracks, fps, label=""):
        if not 2 <= len(tracks) <= len(COLORS) or fps <= 0:
            raise ValueError("Overview requires 2–10 players and positive FPS")
        self.tracks, self.fps, self.label = tracks, fps, str(label)
        self.count = len(tracks[0]["pos"])
        if self.count == 0 or any(
            len(t[key]) != self.count
            for t in tracks
            for key in ("pos", "yaw", "pitch", "keyboard", "mouse", "weapon")
        ):
            raise ValueError("Empty or unequal visualization track lengths")
        self.cols = math.ceil(len(tracks) / 2)
        self.width, self.height = self.cols * 720 + 480, 1080
        points = np.concatenate([t["pos"] for t in tracks])
        low, high = points.min(0), points.max(0)
        self.center = (low + high) / 2
        self.span = max(float(max(high - low)) * 1.30, 10.0)
        self.scale = 330 / self.span
        self.mx, self.my = self.width - 426, 344

    def project(self, xy):
        x, y = (np.asarray(xy) - self.center) * self.scale
        return self.mx + 188 + x, self.my + 188 - y

    def frame(self, frames, index):
        if len(frames) != len(self.tracks) or not 0 <= index < self.count:
            raise ValueError("Frame/player count does not match input tracks")
        image = Image.new("RGB", (self.width, self.height), BG)
        draw = ImageDraw.Draw(image)
        _text(draw, (24, 18), "MULTICS", 27, bold=True)
        _text(draw, (187, 25), f"{len(frames)} PLAYERS  /  {self.label}", 15, MUTED)
        _text(
            draw,
            (self.width - 25, 23),
            f"Frame {index + 1}/{self.count}   Playback {index / self.fps:.2f}s",
            17,
            MUTED,
            anchor="ra",
        )
        for p, (frame, track) in enumerate(zip(frames, self.tracks)):
            x, y = 24 + p % self.cols * 720, 66 + p // self.cols * 506
            color, keys = COLORS[p], track["keyboard"][index]
            draw.rounded_rectangle(
                (x, y + 25, x + 704, y + 500), radius=5, fill=PANEL, outline=LINE
            )
            _text(draw, (x, y - 1), f"Player {p + 1}", 19, color, True)
            _text(draw, (x + 704, y + 4), "GENERATED", 11, MUTED, anchor="ra")
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.asarray(frame))
            # Letterbox custom aspect ratios instead of stretching them.
            frame = ImageOps.contain(
                frame.convert("RGB"), (704, 387), Image.Resampling.LANCZOS
            )
            image.paste(
                frame,
                (x + (704 - frame.width) // 2, y + 25 + (387 - frame.height) // 2),
            )
            draw.line((x, y + 25, x + 704, y + 25), fill=color, width=2)
            cy = y + 417
            _text(draw, (x + 12, cy), "KEYBOARD INPUT", 9, MUTED)
            ky = cy + 13
            for name, dim, col, row in (
                ("R", 22, 0, 0),
                ("W", 11, 1, 0),
                ("E", 20, 2, 0),
                ("A", 13, 0, 1),
                ("S", 12, 1, 1),
                ("D", 14, 2, 1),
                ("SHIFT", 17, 0, 2),
                ("SPACE", 15, 1, 2),
                ("CTRL", 16, 2, 2),
            ):
                bx, by = x + 12 + col * 49, ky + row * 21
                _key(draw, (bx, by, bx + 44, by + 18), name, keys[dim] > 0, color, 11)
            for j, (name, dim) in enumerate(
                (("Attack", 19), ("Secondary", 21), ("Switch", 18))
            ):
                by = ky + j * 21
                _key(draw, (x + 174, by, x + 295, by + 18), name, keys[dim] > 0, color)
            draw.line((x + 315, cy + 4, x + 315, y + 490), fill=LINE)
            _text(draw, (x + 333, cy), "MOUSE INPUT", 9, MUTED)
            ax, ay = x + 357, cy + 40
            draw.rounded_rectangle(
                (ax - 24, ay - 24, ax + 24, ay + 24), radius=4, fill=BG, outline=LINE
            )
            draw.line((ax - 23, ay, ax + 23, ay), fill=LINE)
            draw.line((ax, ay - 23, ax, ay + 23), fill=LINE)
            dx, dy = track["mouse"][index]
            ex, ey = (
                ax - float(np.clip(dx * 5, -21, 21)),
                ay + float(np.clip(dy * 5, -21, 21)),
            )
            draw.line((ax, ay, ex, ey), fill=color, width=2)
            draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=color)
            _text(draw, (x + 398, cy + 20), f"delta yaw   {dx:+.1f}°", 13, color)
            _text(draw, (x + 398, cy + 43), f"delta pitch {dy:+.1f}°", 13, color)
            weapon = str(track["weapon"][index] or "none").upper()
            weapon = weapon.replace("M4A1_SILENCER", "M4A1-S")
            _text(draw, (x + 692, cy + 2), "WEAPON INPUT", 9, MUTED, anchor="ra")
            _text(draw, (x + 692, cy + 22), weapon, 11, anchor="ra")
            _text(
                draw,
                (x + 692, cy + 46),
                f"Yaw {track['yaw'][index]:.1f}°",
                11,
                MUTED,
                anchor="ra",
            )
        self._map(draw, index)
        return image

    def _map(self, draw, index):
        mx, my = self.mx, self.my
        draw.rounded_rectangle(
            (mx - 20, 277, mx + 396, 834), radius=6, fill=PANEL, outline=LINE
        )
        _text(draw, (mx, 295), "SHARED TOP-DOWN MAP", 18, bold=True)
        _text(draw, (mx, 322), "Input positions & headings", 12, MUTED)
        draw.rectangle((mx, my, mx + 376, my + 376), fill=BG, outline=LINE)
        for axis in range(2):
            low, high = (
                self.center[axis] - self.span / 2,
                self.center[axis] + self.span / 2,
            )
            # Adaptive spacing also bounds rendering cost for large maps.
            step = 10 ** math.floor(math.log10(self.span / 4))
            for value in np.arange(math.ceil(low / step) * step, high, step):
                point = self.center.copy()
                point[axis] = value
                px, py = self.project(point)
                line = (px, my, px, my + 376) if axis == 0 else (mx, py, mx + 376, py)
                draw.line(line, fill="#20262d")
        for p, track in enumerate(self.tracks):
            color = COLORS[p]
            points = [self.project(pos) for pos in track["pos"][: index + 1]]
            if len(points) > 1:
                rgb = tuple(int(v * 0.35) for v in bytes.fromhex(color[1:]))
                draw.line(points, fill=rgb, width=2)
                draw.line(points[max(0, index - 63) :], fill=color, width=2)
            px, py = points[-1]
            angle = math.radians(float(track["yaw"][index]))
            vx, vy = math.cos(angle), -math.sin(angle)
            draw.polygon(
                [
                    (px + 24 * vx, py + 24 * vy),
                    (px + 8 * vx - 7 * vy, py + 8 * vy + 7 * vx),
                    (px + 8 * vx + 7 * vy, py + 8 * vy - 7 * vx),
                ],
                fill=color,
            )
            draw.ellipse(
                (px - 10, py - 10, px + 10, py + 10), fill=color, outline=BG, width=2
            )
            _text(draw, (px, py), str(p + 1), 11, BG, True, anchor="mm")
        occupied = []
        for p, track in enumerate(self.tracks):
            px, py = self.project(track["pos"][index])
            for ox, oy in (
                (15, -22),
                (15, 9),
                (-35, -22),
                (-35, 9),
                (4, -40),
                (4, 27),
                (30, -10),
                (-50, -10),
            ):
                xx, yy = (
                    float(np.clip(px + ox, mx + 3, mx + 338)),
                    float(np.clip(py + oy, my + 3, my + 354)),
                )
                box = (xx, yy, xx + 31, yy + 17)
                if not any(
                    box[0] < b[2] + 2
                    and box[2] + 2 > b[0]
                    and box[1] < b[3] + 2
                    and box[3] + 2 > b[1]
                    for b in occupied
                ):
                    break
            occupied.append(box)
            draw.line((px, py, xx + 15, yy + 8), fill=COLORS[p])
            draw.rounded_rectangle(box, radius=3, fill=BG)
            _text(draw, (xx + 15, yy + 8), f"P{p + 1}", 11, COLORS[p], True, "mm")
        length = 10 ** math.floor(math.log10(self.span / 4))
        bx, by = mx + 14, my + 358
        draw.line((bx, by, bx + length * self.scale, by), fill=MUTED, width=2)
        _text(draw, (bx, by - 18), f"{length:g} m", 10, MUTED)
        for p in range(len(self.tracks)):
            cols = min(len(self.tracks), 5)
            _text(
                draw,
                (mx + 10 + p % cols * (356 / cols), 742 + p // cols * 26),
                f"P{p + 1}",
                13,
                COLORS[p],
                True,
            )
        _text(draw, (mx, 808), "Input trails / schematic, no floorplan", 12, MUTED)
        _text(
            draw,
            (mx + 188, 856),
            "Arrow = input camera yaw; +Y points up.",
            12,
            MUTED,
            anchor="ma",
        )


def save_action_camera(
    videos,
    row,
    dataset_path,
    output_path,
    fps,
    target_rgb_frames,
    num_frames=None,
    label="",
):
    """Stream PIL-frame sequences or saved player MP4s to an atomic output."""
    output_path = Path(output_path)
    with ExitStack() as stack:
        streams = []
        counts = []
        for video in videos:
            if isinstance(video, (str, Path)):
                reader = stack.enter_context(imageio.get_reader(str(video)))
                if not np.isclose(reader.get_meta_data()["fps"], fps):
                    raise ValueError(f"Playback FPS differs for {video}")
                counts.append(reader.count_frames())
                streams.append(iter(reader))
            else:
                counts.append(len(video))
                streams.append(iter(video))
        if not counts or len(set(counts)) != 1 or counts[0] <= 0:
            raise ValueError(f"Player video lengths differ or are empty: {counts}")
        num_frames = counts[0] if num_frames is None else num_frames
        if num_frames != counts[0]:
            raise ValueError(f"Expected {num_frames} frames, got {counts}")
        tracks = load_tracks(
            row, dataset_path, len(videos), num_frames, target_rgb_frames
        )
        overview = ActionCameraOverview(tracks, fps, label)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".overview-", dir=output_path.parent
        ) as temporary:
            path = Path(temporary) / output_path.name
            with imageio.get_writer(
                str(path),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_params=["-preset", "fast", "-movflags", "+faststart"],
            ) as writer:
                for index in range(num_frames):
                    frame = overview.frame([next(stream) for stream in streams], index)
                    writer.append_data(np.asarray(frame))
            path.replace(output_path)
    return output_path
