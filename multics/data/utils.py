import hashlib
import cv2
import torch
import numpy as np
from PIL import Image

"\nSource: https://github.com/willisma/jax_measure_transport/blob/main/data/utils.py\n"


def anything_to_seed(*args):
    serialized_args = []
    for arg in args:
        if isinstance(arg, int):
            type_code = "int"
            value_repr = str(arg)
        elif isinstance(arg, float):
            type_code = "float"
            value_repr = repr(arg)
        elif isinstance(arg, bool):
            type_code = "bool"
            value_repr = str(arg)
        elif isinstance(arg, str):
            type_code = "str"
            value_repr = repr(arg)
        else:
            raise TypeError(f"Unsupported type: {type(arg).__name__}")
        serialized_arg = f"{type_code}:{value_repr}"
        serialized_args.append(serialized_arg)
    serialized_str = "|".join(serialized_args)
    serialized_bytes = serialized_str.encode("utf-8")
    hash_bytes = hashlib.sha256(serialized_bytes).digest()
    seed_int = int.from_bytes(hash_bytes, "big")
    return seed_int % (1 << 64)


def resize_letterbox(image, desired_height, desired_width):
    h, w = image.shape[:2]
    if h == desired_height and w == desired_width:
        return image
    resized = cv2.resize(
        image, (desired_width, desired_height), interpolation=cv2.INTER_AREA
    )
    letterboxed = resized
    return letterboxed


def tensor_bchw_to_pil_list(video: torch.Tensor):
    """[1,3,T,H,W] or [3,T,H,W], values in ~[-1,1] → list of PIL RGB (H,W)."""
    if video.dim() == 5:
        video = video[0]
    video = video.detach().float().cpu().clamp(-1, 1)
    arr = (video.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return [Image.fromarray(arr[t]) for t in range(arr.shape[0])]


def pad_pil_list_to_length(pils, t_grid: int):
    """Truncate to t_grid or pad by repeating the last frame (freeze)."""
    if not pils:
        raise ValueError("pad_pil_list_to_length: empty list")
    if len(pils) >= t_grid:
        return pils[:t_grid]
    out = list(pils)
    last = out[-1]
    while len(out) < t_grid:
        out.append(last.copy())
    return out


def black_placeholder_list(t_grid: int, width: int, height: int):
    blank = Image.new("RGB", (width, height), color=(32, 32, 32))
    return [blank.copy() for _ in range(t_grid)]


def build_grid_frame(hist_a, hist_b, gen_a, gen_b, gt_a, gt_b, t: int, border: int = 1):
    """One timestep: 3 rows × 2 cols; each cell is H×W RGB."""
    cells = [[hist_a[t], hist_b[t]], [gen_a[t], gen_b[t]], [gt_a[t], gt_b[t]]]
    w0, h0 = cells[0][0].size
    bw = border
    total_w = 2 * w0 + bw
    total_h = 3 * h0 + 2 * bw
    canvas = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))
    for r in range(3):
        for c in range(2):
            x = c * (w0 + bw)
            y = r * (h0 + bw)
            canvas.paste(cells[r][c], (x, y))
    return canvas


def build_grid_video(hist_a, hist_b, gen_a, gen_b, gt_a, gt_b, t_grid: int):
    return [
        build_grid_frame(hist_a, hist_b, gen_a, gen_b, gt_a, gt_b, t)
        for t in range(t_grid)
    ]
