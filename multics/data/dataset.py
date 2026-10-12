import os, re, random, warnings
import torch
import numpy as np
import pandas as pd
from pathlib import Path

_REPO_DIR = str(Path(__file__).resolve().parents[2] / "asset")
_BLANK_PROMPT_EMB_FILENAME = "blank_string_prompt_emb.pth"


def _default_blank_prompt_emb_path() -> str:
    return os.path.join(_REPO_DIR, _BLANK_PROMPT_EMB_FILENAME)


def _parse_int_csv(value, *, default=None, name="value"):
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _prompt_emb_to_cpu(prompt_emb: dict) -> dict:
    if not isinstance(prompt_emb, dict):
        raise TypeError(f"prompt emb should be dict, got {type(prompt_emb)}")
    return {
        k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prompt_emb.items()
    }


def _save_prompt_emb_atomic(prompt_emb: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rank = os.environ.get("RANK", "0")
    tmp_path = f"{path}.tmp.rank{rank}.pid{os.getpid()}"
    try:
        torch.save(prompt_emb, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _resolve_path(base_path, p):
    """CSV 路径可为绝对路径；否则相对 dataset_path。"""
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(base_path, p)


def _camera12_from_absolute_c2w(
    c2w_abs: torch.Tensor, pos_scale: float
) -> torch.Tensor:
    """Convert absolute cameras to the model's shared-relative representation.

    Args:
        c2w_abs: Float tensor ``[P,T,4,4]`` in the shared CS world frame.
        pos_scale: Positive translation normalization in metres.

    Returns:
        Float32 tensor ``[P,T,12]``, anchored at player 0/frame 0.
    """
    if c2w_abs.ndim != 4 or tuple(c2w_abs.shape[-2:]) != (4, 4):
        raise ValueError(f"absolute c2w 应为 [P,T,4,4]，got {tuple(c2w_abs.shape)}")
    if c2w_abs.shape[0] < 1 or c2w_abs.shape[1] < 1:
        raise ValueError(f"absolute c2w 的 P/T 必须为正，got {tuple(c2w_abs.shape)}")
    if not torch.isfinite(c2w_abs).all():
        raise ValueError("absolute c2w 包含 NaN/Inf")
    if not np.isfinite(pos_scale) or float(pos_scale) <= 0:
        raise ValueError(f"camera pos_scale 必须为正，got {pos_scale}")
    c2w = c2w_abs.to(dtype=torch.float64)
    expected_bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=c2w.dtype, device=c2w.device
    )
    if not torch.allclose(c2w[..., 3, :], expected_bottom, atol=1e-09, rtol=0.0):
        raise ValueError("absolute c2w 齐次矩阵最后一行不是 [0,0,0,1]")
    rotations = c2w[..., :3, :3]
    eye3 = torch.eye(3, dtype=c2w.dtype, device=c2w.device)
    if not torch.allclose(
        rotations.transpose(-1, -2) @ rotations, eye3, atol=2e-07, rtol=0.0
    ):
        raise ValueError("absolute c2w rotation 不是正交矩阵")
    det = torch.linalg.det(rotations)
    if not torch.allclose(det, torch.ones_like(det), atol=2e-07, rtol=0.0):
        raise ValueError("absolute c2w rotation determinant 不是 +1")
    anchor = c2w[0, 0]
    anchor_w2c = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    anchor_w2c[:3, :3] = anchor[:3, :3].transpose(0, 1)
    anchor_w2c[:3, 3] = -(anchor[:3, :3].transpose(0, 1) @ anchor[:3, 3])
    c2w_rel = anchor_w2c.view(1, 1, 4, 4) @ c2w
    c2w_rel[..., :3, 3] /= float(pos_scale)
    return c2w_rel[..., :3, :].reshape(c2w.shape[0], c2w.shape[1], 12).float()


def _ensure_prompt_batch_dim(t: torch.Tensor, *, path: str, key: str) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError(f"prompt emb {key!r} 不是 tensor: {path}")
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim >= 3:
        return t
    raise ValueError(f"prompt emb {key!r} 形状异常 {tuple(t.shape)}: {path}")


def _load_prompt_emb_many(paths: list[str]) -> dict:
    """Load per-player prompt embeddings and stack on P.

    Each file is expected to contain one player's context, e.g. [1,S,4096].
    Return shape per key: [P,S,4096] before DataLoader adds B -> [B,P,S,4096].
    """
    embs = [torch.load(p, weights_only=True, map_location="cpu") for p in paths]
    if any((not isinstance(e, dict) for e in embs)):
        raise TypeError(f"prompt emb 文件应为 dict: {paths}")
    keys = list(embs[0].keys())
    out = {}
    for k in keys:
        vals = []
        for path, emb in zip(paths, embs):
            if k not in emb:
                raise KeyError(f"prompt emb key {k!r} 不存在于 {path}")
            v = _ensure_prompt_batch_dim(emb[k], path=path, key=k)
            if v.shape[0] != 1:
                raise ValueError(
                    f"prompt_emb_path* 应各自只含 1 条 player emb，但 key {k!r} 在 {path} 的形状为 {tuple(v.shape)}"
                )
            if vals and tuple(v.shape[1:]) != tuple(vals[0].shape[1:]):
                raise ValueError(
                    f"prompt emb key {k!r} 除 player 维外形状不一致: {tuple(v.shape)} vs {tuple(vals[0].shape)} ({path})"
                )
            vals.append(v)
        out[k] = torch.cat(vals, dim=0)
    return out


def consecutive_precomputed_start_rows(
    metadata, consecutive_rows: int, num_players: int
) -> list[int]:
    """Return CSV row starts whose following rows are the same continuous clip."""
    consecutive_rows = int(consecutive_rows)
    if consecutive_rows < 1:
        raise ValueError(f"consecutive_rows must be >= 1, got {consecutive_rows}")
    n_rows = len(metadata)
    if consecutive_rows == 1:
        return list(range(n_rows))
    required = ["start_frame"]
    for pidx in range(1, int(num_players) + 1):
        required.extend((f"video_path{pidx}", f"video{pidx}_frame"))
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"consecutive precomputed rows require CSV columns {missing} to verify continuity"
        )
    valid = []
    for start in range(0, n_rows - consecutive_rows + 1):
        ok = True
        for row in range(start, start + consecutive_rows - 1):
            lengths = [
                int(metadata.iloc[row][f"video{pidx}_frame"])
                for pidx in range(1, num_players + 1)
            ]
            if min(lengths) <= 0 or len(set(lengths)) != 1:
                ok = False
                break
            if (
                int(metadata.iloc[row + 1]["start_frame"])
                != int(metadata.iloc[row]["start_frame"]) + lengths[0]
            ):
                ok = False
                break
            for pidx in range(1, num_players + 1):
                current_dir = os.path.dirname(
                    str(metadata.iloc[row][f"video_path{pidx}"]).replace("\\", "/")
                )
                next_dir = os.path.dirname(
                    str(metadata.iloc[row + 1][f"video_path{pidx}"]).replace("\\", "/")
                )
                if current_dir != next_dir:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(start)
    return valid


class PrecomputedLatentDataset(torch.utils.data.Dataset):
    """直接从预处理好的 CSV 读取压好的 VAE latent + action（跳过在线读视频 + VAE 编码）。

    CSV 列（路径相对 base_path=preprocess root；绝对路径则原样用）：
      latent_path1..latent_pathP（或 vae_path1..P）, action_path1..action_pathP
    camera pose（仅 --use_camera_pose 时读取）：
      - 只读预计算好的 camera_pose_path/camera12_path，或 per-player camera_pose_path1..P；
      - per-player 文件可为旧 [T,12]，也可为 canonical absolute c2w [T,4,4]；后者在
        P 路全部加载后，以 compact player 0 / frame 0 统一计算 shared-relative camera12；
      - 训练时不再从 json_file1/2 现场生成 camera12，避免 dataloader 反复读 action JSON。

    每段（CSV 一行）对应的物理文件：
      *.latent.pt : torch.save 的张量 [C=48, T_lat, h, w]（bf16）
      *.action.pt : torch.save({"keyboard": float[T,23], "mouse": float[T,2]})（mouse 已 compress_mouse_linear）

    返回的 batch key 与 TensorDataset 默认路径下游一致：
      latents[P,48,T_lat,h,w]、keyboard[P,T,23]、mouse[P,T,2]、prompt_emb[P,S,4096]。
      为兼容旧 2-player 路径，当 P=2 时也会额外返回 latents1/latents2。
    latent 与 action 由同一 VAE 和同一切片离线算出。
    """

    def __init__(
        self,
        base_path,
        metadata_path,
        steps_per_epoch=2000,
        num_players=2,
        target_rgb_frames=165,
        blank_prompt_emb_path=None,
        vae_latent_channels=48,
        debug_no_retry=False,
        num_frame_per_block=3,
        local_attn_size=24,
        use_camera_pose=False,
        use_group_labels=False,
        num_groups=1,
        group_path_column="group_path",
        group_label_player_ids=None,
        hier_group_mode="csv",
        hier_group_seed=0,
        hier_random_preserve_sizes=True,
        hier_group_format="labels",
        consecutive_rows=1,
        grouping_sample_indices=None,
        grouping_cache_dir="outputs/grouping_cache",
        grouping_model_profile=(5, 6, 1.0, 3),
        allow_first_frame_latents=False,
    ):
        metadata = pd.read_csv(metadata_path)
        self.num_players = int(num_players)
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")
        latent_cols = []
        action_cols = []
        missing = []
        for pidx in range(1, self.num_players + 1):
            latent_col = None
            for candidate in (f"latent_path{pidx}", f"vae_path{pidx}"):
                if candidate in metadata.columns:
                    latent_col = candidate
                    break
            if latent_col is None:
                missing.append(f"latent_path{pidx}/vae_path{pidx}")
            latent_cols.append(latent_col)
            action_col = f"action_path{pidx}"
            if action_col not in metadata.columns:
                missing.append(action_col)
            action_cols.append(action_col)
        if missing:
            raise ValueError(
                f"[PrecomputedLatentDataset] CSV 缺少预处理列 {missing}；该 CSV 似乎不是 *.preprocessed.csv。现有列：{list(metadata.columns)}"
            )
        self.metadata = metadata
        self.base_path = base_path
        self.latent_column_names = latent_cols
        self.action_column_names = action_cols
        self.latent_paths = [
            [_resolve_path(base_path, x) for x in metadata[col]]
            for col in self.latent_column_names
        ]
        self.action_paths = [
            [_resolve_path(base_path, x) for x in metadata[col]]
            for col in self.action_column_names
        ]
        self.latent_path1 = self.latent_paths[0]
        self.latent_path2 = self.latent_paths[1]
        self.action_path1 = self.action_paths[0]
        self.action_path2 = self.action_paths[1]
        self.use_camera_pose = bool(use_camera_pose)
        self.camera_pose_path = None
        self.camera_pose_paths = None
        combined_camera_col = next(
            (c for c in ("camera_pose_path", "camera12_path") if c in metadata.columns),
            None,
        )
        pair_camera_cols = next(
            (
                prefix
                for prefix in ("camera_pose_path", "camera12_path")
                if all(
                    (
                        f"{prefix}{pidx}" in metadata.columns
                        for pidx in range(1, self.num_players + 1)
                    )
                )
            ),
            None,
        )
        if combined_camera_col is not None and pair_camera_cols is not None:
            raise ValueError(
                "CSV camera pose 列不能同时使用 combined 和 per-player 两种格式"
            )
        for prefix in ("camera_pose_path", "camera12_path"):
            present = [
                f"{prefix}{pidx}" in metadata.columns
                for pidx in range(1, self.num_players + 1)
            ]
            if any(present) and (not all(present)):
                raise ValueError(
                    f"CSV 若使用 per-player camera pose，必须同时包含 {prefix}1..{prefix}{self.num_players}"
                )
        if combined_camera_col is not None:
            self.camera_pose_path = [
                _resolve_path(base_path, x) for x in metadata[combined_camera_col]
            ]
        elif pair_camera_cols is not None:
            self.camera_pose_paths = [
                [
                    _resolve_path(base_path, x)
                    for x in metadata[f"{pair_camera_cols}{pidx}"]
                ]
                for pidx in range(1, self.num_players + 1)
            ]
        if (
            self.use_camera_pose
            and self.camera_pose_path is None
            and (self.camera_pose_paths is None)
        ):
            raise ValueError(
                f"--use_camera_pose requires camera_pose_path/camera12_path, or per-player camera_pose_path1..{self.num_players} / camera12_path1..{self.num_players}. Camera poses must already be precomputed."
            )
        self.prompt_emb_paths = None
        prompt_cols = [
            f"prompt_emb_path{pidx}" for pidx in range(1, self.num_players + 1)
        ]
        prompt_present = [c in metadata.columns for c in prompt_cols]
        if any(prompt_present) and (not all(prompt_present)):
            raise ValueError(
                f"CSV 若使用 prompt emb，必须同时包含 prompt_emb_path1..prompt_emb_path{self.num_players}"
            )
        if all(prompt_present):
            self.prompt_emb_paths = [
                [_resolve_path(base_path, x) for x in metadata[col]]
                for col in prompt_cols
            ]
        self.prompt_emb_path1 = (
            self.prompt_emb_paths[0] if self.prompt_emb_paths else None
        )
        self.prompt_emb_path2 = (
            self.prompt_emb_paths[1] if self.prompt_emb_paths else None
        )
        self.captions = None
        caption_cols = [f"caption{pidx}" for pidx in range(1, self.num_players + 1)]
        caption_present = [c in metadata.columns for c in caption_cols]
        if any(caption_present) and (not all(caption_present)):
            raise ValueError(
                f"CSV 若使用 caption，必须同时包含 caption1..caption{self.num_players}"
            )
        if all(caption_present):
            self.captions = [
                [str(v).strip() if pd.notna(v) else "" for v in metadata[col]]
                for col in caption_cols
            ]
            print(
                f"[PrecomputedLatentDataset] 使用 caption1..caption{self.num_players} 作为 validation prompt。"
            )
        self.caption1 = self.captions[0] if self.captions else None
        self.caption2 = self.captions[1] if self.captions else None
        self.use_group_labels = bool(use_group_labels)
        self.num_groups = int(num_groups)
        self.group_path_column = str(group_path_column)
        self.hier_group_mode = str(hier_group_mode)
        self.hier_group_seed = int(hier_group_seed)
        self.hier_random_preserve_sizes = bool(hier_random_preserve_sizes)
        self.hier_group_format = str(hier_group_format)
        if self.hier_group_mode not in ("csv", "random_static", "random_block"):
            raise ValueError(
                f"hier_group_mode must be csv/random_static/random_block, got {self.hier_group_mode!r}"
            )
        if self.hier_group_format not in ("labels", "membership"):
            raise ValueError(
                f"hier_group_format must be labels or membership, got {self.hier_group_format!r}"
            )
        if self.hier_group_format == "membership" and self.hier_group_mode != "csv":
            raise ValueError(
                f"hier_group_format=membership does not support hier_group_mode={self.hier_group_mode!r}; use csv. Random modes only support the labels format."
            )
        self.group_label_player_ids = _parse_int_csv(
            group_label_player_ids,
            default=list(range(1, self.num_players + 1)),
            name="group_label_player_ids",
        )
        if len(self.group_label_player_ids) != self.num_players:
            raise ValueError(
                f"group_label_player_ids length {len(self.group_label_player_ids)} must equal num_players={self.num_players}: {self.group_label_player_ids}"
            )
        if any((pid < 1 or pid > 10 for pid in self.group_label_player_ids)):
            raise ValueError(
                f"group_label_player_ids must be 1-based ids in [1,10], got {self.group_label_player_ids}"
            )
        self.group_paths = None
        self.source_n_rows = len(metadata)
        self.consecutive_rows = int(consecutive_rows)
        if self.consecutive_rows < 1:
            raise ValueError(
                f"consecutive_rows must be >= 1, got {self.consecutive_rows}"
            )
        self.sample_start_rows = consecutive_precomputed_start_rows(
            metadata, self.consecutive_rows, self.num_players
        )
        self.n_rows = len(self.sample_start_rows)
        if self.n_rows == 0:
            raise ValueError(
                f"[PrecomputedLatentDataset] no valid {self.consecutive_rows}-row continuous windows in {metadata_path}"
            )
        self.allow_first_frame_latents = bool(allow_first_frame_latents)
        if self.allow_first_frame_latents and self.consecutive_rows != 1:
            raise ValueError("First-frame latents require consecutive_rows=1")
        self.steps_per_epoch = steps_per_epoch
        self.target_rgb_frames = int(target_rgb_frames)
        self.vae_latent_channels = int(vae_latent_channels)
        self.num_frame_per_block = int(num_frame_per_block)
        self.local_attn_size = int(local_attn_size)
        self.blank_prompt_emb_path = (
            blank_prompt_emb_path or _default_blank_prompt_emb_path()
        )
        self.blank_prompt_emb = None
        self.debug_no_retry = debug_no_retry
        if self.use_group_labels:
            from multics.grouping.prepare import prepare_dataset_groups

            prepare_dataset_groups(
                self, grouping_sample_indices, grouping_cache_dir,
                (*grouping_model_profile[:3], self.num_frame_per_block),
            )
        print(
            f"[PrecomputedLatentDataset] rows={self.n_rows} target_rgb_frames={self.target_rgb_frames} latent_channels={self.vae_latent_channels} use_camera_pose={self.use_camera_pose} camera_source={('path' if self.camera_pose_path or self.camera_pose_paths else 'none')} prompt_emb={self.prompt_emb_paths is not None} num_players={self.num_players} use_group_labels={self.use_group_labels} group_label_player_ids={(self.group_label_player_ids if self.use_group_labels else 'none')} hier_group_mode={(self.hier_group_mode if self.use_group_labels else 'none')} hier_group_format={(self.hier_group_format if self.use_group_labels else 'none')} hier_group_seed={self.hier_group_seed} hier_random_preserve_sizes={self.hier_random_preserve_sizes} consecutive_rows={self.consecutive_rows} valid_windows={self.n_rows}"
        )

    def __len__(self):
        return max(self.n_rows, self.steps_per_epoch)

    def _load_action(self, path):
        d = torch.load(path, weights_only=True, map_location="cpu")
        return (d["keyboard"].float(), d["mouse"].float())

    def _load_camera_pose_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            obj = np.load(path)
        elif ext == ".npz":
            with np.load(path) as npz:
                if "c2w_abs" in npz and (
                    not any(
                        (
                            key in npz
                            for key in ("camera_pose", "camera12", "pose", "camera")
                        )
                    )
                ):
                    obj = {key: np.array(npz[key], copy=True) for key in npz.files}
                else:
                    key = next(
                        (
                            k
                            for k in ("camera_pose", "camera12", "pose", "camera")
                            if k in npz
                        ),
                        npz.files[0],
                    )
                    obj = np.array(npz[key], copy=True)
        else:
            obj = torch.load(path, weights_only=True, map_location="cpu")
        if isinstance(obj, dict):
            if "c2w_abs" in obj and (
                not any(
                    (
                        key in obj
                        for key in ("camera_pose", "camera12", "pose", "camera")
                    )
                )
            ):
                return obj
            key = next(
                (k for k in ("camera_pose", "camera12", "pose", "camera") if k in obj),
                None,
            )
            if key is None:
                raise KeyError(
                    f"camera pose 文件缺少 camera_pose/camera12/pose/camera key: {path}"
                )
            obj = obj[key]
        if not torch.is_tensor(obj):
            obj = torch.as_tensor(obj)
        return obj.float()

    @staticmethod
    def _camera_payload_tensor(payload: dict, key: str, *, path: str) -> torch.Tensor:
        """Read one required sidecar field as a tensor.

        Args:
            payload: Loaded numeric per-video camera dictionary.
            key: Required dictionary field name.
            path: Source path included in validation errors.

        Returns:
            The field as a torch tensor without changing its dtype.
        """
        if key not in payload:
            raise KeyError(f"per-video camera 文件缺少 {key!r}: {path}")
        value = payload[key]
        return value if torch.is_tensor(value) else torch.as_tensor(value)

    def _camera12_from_per_video_absolute(
        self, row: int, payloads: list[dict]
    ) -> torch.Tensor:
        """Stack one row's physical-video sidecars and choose a shared anchor.

        Args:
            row: Metadata row index used for CSV/path consistency checks.
            payloads: Per-video absolute-camera dictionaries in compact CSV order.

        Returns:
            Float32 ``camera12[P,T_lat,12]`` anchored at the first compact
            player's first latent frame. Physical ``player_key`` is checked
            against each corresponding ``video_pathN``.
        """
        paths = [player_paths[row] for player_paths in self.camera_pose_paths]
        c2w_players = []
        reference_ticks = None
        reference_indices = None
        reference_start = None
        reference_stride = None
        reference_scale = None
        reference_unit = None
        csv_start = None
        if "start_frame" in self.metadata.columns:
            csv_start = int(self.metadata.iloc[row]["start_frame"])
        for player_index, (payload, path) in enumerate(zip(payloads, paths), start=1):
            version = int(
                self._camera_payload_tensor(payload, "format_version", path=path)
                .reshape(-1)[0]
                .item()
            )
            if version not in (2, 3):
                raise ValueError(
                    f"per-video absolute camera format_version 应为 2/3，got {version}: {path}"
                )
            c2w = self._camera_payload_tensor(payload, "c2w_abs", path=path)
            if c2w.ndim != 3 or tuple(c2w.shape[-2:]) != (4, 4):
                raise ValueError(
                    f"per-video c2w_abs 应为 [T,4,4]，got {tuple(c2w.shape)}: {path}"
                )
            c2w_players.append(c2w)
            ticks = self._camera_payload_tensor(
                payload, "latent_ticks", path=path
            ).long()
            indices = self._camera_payload_tensor(
                payload, "latent_rgb_indices", path=path
            ).long()
            start = int(
                self._camera_payload_tensor(payload, "start_frame", path=path)
                .reshape(-1)[0]
                .item()
            )
            stride = int(
                self._camera_payload_tensor(payload, "latent_stride", path=path)
                .reshape(-1)[0]
                .item()
            )
            scale = float(
                self._camera_payload_tensor(payload, "pos_scale", path=path)
                .reshape(-1)[0]
                .item()
            )
            unit = float(
                self._camera_payload_tensor(payload, "world_unit_to_metre", path=path)
                .reshape(-1)[0]
                .item()
            )
            player_key = int(
                self._camera_payload_tensor(payload, "player_key", path=path)
                .reshape(-1)[0]
                .item()
            )
            video_col = f"video_path{player_index}"
            if video_col in self.metadata.columns:
                video_path = str(self.metadata.iloc[row][video_col]).replace("\\", "/")
                match = _CAMERA_PLAYER_PART.search(video_path)
                if match is not None and int(match.group(1)) != player_key:
                    raise ValueError(
                        f"camera player_key={player_key:02d} 与 {video_col} 中的 player={int(match.group(1)):02d} 不一致: {path}"
                    )
            if csv_start is not None and start != csv_start:
                raise ValueError(
                    f"camera start_frame={start} 与 CSV start_frame={csv_start} 不一致: {path}"
                )
            if reference_ticks is None:
                reference_ticks = ticks
                reference_indices = indices
                reference_start = start
                reference_stride = stride
                reference_scale = scale
                reference_unit = unit
            else:
                if not torch.equal(ticks, reference_ticks):
                    raise ValueError(
                        f"不同 player 的 camera latent_ticks 不一致: {path}"
                    )
                if not torch.equal(indices, reference_indices):
                    raise ValueError(
                        f"不同 player 的 camera latent_rgb_indices 不一致: {path}"
                    )
                if start != reference_start or stride != reference_stride:
                    raise ValueError(
                        f"不同 player 的 camera start/stride 不一致: {path}"
                    )
                if scale != reference_scale or unit != reference_unit:
                    raise ValueError(
                        f"不同 player 的 camera unit/pos_scale 不一致: {path}"
                    )
        if len({tuple(c2w.shape) for c2w in c2w_players}) != 1:
            raise ValueError(
                f"不同 player 的 per-video c2w_abs shape 不一致: {[tuple(c2w.shape) for c2w in c2w_players]}"
            )
        c2w_abs = torch.stack(c2w_players, dim=0)
        return _camera12_from_absolute_c2w(c2w_abs, float(reference_scale))

    def _load_camera_pose(self, row: int, t_lat: int) -> torch.Tensor | None:
        if self.camera_pose_path is not None:
            cam = self._load_camera_pose_file(self.camera_pose_path[row])
            if isinstance(cam, dict):
                raise ValueError(
                    "combined camera_pose_path 不能指向 per-video absolute c2w；请使用 camera_pose_path1..N"
                )
        elif self.camera_pose_paths is not None:
            cams = [
                self._load_camera_pose_file(paths[row])
                for paths in self.camera_pose_paths
            ]
            absolute = [isinstance(cam_i, dict) for cam_i in cams]
            if any(absolute) and (not all(absolute)):
                raise ValueError(
                    "per-player camera 文件不能混用 absolute c2w 与旧 camera12 格式"
                )
            if all(absolute):
                cam = self._camera12_from_per_video_absolute(row, cams)
            else:
                if any((cam_i.dim() != 2 for cam_i in cams)):
                    raise ValueError("旧 per-player camera pose 文件应为 [T,12]")
                cam = torch.stack(cams, dim=0)
        else:
            return None
        if cam.dim() != 3:
            raise ValueError(
                f"camera_pose 应为 [P,T,12] 或 [T,P,12]，got {tuple(cam.shape)}"
            )
        if cam.shape[0] != self.num_players and cam.shape[1] == self.num_players:
            cam = cam.permute(1, 0, 2).contiguous()
        if cam.shape[0] != self.num_players or cam.shape[-1] != 12:
            raise ValueError(
                f"camera_pose 形状应为 [P={self.num_players},T,12]，got {tuple(cam.shape)}"
            )
        if cam.shape[1] < t_lat:
            raise ValueError(
                f"camera_pose latent T={cam.shape[1]} 短于 latent T={t_lat}"
            )
        return cam[:, :t_lat, :].contiguous()

    def _randomize_group_labels(
        self, group_labels: torch.Tensor, row: int
    ) -> torch.Tensor:
        """Apply a deterministic random grouping while preserving [T,P]."""
        if self.hier_group_mode == "csv":
            return group_labels
        t_lat, num_players = group_labels.shape
        n_block = max(1, int(self.num_frame_per_block))
        generator = torch.Generator()
        seed = (self.hier_group_seed + int(row) * 1000003) % ((1 << 63) - 1)
        generator.manual_seed(seed)
        if self.hier_random_preserve_sizes:
            if self.hier_group_mode == "random_static":
                perm = torch.randperm(num_players, generator=generator)
                return group_labels.index_select(1, perm).contiguous()
            randomized = torch.empty_like(group_labels)
            for start in range(0, t_lat, n_block):
                end = min(start + n_block, t_lat)
                perm = torch.randperm(num_players, generator=generator)
                assignment = group_labels[start].index_select(0, perm)
                randomized[start:end] = assignment
            return randomized
        randomized = torch.empty_like(group_labels)
        if self.hier_group_mode == "random_static":
            assignment = torch.randint(
                self.num_groups, (num_players,), generator=generator
            )
            randomized[:] = assignment
        else:
            for start in range(0, t_lat, n_block):
                end = min(start + n_block, t_lat)
                assignment = torch.randint(
                    self.num_groups, (num_players,), generator=generator
                )
                randomized[start:end] = assignment
        return randomized

    def _select_group_player_columns(
        self, tensor: torch.Tensor, player_dim: int, path: str, name: str
    ) -> torch.Tensor:
        """Select compact player columns using 1-based group_label_player_ids.

        tensor player axis may be the full 10-player roster or already-compact P.
        """
        cols = [pid - 1 for pid in self.group_label_player_ids]
        n_src = int(tensor.shape[player_dim])
        max_id = max(self.group_label_player_ids)
        if n_src < max_id:
            already_compact = (
                n_src == self.num_players
                and self.group_label_player_ids == list(range(1, self.num_players + 1))
            )
            if not already_compact:
                raise ValueError(
                    f"{name} player dim {n_src} cannot cover ids {self.group_label_player_ids} at {path}"
                )
            return tensor
        return tensor.index_select(player_dim, torch.tensor(cols, dtype=torch.long))

    @staticmethod
    def _assert_block_constant(
        tensor: torch.Tensor, n_block: int, path: str, name: str
    ) -> None:
        """Grouping must be constant inside each latent temporal block (n=3)."""
        if n_block <= 1:
            return
        t_lat = int(tensor.shape[0])
        for start in range(0, t_lat, n_block):
            block = tensor[start : start + n_block]
            if block.numel() and (not torch.equal(block, block[:1].expand_as(block))):
                raise ValueError(
                    f"{name} must be constant inside each latent temporal block; path={path} block=[{start},{min(start + n_block, t_lat)})"
                )

    def _load_group_labels(self, row: int, t_lat: int) -> dict:
        """Load grouping tensors for one CSV row.

        labels mode returns:
          group_labels [T,P] long in [0,N), active_group [T,N] bool
        membership mode additionally returns:
          membership [T,N,P] bool, global_only [T,P] bool
          group_labels is debug-only (argmax; -1 = not in any local group)
        eligible stays in the .pt and is only asserted, never returned.
        """
        if self.group_paths is None:
            raise RuntimeError(
                "use_group_labels=True but group_paths is not initialized"
            )
        path = self.group_paths[row]
        d = torch.load(path, weights_only=True, map_location="cpu")
        n_block = max(1, int(self.num_frame_per_block))
        if self.hier_group_format == "membership":
            if not isinstance(d, dict) or "membership" not in d:
                raise ValueError(
                    f"hier_group_format=membership requires key 'membership' in {path}; refusing to silently fall back to labels (anchors / orphans would be wrong)"
                )
            membership = d["membership"]
            if not torch.is_tensor(membership):
                membership = torch.as_tensor(membership)
            if membership.dim() != 3:
                raise ValueError(
                    f"membership must be [T,N,P] bool, got {tuple(membership.shape)} at {path}"
                )
            if membership.shape[0] < t_lat:
                raise ValueError(
                    f"membership T={membership.shape[0]} shorter than latent T={t_lat}: {path}"
                )
            if int(membership.shape[1]) != self.num_groups:
                raise ValueError(
                    f"membership N={int(membership.shape[1])} must equal num_groups={self.num_groups} at {path}"
                )
            membership = membership[:t_lat].to(dtype=torch.bool)
            membership = self._select_group_player_columns(
                membership, player_dim=2, path=path, name="membership"
            )
            if membership.shape[2] != self.num_players:
                raise ValueError(
                    f"membership P={membership.shape[2]} must equal num_players={self.num_players} at {path}"
                )
            if "global_only" not in d:
                raise ValueError(
                    f"hier_group_format=membership requires key 'global_only' in {path}"
                )
            global_only = d["global_only"]
            if not torch.is_tensor(global_only):
                global_only = torch.as_tensor(global_only)
            if global_only.dim() != 2:
                raise ValueError(
                    f"global_only must be [T,P] bool, got {tuple(global_only.shape)} at {path}"
                )
            if global_only.shape[0] < t_lat:
                raise ValueError(
                    f"global_only T={global_only.shape[0]} shorter than latent T={t_lat}: {path}"
                )
            global_only = global_only[:t_lat].to(dtype=torch.bool)
            global_only = self._select_group_player_columns(
                global_only, player_dim=1, path=path, name="global_only"
            )
            if global_only.shape[1] != self.num_players:
                raise ValueError(
                    f"global_only P={global_only.shape[1]} must equal num_players={self.num_players} at {path}"
                )
            if "eligible" not in d:
                raise ValueError(
                    f"hier_group_format=membership requires key 'eligible' [P] in {path}"
                )
            eligible = d["eligible"]
            if not torch.is_tensor(eligible):
                eligible = torch.as_tensor(eligible)
            eligible = eligible.to(dtype=torch.bool).reshape(-1)
            eligible = self._select_group_player_columns(
                eligible.unsqueeze(0), player_dim=1, path=path, name="eligible"
            ).squeeze(0)
            if eligible.numel() != self.num_players:
                raise ValueError(
                    f"eligible P={eligible.numel()} must equal num_players={self.num_players} at {path}"
                )
            if not bool(eligible.all().item()):
                raise ValueError(
                    f"eligible must be all True for this training CSV (not used in the mask); got {eligible.tolist()} at {path}"
                )
            self._assert_block_constant(membership, n_block, path, "membership")
            self._assert_block_constant(global_only, n_block, path, "global_only")
            active_group = membership.any(dim=-1).contiguous()
            if "labels" in d:
                labels = d["labels"]
                if not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels)
                group_labels = labels[:t_lat].to(dtype=torch.long)
                group_labels = self._select_group_player_columns(
                    group_labels, player_dim=1, path=path, name="labels"
                ).contiguous()
            else:
                any_g = membership.any(dim=1)
                group_labels = membership.to(dtype=torch.long).argmax(dim=1)
                group_labels = torch.where(
                    any_g, group_labels, torch.full_like(group_labels, -1)
                )
            return {
                "group_labels": group_labels,
                "active_group": active_group,
                "membership": membership.contiguous(),
                "global_only": global_only.contiguous(),
            }
        if isinstance(d, dict) and "labels" not in d:
            raise ValueError(
                f"hier_group_format=labels requires key 'labels' in {path}; this looks like a membership-only file. Pass --hier_group_format membership."
            )
        labels = d["labels"] if isinstance(d, dict) else d
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() != 2 or labels.shape[1] < max(self.group_label_player_ids):
            already_compact = (
                labels.dim() == 2
                and labels.shape[1] == self.num_players
                and (
                    self.group_label_player_ids == list(range(1, self.num_players + 1))
                )
            )
            if not already_compact:
                raise ValueError(
                    f"group labels should be [T,10] and cover ids {self.group_label_player_ids}, got {tuple(labels.shape)} at {path}"
                )
        if labels.shape[0] < t_lat:
            raise ValueError(
                f"group labels T={labels.shape[0]} shorter than latent T={t_lat}: {path}"
            )
        group_labels = labels[:t_lat].to(dtype=torch.long)
        group_labels = self._select_group_player_columns(
            group_labels, player_dim=1, path=path, name="labels"
        ).contiguous()
        if group_labels.numel() and (
            int(group_labels.min().item()) < 0
            or int(group_labels.max().item()) >= self.num_groups
        ):
            raise ValueError(
                f"group_labels values out of range [0,{self.num_groups}) at {path}: min={int(group_labels.min().item())} max={int(group_labels.max().item())}"
            )
        self._assert_block_constant(group_labels, n_block, path, "group_labels")
        group_labels = self._randomize_group_labels(group_labels, row)
        active_group = torch.zeros(t_lat, self.num_groups, dtype=torch.bool)
        active_group.scatter_(1, group_labels, True)
        return {"group_labels": group_labels, "active_group": active_group}

    def _load_camera_pose_rows(
        self, rows: list[int], latent_lengths: list[int], target_t_lat: int
    ) -> torch.Tensor | None:
        """Load consecutive camera segments in one shared relative frame.

        Multi-row inference requires canonical per-player absolute c2w files.
        Concatenating legacy camera12 tensors would silently reset the relative
        anchor at every CSV boundary, so that ambiguous case is rejected.
        """
        if len(rows) == 1:
            return self._load_camera_pose(rows[0], target_t_lat)
        if self.camera_pose_paths is None:
            raise ValueError(
                "consecutive_rows>1 with camera pose requires per-player canonical camera_pose_path1..P containing absolute c2w"
            )
        segments = []
        reference_scale = None
        for row, t_lat in zip(rows, latent_lengths):
            payloads = [
                self._load_camera_pose_file(paths[row])
                for paths in self.camera_pose_paths
            ]
            if not all((isinstance(payload, dict) for payload in payloads)):
                raise ValueError(
                    "consecutive_rows>1 cannot concatenate legacy relative camera12; use canonical per-player absolute c2w sidecars"
                )
            self._camera12_from_per_video_absolute(row, payloads)
            scales = [
                float(
                    self._camera_payload_tensor(payload, "pos_scale", path=paths[row])
                    .reshape(-1)[0]
                    .item()
                )
                for payload, paths in zip(payloads, self.camera_pose_paths)
            ]
            if len(set(scales)) != 1:
                raise ValueError(
                    f"camera pos_scale differs between players at CSV row {row}: {scales}"
                )
            if reference_scale is None:
                reference_scale = scales[0]
            elif scales[0] != reference_scale:
                raise ValueError(
                    f"camera pos_scale differs across consecutive rows: {reference_scale} vs {scales[0]}"
                )
            c2w = torch.stack(
                [
                    self._camera_payload_tensor(payload, "c2w_abs", path=paths[row])
                    for payload, paths in zip(payloads, self.camera_pose_paths)
                ],
                dim=0,
            )
            if c2w.shape[1] < t_lat:
                raise ValueError(
                    f"camera latent T={c2w.shape[1]} shorter than row latent T={t_lat} at CSV row {row}"
                )
            segments.append(c2w[:, :t_lat])
        absolute = torch.cat(segments, dim=1)
        if absolute.shape[1] < target_t_lat:
            raise ValueError(
                f"combined camera latent T={absolute.shape[1]} shorter than target {target_t_lat}"
            )
        return _camera12_from_absolute_c2w(
            absolute[:, :target_t_lat], float(reference_scale)
        ).contiguous()

    def _load_group_rows(
        self, rows: list[int], latent_lengths: list[int], target_t_lat: int
    ) -> dict:
        parts = [
            self._load_group_labels(row, t_lat)
            for row, t_lat in zip(rows, latent_lengths)
        ]
        keys = set(parts[0])
        if any((set(part) != keys for part in parts[1:])):
            raise ValueError("group tensor keys differ across consecutive CSV rows")
        result = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
        for key, tensor in result.items():
            if tensor.shape[0] < target_t_lat:
                raise ValueError(
                    f"combined {key} T={tensor.shape[0]} shorter than target latent T={target_t_lat}"
                )
            result[key] = tensor[:target_t_lat].contiguous()
        return result

    def __getitem__(self, index):
        for _ in range(20):
            sample_index = index % self.n_rows
            row = self.sample_start_rows[sample_index]
            if self.use_group_labels and any(
                r not in self.prepared_group_rows
                for r in range(row, row + self.consecutive_rows)
            ):
                raise ValueError(f"Grouping was not prepared for sample {sample_index}")
            try:
                rows = list(range(row, row + self.consecutive_rows))
                row_latents = [
                    [
                        torch.load(paths[r], weights_only=True, map_location="cpu")
                        for paths in self.latent_paths
                    ]
                    for r in rows
                ]
                row_actions = [
                    [self._load_action(paths[r]) for paths in self.action_paths]
                    for r in rows
                ]
                latent_lengths = []
                for segment_idx, (
                    source_row,
                    segment_latents,
                    segment_actions,
                ) in enumerate(zip(rows, row_latents, row_actions)):
                    for pidx, lat in enumerate(segment_latents, start=1):
                        if lat.dim() != 4 or lat.shape[0] != self.vae_latent_channels:
                            raise RuntimeError(
                                f"latent{pidx} shape {tuple(lat.shape)}, expected [{self.vae_latent_channels}, T_lat, h, w]: {self.latent_paths[pidx - 1][source_row]}"
                            )
                        if tuple(lat.shape) != tuple(segment_latents[0].shape):
                            raise RuntimeError(
                                f"latent{pidx} shape {tuple(lat.shape)} differs from latent1 {tuple(segment_latents[0].shape)} at CSV row {source_row}"
                            )
                    segment_rgb_t = int(segment_actions[0][0].shape[0])
                    expected_latent_t = (segment_rgb_t - 1) // 4 + 1
                    first_frame_only = (
                        self.allow_first_frame_latents
                        and int(segment_latents[0].shape[1]) == 1
                    )
                    if (
                        not first_frame_only
                        and int(segment_latents[0].shape[1]) != expected_latent_t
                    ):
                        raise RuntimeError(
                            f"CSV row {source_row} latent T={segment_latents[0].shape[1]} does not match action T={segment_rgb_t} (expected {expected_latent_t})"
                        )
                    for pidx, (keyboard_i, mouse_i) in enumerate(
                        segment_actions, start=1
                    ):
                        if keyboard_i.shape != (segment_rgb_t, 23) or mouse_i.shape != (
                            segment_rgb_t,
                            2,
                        ):
                            raise RuntimeError(
                                f"action{pidx} shape invalid at CSV row {source_row}: keyboard={tuple(keyboard_i.shape)} mouse={tuple(mouse_i.shape)}"
                            )
                    latent_lengths.append(expected_latent_t)
                target_t_lat = (self.target_rgb_frames - 1) // 4 + 1
                latents_per_player = [
                    torch.cat([segment[pidx] for segment in row_latents], dim=1)[
                        :, :target_t_lat
                    ]
                    for pidx in range(self.num_players)
                ]
                keyboards = [
                    torch.cat([segment[pidx][0] for segment in row_actions], dim=0)[
                        : self.target_rgb_frames
                    ]
                    for pidx in range(self.num_players)
                ]
                mice = [
                    torch.cat([segment[pidx][1] for segment in row_actions], dim=0)[
                        : self.target_rgb_frames
                    ]
                    for pidx in range(self.num_players)
                ]
                got_T = int(keyboards[0].shape[0])
                if got_T < self.target_rgb_frames:
                    warnings.warn(
                        f"[PrecomputedLatentDataset] action 比 target 短：got={got_T} want={self.target_rgb_frames} (row={row}, {self.action_paths[0][row]})。"
                    )
                for pidx, (kb, ms) in enumerate(zip(keyboards, mice), start=1):
                    if kb.dim() != 2 or ms.dim() != 2:
                        raise RuntimeError(
                            f"action{pidx} 形状异常 keyboard={tuple(kb.shape)} mouse={tuple(ms.shape)}"
                        )
                    if kb.shape[0] != got_T or ms.shape[0] != got_T:
                        raise RuntimeError(
                            f"action{pidx} T 与 action1 不一致: keyboard={tuple(kb.shape)} mouse={tuple(ms.shape)}, action1_T={got_T}"
                        )
                    if kb.shape[-1] != 23 or ms.shape[-1] != 2:
                        raise RuntimeError(
                            f"action{pidx} feature dim 异常，期望 keyboard[...,23] / mouse[...,2]，got {tuple(kb.shape)} / {tuple(ms.shape)}"
                        )
                exp_T_lat = (got_T - 1) // 4 + 1
                if latents_per_player[0].shape[1] != exp_T_lat and not (
                    self.allow_first_frame_latents
                    and latents_per_player[0].shape[1] == 1
                ):
                    raise RuntimeError(
                        f"latent T({latents_per_player[0].shape[1]}) 与 action T({got_T}) 不一致：期望 latent_T=(T-1)//4+1={exp_T_lat}（{self.latent_paths[0][row]}）"
                    )
                latents = torch.stack(latents_per_player, dim=0)
                keyboard = torch.stack(keyboards, dim=0)
                mouse = torch.stack(mice, dim=0)
                data = {
                    "latents": latents,
                    "keyboard": keyboard,
                    "mouse": mouse,
                    "source_rows": torch.tensor(rows, dtype=torch.long),
                }
                if self.num_players == 2:
                    data["latents1"] = latents[0]
                    data["latents2"] = latents[1]
                if self.use_camera_pose:
                    camera_pose = self._load_camera_pose_rows(
                        rows, latent_lengths, exp_T_lat
                    )
                    if camera_pose is None:
                        raise RuntimeError(
                            "use_camera_pose=True 但 CSV 没有可读取的 camera_pose_path/camera12_path"
                        )
                    data["camera_pose"] = camera_pose
                if self.use_group_labels:
                    data.update(
                        self._load_group_rows(rows, latent_lengths, exp_T_lat)
                    )
                if self.prompt_emb_paths is not None:
                    data["prompt_emb"] = _load_prompt_emb_many(
                        [paths[row] for paths in self.prompt_emb_paths]
                    )
                else:
                    data["prompt_emb"] = {}
                if self.captions is not None:
                    data["prompt"] = [caps[row] for caps in self.captions]
                return data
            except Exception as e:
                print(
                    f"ERROR WHEN LOADING (precomputed) sample={sample_index} source_row={row}: {e}"
                )
                if self.debug_no_retry:
                    raise
                if self.use_group_labels:
                    index = self.grouping_sample_indices[
                        random.randrange(len(self.grouping_sample_indices))
                    ]
                else:
                    index = random.randrange(self.n_rows)
        raise RuntimeError("PrecomputedLatentDataset: 连续加载失败")


_CAMERA_PLAYER_PART = re.compile("(?:^|/)player=(\\d+)(?:/|$)")
