from .config import select_case_rows, case_output_id
from .inference_args import parse_args
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from .utils import ModelConfig
from .pipelines.wan_video_Nplayers import WanVideoNPlayersPipeline
from .data.video import save_video
from multics.models.wan_video_dit import prepare_hier_local_hub_state_dict_for_load
from multics.data.dataset import (
    PrecomputedLatentDataset,
    consecutive_precomputed_start_rows,
)
from multics.data.raw_inputs import (
    encode_raw_first_frames,
    is_raw_metadata,
    prepare_raw_inputs,
)
import glob
from multics.data.utils import tensor_bchw_to_pil_list, pad_pil_list_to_length


def _timed(section: str):
    """Context manager: print load-stage timing (flush for tee/log)."""

    class _Ctx:
        def __enter__(self):
            print(f"[timing] >> {section}", flush=True)
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self._t0
            if exc_type is None:
                print(f"[timing] << {section} {dt:.2f}s", flush=True)
            else:
                print(
                    f"[timing] !! {section} failed after {dt:.2f}s: {exc}", flush=True
                )
            return False

    return _Ctx()


def _load_dit_checkpoint_state(ckpt_path: str):
    """Load a DiT checkpoint and fail fast when any saved weight is non-finite."""
    partial_ckpt = False
    if os.path.isdir(ckpt_path):
        index_path = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Checkpoint directory has no pytorch_model.bin.index.json: {ckpt_path}"
            )
        with open(index_path, "r") as f:
            index = json.load(f)
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise RuntimeError(f"No shards listed in checkpoint index: {index_path}")
        state_dict = {}
        for shard in shards:
            shard_path = os.path.join(ckpt_path, shard)
            with _timed(f"torch.load DiT ckpt shard {shard}"):
                shard_state = torch.load(
                    shard_path, map_location="cpu", weights_only=False
                )
            if not isinstance(shard_state, dict):
                raise RuntimeError(
                    f"Checkpoint shard is not a state_dict: {shard_path}"
                )
            state_dict.update(shard_state)
        partial_ckpt = True
    else:
        with _timed(f"torch.load DiT ckpt {os.path.basename(ckpt_path)}"):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if (
        isinstance(state_dict, dict)
        and "state_dict" in state_dict
        and isinstance(state_dict["state_dict"], dict)
    ):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"DiT checkpoint is not a state_dict: {ckpt_path}")
    if any((k.startswith("pipe.dit.") for k in state_dict.keys())):
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("pipe.dit."):
                stripped[key[len("pipe.dit.") :]] = value
        state_dict = stripped
        partial_ckpt = True
    bad = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or not (
            value.is_floating_point() or value.is_complex()
        ):
            continue
        try:
            finite = bool(torch.isfinite(value).all().item())
        except RuntimeError as exc:
            bad.append(
                f"{key}: shape={tuple(value.shape)} dtype={value.dtype} check_failed={exc}"
            )
            continue
        if not finite:
            bad.append(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            if len(bad) >= 12:
                break
    if bad:
        raise RuntimeError(
            f"DiT checkpoint contains non-finite tensor(s); refusing to run inference because this will generate NaN/black video.\nckpt_path={ckpt_path}\n"
            + "\n".join((f"  - {item}" for item in bad))
        )
    return (state_dict, partial_ckpt)


def _patch_model_manager_load_timing():
    """Log each Wan base weight file load inside from_pretrained."""
    from multics.models import model_manager as mm

    if getattr(mm.ModelManager.load_model, "_infer_timing_patched", False):
        return
    _orig = mm.ModelManager.load_model

    def _load_model_timed(
        self, file_path, model_names=None, device=None, torch_dtype=None
    ):
        if isinstance(file_path, list):
            label = f"ModelManager.load_model [{len(file_path)} shards]"
        else:
            label = f"ModelManager.load_model {os.path.basename(str(file_path))}"
        with _timed(label):
            return _orig(
                self,
                file_path,
                model_names=model_names,
                device=device,
                torch_dtype=torch_dtype,
            )

    _load_model_timed._infer_timing_patched = True
    mm.ModelManager.load_model = _load_model_timed


NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def _batch_output_names(batch_idx: int, args) -> list:
    """Filenames written for one dataloader row. batch_idx is the CSV row index."""
    names = []
    if not args.no_save_separate_players:
        for pidx in range(args.num_players):
            names.append(
                f"video{batch_idx}_{_player_file_tag(pidx, args.num_players)}.mp4"
            )
    if not args.no_save_grid:
        names.append(f"video{batch_idx}_grid.mp4")
    return names


def _list_mp4_basenames(output_dir, s3_uri=None):
    return (
        {name for name in os.listdir(output_dir) if name.endswith(".mp4")}
        if os.path.isdir(output_dir)
        else set()
    )


def _existing_batch_indices(
    n_planned: int, args, existing: set, output_ids=None
) -> set:
    done = set()
    for batch_idx in range(n_planned):
        expected = _batch_output_names(
            output_ids[batch_idx] if output_ids is not None else batch_idx, args
        )
        if expected and all((name in existing for name in expected)):
            done.add(batch_idx)
    return done


def _infer_rank_world():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK={rank} out of range for WORLD_SIZE={world_size}")
    return (rank, world_size)


def _infer_shard_rows(
    n_planned: int, skip_batches: set, rank: int, world_size: int
) -> list:
    """CSV row ids this rank should generate. Names stay video{row}_*.mp4."""
    return [
        i for i in range(n_planned) if i % world_size == rank and i not in skip_batches
    ]


def _infer_row_bounds(n_planned: int) -> tuple[int, int]:
    """Optional half-open row range for splitting one result repo across jobs."""
    start = int(os.environ.get("INFER_ROW_START", "0"))
    end = int(os.environ.get("INFER_ROW_END", str(n_planned)))
    if start < 0 or end < start or end > n_planned:
        raise ValueError(
            f"invalid inference row range [{start}, {end}) for n_planned={n_planned}"
        )
    return (start, end)


TRAIN_HEIGHT = 352
TRAIN_WIDTH = 640
TILE_SIZE = (34, 34)
TILE_STRIDE = (18, 16)


def _resolve_metadata_path(dataset_path, metadata_file_name):
    if os.path.isabs(metadata_file_name):
        return metadata_file_name
    return os.path.join(dataset_path, metadata_file_name)


def _precomputed_video_latents(batch) -> torch.Tensor:
    """[P,48,T_lat,h,w] from PrecomputedLatentDataset batch (batch_size=1)."""
    if "latents" in batch:
        if batch["latents"].shape[0] != 1:
            raise ValueError(
                "precomputed inference currently expects DataLoader batch_size=1"
            )
        return batch["latents"].squeeze(0)
    return torch.cat([batch["latents1"], batch["latents2"]], dim=0)


def _decode_latents_to_video(
    pipe, latents_b1: torch.Tensor, tiler_kwargs: dict
) -> torch.Tensor:
    """[1, 48, T_lat, h, w] -> [1, 3, T_rgb, H, W] for GT grid / parity."""
    pipe.load_models_to_device(["vae"])
    return pipe.decode_video(
        latents_b1.to(dtype=pipe.torch_dtype, device=pipe.device), **tiler_kwargs
    )


def _decode_latents_to_videos(
    pipe, latents_p: torch.Tensor, tiler_kwargs: dict
) -> torch.Tensor:
    """[P,48,T_lat,h,w] -> [P,3,T_rgb,H,W] for GT grids."""
    pipe.load_models_to_device(["vae"])
    return pipe.decode_video(
        latents_p.to(dtype=pipe.torch_dtype, device=pipe.device), **tiler_kwargs
    )


def _as_player_videos(outputs):
    """Normalize pipeline return to list length P; P=2 may be returned as a tuple for compatibility."""
    if isinstance(outputs, tuple):
        return list(outputs)
    return outputs


def _batch_prompt_strings(batch, num_players: int):
    """Return per-player prompt strings from a collated precomputed batch."""
    if "prompt" not in batch:
        return None
    prompt = batch["prompt"]
    if isinstance(prompt, (list, tuple)):
        out = [
            p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else p for p in prompt
        ]
    else:
        out = [prompt]
    out = [str(p).strip() for p in out]
    if len(out) == 1 and num_players > 1:
        out = out * num_players
    if len(out) != num_players:
        raise ValueError(
            f"prompt count {len(out)} must match num_players={num_players}"
        )
    return out


def _prompt_context_from_batch(pipe, args, batch, prompt_strings):
    """Build T5 context [P,S,4096] or [1,S,4096] for full-sequence inference."""
    prompt_emb = batch.get("prompt_emb", {})
    context = prompt_emb.get("context") if isinstance(prompt_emb, dict) else None
    if context is not None:
        if context.dim() == 4:
            if context.shape[0] != 1:
                raise ValueError(
                    f"prompt context batch must be 1, got {tuple(context.shape)}"
                )
            context = context[0]
        return context.to(dtype=pipe.torch_dtype, device=pipe.device)
    pipe.load_models_to_device(["text_encoder"])
    if prompt_strings is not None and any(prompt_strings):
        return pipe.encode_prompt(prompt_strings, positive=True)["context"].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
    return pipe.encode_prompt("", positive=True)["context"].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )


@torch.no_grad()
def _hier_full_sequence_inference(
    pipe,
    args,
    batch,
    video_latents: torch.Tensor,
    mouse: torch.Tensor,
    keyboard: torch.Tensor,
    camera_pose: torch.Tensor,
    agent_vertex_ids,
):
    """Hierarchical inference with a full-sequence block-causal forward, no KV cache.

    Key tensors:
      video_latents: [P,C,T,h,w], clean GT latents used only for frame-0 anchor.
      latents:       [P,C,T,h,w], denoising state.
      timestep:      [P,T], frame 0 pinned to t=0; other frames use current sigma.
      group_labels:  [1,T,P] from dataloader, passed to WanModel.forward.
      active_group:  [1,T,N] from dataloader, passed to WanModel.forward.
    """
    if "group_labels" not in batch or "active_group" not in batch:
        raise KeyError(
            "hier_hub full-sequence inference requires group_labels and active_group"
        )
    if video_latents is None:
        raise ValueError(
            "hier_hub full-sequence inference requires --precomputed_latents"
        )
    video_latents = video_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
    P, C, T, h, w = video_latents.shape
    if T == 1:
        T = int(batch["group_labels"].shape[1])
    if P != args.num_players:
        raise ValueError(
            f"video_latents P={P} must match --num_players={args.num_players}"
        )
    group_labels = batch["group_labels"].to(device=pipe.device, dtype=torch.long)
    active_group = batch["active_group"].to(device=pipe.device, dtype=torch.bool)
    membership = batch.get("membership")
    global_only = batch.get("global_only")
    if membership is not None:
        membership = membership.to(device=pipe.device, dtype=torch.bool)
        if global_only is None:
            raise KeyError(
                "hier_group_format=membership requires batch key global_only"
            )
        global_only = global_only.to(device=pipe.device, dtype=torch.bool)
    if group_labels.shape[1] != T or group_labels.shape[2] != P:
        raise ValueError(
            f"group_labels must be [1,T={T},P={P}], got {tuple(group_labels.shape)}"
        )
    if active_group.shape[1] != T or active_group.shape[2] != args.num_groups:
        raise ValueError(
            f"active_group must be [1,T={T},N={args.num_groups}], got {tuple(active_group.shape)}"
        )
    mouse = mouse.to(dtype=pipe.torch_dtype, device=pipe.device)
    keyboard = keyboard.to(dtype=pipe.torch_dtype, device=pipe.device)
    if camera_pose is not None:
        camera_pose = camera_pose.to(dtype=pipe.torch_dtype, device=pipe.device)
    pipe.scheduler.set_timesteps(
        args.num_inference_steps, denoising_strength=1.0, shift=5.0
    )
    noise = pipe.generate_noise(
        (P, C, T, h, w), seed=args.seed, device="cpu", dtype=torch.float32
    )
    latents = noise.to(dtype=pipe.torch_dtype, device=pipe.device)
    latents[:, :, 0:1, ...] = video_latents[:, :, 0:1, ...].clone()
    prompt_strings = _batch_prompt_strings(batch, args.num_players)
    prompt_context = _prompt_context_from_batch(pipe, args, batch, prompt_strings)
    has_prompt = bool(prompt_strings is not None and any(prompt_strings))
    has_prompt_emb = "prompt_emb" in batch and batch["prompt_emb"] is not None
    cfg_scale = args.cfg_scale if has_prompt else 1.0
    negative_context = None
    if cfg_scale != 1.0:
        pipe.load_models_to_device(["text_encoder"])
        negative_context = pipe.encode_prompt([NEGATIVE_PROMPT] * P, positive=False)[
            "context"
        ].to(dtype=pipe.torch_dtype, device=pipe.device)
    print(
        f"[full_sequence] P={P} latent_T={T} cfg_scale={cfg_scale} prompt={('caption/emb' if has_prompt or has_prompt_emb else 'blank')}",
        flush=True,
    )
    pipe.load_models_to_device(["dit"])
    for progress_id, step_t in enumerate(pipe.scheduler.timesteps):
        timestep = torch.full((P, T), step_t, dtype=torch.float32, device=pipe.device)
        timestep[:, 0] = 0.0
        noise_pred_pos = pipe.dit(
            latents,
            timestep=timestep,
            context=prompt_context,
            mouse=mouse,
            keyboard=keyboard,
            loss_type="diffusion_forcing",
            agent_vertex_ids=agent_vertex_ids,
            camera_pose=camera_pose,
            group_labels=group_labels,
            active_group=active_group,
            membership=membership,
            global_only=global_only,
        )
        if cfg_scale != 1.0:
            noise_pred_neg = pipe.dit(
                latents,
                timestep=timestep,
                context=negative_context,
                mouse=mouse,
                keyboard=keyboard,
                loss_type="diffusion_forcing",
                agent_vertex_ids=agent_vertex_ids,
                camera_pose=camera_pose,
                group_labels=group_labels,
                active_group=active_group,
                membership=membership,
                global_only=global_only,
            )
            noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
        else:
            noise_pred = noise_pred_pos
        latents = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[progress_id], latents
        )
        latents[:, :, 0:1, ...] = video_latents[:, :, 0:1, ...].clone()
    player_videos = pipe._player_videos_from_latents(
        latents, {"tiled": True, "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE}
    )
    pipe.load_models_to_device([])
    return _as_player_videos(pipe._return_player_videos(player_videos))


def _player_file_tag(player_idx: int, num_players: int) -> str:
    if num_players == 2:
        return "playerA" if player_idx == 0 else "playerB"
    return f"player{player_idx + 1:02d}"


def _build_player_rows_grid_video(rows, t_grid: int, border: int = 1):
    """Build a generic rows x P grid.

    Tensor/PIL convention at caller:
      rows: List[R][P][T] of PIL frames.
      Current N-player inference uses R=2:
        row 0 = generated players [P,T]
        row 1 = GT players        [P,T]
    """
    if not rows or not rows[0]:
        raise ValueError("_build_player_rows_grid_video: empty rows")
    num_rows = len(rows)
    num_players = len(rows[0])
    for ridx, row in enumerate(rows):
        if len(row) != num_players:
            raise ValueError(
                f"_build_player_rows_grid_video: row {ridx} has {len(row)} players, expected {num_players}"
            )
        for pidx, frames in enumerate(row):
            if len(frames) < t_grid:
                raise ValueError(
                    f"_build_player_rows_grid_video: row={ridx} player={pidx} has {len(frames)} frames, expected >= {t_grid}"
                )
    w0, h0 = rows[0][0][0].size
    total_w = num_players * w0 + (num_players - 1) * border
    total_h = num_rows * h0 + (num_rows - 1) * border
    grid_frames = []
    for tidx in range(t_grid):
        canvas = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))
        for ridx, row in enumerate(rows):
            for pidx, frames in enumerate(row):
                x = pidx * (w0 + border)
                y = ridx * (h0 + border)
                canvas.paste(frames[tidx], (x, y))
        grid_frames.append(canvas)
    return grid_frames


def main():
    _t_main = time.perf_counter()
    print("[timing] main() start", flush=True)
    with _timed("parse_args"):
        args = parse_args()
    if args.num_players < 2:
        raise ValueError(f"--num_players must be >= 2, got {args.num_players}")
    agent_vertex_ids = args.agent_vertex_ids
    if agent_vertex_ids is None:
        agent_vertex_ids = list(range(args.num_players))
    elif len(agent_vertex_ids) != args.num_players:
        raise ValueError(
            f"--agent_vertex_ids length ({len(agent_vertex_ids)}) must match --num_players ({args.num_players})"
        )
    args.loss_type = "diffusion_forcing"
    with _timed("read metadata csv"):
        meta_path = _resolve_metadata_path(args.dataset_path, args.metadata_file_name)
        metadata_table = pd.read_csv(meta_path)
        n_rows = len(metadata_table)
    if args.precomputed_consecutive_rows < 1:
        raise ValueError("--precomputed_consecutive_rows must be >= 1")
    precomputed_start_rows = None
    precomputed_start_rows = consecutive_precomputed_start_rows(
        metadata_table, args.precomputed_consecutive_rows, args.num_players
    )
    available_rows = len(precomputed_start_rows)
    if available_rows < 1:
        raise ValueError(
            f"metadata has no valid window for precomputed_consecutive_rows={args.precomputed_consecutive_rows} (source rows={n_rows})"
        )
    n_planned = (
        available_rows
        if args.max_batches is None
        else min(available_rows, args.max_batches)
    )
    sample_metadata = metadata_table.iloc[precomputed_start_rows].reset_index(drop=True)
    output_ids = [case_output_id(metadata_table, row) for row in precomputed_start_rows]
    sample_metadata["case_id"] = output_ids
    rank, world_size = _infer_rank_world()
    row_start, row_end = _infer_row_bounds(n_planned)
    eligible = select_case_rows(args, sample_metadata, list(range(row_start, row_end)))
    selected = set(eligible)
    skip_batches = set()
    if os.environ.get("FORCE_INFER", "0") != "1":
        existing = _list_mp4_basenames(args.output_dir, None)
        skip_batches = _existing_batch_indices(n_planned, args, existing, output_ids)
        remaining = n_planned - len(skip_batches)
        print(
            f"[skip] {len(skip_batches)}/{n_planned} batches already have mp4 (local); remaining={remaining}; FORCE_INFER=1 to overwrite",
            flush=True,
        )
        if skip_batches and remaining == 0:
            print("[skip] all planned videos exist; nothing to generate", flush=True)
            return
        if skip_batches:
            print(f"[skip] done batches={sorted(skip_batches)}", flush=True)
    my_rows = [
        i
        for i in _infer_shard_rows(n_planned, skip_batches, rank, world_size)
        if i in selected
    ]
    print(
        f"[shard] rank={rank}/{world_size} n_planned={n_planned} range=[{row_start},{row_end}) skip={len(skip_batches)} my_rows={my_rows}",
        flush=True,
    )
    if os.environ.get("INFER_SHARD_DRY", "0") == "1":
        print("[shard] INFER_SHARD_DRY=1, exit before model load", flush=True)
        return
    if not my_rows:
        print("[shard] nothing left on this rank; skip model load", flush=True)
        return
    raw_inputs = is_raw_metadata(metadata_table)
    if raw_inputs:
        if args.precomputed_consecutive_rows != 1:
            raise ValueError("Raw-input CSVs require precomputed_consecutive_rows=1")
        with _timed("prepare actions, cameras, and grouping from tick logs"):
            meta_path = prepare_raw_inputs(
                metadata_table,
                args.dataset_path,
                my_rows,
                args.num_players,
                args.target_rgb_frames,
                args.prepared_inputs_dir,
            )
        print(f"[infer] prepared raw inputs: {meta_path}", flush=True)
    with _timed("dataset init"):
        dataset = PrecomputedLatentDataset(
            args.dataset_path,
            meta_path,
            steps_per_epoch=available_rows,
            num_players=args.num_players,
            target_rgb_frames=args.target_rgb_frames,
            blank_prompt_emb_path=args.blank_prompt_emb_path,
            vae_latent_channels=args.vae_latent_channels,
            debug_no_retry=os.environ.get("INFER_SKIP_BAD_ROWS") != "1",
            use_camera_pose=args.use_camera_pose,
            use_group_labels=True,
            num_groups=args.num_groups,
            group_path_column=args.group_path_column,
            group_label_player_ids=args.group_label_player_ids,
            hier_group_mode=args.hier_group_mode,
            hier_group_seed=args.hier_group_seed,
            hier_random_preserve_sizes=bool(args.hier_random_preserve_sizes),
            hier_group_format=args.hier_group_format,
            num_frame_per_block=args.num_frame_per_block,
            consecutive_rows=args.precomputed_consecutive_rows,
            grouping_sample_indices=my_rows,
            grouping_cache_dir=args.grouping_cache_dir,
            grouping_model_profile=(args.num_local_hub_tokens, args.num_global_hub_tokens,
                                    args.hier_direct_frac, args.num_frame_per_block),
            allow_first_frame_latents=raw_inputs,
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    wan_root = args.wan_model_root.rstrip("/")
    print(f"[timing] wan_model_root={wan_root}", flush=True)
    print(f"[timing] ckpt_path={args.ckpt_path}", flush=True)
    with _timed("glob DiT safetensors"):
        dit_files = sorted(
            glob.glob(f"{wan_root}/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors")
        )
        print(f"[timing]   found {len(dit_files)} shards", flush=True)
    _patch_model_manager_load_timing()
    with _timed("WanVideoNPlayersPipeline.from_pretrained (T5+DiT+VAE+tokenizer)"):
        pipe = WanVideoNPlayersPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            tokenizer_config=ModelConfig(
                path=f"{wan_root}/Wan2.1-T2V-1.3B/google/umt5-xxl"
            ),
            model_configs=[
                ModelConfig(
                    path=f"{wan_root}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    offload_device="cpu",
                ),
                ModelConfig(path=dit_files, offload_device="cpu"),
                ModelConfig(
                    path=f"{wan_root}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                    offload_device="cpu",
                ),
            ],
        )
        pipe.num_players = int(args.num_players)
    with _timed("patch DiT (keyboard/action modules)"):
        pipe.dit.keyboard_embedding = nn.Linear(23, 3072)
        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in pipe.dit.blocks:
            block.action_mlp = nn.Sequential(
                nn.LayerNorm(dim + 24, eps=1e-05, elementwise_affine=True),
                nn.Linear(dim + 24, dim + 24),
                nn.GELU(),
                nn.Linear(dim + 24, dim),
                nn.LayerNorm(dim, eps=1e-05, elementwise_affine=True),
            )
            block.action_temporal_attn = copy.deepcopy(block.self_attn)
            block.action_cross_attn = copy.deepcopy(block.cross_attn)
        pipe.dit.enable_simplex_rope(
            pool_size=args.simplex_pool_size, gamma=args.simplex_gamma
        )
    latent_t = (args.target_rgb_frames - 1) // 4 + 1
    if latent_t % args.num_frame_per_block != 0:
        raise ValueError(
            f"latent_T={latent_t} must divide num_frame_per_block={args.num_frame_per_block}"
        )
    if args.local_attn_size != -1 and args.local_attn_size < args.num_frame_per_block:
        raise ValueError(
            f"local_attn_size must be -1 or >= num_frame_per_block; got local={args.local_attn_size}, n={args.num_frame_per_block}"
        )
    kg = args.num_global_hub_tokens
    if kg is None:
        kg = args.num_hub_tokens - args.num_groups * args.num_local_hub_tokens
    if args.num_groups * args.num_local_hub_tokens + kg != args.num_hub_tokens:
        raise ValueError(
            f"hier_hub requires num_groups*num_local_hub_tokens + num_global_hub_tokens == num_hub_tokens; got {args.num_groups}*{args.num_local_hub_tokens}+{kg} != {args.num_hub_tokens}"
        )
    if args.hier_no_local_hub:
        if args.hier_local_hub_mode != "shared":
            raise ValueError(
                "--hier_no_local_hub requires --hier_local_hub_mode shared"
            )
        if args.num_groups <= 0 or args.num_local_hub_tokens != 0 or kg <= 0:
            raise ValueError(
                f"--hier_no_local_hub requires num_groups>0, K_L=0, and K_G>0; got N={args.num_groups}, K_L={args.num_local_hub_tokens}, K_G={kg}"
            )
        infer_kernel = os.environ.get("WAN_HIER_INFER_KERNEL", "sdpa").strip().lower()
        if infer_kernel not in ("sdpa", "packed_varlen", "paged_kv"):
            raise ValueError(
                "--hier_no_local_hub supports WAN_HIER_INFER_KERNEL=sdpa, packed_varlen, or paged_kv; got "
                + infer_kernel
            )
    elif args.num_local_hub_tokens <= 0:
        raise ValueError("ordinary hier_hub requires num_local_hub_tokens>0")
    pipe.enable_stage2(
        num_hub_tokens=args.num_hub_tokens,
        local_attn_size=args.local_attn_size,
        num_frame_per_block=args.num_frame_per_block,
        attention_mode="hier_hub",
        num_groups=args.num_groups,
        num_local_hub_tokens=args.num_local_hub_tokens,
        num_global_hub_tokens=args.num_global_hub_tokens,
        hier_direct_schedule=args.hier_direct_schedule,
        hier_direct_frac=args.hier_direct_frac,
        hier_direct_layers=args.hier_direct_layers,
        attn_impl=args.attn_impl,
        hier_no_local_hub=args.hier_no_local_hub,
        hier_local_hub_mode=args.hier_local_hub_mode,
    )
    if args.use_camera_pose:
        with _timed("dit.enable_camera_pose_alignment"):
            pipe.dit.enable_camera_pose_alignment(
                cam_dim=args.camera_dim, dropout_p=args.camera_dropout_p
            )
    state_dict, partial_ckpt = _load_dit_checkpoint_state(args.ckpt_path)
    if not args.hier_no_local_hub:
        state_dict, migration = prepare_hier_local_hub_state_dict_for_load(
            state_dict,
            target_mode=args.hier_local_hub_mode,
            target_num_groups=pipe.dit.num_groups,
            target_num_local_hub_tokens=pipe.dit.num_local_hub_tokens,
            model_dim=pipe.dit.dim,
        )
        if migration is not None:
            print(
                f"[hier-local-hub] inference migration: {migration['source_mode']} {migration['source_shape']} -> {migration['target_mode']} {migration['target_shape']}",
                flush=True,
            )
    if args.hier_no_local_hub:
        if "local_hub_embed" in state_dict:
            raise RuntimeError(
                "No-local inference refuses to migrate an old hierarchical checkpoint. Warm-start it through training with --hier_no_local_hub, save a new checkpoint, then infer from that checkpoint."
            )
        global_seed = state_dict.get("global_hub_embed")
        expected = tuple(pipe.dit.global_hub_embed.shape)
        if not torch.is_tensor(global_seed) or tuple(global_seed.shape) != expected:
            raise RuntimeError(
                f"No-local checkpoint has an incompatible global_hub_embed: expected {expected}, got {(None if global_seed is None else tuple(global_seed.shape))}"
            )
    with _timed("dit.load_state_dict(ckpt)"):
        missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
    if missing:
        _hub_related = [k for k in missing if k.endswith("hub_embed")]
        _hier_hub_related = [
            k for k in missing if k in ("local_hub_embed", "global_hub_embed")
        ]
        if _hier_hub_related:
            raise RuntimeError(
                f"hier_hub ckpt is missing required hub embeddings: {_hier_hub_related}. This usually means CKPT_PATH points to the wrong experiment/version or to a non-hier checkpoint."
            )
        _cam_related = [k for k in missing if ".cam_q." in k or ".cam_k." in k]
        _other = [k for k in missing if k not in _hub_related and k not in _cam_related]
        if _other and (not partial_ckpt):
            raise RuntimeError(
                f"Unexpected missing keys in DiT ckpt: {_other[:10]} ..."
            )
        if _other and partial_ckpt:
            print(
                f"[ckpt] partial DiT ckpt missing {len(_other)} key(s); keeping those values from the pretrained/base model.",
                flush=True,
            )
        if _hub_related:
            print(
                f"[checkpoint] {len(_hub_related)} hub_embed param(s) absent in DiT ckpt (zeros init)."
            )
        if _cam_related:
            print(
                f"[camera-pose] {len(_cam_related)} camera adapter param(s) absent in DiT ckpt (zero-init)."
            )
    if unexpected:
        _ff_keys = (".k_ff", ".v_ff", ".norm_k_ff", ".ff_gate")
        _ff_related = [k for k in unexpected if any((s in k for s in _ff_keys))]
        _handled_unexpected = list(_ff_related)
        _old_hub = [k for k in unexpected if k.endswith(".hub_embed")]
        if _old_hub:
            print(
                f"[checkpoint] ignoring {len(_old_hub)} legacy per-block hub_embed key(s) in ckpt."
            )
        _handled_unexpected += _old_hub
        _legacy_cam_bias = [
            k
            for k in unexpected
            if k.endswith(".cam_q.bias") or k.endswith(".cam_k.bias")
        ]
        if _legacy_cam_bias:
            print(
                f"[camera-pose] dropping {len(_legacy_cam_bias)} legacy camera projection bias key(s) for the bias-free adapter."
            )
        _disabled_cam = [
            k
            for k in unexpected
            if not args.use_camera_pose
            and (".cam_q." in k or ".cam_k." in k)
            and (k not in _legacy_cam_bias)
        ]
        if _disabled_cam:
            print(
                f"[camera-pose] ignoring {len(_disabled_cam)} camera adapter key(s) because camera conditioning is disabled."
            )
        _handled_unexpected += _legacy_cam_bias + _disabled_cam
        if _ff_related:
            print(
                f"[WARN] DiT ckpt 中检测到 {len(_ff_related)} 个 first-frame cross-attn 分支参数 （k_ff/v_ff/norm_k_ff/ff_gate），当前架构已回退到 latent-overwrite 方案，这些参数会被忽略。如不需要可重新存一份干净的 ckpt。示例: {_ff_related[:3]}"
            )
        _other_unexpected = [k for k in unexpected if k not in _handled_unexpected]
        if _other_unexpected:
            print(f"[WARN] unexpected keys in DiT ckpt: {_other_unexpected[:10]} ...")
    with _timed("pipe.to(cuda) + bfloat16"):
        pipe.to("cuda")
        pipe.to(dtype=torch.bfloat16)
    if raw_inputs:
        with _timed("encode first frames with the VAE"):
            encode_raw_first_frames(pipe, meta_path, my_rows, args.num_players)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.precomputed_consecutive_rows > 1:
        manifest = {
            "metadata_file": meta_path,
            "target_rgb_frames": int(args.target_rgb_frames),
            "consecutive_rows": int(args.precomputed_consecutive_rows),
            "samples": [
                {
                    "sample_index": sample_index,
                    "case_id": output_ids[sample_index],
                    "output_index": output_ids[sample_index],
                    "source_rows": list(
                        range(
                            int(precomputed_start_rows[sample_index]),
                            int(precomputed_start_rows[sample_index])
                            + args.precomputed_consecutive_rows,
                        )
                    ),
                }
                for sample_index in range(n_planned)
            ],
        }
        with open(
            os.path.join(args.output_dir, "consecutive_rows_manifest.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(
        f"[timing] model load total {time.perf_counter() - _t_main:.2f}s",
        flush=True,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, my_rows),
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    tiler_kwargs = {"tiled": True, "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE}
    target_text = ""
    _t_first_batch = time.perf_counter()
    for local_i, batch in enumerate(loader):
        batch_idx = output_ids[my_rows[local_i]]
        if local_i == 0:
            src = "precomputed latent/action"
            print(
                f"[timing] first dataloader batch {time.perf_counter() - _t_first_batch:.2f}s ({src})",
                flush=True,
            )
        video_latents = None
        gt_all = None
        keyboard = batch["keyboard"].squeeze(0)[:, :-1, :]
        mouse = batch["mouse"].squeeze(0)[:, :-1, :]
        video_latents = _precomputed_video_latents(batch).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        if video_latents.shape[0] != args.num_players:
            raise ValueError(
                f"batch video_latents P={video_latents.shape[0]} does not match --num_players={args.num_players}"
            )
        # First-frame-only latents (raw inputs) have no ground-truth clip.
        if video_latents.shape[2] > 1:
            gt_all = _decode_latents_to_videos(pipe, video_latents, tiler_kwargs)
            video1 = gt_all[0:1]
            video2 = gt_all[1:2]
        else:
            video1 = video2 = None
        camera_pose = None
        if args.use_camera_pose:
            if "camera_pose" not in batch:
                raise KeyError(
                    "--use_camera_pose 需要 batch 含 camera_pose（CSV camera_pose_path* / camera12_path*）"
                )
            camera_pose = (
                batch["camera_pose"]
                .squeeze(0)
                .to(dtype=pipe.torch_dtype, device=pipe.device)
            )
        prompt = target_text
        negative_prompt = NEGATIVE_PROMPT
        has_caption_prompt = False
        if "prompt" in batch:
            prompt = batch["prompt"]
            if isinstance(prompt, (list, tuple)):
                prompt = [
                    p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else p
                    for p in prompt
                ]
            else:
                prompt = [prompt]
            prompt = [str(p).strip() for p in prompt]
            has_caption_prompt = any(prompt)
            if has_caption_prompt:
                negative_prompt = [
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                ] * len(prompt)
            else:
                prompt = target_text
        causal_cfg_scale = args.cfg_scale if has_caption_prompt else 1.0
        print(
            f"[infer] prompt={('caption' if has_caption_prompt else 'blank')} cfg_scale={causal_cfg_scale}",
            flush=True,
        )
        if args.hier_infer_mode == "full_sequence":
            gen_players = _hier_full_sequence_inference(
                pipe,
                args,
                batch,
                video_latents,
                mouse,
                keyboard,
                camera_pose,
                agent_vertex_ids,
            )
        else:
            if "group_labels" not in batch or "active_group" not in batch:
                raise KeyError(
                    "hier_hub cached inference requires group_labels and active_group"
                )
            gen_players = _as_player_videos(
                pipe.causal_inference(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    video1=video1,
                    video2=video2,
                    mouse=mouse,
                    keyboard=keyboard,
                    num_frames=args.causal_num_frames,
                    height=TRAIN_HEIGHT,
                    width=TRAIN_WIDTH,
                    cfg_scale=causal_cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    tiled=True,
                    tile_size=TILE_SIZE,
                    tile_stride=TILE_STRIDE,
                    loss_type="diffusion_forcing",
                    return_both_players=True,
                    agent_vertex_ids=agent_vertex_ids,
                    video_latents=video_latents,
                    camera_pose=camera_pose,
                    overlap_path=None,
                    group_labels=batch["group_labels"],
                    active_group=batch["active_group"],
                    membership=batch.get("membership"),
                    global_only=batch.get("global_only"),
                )
            )
        if len(gen_players) != args.num_players:
            raise RuntimeError(
                f"Generated {len(gen_players)} player videos, expected P={args.num_players}"
            )
        fps = args.grid_fps
        if not args.no_save_separate_players:
            for pidx, gen_p in enumerate(gen_players):
                save_video(
                    gen_p,
                    os.path.join(
                        args.output_dir,
                        f"video{batch_idx}_{_player_file_tag(pidx, args.num_players)}.mp4",
                    ),
                    fps=fps,
                    quality=5,
                )
        if not args.no_save_grid:
            T_gen = len(gen_players[0])
            for pidx, gen_p in enumerate(gen_players):
                if len(gen_p) != T_gen:
                    raise RuntimeError(
                        f"Player outputs length mismatch: player0={T_gen} vs player{pidx}={len(gen_p)}"
                    )
            grid_rows = [gen_players]
            if gt_all is not None:
                grid_rows.append(
                    [
                        tensor_bchw_to_pil_list(gt_all[pidx])
                        for pidx in range(args.num_players)
                    ]
                )
            T_grid = max(len(frames) for row in grid_rows for frames in row)
            grid_frames = _build_player_rows_grid_video(
                [
                    [pad_pil_list_to_length(frames, T_grid) for frames in row]
                    for row in grid_rows
                ],
                T_grid,
            )
            save_video(
                grid_frames,
                os.path.join(args.output_dir, f"video{batch_idx}_grid.mp4"),
                fps=fps,
                quality=5,
            )
