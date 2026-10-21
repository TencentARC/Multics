import argparse
import os
from .config import resolve_args


def parse_args():
    p = argparse.ArgumentParser(
        description="Multics hierarchical multi-player inference"
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset root (CSV paths resolved relative to this).",
    )
    p.add_argument(
        "--metadata_file_name",
        type=str,
        default=None,
        help="CSV path or name under dataset_path.",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="DiT state_dict (.ckpt from Lightning on_save_checkpoint).",
    )
    p.add_argument(
        "--wan_model_root",
        type=str,
        default=None,
        help="Directory containing Wan2.1-T2V-1.3B and Wan2.2-TI2V-5B.",
    )
    p.add_argument(
        "--target_rgb_frames",
        type=int,
        default=81,
        help="RGB frame count. Must be 4n+1 and match training. Default 81.",
    )
    p.add_argument(
        "--blank_prompt_emb_path",
        type=str,
        default=None,
        help="Unused at inference. Prompts are encoded from the CSV. Training reads asset/blank_string_prompt_emb.pth when text is dropped or missing.",
    )
    p.add_argument("--vae_latent_channels", type=int, default=48)
    p.add_argument("--num_players", type=int, default=2)
    p.add_argument(
        "--use_simplex_rope",
        action="store_true",
        default=True,
        help="Use simplex 4D RoPE instead of learned player embeddings.",
    )
    p.add_argument("--simplex_pool_size", type=int, default=4)
    p.add_argument("--simplex_gamma", type=float, default=1.0)
    p.add_argument(
        "--agent_vertex_ids",
        type=int,
        nargs="+",
        default=None,
        help="Simplex vertex id for each player. Defaults to 0 .. num_players-1.",
    )
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--training_stage",
        type=str,
        default="stage2",
        choices=["stage2"],
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--attention_mode", type=str, default="hier_hub", choices=["hier_hub"]
    )
    p.add_argument("--local_attn_size", type=int, default=24)
    p.add_argument("--num_frame_per_block", type=int, default=7)
    p.add_argument("--num_local_hub_tokens", type=int, default=5)
    p.add_argument("--num_global_hub_tokens", type=int, default=6)
    p.add_argument(
        "--hier_no_local_hub",
        action="store_true",
        help="Ablation: remove local hubs and connect every player directly to global hub.",
    )
    p.add_argument(
        "--hier_local_hub_mode",
        type=str,
        default="shared",
        choices=["shared", "per_group"],
        help="hier_hub local seed parameterization: shared=[K_L,D], per_group=[N,K_L,D].",
    )
    p.add_argument(
        "--hier_direct_schedule",
        type=str,
        default="none",
        choices=["none", "uniform", "explicit"],
    )
    p.add_argument("--hier_direct_frac", type=float, default=0.0)
    p.add_argument("--hier_direct_layers", type=str, default=None)
    p.add_argument(
        "--hier_infer_mode",
        type=str,
        default="cached",
        choices=["full_sequence", "cached"],
        help="hier_hub inference mode: cached uses the KV cache; full_sequence is a slower reference path without it.",
    )
    p.add_argument("--group_path_column", type=str, default="group_path")
    p.add_argument(
        "--hier_group_format",
        type=str,
        default="labels",
        choices=["labels", "membership"],
        help="hier_hub grouping tensor format. labels=partition (default); membership=cover.",
    )
    p.add_argument(
        "--hier_group_mode",
        type=str,
        default="csv",
        choices=["csv", "random_static", "random_block"],
    )
    p.add_argument("--hier_group_seed", type=int, default=0)
    p.add_argument("--hier_random_preserve_sizes", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--group_label_player_ids",
        type=str,
        default=None,
        help="1-based player ids used to select columns from group labels, e.g. 1,2,3,4.",
    )
    p.add_argument(
        "--use_camera_pose",
        action="store_true",
        help="Add relative camera conditioning to self-attention Q and K. The CSV must contain camera_pose_path* or camera12_path*.",
    )
    p.add_argument(
        "--camera_dim", type=int, default=12, help="Camera-pose embedding size."
    )
    p.add_argument(
        "--camera_dropout_p",
        type=float,
        default=0.1,
        help="Camera-adapter dropout used in training. Disabled when the model is evaluated.",
    )
    p.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "flex"])
    p.add_argument(
        "--loss_type",
        type=str,
        default="diffusion_forcing",
        choices=["diffusion_forcing"],
        help="Training objective. Fixed to diffusion forcing.",
    )
    p.add_argument(
        "--causal_num_frames",
        type=int,
        default=257,
        help="RGB frame count for diffusion_forcing / causal_inference only.",
    )
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="If set, only run this many batches.",
    )
    p.add_argument(
        "--precomputed_consecutive_rows",
        type=int,
        default=1,
        help="For precomputed N-player data, concatenate this many consecutive CSV rows along time before trimming to target_rgb_frames. Default 1 preserves existing behavior.",
    )
    p.add_argument(
        "--parity_atol",
        type=float,
        default=0.5,
        help="Relative threshold for the parity check, max|delta| / block mean. Default 0.5.",
    )
    p.add_argument(
        "--no_save_grid",
        action="store_true",
        help="Do not write the grid MP4 of generated player views.",
    )
    p.add_argument(
        "--no_save_separate_players",
        action="store_true",
        help="Do not write per-player generated MP4s.",
    )
    p.add_argument("--grid_fps", type=int, default=20, help="FPS for written MP4s.")
    args = resolve_args(p, "inference")
    if args.hier_group_format == "membership":
        infer_kernel = os.environ.get("WAN_HIER_INFER_KERNEL", "sdpa").strip().lower()
        if (
            infer_kernel in ("segmented", "tiled128")
            or args.attn_impl == "hier_tiled_ste"
        ):
            raise ValueError(
                f"hier_group_format=membership is not supported with WAN_HIER_INFER_KERNEL={infer_kernel} or attn_impl={args.attn_impl}; use WAN_HIER_INFER_KERNEL=sdpa or packed_varlen with attn_impl=flex"
            )
    return args
