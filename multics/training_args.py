import argparse
from .config import resolve_args


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multics")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=False,
        choices=["train"],
        help="Train the hierarchical model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./", help="Path to save the model."
    )
    parser.add_argument(
        "--ckpt_local_dir",
        type=str,
        default=None,
        help="If set, torch.save writes checkpoints here instead of the Lightning checkpoint directory.",
    )
    parser.add_argument(
        "--run_version",
        type=str,
        default=None,
        help="Lightning log directory name, for example version_76. If omitted, the next local version_N under the output path is used.",
    )
    parser.add_argument(
        "--text_encoder_path", type=str, default=None, help="Path of text encoder."
    )
    parser.add_argument(
        "--image_encoder_path", type=str, default=None, help="Path of image encoder."
    )
    parser.add_argument("--vae_path", type=str, default=None, help="Path of VAE.")
    parser.add_argument("--dit_path", type=str, default=None, help="Path of DiT.")
    parser.add_argument(
        "--steps_per_epoch", type=int, default=500, help="Number of steps per epoch."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-05, help="Learning rate."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Global gradient-norm clip. 0 disables clipping.",
    )
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps for this run; -1 disables the step limit.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode", default=None, help="SwanLab mode (cloud or local)."
    )
    parser.add_argument(
        "--val_infer",
        action="store_true",
        help="Run causal inference on the first --val_num_samples training rows and log the generated and reference videos.",
    )
    parser.add_argument(
        "--val_num_samples", type=int, default=4, help="Number of leading training samples used for validation."
    )
    parser.add_argument(
        "--val_every_n_steps",
        type=int,
        default=500,
        help="Run validation every this many training batches.",
    )
    parser.add_argument(
        "--val_num_inference_steps",
        type=int,
        default=50,
        help="Denoising steps used during validation inference.",
    )
    parser.add_argument(
        "--val_causal_num_frames",
        type=int,
        default=None,
        help="Frames to generate during validation. Defaults to target_rgb_frames.",
    )
    parser.add_argument(
        "--val_output_dir",
        type=str,
        default="outputs/validation",
        help="Directory for validation MP4 files.",
    )
    parser.add_argument(
        "--val_grid_fps", type=int, default=15, help="Frames per second for saved validation videos."
    )
    parser.add_argument("--metadata_file_name", type=str, default="metadata.csv")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--train_step_offset",
        type=int,
        default=None,
        help="Completed optimizer updates before this run. Default: infer from the resume checkpoint's parent hparams.yaml and stepN filename.",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=4,
        help="Number of player streams in the precomputed dataset.",
    )
    parser.add_argument(
        "--use_simplex_rope",
        action="store_true",
        default=True,
        help="Use simplex 4D RoPE instead of learned player embeddings.",
    )
    parser.add_argument(
        "--simplex_pool_size",
        type=int,
        default=4,
        help="Simplex vertex-pool size. Training draws num_players vertices from this pool.",
    )
    parser.add_argument(
        "--simplex_gamma", type=float, default=1.0, help="Rotation strength of the simplex player axis."
    )
    parser.add_argument(
        "--simplex_train_permute",
        action="store_true",
        help="Randomly permute player slots at each step, together with the sampled vertices.",
    )
    parser.add_argument(
        "--no_simplex_train_permute",
        action="store_false",
        dest="simplex_train_permute",
        help="Do not permute player streams. Vertex sampling stays on its own switch.",
    )
    parser.add_argument(
        "--simplex_train_sample_vertex_ids",
        action="store_true",
        help="Sample P vertex IDs from the simplex pool without replacement, without reordering player streams.",
    )
    parser.add_argument(
        "--training_stage",
        type=str,
        default="stage2",
        choices=["stage2"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="diffusion_forcing",
        choices=["diffusion_forcing"],
        help="Training objective. Fixed to diffusion forcing.",
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        default="hier_hub",
        choices=["hier_hub"],
        help="Attention layout. Fixed to hierarchical hubs.",
    )
    parser.add_argument("--local_attn_size", type=int, default=7)
    parser.add_argument("--local_attn_size_after", type=int, default=21)
    parser.add_argument("--local_attn_switch_step", type=int, default=10000)
    parser.add_argument("--num_frame_per_block", type=int, default=7)
    parser.add_argument(
        "--num_local_hub_tokens",
        type=int,
        default=5,
        help="hier_hub: local hub tokens K_L.",
    )
    parser.add_argument(
        "--num_global_hub_tokens",
        type=int,
        default=6,
        help="hier_hub: global hub tokens K_G.",
    )
    parser.add_argument(
        "--hier_no_local_hub",
        action="store_true",
        help="Ablation: remove local hubs and connect every player directly to the global hub.",
    )
    parser.add_argument(
        "--hier_local_hub_mode",
        type=str,
        default="shared",
        choices=["shared", "per_group"],
        help="hier_hub local seed parameterization: shared uses [K_L,D] for all dynamic group slots; per_group learns independent [N,K_L,D] seeds.",
    )
    parser.add_argument(
        "--hier_direct_schedule",
        type=str,
        default="none",
        choices=["none", "uniform", "explicit"],
        help="hier_hub: selected layers with same-block same-group direct player attention.",
    )
    parser.add_argument(
        "--hier_direct_frac",
        type=float,
        default=0.0,
        help="hier_hub uniform schedule only: fraction in [0,1] of DiT layers that enable direct same-group, same-latent-block player-player attention. Ignored when --hier_direct_schedule is none or explicit; same-player self-history is always controlled by the base mask.",
    )
    parser.add_argument(
        "--hier_direct_layers",
        type=str,
        default=None,
        help="hier_hub explicit zero-based layer ids, e.g. 2,5,8,11.",
    )
    parser.add_argument("--group_path_column", type=str, default="group_path")
    parser.add_argument(
        "--hier_group_format",
        type=str,
        default="labels",
        choices=["labels", "membership"],
        help="hier_hub grouping tensor format. labels: partition ids in [0,N), no player↔global (default, bit-identical to previous runs). membership: cover with membership[T,N,P] + global_only[T,P].",
    )
    parser.add_argument(
        "--hier_group_mode",
        type=str,
        default="csv",
        choices=["csv", "random_static", "random_block"],
        help="hier_hub grouping: CSV labels, one fixed random permutation, or one permutation per latent block.",
    )
    parser.add_argument("--hier_group_seed", type=int, default=0)
    parser.add_argument(
        "--hier_random_preserve_sizes",
        type=int,
        default=1,
        choices=[0, 1],
        help="Random grouping: preserve each source block's group-size histogram when set to 1.",
    )
    parser.add_argument(
        "--group_label_player_ids",
        type=str,
        default=None,
        help="1-based original player ids for group label columns; default 1..P.",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flex", "hier_tiled_ste"],
        help="Self-attention kernel: sdpa, flex, or hier_tiled_ste. hier_tiled_ste uses tiled forward and Flex backward, and requires hier_hub.",
    )
    parser.add_argument(
        "--use_camera_pose",
        action="store_true",
        help="Add relative camera conditioning to self-attention Q and K. The CSV must contain precomputed camera_pose_path* or camera12_path*.",
    )
    parser.add_argument(
        "--camera_dim",
        type=int,
        default=12,
        help="Camera-pose embedding size. The default is the 12-D relative camera.",
    )
    parser.add_argument(
        "--camera_dropout_p",
        type=float,
        default=0.1,
        help="Shared dropout on the camera Q/K adapter. Used only while training. Default 0.1.",
    )
    parser.add_argument(
        "--unfreeze_self_attn",
        action="store_true",
        help="Train the self-attention weights. On by default.",
    )
    parser.add_argument(
        "--causal_forcing_training",
        action="store_true",
        default=False,
        help="Causal-forcing block training. A supervised block is chosen at random; earlier blocks are teacher-forced from clean latents. Inference context noise should stay at 0 when this flag is active.",
    )
    parser.add_argument(
        "--clean_prefix_prob",
        type=float,
        default=0.0,
        help="Probability of training on a clean prefix and supervising only the current block. 0 disables it. A typical range is 0.5 to 0.8.",
    )
    parser.add_argument(
        "--clean_prefix_context_noise",
        type=int,
        default=0,
        help="Context timestep used for clean-prefix blocks. Keep it equal to the inference context noise. 0 means fully clean.",
    )
    parser.add_argument(
        "--generated_history_prob",
        type=float,
        default=0.0,
        help="Fraction of batches that train on generated history: a ground-truth prefix, one to three generated blocks, and loss only on the next ground-truth block. 0 disables it.",
    )
    parser.add_argument(
        "--df_per_frame_noise",
        action="store_true",
        help="Sample an independent noise level for each latent frame instead of sharing one level across a block.",
    )
    parser.add_argument(
        "--target_rgb_frames",
        type=int,
        default=81,
        help="RGB frames in the supervised clip. Must be 4n+1. Default 81. The diffusion loss and action slices use this window.",
    )
    parser.add_argument(
        "--blank_prompt_emb_path",
        type=str,
        default=None,
        help="Cached embedding of the empty text prompt. Defaults to asset/blank_string_prompt_emb.pth. If the file is missing, training encodes the empty string and writes the cache.",
    )
    parser.add_argument(
        "--text_prompt_dropout_p",
        type=float,
        default=0.0,
        help="Probability of replacing a sample's text embedding with the empty-prompt embedding. 0 disables it. The bundled training configs use 0.1.",
    )
    parser.add_argument(
        "--vae_latent_channels",
        type=int,
        default=48,
        help="VAE latent channels. 48 for the Wan 5B VAE.",
    )
    parser.add_argument(
        "--first_frame_dropout_p",
        type=float,
        default=0.0,
        help="Probability of dropping the clean first frame during training and applying loss to every frame. Inference still uses a clean first frame. 0 disables it.",
    )
    parser.add_argument(
        "--first_frame_noise_p",
        type=float,
        default=0.0,
        help="Probability of noising the first latent frame of the anchor. 0 disables it.",
    )
    parser.add_argument(
        "--first_frame_noise_corrupt_ratio",
        type=float,
        default=1 / 3,
        help="Upper bound of the uniform noise scale applied to the anchor's first frame. Default 1/3.",
    )
    parser.add_argument(
        "--first_frame_noise_min_seg",
        type=int,
        default=1,
        help="Apply anchor corruption only at or after this segment index.",
    )
    parser.add_argument(
        "--perf_log_every_n_steps",
        type=int,
        default=100,
        help="Print averaged timing and the bottleneck every N steps.",
    )
    parser.add_argument(
        "--no_perf_timing",
        action="store_true",
        help="Disable step timing. Skips the extra CUDA sync and the perf logs.",
    )
    args = resolve_args(parser, "train")
    if args.hier_group_format == "membership" and args.attn_impl == "hier_tiled_ste":
        raise ValueError(
            "hier_group_format=membership is not supported with --attn_impl hier_tiled_ste; use flex or sdpa"
        )
    return args
