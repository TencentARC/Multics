"""Construct the complete Multics DiT without loading Wan backbone weights."""

import copy

import torch
from torch import nn

from .models.utils import init_weights_on_device
from .models.wan_video_dit import WanModel


# Architecture of the Wan2.2 TI2V 5B backbone used to train Multics.
DIT_CONFIG = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


def stage2_options(args):
    return {
        name: getattr(args, name)
        for name in (
            "num_hub_tokens", "local_attn_size", "num_frame_per_block",
            "attention_mode", "num_groups", "num_local_hub_tokens",
            "num_global_hub_tokens", "hier_direct_schedule", "hier_direct_frac",
            "hier_direct_layers", "attn_impl", "hier_no_local_hub",
            "hier_local_hub_mode",
        )
    }


def build_dit(args):
    """Allocate parameters on meta; keep computed RoPE buffers on CPU."""
    with init_weights_on_device():
        model = WanModel(**DIT_CONFIG)
        dim = model.dim
        model.keyboard_embedding = nn.Linear(23, dim)
        for block in model.blocks:
            block.action_mlp = nn.Sequential(
                nn.LayerNorm(dim + 24, eps=1e-5, elementwise_affine=True),
                nn.Linear(dim + 24, dim + 24),
                nn.GELU(),
                nn.Linear(dim + 24, dim),
                nn.LayerNorm(dim, eps=1e-5, elementwise_affine=True),
            )
            block.action_temporal_attn = copy.deepcopy(block.self_attn)
            block.action_cross_attn = copy.deepcopy(block.cross_attn)
        model.enable_simplex_rope(
            pool_size=args.simplex_pool_size, gamma=args.simplex_gamma
        )
        model.enable_stage2(**stage2_options(args))
        if args.use_camera_pose:
            model.enable_camera_pose_alignment(
                cam_dim=args.camera_dim, dropout_p=args.camera_dropout_p
            )
    return model


def load_complete_state_dict(model, state_dict, dtype=torch.bfloat16):
    """Reject incomplete/incompatible checkpoints before materializing parameters."""
    expected = model.state_dict()
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    mismatched = [
        key for key in expected.keys() & state_dict.keys()
        if not isinstance(state_dict[key], torch.Tensor)
        or expected[key].shape != state_dict[key].shape
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Inference requires a complete Multics DiT checkpoint. "
            f"Missing keys ({len(missing)}): {missing[:8]}; "
            f"unexpected keys ({len(unexpected)}): {unexpected[:8]}; "
            f"incompatible tensors ({len(mismatched)}): {sorted(mismatched)[:8]}"
        )
    model.load_state_dict(state_dict, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Checkpoint left uninitialized DiT parameters")
    return model.to(dtype=dtype).eval()
