from .data.dataset import (
    PrecomputedLatentDataset,
    _default_blank_prompt_emb_path,
    _prompt_emb_to_cpu,
    _save_prompt_emb_atomic,
)
import copy
import glob
import os
import re
import time
import torch, os
import lightning as pl
from .pipelines.wan_video_Nplayers import WanVideoNPlayersPipeline
from .models.model_manager import ModelManager
from multics.pipelines.wan_video_Nplayers import model_fn_wan_video_causal
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import glob
from lightning.pytorch.loggers import TensorBoardLogger
from multics.models.wan_video_dit import (
    prepare_hier_local_hub_state_dict_for_load,
    migrate_hier_no_local_hub_state_dict,
)

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_BLANK_PROMPT_EMB_FILENAME = "blank_string_prompt_emb.pth"
_CAMERA_PLAYER_PART = re.compile("(?:^|/)player=(\\d+)(?:/|$)")
_GENERATED_HISTORY_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def get_corrupt_noise_sigma(
    model_input,
    batch_size,
    corrupt_ratio=1 / 3,
    num_frames=None,
    is_frame_independent=False,
):
    """DF-style σ ~ U(0, corrupt_ratio) for (1-σ)*x + σ*noise. Broadcast to model_input.ndim."""
    if is_frame_independent:
        noise_sigma_shape = (batch_size, 1, num_frames)
    else:
        noise_sigma_shape = (batch_size,)
    noise_sigma = (
        torch.rand(
            size=noise_sigma_shape, device=model_input.device, dtype=model_input.dtype
        )
        * corrupt_ratio
    )
    while len(noise_sigma.shape) < model_input.ndim:
        noise_sigma = noise_sigma.unsqueeze(-1)
    return noise_sigma


def _attach_hier_group_dit_kwargs(dit_kwargs: dict, batch: dict, device) -> None:
    """Copy grouping tensors from the dataloader batch into DiT kwargs.

    labels mode: group_labels [1,T,P], active_group [1,T,N]
    membership mode also: membership [1,T,N,P], global_only [1,T,P]
    """
    if "group_labels" not in batch or "active_group" not in batch:
        raise KeyError(
            "hier_hub requires dataset batch keys group_labels and active_group"
        )
    dit_kwargs["group_labels"] = batch["group_labels"].to(
        device=device, dtype=torch.long
    )
    dit_kwargs["active_group"] = batch["active_group"].to(
        device=device, dtype=torch.bool
    )
    if "membership" in batch:
        if "global_only" not in batch:
            raise KeyError("hier_group_format=membership batch requires global_only")
        dit_kwargs["membership"] = batch["membership"].to(
            device=device, dtype=torch.bool
        )
        dit_kwargs["global_only"] = batch["global_only"].to(
            device=device, dtype=torch.bool
        )


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path,
        dit_path,
        learning_rate=1e-05,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        output_path="./",
        num_players=2,
        vae_latent_channels=48,
        perf_log_every_n_steps=100,
        perf_timing=True,
        target_rgb_frames=81,
        first_frame_dropout_p=0.0,
        first_frame_noise_p=0.0,
        first_frame_noise_corrupt_ratio=1 / 3,
        first_frame_noise_min_seg=1,
        ckpt_local_dir=None,
        run_version=None,
        blank_prompt_emb_path: str = None,
        text_prompt_dropout_p: float = 0.0,
        use_simplex_rope: bool = True,
        simplex_pool_size: int = 4,
        simplex_gamma: float = 1.0,
        simplex_train_permute: bool = True,
        simplex_train_sample_vertex_ids: bool = False,
        training_stage: str = "stage2",
        loss_type: str = "diffusion_forcing",
        attention_mode: str = "hier_hub",
        attn_impl: str = "sdpa",
        num_hub_tokens: int = 8,
        local_attn_size: int = 24,
        local_attn_size_after: int = None,
        local_attn_switch_step: int = None,
        train_step_offset: int = 0,
        num_frame_per_block: int = 7,
        num_groups: int = 1,
        num_local_hub_tokens: int = 0,
        num_global_hub_tokens: int = None,
        hier_no_local_hub: bool = False,
        hier_local_hub_mode: str = "shared",
        hier_direct_schedule: str = "none",
        hier_direct_frac: float = 0.0,
        hier_direct_layers=None,
        unfreeze_self_attn: bool = False,
        clean_prefix_prob: float = 0.0,
        clean_prefix_context_noise: int = 0,
        generated_history_prob: float = 0.0,
        df_per_frame_noise: bool = False,
        causal_forcing_training: bool = False,
        use_camera_pose: bool = False,
        camera_dim: int = 12,
        camera_dropout_p: float = 0.1,
        val_infer: bool = False,
        val_every_n_steps: int = 500,
        val_num_samples: int = 4,
        val_num_inference_steps: int = 50,
        val_causal_num_frames: int = None,
        val_output_dir: str = "outputs/validation",
        val_grid_fps: int = 15,
    ):
        super().__init__()
        if (
            training_stage != "stage2"
            or loss_type != "diffusion_forcing"
            or attention_mode != "hier_hub"
            or (not use_simplex_rope)
        ):
            raise ValueError(
                "Multics supports hierarchical attention with diffusion forcing and simplex RoPE"
            )
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if "*" in dit_path:
            dit_files = sorted(glob.glob(dit_path))
            model_manager.load_model(dit_files)
        elif os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager.load_models(model_path)
        self.pipe = WanVideoNPlayersPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.num_players = num_players
        self.use_simplex_rope = bool(True)
        self.simplex_pool_size = int(simplex_pool_size)
        self.simplex_gamma = float(simplex_gamma)
        self.simplex_train_permute = bool(simplex_train_permute)
        self.simplex_train_sample_vertex_ids = bool(simplex_train_sample_vertex_ids)
        self.training_stage = str("stage2")
        self.loss_type = str("diffusion_forcing")
        self.attention_mode = str("hier_hub")
        self.attn_impl = str(attn_impl)
        self.num_hub_tokens = int(num_hub_tokens)
        self.local_attn_size_initial = int(local_attn_size)
        self.local_attn_size_after = (
            None if local_attn_size_after is None else int(local_attn_size_after)
        )
        self.local_attn_switch_step = (
            None if local_attn_switch_step is None else int(local_attn_switch_step)
        )
        self.train_step_offset = int(train_step_offset)
        if (self.local_attn_size_after is None) != (
            self.local_attn_switch_step is None
        ):
            raise ValueError(
                "local_attn_size_after and local_attn_switch_step must be set together"
            )
        if self.local_attn_switch_step is not None and self.local_attn_switch_step < 0:
            raise ValueError("local_attn_switch_step must be >= 0")
        if self.train_step_offset < 0:
            raise ValueError("train_step_offset must be >= 0")
        self._last_logged_local_attn_size = None
        self.num_frame_per_block = int(num_frame_per_block)
        for name, value in (
            ("local_attn_size", self.local_attn_size_initial),
            ("local_attn_size_after", self.local_attn_size_after),
        ):
            if value is not None and value != -1 and (value < self.num_frame_per_block):
                raise ValueError(
                    f"{name} must be -1 or >= num_frame_per_block; got {value} and {self.num_frame_per_block}"
                )
        self.local_attn_size = self._local_attn_size_for_effective_step(
            self.train_step_offset
        )
        self.num_groups = int(num_groups)
        self.num_local_hub_tokens = int(num_local_hub_tokens)
        self.num_global_hub_tokens = (
            None if num_global_hub_tokens is None else int(num_global_hub_tokens)
        )
        self.hier_no_local_hub = bool(hier_no_local_hub)
        self.hier_local_hub_mode = str(hier_local_hub_mode)
        self.hier_direct_schedule = str(hier_direct_schedule)
        self.hier_direct_frac = float(hier_direct_frac)
        self.hier_direct_layers = hier_direct_layers
        self.unfreeze_self_attn = bool(unfreeze_self_attn)
        self.target_rgb_frames = int(target_rgb_frames)
        self.val_infer = bool(val_infer)
        self.val_every_n_steps = int(val_every_n_steps)
        self.val_num_samples = int(val_num_samples)
        self.val_num_inference_steps = int(val_num_inference_steps)
        self.val_causal_num_frames = int(val_causal_num_frames or target_rgb_frames)
        self.val_output_dir = val_output_dir
        self.val_grid_fps = int(val_grid_fps)
        self.val_agent_vertex_ids = list(range(int(num_players)))
        self._manual_val_loader = None
        self._last_val_step = -1
        self.clean_prefix_prob = float(clean_prefix_prob)
        self.clean_prefix_context_noise = int(clean_prefix_context_noise)
        self.generated_history_prob = float(generated_history_prob)
        if not 0.0 <= self.generated_history_prob <= 1.0:
            raise ValueError(
                f"generated_history_prob must be in [0,1], got {self.generated_history_prob}"
            )
        if self.generated_history_prob > 0.0:
            if causal_forcing_training:
                raise ValueError(
                    "Generated history uses the parallel target path and is incompatible with --causal_forcing_training"
                )
            print(
                f"[generated-history] enabled prob={self.generated_history_prob} blocks=1..3 steps={self.val_num_inference_steps} cfg=5",
                flush=True,
            )
        self._generated_history_negative_context_cpu = None
        self.df_per_frame_noise = bool(df_per_frame_noise)
        self.causal_forcing_training = bool(causal_forcing_training)
        self.use_camera_pose = bool(use_camera_pose)
        self.camera_dim = int(camera_dim)
        self.camera_dropout_p = float(camera_dropout_p)
        self.pipe.dit.keyboard_embedding = nn.Linear(23, 3072)
        nn.init.normal_(self.pipe.dit.keyboard_embedding.weight, mean=0, std=0.05)
        nn.init.zeros_(self.pipe.dit.keyboard_embedding.bias)
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.action_mlp = nn.Sequential(
                nn.LayerNorm(dim + 24, eps=1e-05, elementwise_affine=True),
                nn.Linear(dim + 24, dim + 24),
                nn.GELU(),
                nn.Linear(dim + 24, dim),
                nn.LayerNorm(dim, eps=1e-05, elementwise_affine=True),
            )
            block.action_temporal_attn = copy.deepcopy(block.self_attn)
            block.action_cross_attn = copy.deepcopy(block.cross_attn)
            nn.init.zeros_(block.action_temporal_attn.o.weight)
            nn.init.zeros_(block.action_temporal_attn.o.bias)
            nn.init.zeros_(block.action_cross_attn.o.weight)
            nn.init.zeros_(block.action_cross_attn.o.bias)
        self.pipe.dit.enable_simplex_rope(
            pool_size=self.simplex_pool_size, gamma=self.simplex_gamma
        )
        latent_t = (self.target_rgb_frames - 1) // 4 + 1
        if latent_t % self.num_frame_per_block != 0:
            raise ValueError(
                f"Requires latent_T % num_frame_per_block == 0; got latent_T={latent_t}, n={self.num_frame_per_block}"
            )
        stage2_attention_mode = "hier_hub"
        if self.simplex_train_permute:
            raise ValueError(
                "hier_hub requires --no_simplex_train_permute so group_labels stay aligned"
            )
        self.pipe.enable_stage2(
            num_hub_tokens=self.num_hub_tokens,
            local_attn_size=self.local_attn_size,
            num_frame_per_block=self.num_frame_per_block,
            attention_mode="hier_hub",
            num_groups=self.num_groups,
            num_local_hub_tokens=self.num_local_hub_tokens,
            num_global_hub_tokens=self.num_global_hub_tokens,
            hier_direct_schedule=self.hier_direct_schedule,
            hier_direct_frac=self.hier_direct_frac,
            hier_direct_layers=self.hier_direct_layers,
            attn_impl=self.attn_impl,
            hier_no_local_hub=self.hier_no_local_hub,
            hier_local_hub_mode=self.hier_local_hub_mode,
        )
        self.loss_type = "diffusion_forcing"
        self.attention_mode = "hier_hub"
        self.unfreeze_self_attn = True
        if self.use_camera_pose:
            self.pipe.dit.enable_camera_pose_alignment(
                cam_dim=self.camera_dim, dropout_p=self.camera_dropout_p
            )
            print(
                f"[camera-pose] enabled bias-free Q/K-only camera conditioning (dim={self.camera_dim}, dropout_p={self.camera_dropout_p})"
            )
        assert 0.0 <= float(first_frame_dropout_p) <= 1.0, first_frame_dropout_p
        self.first_frame_dropout_p = float(first_frame_dropout_p)
        assert 0.0 <= float(first_frame_noise_p) <= 1.0, first_frame_noise_p
        assert 0.0 <= float(first_frame_noise_corrupt_ratio) <= 1.0, (
            first_frame_noise_corrupt_ratio
        )
        self.first_frame_noise_p = float(first_frame_noise_p)
        self.first_frame_noise_corrupt_ratio = float(first_frame_noise_corrupt_ratio)
        self.first_frame_noise_min_seg = int(first_frame_noise_min_seg)
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            if not self.hier_no_local_hub:
                state_dict, migration = prepare_hier_local_hub_state_dict_for_load(
                    state_dict,
                    target_mode=self.hier_local_hub_mode,
                    target_num_groups=self.pipe.dit.num_groups,
                    target_num_local_hub_tokens=self.pipe.dit.num_local_hub_tokens,
                    model_dim=self.pipe.dit.dim,
                )
                if migration is not None:
                    print(
                        f"[hier-local-hub] training warm-start migration: {migration['source_mode']} {migration['source_shape']} -> {migration['target_mode']} {migration['target_shape']}; optimizer state is not migrated",
                        flush=True,
                    )
            old_hub = state_dict.get("hub_embed")
            new_hub = self.pipe.dit.hub_embed
            if (
                torch.is_tensor(old_hub)
                and torch.is_tensor(new_hub)
                and (old_hub.ndim == 2)
                and (tuple(old_hub.shape) != tuple(new_hub.shape))
            ):
                resized = new_hub.detach().cpu().clone()
                n = min(old_hub.shape[0], resized.shape[0])
                resized[:n].copy_(old_hub[:n].to(dtype=resized.dtype))
                state_dict["hub_embed"] = resized
                print(
                    f"[checkpoint] resized hub_embed {tuple(old_hub.shape)} -> {tuple(resized.shape)}; copied {n} slots",
                    flush=True,
                )
            if self.hier_no_local_hub:
                state_dict, migration = migrate_hier_no_local_hub_state_dict(
                    state_dict,
                    target_num_global_hub_tokens=self.pipe.dit.num_global_hub_tokens,
                    model_dim=self.pipe.dit.dim,
                )
                if migration is not None:
                    print(
                        f"[hier-no-local] training warm-start migration: {migration['source']} {migration['source_shape']} -> global_hub_embed {migration['target_shape']} (replaced old global {migration['replaced_global_shape']}); optimizer state is not migrated",
                        flush=True,
                    )
            missing, unexpected = self.pipe.dit.load_state_dict(
                state_dict, strict=False
            )
            if missing:
                _hub_related = [k for k in missing if k.endswith("hub_embed")]
                _cam_keys = (".cam_q.", ".cam_k.")
                _cam_related = [k for k in missing if any((s in k for s in _cam_keys))]
                _other_missing = [
                    k
                    for k in missing
                    if k not in _hub_related and k not in _cam_related
                ]
                if _other_missing:
                    raise RuntimeError(
                        f"Unexpected missing keys when loading DiT: {_other_missing[:10]} ..."
                    )
                if _hub_related:
                    print(
                        f"[checkpoint] {len(_hub_related)} hub_embed param(s) absent in ckpt (zeros init, will train from seed)."
                    )
                if _cam_related:
                    print(
                        f"[camera-pose] {len(_cam_related)} camera adapter params absent in ckpt (zeros init)."
                    )
            if unexpected:
                _ff_keys = (".k_ff", ".v_ff", ".norm_k_ff", ".ff_gate")
                _ff_related = [k for k in unexpected if any((s in k for s in _ff_keys))]
                _handled_unexpected = list(_ff_related)
                _old_hub = [
                    k
                    for k in unexpected
                    if k == "hub_embed" or k.endswith(".hub_embed")
                ]
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
                    if not self.use_camera_pose
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
                _other_unexpected = [
                    k for k in unexpected if k not in _handled_unexpected
                ]
                if _other_unexpected:
                    print(
                        f"[WARN] unexpected keys in DiT ckpt: {_other_unexpected[:10]} ..."
                    )
            if "hub_embed" in state_dict and (
                "local_hub_embed" not in state_dict
                or "global_hub_embed" not in state_dict
            ):
                old_hub = state_dict["hub_embed"]
                if (
                    torch.is_tensor(old_hub)
                    and old_hub.dim() == 2
                    and (old_hub.shape[1] == self.pipe.dit.dim)
                ):
                    with torch.no_grad():
                        kl = int(self.pipe.dit.num_local_hub_tokens)
                        kg = int(self.pipe.dit.num_global_hub_tokens)
                        if kl > 0:
                            n = min(kl, old_hub.shape[0])
                            local_target = self.pipe.dit.local_hub_embed
                            local_values = old_hub[:n].to(local_target)
                            if local_target.dim() == 2:
                                local_target[:n].copy_(local_values)
                            elif local_target.dim() == 3:
                                local_target[:, :n].copy_(
                                    local_values.unsqueeze(0).expand(
                                        local_target.shape[0], -1, -1
                                    )
                                )
                            else:
                                raise RuntimeError(
                                    f"local_hub_embed must be rank 2 or 3, got {tuple(local_target.shape)}"
                                )
                        if kg > 0:
                            n = min(kg, old_hub.shape[0])
                            self.pipe.dit.global_hub_embed[-n:].copy_(
                                old_hub[-n:].to(self.pipe.dit.global_hub_embed)
                            )
                    print(
                        f"[hier_hub] initialized local/global hub seeds from central hub_embed {tuple(old_hub.shape)} -> local={(None if self.pipe.dit.local_hub_embed is None else tuple(self.pipe.dit.local_hub_embed.shape))} global={tuple(self.pipe.dit.global_hub_embed.shape)}"
                    )
        self.freeze_parameters()
        trainable_keywords = [
            "keyboard_embedding",
            "action_mlp",
            "action_temporal_attn",
            "action_cross_attn",
        ]
        if self.unfreeze_self_attn:
            trainable_keywords.extend(["self_attn", "hub_embed"])
        if self.use_camera_pose:
            trainable_keywords.extend(["cam_q", "cam_k"])
        for name, module in self.pipe.denoising_model().named_modules():
            if any((keyword in name for keyword in trainable_keywords)):
                print(f"Trainable: {name}")
                for param in module.parameters():
                    param.requires_grad = True
            if name.endswith(".cross_attn"):
                print(f"Trainable (text cross-attention): {name}")
                for param in module.parameters():
                    param.requires_grad = True
        if self.unfreeze_self_attn:
            _dit = self.pipe.denoising_model()
            if hasattr(_dit, "hub_embed") and isinstance(_dit.hub_embed, nn.Parameter):
                _dit.hub_embed.requires_grad_(True)
                print(
                    f"Trainable (parameter, explicit): model-level hub_embed {tuple(_dit.hub_embed.shape)}"
                )
            for _name in ("local_hub_embed", "global_hub_embed"):
                _p = getattr(_dit, _name, None)
                if isinstance(_p, nn.Parameter):
                    _p.requires_grad_(True)
                    print(f"Trainable (parameter, explicit): {_name} {tuple(_p.shape)}")
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.ckpt_local_dir = ckpt_local_dir
        self.run_version = run_version
        self._prompt_encode_cache: dict = {}
        self._blank_prompt_ctx: torch.Tensor | None = None
        self._blank_prompt_emb_path = (
            blank_prompt_emb_path or _default_blank_prompt_emb_path()
        )
        self.text_prompt_dropout_p = float(text_prompt_dropout_p)
        if not 0.0 <= self.text_prompt_dropout_p <= 1.0:
            raise ValueError(
                f"--text_prompt_dropout_p must be in [0,1], got {self.text_prompt_dropout_p}"
            )
        if self.text_prompt_dropout_p > 0:
            print(
                f"[text-cfg] text_prompt_dropout_p={self.text_prompt_dropout_p}",
                flush=True,
            )
        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        tiled = True
        tile_size = (34, 34)
        tile_stride = (18, 16)
        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        self._perf_timing_enabled = bool(perf_timing)
        self.perf_log_every_n_steps = max(1, int(perf_log_every_n_steps))
        self._iter_start_time = None
        self._prev_batch_end_time = None
        self._current_data_wait_time = 0.0
        self._backward_start_time = None
        self._last_backward_time = 0.0
        self._last_optimizer_time = 0.0
        self._latest_step_part_times = {"encode_target": 0.0, "dit_forward_loss": 0.0}
        self._perf_window = {
            "count": 0,
            "iter_wall": 0.0,
            "data_wait": 0.0,
            "compute": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
            "encode_target": 0.0,
            "dit_forward_loss": 0.0,
        }

    def _synced_time(self):
        if self._perf_timing_enabled and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _classify_bottleneck(avg_data_wait, avg_compute, avg_backward):
        total_iter = avg_data_wait + avg_compute
        if total_iter <= 0:
            return "unknown"
        data_wait_ratio = avg_data_wait / total_iter
        compute_ratio = avg_compute / total_iter
        backward_ratio = avg_backward / max(avg_compute, 1e-08)
        if data_wait_ratio >= 0.4 and data_wait_ratio >= compute_ratio:
            return "data"
        if compute_ratio >= 0.6 and backward_ratio >= 0.55:
            return "communication"
        if compute_ratio >= 0.6:
            return "compute"
        if backward_ratio >= 0.65:
            return "communication"
        if data_wait_ratio >= 0.35:
            return "data"
        return "compute"

    def on_train_batch_start(self, batch, batch_idx):
        self._maybe_update_local_attn_size()
        if not self._perf_timing_enabled:
            return
        now = self._synced_time()
        if self._prev_batch_end_time is None:
            self._current_data_wait_time = 0.0
        else:
            self._current_data_wait_time = max(0.0, now - self._prev_batch_end_time)
        self._iter_start_time = now
        self._backward_start_time = None
        self._last_backward_time = 0.0
        self._last_optimizer_time = 0.0

    def _local_attn_size_for_effective_step(self, effective_step: int) -> int:
        if (
            self.local_attn_switch_step is not None
            and int(effective_step) >= self.local_attn_switch_step
        ):
            return int(self.local_attn_size_after)
        return int(self.local_attn_size_initial)

    def _maybe_update_local_attn_size(self):
        """Apply the attention window selected for the next optimizer update."""
        if self.local_attn_switch_step is None:
            return
        local_step = int(self.global_step)
        effective_step = self.train_step_offset + local_step
        target = self._local_attn_size_for_effective_step(effective_step)
        if int(self.local_attn_size) != target:
            self.pipe.set_stage2_local_attn_size(target, reset_training_cache=True)
            self.local_attn_size = target
        values = {
            int(self.local_attn_size),
            int(self.pipe.local_attn_size),
            int(self.pipe.dit.local_attn_size),
            *(int(block.local_attn_size) for block in self.pipe.dit.blocks),
            *(int(block.self_attn.local_attn_size) for block in self.pipe.dit.blocks),
        }
        if values != {target}:
            raise RuntimeError(
                f"local-attn schedule propagation failed: expected {target}, observed {sorted(values)}"
            )
        if self._last_logged_local_attn_size != target:
            if int(getattr(self, "global_rank", 0)) == 0:
                print(
                    f"[local-attn-schedule] local_step={local_step} effective_step={effective_step} local_attn_size={target}",
                    flush=True,
                )
            self._last_logged_local_attn_size = target

    def on_before_backward(self, loss):
        if not self._perf_timing_enabled:
            return
        self._backward_start_time = self._synced_time()

    def on_after_backward(self):
        if not self._perf_timing_enabled:
            return
        if self._backward_start_time is None:
            self._last_backward_time = 0.0
            return
        self._last_backward_time = max(
            0.0, self._synced_time() - self._backward_start_time
        )

    def optimizer_step(self, *args, **kwargs):
        if not self._perf_timing_enabled:
            return super().optimizer_step(*args, **kwargs)
        start_time = self._synced_time()
        out = super().optimizer_step(*args, **kwargs)
        self._last_optimizer_time = max(0.0, self._synced_time() - start_time)
        return out

    def on_train_batch_end(self, outputs, batch, batch_idx):
        end_time = self._synced_time() if self._perf_timing_enabled else None
        self._maybe_run_manual_validation()
        if not self._perf_timing_enabled:
            return
        compute_time = (
            0.0
            if self._iter_start_time is None
            else max(0.0, end_time - self._iter_start_time)
        )
        data_wait_time = max(0.0, self._current_data_wait_time)
        iter_wall_time = data_wait_time + compute_time
        self._prev_batch_end_time = end_time
        self._perf_window["count"] += 1
        self._perf_window["iter_wall"] += iter_wall_time
        self._perf_window["data_wait"] += data_wait_time
        self._perf_window["compute"] += compute_time
        self._perf_window["backward"] += self._last_backward_time
        self._perf_window["optimizer"] += self._last_optimizer_time
        for key in ("encode_target", "dit_forward_loss"):
            self._perf_window[key] += self._latest_step_part_times.get(key, 0.0)
        if self._perf_window["count"] < self.perf_log_every_n_steps:
            return
        c = float(self._perf_window["count"])
        avg_iter_wall = self._perf_window["iter_wall"] / c
        avg_data_wait = self._perf_window["data_wait"] / c
        avg_compute = self._perf_window["compute"] / c
        avg_backward = self._perf_window["backward"] / c
        avg_optimizer = self._perf_window["optimizer"] / c
        avg_encode_target = self._perf_window["encode_target"] / c
        avg_dit_forward_loss = self._perf_window["dit_forward_loss"] / c
        total_iter = max(avg_iter_wall, 1e-08)
        data_wait_ratio = avg_data_wait / total_iter
        compute_ratio = avg_compute / total_iter
        backward_ratio = avg_backward / max(avg_compute, 1e-08)
        bottleneck = self._classify_bottleneck(avg_data_wait, avg_compute, avg_backward)
        if self.trainer is not None and self.trainer.is_global_zero:
            print(
                f"[Perf][step={int(self.global_step)}][N={int(c)}] iter={avg_iter_wall:.4f}s data_wait={avg_data_wait:.4f}s({data_wait_ratio:.1%}) compute={avg_compute:.4f}s({compute_ratio:.1%}) backward={avg_backward:.4f}s(backward/compute={backward_ratio:.1%}) optimizer={avg_optimizer:.4f}s encode_target={avg_encode_target:.4f}s dit_forward_loss={avg_dit_forward_loss:.4f}s bottleneck={bottleneck}"
            )
        self.log(
            "perf/iter_wall_s",
            avg_iter_wall,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/data_wait_s",
            avg_data_wait,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/compute_s",
            avg_compute,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/backward_s",
            avg_backward,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/optimizer_s",
            avg_optimizer,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/encode_target_s",
            avg_encode_target,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/dit_forward_loss_s",
            avg_dit_forward_loss,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/data_wait_ratio",
            data_wait_ratio,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/compute_ratio",
            compute_ratio,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/backward_in_compute_ratio",
            backward_ratio,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/bottleneck_data",
            1.0 if bottleneck == "data" else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/bottleneck_compute",
            1.0 if bottleneck == "compute" else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "perf/bottleneck_communication",
            1.0 if bottleneck == "communication" else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        for key in self._perf_window.keys():
            self._perf_window[key] = 0.0 if key != "count" else 0

    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def _encode_prompt_with_cache(self, text: str) -> torch.Tensor:
        """将文本编码为 T5 context tensor (seq_len, 4096)，结果缓存在 CPU。
        text 为空字符串时优先读取 repo-local blank cache；缺失则 encode 后写回。
        返回值为 CPU tensor；调用侧负责 .to(device)。
        """
        text = text.strip() if text else ""
        if not text:
            if self._blank_prompt_ctx is None:
                if os.path.isfile(self._blank_prompt_emb_path):
                    blank = torch.load(
                        self._blank_prompt_emb_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                else:
                    with torch.no_grad():
                        blank = self.pipe.encode_prompt("", positive=True)
                    blank = _prompt_emb_to_cpu(blank)
                    _save_prompt_emb_atomic(blank, self._blank_prompt_emb_path)
                    print(
                        f"[text-cfg] encoded blank prompt and cached to {self._blank_prompt_emb_path}",
                        flush=True,
                    )
                ctx = blank["context"]
                if ctx.ndim == 3:
                    ctx = ctx[0]
                self._blank_prompt_ctx = ctx.cpu()
            return self._blank_prompt_ctx
        if text not in self._prompt_encode_cache:
            with torch.no_grad():
                emb = self.pipe.encode_prompt(text)
            ctx = emb["context"]
            if ctx.ndim == 3:
                ctx = ctx[0]
            self._prompt_encode_cache[text] = ctx.cpu()
        return self._prompt_encode_cache[text]

    def _get_context_for_seg(self, batch: dict, seg_idx: int) -> torch.Tensor:
        """返回 segment seg_idx 的 DiT context tensor。
        - use_text_prompt=True：返回 (2, seq_len, 4096)，legacy 2-player 文本路径；空字符串 fallback 到 blank。
        - use_text_prompt=False：返回 dataset 提供的 context，去掉 DataLoader batch 维；
          通常为 per-player (P, seq_len, 4096)，没有 CSV prompt_emb_path 时 P 个 player 共用 blank。
        注意：返回的是 T5 原始 embedding（4096 dim），DiT 内部 text_embedding MLP 负责投影到 3072。
        """
        _dtype = self.pipe.torch_dtype
        _dev = self.device
        prompt_emb = batch.get("prompt_emb") or {}
        context = prompt_emb.get("context") if isinstance(prompt_emb, dict) else None
        if context is None:
            return self._blank_context_for_players(
                self.pipe.num_players, dtype=_dtype, device=_dev
            )
        return context[0].to(dtype=_dtype, device=_dev)

    def _blank_context_for_players(
        self, num_players: int, *, dtype, device
    ) -> torch.Tensor:
        blank = self._encode_prompt_with_cache("").to(dtype=dtype, device=device)
        if blank.ndim != 2:
            raise ValueError(
                f"blank prompt context must be [S,D], got {tuple(blank.shape)}"
            )
        return blank.unsqueeze(0).expand(int(num_players), -1, -1).contiguous()

    def _blank_context_like(self, context: torch.Tensor) -> torch.Tensor:
        blank = self._blank_context_for_players(
            int(context.shape[0]), dtype=context.dtype, device=context.device
        )
        if tuple(blank.shape[1:]) != tuple(context.shape[1:]):
            raise ValueError(
                f"blank prompt context shape {tuple(blank.shape)} does not match training context {tuple(context.shape)}"
            )
        return blank

    def _maybe_drop_text_prompt_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        p = self.text_prompt_dropout_p
        if p <= 0.0 or not self.training:
            return (context, False)
        if bool(torch.rand((), device=context.device) < p):
            return (self._blank_context_like(context), True)
        return (context, False)

    @staticmethod
    def _encode_latents(pipe, video, tiler_kwargs):
        """VAE encode 可能返回 tensor 或 list/tuple，统一为 tensor。"""
        z = pipe.encode_video(video, **tiler_kwargs)
        if isinstance(z, (list, tuple)):
            z = z[0]
        return z

    def _should_corrupt_first_frame_anchor(self, seg_idx: int) -> bool:
        if self.first_frame_noise_p <= 0.0:
            return False
        if seg_idx < self.first_frame_noise_min_seg:
            return False
        return bool(torch.rand(1, device=self.device).item() < self.first_frame_noise_p)

    def _corrupt_origin_frame0(self, origin_latents: torch.Tensor) -> torch.Tensor:
        """(1-σ)*z + σ*ε on latent frame 0; σ ~ U(0, corrupt_ratio). Aligns with flow_match.add_noise."""
        out = origin_latents.clone()
        z0 = out[:, :, 0:1]
        eps = torch.randn_like(z0)
        sigma = get_corrupt_noise_sigma(
            z0,
            batch_size=z0.shape[0],
            corrupt_ratio=self.first_frame_noise_corrupt_ratio,
            num_frames=1,
            is_frame_independent=False,
        )
        out[:, :, 0:1] = (1.0 - sigma) * z0 + sigma * eps
        return out

    def _prepare_simplex_agents(self, latents, keyboard, mouse):
        """Select simplex vertices, then optionally permute aligned player streams."""
        num_players = latents.shape[0]
        pool = self.simplex_pool_size
        if pool < num_players:
            raise ValueError(f"simplex_pool_size={pool} < num_players={num_players}")
        if self.simplex_train_sample_vertex_ids:
            vertex_ids = random.sample(range(pool), num_players)
        else:
            vertex_ids = list(range(num_players))
        if self.simplex_train_permute:
            perm = torch.randperm(num_players).tolist()
            vertex_ids = [vertex_ids[i] for i in perm]
            latents = latents[perm]
            keyboard = keyboard[perm]
            mouse = mouse[perm]
        return (latents, keyboard, mouse, vertex_ids)

    def _apply_first_frame_conditioning(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        seg_idx: int = 0,
        allow_drop: bool = True,
    ):
        """Build noisy_latents with optional DF-style corrupt on anchor frame 0."""
        origin_latents = latents.clone().detach()
        if timestep.ndim >= 1 and timestep.numel() > 1:
            t_bf = timestep.reshape(latents.shape[0], latents.shape[2])
            sigma, _ = self.pipe.scheduler._sigma_from_timestep(
                t_bf, device=latents.device, dtype=latents.dtype
            )
            sigma = sigma.view(latents.shape[0], 1, latents.shape[2], 1, 1)
            noisy_latents = (1.0 - sigma) * latents + sigma * noise
        else:
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        drop_first_frame = (
            allow_drop
            and self.first_frame_dropout_p > 0.0
            and bool(torch.rand(1).item() < self.first_frame_dropout_p)
        )
        corrupt_anchor = False
        if not drop_first_frame:
            corrupt_anchor = self._should_corrupt_first_frame_anchor(seg_idx)
            if corrupt_anchor:
                origin_latents = self._corrupt_origin_frame0(origin_latents)
            noisy_latents[:, :, 0:1, ...] = origin_latents[:, :, 0:1, ...]
        return (noisy_latents, drop_first_frame, corrupt_anchor)

    def _stage2_causal_training_step(
        self,
        latents: torch.Tensor,
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
        prompt_emb: dict,
        agent_vertex_ids,
        image_emb: dict,
        batch: dict,
    ) -> torch.Tensor:
        """Causal-Forcing teacher-forcing training for Stage 2.

        Uses a SINGLE parallel forward (WanModel.forward, same as standard training),
        NOT the KV-cache causal path.  Using KV cache during the supervised block's
        forward breaks gradient flow to Wk, Wv, and hub_embed via in-place cache
        writes (cache tensors are requires_grad=False, so the autograd graph is severed
        at the slice-assignment boundary).

        Algorithm:
          - Sample supervised block k ∈ [0, num_blocks-1].
          - Context blocks 0..k-1: timestep = 0  →  sigma = 0  →  noisy = clean latents.
          - Supervised block k:    timestep = random t  →  noisy = latents + noise.
          - Frame 0: always clean (I2V anchor), timestep = 0.
          - Single WanModel.forward over the full sequence (block-causal mask handles
            attention restriction).  Loss computed only on supervised block k.

        This gives full gradient flow to all trainable parameters (Wq, Wk, Wv, Wo,
        hub_embed, action layers) and matches inference semantics: block k is denoised
        while attending to clean context from earlier blocks.
        """
        B, C, f_latent, H, W = latents.shape
        n = self.num_frame_per_block
        num_blocks = f_latent // n
        device = latents.device
        k = int(torch.randint(0, num_blocks, (1,)).item())
        t_idx = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        t_val = self.pipe.scheduler.timesteps[t_idx].to(
            dtype=self.pipe.torch_dtype, device=device
        )
        timestep = torch.zeros(
            [B, f_latent], dtype=self.pipe.torch_dtype, device=device
        )
        timestep[:, k * n : (k + 1) * n] = t_val
        timestep[:, 0] = 0.0
        noise = torch.randn_like(latents)
        sigma, _ = self.pipe.scheduler._sigma_from_timestep(
            timestep, device=device, dtype=latents.dtype
        )
        sigma_view = sigma.view(B, 1, f_latent, 1, 1)
        noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise
        noisy_latents[:, :, 0:1] = latents[:, :, 0:1]
        df_supervise_mask = torch.zeros(f_latent, dtype=torch.bool, device=device)
        df_supervise_mask[k * n : (k + 1) * n] = True
        df_supervise_mask[0] = False
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        extra_input = self.pipe.prepare_extra_input(latents)
        dit_kwargs = dict(
            mouse=mouse,
            keyboard=keyboard,
            loss_type="diffusion_forcing",
            **prompt_emb,
            **extra_input,
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        if agent_vertex_ids is not None:
            dit_kwargs["agent_vertex_ids"] = agent_vertex_ids
        _attach_hier_group_dit_kwargs(dit_kwargs, batch, self.device)
        if self.use_camera_pose:
            if "camera_pose" not in batch:
                raise KeyError(
                    "--use_camera_pose 需要 dataset 返回 camera_pose（CSV camera_pose_path* / camera12_path*）"
                )
            dit_kwargs["camera_pose"] = batch["camera_pose"][0].to(
                device=self.device, dtype=self.pipe.torch_dtype
            )
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **dit_kwargs
        )
        sel = df_supervise_mask
        loss = F.mse_loss(
            noise_pred[:, :, sel, ...].float(), training_target[:, :, sel, ...].float()
        )
        loss_weight = self.pipe.scheduler.training_weight(timestep[:, sel])
        loss = loss * loss_weight
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError(
                f"Non-finite training loss at step={int(self.global_step)} loss={loss.detach().float().cpu()}"
            )
        return loss

    def _select_generated_history_plan(self, num_blocks: int):
        """Return a rank-synchronous (target k, history m, start g0) plan or None.

        The decision uses only global_step, so every DDP rank takes the same branch and
        uses the same block counts.  It does not consume the training RNG.
        """
        if self.generated_history_prob <= 0.0 or num_blocks < 2:
            return None
        step = int(self.global_step)
        mix_hash = step * 1103515245 + 12345 & 4294967295
        if mix_hash / float(1 << 32) >= self.generated_history_prob:
            return None
        plan_hash = (step + 1) * 2654435761 + 1013904223 & 4294967295
        k = 1 + plan_hash % (num_blocks - 1)
        max_m = min(3, k)
        m = 1 + (plan_hash >> 16) % max_m
        return (int(k), int(m), int(k - m))

    @torch.inference_mode()
    def _generated_history_negative_context(self, num_players: int, *, dtype, device):
        """Encode the deployment negative prompt once and keep its cache on CPU."""
        if self._generated_history_negative_context_cpu is None:
            self.pipe.text_encoder.to(device)
            encoded = self.pipe.encode_prompt(
                [_GENERATED_HISTORY_NEGATIVE_PROMPT] * int(num_players), positive=False
            )["context"]
            self._generated_history_negative_context_cpu = encoded.detach().cpu()
            self.pipe.text_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        context = self._generated_history_negative_context_cpu
        if int(context.shape[0]) != int(num_players):
            raise ValueError(
                f"negative prompt cache P={context.shape[0]} does not match P={num_players}"
            )
        return context.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def _rollout_generated_history_online(
        self,
        *,
        latents,
        keyboard,
        mouse,
        positive_context,
        agent_vertex_ids,
        batch,
        target_block,
        history_blocks,
        batch_idx,
    ):
        """Generate the most recent 1--3 history blocks with the real cached CFG path.

        Input latents are [P,C,T,H,W].  Blocks [0,g0) are committed as clean GT,
        blocks [g0,k) are generated with the deployment sampler, and the returned
        [P,C,m*n,H,W] tensor is detached.  The student target is intentionally not
        evaluated here; it stays on the parallel differentiable training path.
        """
        P, C, T, H, W = latents.shape
        n = int(self.num_frame_per_block)
        k = int(target_block)
        m = int(history_blocks)
        g0 = k - m
        if not (1 <= k < T // n and 1 <= m <= min(3, k)):
            raise ValueError(f"invalid generated-history plan T={T} n={n} k={k} m={m}")
        rollout_scheduler = copy.deepcopy(self.pipe.scheduler)
        rollout_scheduler.set_timesteps(
            self.val_num_inference_steps, denoising_strength=1.0, shift=5.0
        )
        negative_context = self._generated_history_negative_context(
            P, dtype=latents.dtype, device=latents.device
        )
        group_labels = batch["group_labels"][0].to(
            device=latents.device, dtype=torch.long
        )
        active_group = batch["active_group"][0].to(
            device=latents.device, dtype=torch.bool
        )
        membership = batch.get("membership")
        global_only = batch.get("global_only")
        if membership is not None:
            if global_only is None:
                raise KeyError(
                    "membership generated-history rollout requires global_only"
                )
            membership = membership[0].to(device=latents.device, dtype=torch.bool)
            global_only = global_only[0].to(device=latents.device, dtype=torch.bool)
        camera_pose = None
        if self.use_camera_pose:
            camera_pose = batch["camera_pose"][0].to(
                device=latents.device, dtype=latents.dtype
            )

        def group_kwargs(lo, hi):
            out = {
                "group_labels": group_labels[lo:hi],
                "active_group": active_group[lo:hi],
            }
            if membership is not None:
                out.update(
                    membership=membership,
                    global_only=global_only,
                    active_group=active_group,
                )
            return out

        old_kv, old_cross = (self.pipe.kv_cache1, self.pipe.crossattn_cache)

        def new_cache_pair():
            self.pipe.kv_cache1, self.pipe.crossattn_cache = (None, None)
            self.pipe.init_kv_cache_for_training(batch_size=P, device=latents.device)
            return (self.pipe.kv_cache1, self.pipe.crossattn_cache)

        kv_pos, cross_pos = new_cache_pair()
        kv_neg, cross_neg = new_cache_pair()
        if kv_pos is kv_neg or cross_pos is cross_neg:
            raise AssertionError("positive and negative CFG caches must be independent")
        token_stride = int(self.pipe.frame_seq_length) * P
        common = dict(
            mouse=mouse,
            keyboard=keyboard,
            loss_type="diffusion_forcing",
            agent_vertex_ids=agent_vertex_ids,
            camera_pose_full=camera_pose,
        )
        pos_common = dict(
            common, context=positive_context, kv_cache=kv_pos, crossattn_cache=cross_pos
        )
        neg_common = dict(
            common, context=negative_context, kv_cache=kv_neg, crossattn_cache=cross_neg
        )
        module_states = [
            (module, module.training) for module in self.pipe.dit.modules()
        ]
        self.pipe.dit.eval()
        generated = []
        started = time.perf_counter()
        try:
            warm_t = torch.zeros((P, 1), device=latents.device, dtype=torch.int64)
            for branch in (pos_common, neg_common):
                model_fn_wan_video_causal(
                    self.pipe.dit,
                    latents[:, :, :1],
                    timestep=warm_t,
                    current_start=0,
                    cache_start=0,
                    **group_kwargs(0, 1),
                    **branch,
                )
            for block_idx in range(g0):
                lo, hi = (block_idx * n, (block_idx + 1) * n)
                context_timestep = torch.zeros(
                    (P, n), device=latents.device, dtype=torch.int64
                )
                for branch in (pos_common, neg_common):
                    model_fn_wan_video_causal(
                        self.pipe.dit,
                        latents[:, :, lo:hi],
                        timestep=context_timestep,
                        current_start=lo * token_stride,
                        cache_start=lo * token_stride,
                        **group_kwargs(lo, hi),
                        **branch,
                    )
            generator = torch.Generator(device=latents.device)
            seed = (
                12648430
                + 1000003 * int(self.global_rank)
                + 9176 * int(self.global_step)
                + int(batch_idx)
                & 9223372036854775807
            )
            generator.manual_seed(seed)
            for block_idx in range(g0, k):
                lo, hi = (block_idx * n, (block_idx + 1) * n)
                noisy = torch.randn(
                    (P, C, n, H, W),
                    generator=generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                if block_idx == 0:
                    noisy[:, :, :1] = latents[:, :, :1]
                for current_timestep in rollout_scheduler.timesteps:
                    timestep = torch.full(
                        (P, n),
                        current_timestep,
                        device=latents.device,
                        dtype=torch.int64,
                    )
                    if block_idx == 0:
                        timestep[:, 0] = 0
                        noisy[:, :, :1] = latents[:, :, :1]
                    pred_pos = model_fn_wan_video_causal(
                        self.pipe.dit,
                        noisy,
                        timestep=timestep,
                        current_start=lo * token_stride,
                        cache_start=lo * token_stride,
                        **group_kwargs(lo, hi),
                        **pos_common,
                    )
                    pred_neg = model_fn_wan_video_causal(
                        self.pipe.dit,
                        noisy,
                        timestep=timestep,
                        current_start=lo * token_stride,
                        cache_start=lo * token_stride,
                        **group_kwargs(lo, hi),
                        **neg_common,
                    )
                    denoised_pred = pred_neg + 5.0 * (pred_pos - pred_neg)
                    noisy = rollout_scheduler.step(
                        denoised_pred, current_timestep, noisy
                    )
                    if block_idx == 0:
                        noisy[:, :, :1] = latents[:, :, :1]
                generated.append(noisy.detach().clone())
                context_timestep = torch.zeros(
                    (P, n), device=latents.device, dtype=torch.int64
                )
                for branch in (pos_common, neg_common):
                    model_fn_wan_video_causal(
                        self.pipe.dit,
                        noisy,
                        timestep=context_timestep,
                        current_start=lo * token_stride,
                        cache_start=lo * token_stride,
                        **group_kwargs(lo, hi),
                        **branch,
                    )
        finally:
            for module, state in module_states:
                module.training = state
            self.pipe.kv_cache1, self.pipe.crossattn_cache = (old_kv, old_cross)
        history = torch.cat(generated, dim=2).detach()
        expected_shape = (P, C, m * n, H, W)
        if tuple(history.shape) != expected_shape:
            raise AssertionError(
                f"generated history shape={tuple(history.shape)}, expected={expected_shape}"
            )
        if history.requires_grad or history.grad_fn is not None:
            raise AssertionError("generated history must be detached")
        if not torch.isfinite(history).all():
            raise RuntimeError("generated history contains non-finite values")
        if int(self.global_rank) == 0:
            print(
                f"[generated-history] step={int(self.global_step)} k={k} m={m} g0={g0} sampler_steps={self.val_num_inference_steps} cfg=5 rollout_s={time.perf_counter() - started:.3f} shape={tuple(history.shape)}",
                flush=True,
            )
        return history

    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device
        perf = self._perf_timing_enabled
        step_part_times = {"encode_target": 0.0, "dit_forward_loss": 0.0}
        if perf:
            t_encode_target_start = self._synced_time()
        if "latents" in batch:
            if batch["latents"].shape[0] != 1:
                raise ValueError(
                    "N-player precomputed training currently expects DataLoader batch_size=1"
                )
            latents = (
                batch["latents"]
                .squeeze(0)
                .to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            )
            if latents.shape[0] != self.pipe.num_players:
                raise ValueError(
                    f"latents P={latents.shape[0]} does not match num_players={self.pipe.num_players}"
                )
        elif "latents1" in batch:
            latents1 = batch["latents1"].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )
            latents2 = batch["latents2"].to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )
            latents = torch.cat((latents1, latents2), dim=0)
        else:
            video1 = batch["video1"]
            video1 = video1.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents1 = self._encode_latents(self.pipe, video1, self.tiler_kwargs)
            latents1 = latents1.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            video2 = batch["video2"]
            video2 = video2.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents2 = self._encode_latents(self.pipe, video2, self.tiler_kwargs)
            latents2 = latents2.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = torch.cat((latents1, latents2), dim=0)
        if perf:
            step_part_times["encode_target"] = max(
                0.0, self._synced_time() - t_encode_target_start
            )
        image_emb = {}
        keyboard = batch["keyboard"].squeeze(0)[:, :-1, :]
        mouse = batch["mouse"].squeeze(0)[:, :-1, :]
        keyboard = keyboard.to(device=self.pipe.device)
        mouse = mouse.to(device=self.pipe.device)
        if keyboard.shape[0] != latents.shape[0] or mouse.shape[0] != latents.shape[0]:
            raise ValueError(
                f"Player dim mismatch: latents P={latents.shape[0]}, keyboard P={keyboard.shape[0]}, mouse P={mouse.shape[0]}"
            )
        latents, keyboard, mouse, agent_vertex_ids = self._prepare_simplex_agents(
            latents, keyboard, mouse
        )
        generated_history_plan = None
        if self.generated_history_prob > 0.0:
            if latents.shape[2] % self.num_frame_per_block != 0:
                raise ValueError(
                    f"generated-history latent T={latents.shape[2]} is not divisible by block size {self.num_frame_per_block}"
                )
            generated_history_plan = self._select_generated_history_plan(
                int(latents.shape[2]) // self.num_frame_per_block
            )
        prompt_emb = dict(batch.get("prompt_emb", {}))
        prompt_context = self._get_context_for_seg(batch, seg_idx=0)
        if generated_history_plan is None:
            prompt_context, text_prompt_dropped = self._maybe_drop_text_prompt_context(
                prompt_context
            )
        else:
            text_prompt_dropped = False
        prompt_emb["context"] = prompt_context
        self.log(
            "train/text_prompt_dropped",
            1.0 if text_prompt_dropped else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        if perf:
            t_dit_forward_loss_start = self._synced_time()
        if self.causal_forcing_training:
            loss = self._stage2_causal_training_step(
                latents=latents,
                keyboard=keyboard,
                mouse=mouse,
                prompt_emb=prompt_emb,
                agent_vertex_ids=agent_vertex_ids,
                image_emb=image_emb,
                batch=batch,
            )
            loss_weight = torch.tensor(1.0)
            self.log("train_loss", loss, prog_bar=True)
            self.log(
                "train/is_recon_warmup",
                0.0,
                on_step=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
            self._latest_step_part_times = step_part_times
            return loss
        noise = torch.randn_like(latents)
        f_latent = latents.shape[2]
        df_supervise_mask = None
        clean_prefix_block = None
        n_block = self.num_frame_per_block
        num_blocks = f_latent // n_block
        if self.df_per_frame_noise:
            timestep_per_frame = torch.randint(
                0, self.pipe.scheduler.num_train_timesteps, (1, f_latent)
            )
        else:
            timestep_id = torch.randint(
                0, self.pipe.scheduler.num_train_timesteps, (1, num_blocks)
            )
            timestep_per_frame = timestep_id.repeat_interleave(n_block, dim=1)
            if timestep_per_frame.shape[1] < f_latent:
                pad = timestep_id[:, -1:].expand(
                    -1, f_latent - timestep_per_frame.shape[1]
                )
                timestep_per_frame = torch.cat([timestep_per_frame, pad], dim=1)
            elif timestep_per_frame.shape[1] > f_latent:
                timestep_per_frame = timestep_per_frame[:, :f_latent]
        t_idx = self.pipe.scheduler.timesteps[timestep_per_frame[0]].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        timestep = t_idx.unsqueeze(0).expand(latents.shape[0], -1).clone()
        if generated_history_plan is not None:
            k, m, g0 = generated_history_plan
            prefix_end = k * n_block
            timestep[:, :prefix_end] = 0.0
            cur_lo = k * n_block
            cur_hi = min((k + 1) * n_block, f_latent)
            df_supervise_mask = torch.zeros(
                f_latent, dtype=torch.bool, device=timestep.device
            )
            df_supervise_mask[cur_lo:cur_hi] = True
            generated_history = self._rollout_generated_history_online(
                latents=latents,
                keyboard=keyboard,
                mouse=mouse,
                positive_context=prompt_emb["context"],
                agent_vertex_ids=agent_vertex_ids,
                batch=batch,
                target_block=k,
                history_blocks=m,
                batch_idx=batch_idx,
            )
            mixed_latents = latents.clone()
            mixed_latents[:, :, g0 * n_block : k * n_block] = generated_history
            latents = mixed_latents
        elif (
            self.clean_prefix_prob > 0.0
            and num_blocks >= 2
            and bool(torch.rand(1).item() < self.clean_prefix_prob)
        ):
            k = int(torch.randint(1, num_blocks, (1,)).item())
            clean_prefix_block = k
            prefix_end = k * n_block
            timestep[:, :prefix_end] = float(self.clean_prefix_context_noise)
            cur_lo = k * n_block
            cur_hi = min((k + 1) * n_block, f_latent)
            df_supervise_mask = torch.zeros(
                f_latent, dtype=torch.bool, device=timestep.device
            )
            df_supervise_mask[cur_lo:cur_hi] = True
        timestep[:, 0] = 0.0
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents, drop_first_frame, corrupt_anchor = (
            self._apply_first_frame_conditioning(
                latents,
                noise,
                timestep,
                seg_idx=0,
                allow_drop=df_supervise_mask is None,
            )
        )
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        dit_kwargs = dict(
            mouse=mouse,
            keyboard=keyboard,
            loss_type="diffusion_forcing",
            **prompt_emb,
            **extra_input,
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        if agent_vertex_ids is not None:
            dit_kwargs["agent_vertex_ids"] = agent_vertex_ids
        _attach_hier_group_dit_kwargs(dit_kwargs, batch, self.device)
        if self.use_camera_pose:
            if "camera_pose" not in batch:
                raise KeyError(
                    "--use_camera_pose 需要 dataset 返回 camera_pose（CSV camera_pose_path* / camera12_path*）"
                )
            dit_kwargs["camera_pose"] = batch["camera_pose"][0].to(
                device=self.device, dtype=self.pipe.torch_dtype
            )
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **dit_kwargs
        )
        if df_supervise_mask is not None:
            sel = df_supervise_mask
            loss = torch.nn.functional.mse_loss(
                noise_pred[:, :, sel, ...].float(),
                training_target[:, :, sel, ...].float(),
            )
            loss_weight = self.pipe.scheduler.training_weight(timestep[:, sel])
        elif drop_first_frame:
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), training_target.float()
            )
            loss_weight = self.pipe.scheduler.training_weight(timestep)
        else:
            loss = torch.nn.functional.mse_loss(
                noise_pred[:, :, 1:, ...].float(),
                training_target[:, :, 1:, ...].float(),
            )
            loss_weight = self.pipe.scheduler.training_weight(timestep[:, 1:])
        loss = loss * loss_weight
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError(
                f"Non-finite training loss at step={int(self.global_step)} batch_idx={int(batch_idx)} loss={loss.detach().float().cpu()}"
            )
        if perf:
            step_part_times["dit_forward_loss"] = max(
                0.0, self._synced_time() - t_dit_forward_loss_start
            )
        self._latest_step_part_times = step_part_times
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train/is_recon_warmup",
            0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/first_frame_dropped",
            1.0 if drop_first_frame else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/first_frame_corrupted",
            1.0 if corrupt_anchor else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/clean_prefix",
            1.0 if clean_prefix_block is not None else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/generated_history",
            1.0 if generated_history_plan is not None else 0.0,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        return loss

    @torch.no_grad()
    def _maybe_run_manual_validation(self):
        """方案二：按 global_step 触发验证，与 epoch 边界解耦。

        在 on_train_batch_end 里每步调用。所有 rank 依据同一 global_step 判定是否验证
        （DDP 下 global_step 同步，判定一致），只有 rank0 真正跑生成/记录（无集合通信），
        其余 rank 到末尾 barrier 等待，避免跑太远 / logger 竞争。
        """
        if not self.val_infer or self._manual_val_loader is None:
            return
        gs = int(self.global_step)
        if gs <= 0 or gs % self.val_every_n_steps != 0:
            return
        if getattr(self, "_last_val_step", -1) == gs:
            return
        self._last_val_step = gs
        was_training = self.training
        self.eval()
        try:
            if self.trainer.is_global_zero:
                try:
                    with torch.no_grad():
                        for i, batch in enumerate(self._manual_val_loader):
                            self._run_val_sample(batch, i)
                except Exception as e:
                    print(
                        f"[val] manual validation failed @ step {gs}: {e}", flush=True
                    )
        finally:
            if was_training:
                self.train()
            self.pipe.scheduler.set_timesteps(1000, training=True)
        if (
            getattr(self.trainer, "world_size", 1) > 1
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()

    def _run_val_sample(self, batch, batch_idx):
        """Save generated player views for the first N training samples on rank 0.

        依赖 PrecomputedLatentDataset 的 batch 结构（batch_size=1）：
          latents [1,P,48,T_lat,h,w]、keyboard [1,P,T,23]、mouse [1,P,T,2]。
        KV cache 在 causal_inference 内部逐次 reset，故每条样本独立。
        """
        if "latents" not in batch and "latents1" not in batch:
            return
        from .data.video import generated_player_grid, save_video

        gs = int(self.global_step)
        _dtype, _dev = (self.pipe.torch_dtype, self.device)
        if "latents" in batch:
            video_latents = batch["latents"].squeeze(0).to(dtype=_dtype, device=_dev)
        else:
            video_latents = torch.cat([batch["latents1"], batch["latents2"]], dim=0).to(
                dtype=_dtype, device=_dev
            )
        P = int(video_latents.shape[0])
        keyboard = batch["keyboard"].squeeze(0)[:, :-1, :].to(_dev)
        mouse = batch["mouse"].squeeze(0)[:, :-1, :].to(_dev)
        camera_pose = None
        if self.use_camera_pose:
            if "camera_pose" not in batch:
                raise KeyError("--use_camera_pose validation 需要 camera_pose")
            camera_pose = batch["camera_pose"].squeeze(0).to(dtype=_dtype, device=_dev)
        prompt = ""
        negative_prompt = ""
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
                prompt = ""
        val_cfg_scale = 5.0 if has_caption_prompt else 1.0
        if batch_idx == 0:
            print(
                f"[val] prompt={('caption1/caption2' if has_caption_prompt else 'blank')} cfg_scale={val_cfg_scale}",
                flush=True,
            )
        try:
            hier_infer_kwargs = {}
            _attach_hier_group_dit_kwargs(hier_infer_kwargs, batch, self.device)
            gen_players = self.pipe.causal_inference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                mouse=mouse,
                keyboard=keyboard,
                num_frames=self.val_causal_num_frames,
                height=video_latents.shape[-2] * self.pipe.vae.upsampling_factor,
                width=video_latents.shape[-1] * self.pipe.vae.upsampling_factor,
                cfg_scale=val_cfg_scale,
                num_inference_steps=self.val_num_inference_steps,
                seed=0,
                tiled=self.tiler_kwargs["tiled"],
                tile_size=self.tiler_kwargs["tile_size"],
                tile_stride=self.tiler_kwargs["tile_stride"],
                loss_type="diffusion_forcing",
                return_both_players=True,
                agent_vertex_ids=self.val_agent_vertex_ids,
                video_latents=video_latents,
                camera_pose=camera_pose,
                overlap_path=None,
                **hier_infer_kwargs,
            )
        finally:
            self.pipe.scheduler.set_timesteps(1000, training=True)
        if isinstance(gen_players, tuple):
            gen_players = list(gen_players)
        if len(gen_players) != P:
            raise RuntimeError(
                f"validation generated {len(gen_players)} players, expected P={P}"
            )
        grid = generated_player_grid(gen_players)
        os.makedirs(self.val_output_dir, exist_ok=True)
        grid_path = os.path.join(self.val_output_dir, f"step{gs}_s{batch_idx}_grid.mp4")
        save_video(grid, grid_path, fps=self.val_grid_fps, quality=5)

    def configure_optimizers(self):
        params = [
            p for p in self.pipe.denoising_model().parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        return optimizer

    def _resolve_local_version_dir(self):
        """Resolve a local run directory for logs and checkpoints."""
        try:
            ld = self.trainer.logger.log_dir
        except Exception:
            ld = None
        if ld:
            _out = os.path.abspath(self.output_path)
            if os.path.abspath(ld).startswith(_out + os.sep):
                return ld
        ver = self.run_version or "version_0"
        return os.path.join(self.output_path, ver)

    def on_train_start(self):
        """Record trainable parameter names in the local run directory."""
        if not self.trainer.is_global_zero:
            return
        local_version_dir = self._resolve_local_version_dir()
        if not local_version_dir:
            print(
                "[on_train_start] cannot resolve local version dir; skip train_module.txt dump."
            )
            return
        os.makedirs(local_version_dir, exist_ok=True)
        file_path = os.path.join(local_version_dir, "train_module.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for name, param in self.pipe.dit.named_parameters():
                f.write(f"name: {name} grad: {param.requires_grad}\n")
        print(f"[on_train_start] wrote {file_path}")

    def _assert_dit_state_finite_before_save(self):
        bad = []
        for name, param in self.pipe.denoising_model().named_parameters():
            if not param.is_floating_point() and (not param.is_complex()):
                continue
            try:
                finite = bool(torch.isfinite(param.detach()).all().item())
            except RuntimeError as exc:
                bad.append(
                    f"{name}: shape={tuple(param.shape)} dtype={param.dtype} check_failed={exc}"
                )
                continue
            if not finite:
                bad.append(f"{name}: shape={tuple(param.shape)} dtype={param.dtype}")
                if len(bad) >= 12:
                    break
        if bad:
            raise RuntimeError(
                "DiT contains non-finite parameter(s) at checkpoint save time; refusing to save a checkpoint that will produce NaN/black videos.\n"
                + "\n".join((f"  - {item}" for item in bad))
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        if not self.trainer.is_global_zero:
            return
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        if self.ckpt_local_dir:
            save_dir = self.ckpt_local_dir
        else:
            save_dir = os.path.join(self._resolve_local_version_dir(), "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoint save_dir: {save_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")
        saved_files = []
        self._assert_dit_state_finite_before_save()
        fname = f"step{current_step}.ckpt"
        torch.save(
            self.pipe.denoising_model().state_dict(), os.path.join(save_dir, fname)
        )
        saved_files.append(fname)


def _trainer_cluster_kwargs():
    """Match Lightning num_nodes/devices to torchrun (WORLD_SIZE / LOCAL_WORLD_SIZE)."""
    ws = os.environ.get("WORLD_SIZE")
    lws = os.environ.get("LOCAL_WORLD_SIZE")
    if ws is not None and lws is not None:
        try:
            ws_i, lws_i = (int(ws), int(lws))
            if lws_i > 0 and ws_i >= lws_i and (ws_i % lws_i == 0):
                return {"num_nodes": ws_i // lws_i, "devices": lws_i}
        except ValueError:
            pass
    return {"num_nodes": 1, "devices": "auto"}


def _read_int_hparam(path: str, key: str, default: int = 0) -> int:
    if not path or not os.path.isfile(path):
        return int(default)
    with open(path, encoding="utf-8") as f:
        for raw in f:
            field, sep, value = raw.strip().partition(":")
            if not sep or field.strip() != key:
                continue
            value = value.strip()
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value.strip("'\"")
            if parsed in (None, ""):
                return int(default)
            return int(parsed)
    return int(default)


def _resolve_train_step_offset(
    resume_ckpt_path: str, explicit_offset, *, schedule_enabled: bool
) -> int:
    """Resolve cumulative completed updates for weight-only resume checkpoints."""
    if explicit_offset is not None:
        offset = int(explicit_offset)
        if offset < 0:
            raise ValueError(f"train_step_offset must be >= 0, got {offset}")
        print(f"[train-step] using explicit train_step_offset={offset}", flush=True)
        return offset
    if not resume_ckpt_path:
        return 0
    source_path = os.environ.get("WAN_RESUME_CKPT_SOURCE_PATH") or resume_ckpt_path
    match = re.fullmatch("step(\\d+)\\.ckpt", os.path.basename(source_path))
    if match is None:
        message = f"cannot infer train_step_offset: resume checkpoint basename must be stepN.ckpt, got {source_path!r}"
        if schedule_enabled:
            raise ValueError(message + "; pass --train_step_offset explicitly")
        print(f"[train-step][warn] {message}; using 0", flush=True)
        return 0
    checkpoint_step = int(match.group(1))
    parent_hparams = os.path.join(
        os.path.dirname(os.path.dirname(source_path)), "hparams.yaml"
    )
    parent_offset = _read_int_hparam(parent_hparams, "train_step_offset", default=0)
    offset = parent_offset + checkpoint_step
    print(
        f"[train-step] auto-resolved parent_offset={parent_offset} checkpoint_step={checkpoint_step} train_step_offset={offset} source={source_path}",
        flush=True,
    )
    return offset


def _write_hparams_yaml(path, args):
    """Write Lightning-style hparams.yaml next to checkpoints (TB logger does not)."""
    payload = {k: "" if v is None else str(v) for k, v in vars(args).items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, value in payload.items():
            f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")


def _compute_run_version(output_path):
    """Choose the next local lightning_logs/version_N directory."""
    base = os.path.join(output_path, "lightning_logs")
    max_ver = -1
    if os.path.isdir(base):
        for name in os.listdir(base):
            m = re.match("version_(\\d+)$", name)
            if m:
                max_ver = max(max_ver, int(m.group(1)))
    return f"version_{max_ver + 1}"


def train(args):
    schedule_enabled = (
        args.local_attn_size_after is not None
        or args.local_attn_switch_step is not None
    )
    if (args.local_attn_size_after is None) != (args.local_attn_switch_step is None):
        raise ValueError(
            "--local_attn_size_after and --local_attn_switch_step must be set together"
        )
    if args.local_attn_switch_step is not None and args.local_attn_switch_step < 0:
        raise ValueError("--local_attn_switch_step must be >= 0")
    args.train_step_offset = _resolve_train_step_offset(
        args.resume_ckpt_path, args.train_step_offset, schedule_enabled=schedule_enabled
    )
    latent_t = (args.target_rgb_frames - 1) // 4 + 1
    if latent_t % args.num_frame_per_block != 0:
        raise ValueError(
            f"latent_T={latent_t} must be divisible by num_frame_per_block={args.num_frame_per_block}"
        )
    if args.local_attn_size != -1 and args.local_attn_size < args.num_frame_per_block:
        raise ValueError(
            f"local_attn_size must be -1 or >= num_frame_per_block; got local={args.local_attn_size}, n={args.num_frame_per_block}"
        )
    if (
        args.local_attn_size_after is not None
        and args.local_attn_size_after != -1
        and (args.local_attn_size_after < args.num_frame_per_block)
    ):
        raise ValueError(
            f"local_attn_size_after must be -1 or >= num_frame_per_block; got local={args.local_attn_size_after}, n={args.num_frame_per_block}"
        )
    if args.simplex_train_permute:
        raise ValueError("hier_hub requires --no_simplex_train_permute")
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
        if args.attn_impl == "hier_tiled_ste":
            raise ValueError("--hier_no_local_hub does not support hier_tiled_ste")
    elif args.num_local_hub_tokens <= 0:
        raise ValueError("ordinary hier_hub requires num_local_hub_tokens>0")
    print(
        "[train] 使用预压 latent 数据集（PrecomputedLatentDataset），跳过在线视频/VAE 编码。"
    )
    dataset = PrecomputedLatentDataset(
        args.dataset_path,
        args.metadata_file_name
        if os.path.isabs(args.metadata_file_name)
        else os.path.join(args.dataset_path, args.metadata_file_name),
        steps_per_epoch=args.steps_per_epoch,
        num_players=args.num_players,
        target_rgb_frames=args.target_rgb_frames,
        blank_prompt_emb_path=args.blank_prompt_emb_path,
        vae_latent_channels=args.vae_latent_channels,
        num_frame_per_block=args.num_frame_per_block,
        local_attn_size=args.local_attn_size,
        use_camera_pose=args.use_camera_pose,
        use_group_labels=True,
        num_groups=args.num_groups,
        group_path_column=args.group_path_column,
        group_label_player_ids=args.group_label_player_ids,
        hier_group_mode=args.hier_group_mode,
        hier_group_seed=args.hier_group_seed,
        hier_random_preserve_sizes=bool(args.hier_random_preserve_sizes),
        hier_group_format=args.hier_group_format,
        grouping_cache_dir=args.grouping_cache_dir,
        grouping_model_profile=(args.num_local_hub_tokens, args.num_global_hub_tokens,
                                args.hier_direct_frac, args.num_frame_per_block),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=1, num_workers=args.dataloader_num_workers
    )
    val_loader = None
    if args.val_infer:
        from torch.utils.data import Subset

        n_val = min(args.val_num_samples, len(dataset))
        val_loader = torch.utils.data.DataLoader(
            Subset(dataset, list(range(n_val))),
            shuffle=False,
            batch_size=1,
            num_workers=2,
        )
        print(
            f"[val] inline validation enabled: {n_val} samples, every {args.val_every_n_steps} steps."
        )
    run_version = args.run_version or _compute_run_version(args.output_path)
    print(
        f"[run] version = {run_version}"
        + ("  (来自 --run_version)" if args.run_version else "  (扫描 max+1)")
    )
    model = LightningModelForTrain(
        vae_path=args.vae_path,
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        output_path=os.path.join(args.output_path, "lightning_logs"),
        num_players=args.num_players,
        vae_latent_channels=args.vae_latent_channels,
        perf_log_every_n_steps=args.perf_log_every_n_steps,
        perf_timing=not args.no_perf_timing,
        target_rgb_frames=args.target_rgb_frames,
        first_frame_dropout_p=args.first_frame_dropout_p,
        first_frame_noise_p=args.first_frame_noise_p,
        first_frame_noise_corrupt_ratio=args.first_frame_noise_corrupt_ratio,
        first_frame_noise_min_seg=args.first_frame_noise_min_seg,
        ckpt_local_dir=args.ckpt_local_dir,
        run_version=run_version,
        blank_prompt_emb_path=args.blank_prompt_emb_path,
        text_prompt_dropout_p=args.text_prompt_dropout_p,
        use_simplex_rope=True,
        simplex_pool_size=args.simplex_pool_size,
        simplex_gamma=args.simplex_gamma,
        simplex_train_permute=args.simplex_train_permute,
        simplex_train_sample_vertex_ids=args.simplex_train_sample_vertex_ids,
        training_stage="stage2",
        loss_type="diffusion_forcing",
        attention_mode="hier_hub",
        attn_impl=args.attn_impl,
        num_hub_tokens=args.num_hub_tokens,
        local_attn_size=args.local_attn_size,
        local_attn_size_after=args.local_attn_size_after,
        local_attn_switch_step=args.local_attn_switch_step,
        train_step_offset=args.train_step_offset,
        num_frame_per_block=args.num_frame_per_block,
        num_groups=args.num_groups,
        num_local_hub_tokens=args.num_local_hub_tokens,
        num_global_hub_tokens=args.num_global_hub_tokens,
        hier_no_local_hub=args.hier_no_local_hub,
        hier_local_hub_mode=args.hier_local_hub_mode,
        hier_direct_schedule=args.hier_direct_schedule,
        hier_direct_frac=args.hier_direct_frac,
        hier_direct_layers=args.hier_direct_layers,
        unfreeze_self_attn=args.unfreeze_self_attn or True,
        clean_prefix_prob=args.clean_prefix_prob,
        clean_prefix_context_noise=args.clean_prefix_context_noise,
        generated_history_prob=args.generated_history_prob,
        df_per_frame_noise=args.df_per_frame_noise,
        causal_forcing_training=args.causal_forcing_training,
        use_camera_pose=args.use_camera_pose,
        camera_dim=args.camera_dim,
        camera_dropout_p=args.camera_dropout_p,
        val_infer=args.val_infer,
        val_every_n_steps=args.val_every_n_steps,
        val_num_samples=args.val_num_samples,
        val_num_inference_steps=args.val_num_inference_steps,
        val_causal_num_frames=args.val_causal_num_frames,
        val_output_dir=args.val_output_dir,
        val_grid_fps=args.val_grid_fps,
    )
    model._manual_val_loader = val_loader
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name=f"wan_{run_version}",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_path, name="lightning_logs", version=run_version
        )
        logger.log_hyperparams({k: str(v) for k, v in vars(args).items()})
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        every_n_train_steps=1000, save_top_k=2, monitor="step", mode="max"
    )
    _val_trainer_kwargs = {}
    if args.val_infer:
        _val_trainer_kwargs = dict(use_distributed_sampler=False)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        **_trainer_cluster_kwargs(),
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val
        if args.gradient_clip_val > 0
        else None,
        callbacks=[checkpoint_callback],
        logger=logger,
        **_val_trainer_kwargs,
    )
    if trainer.global_rank == 0:
        _primary_logger = logger[0] if isinstance(logger, (list, tuple)) else logger
        _log_dir = getattr(_primary_logger, "log_dir", None) or args.output_path
        checkpoints_dir = os.path.join(_log_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"Manually created checkpoint directory for rank 0: {checkpoints_dir}")
        version_dir = os.path.join(args.output_path, "lightning_logs", run_version)
        hparams_path = os.path.join(version_dir, "hparams.yaml")
        _write_hparams_yaml(hparams_path, args)
        print(f"[hparams] wrote {hparams_path}")
    trainer.fit(model, dataloader)
