from ..models import ModelManager
from ..models.wan_video_dit import (
    WanModel,
    _debug_labels_from_membership,
    expand_hier_local_hub_seed,
    _prepare_hier_membership_tensors,
    _squeeze_hier_batch,
    precompute_freqs_cis,
    rope_apply,
)
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from ..vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
)
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..utils import ModelConfig
from typing import Optional, Union


class WanVideoNPlayersPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.frame_seq_length = 220
        self.num_transformer_blocks = 30
        self.num_frame_per_block = 8
        self.independent_first_frame = True
        self.local_attn_size = 6
        self.num_players = 2
        self.context_noise = 0
        self.attention_mode = "dense"
        self.num_hub_tokens = 8

    def set_stage2_local_attn_size(
        self, local_attn_size: int, *, reset_training_cache: bool = False
    ):
        """Update the attention window while keeping all learned parameters intact."""
        if self.dit is None:
            raise RuntimeError(
                "Cannot set the local attention window before the DiT is loaded"
            )
        local_attn_size = int(local_attn_size)
        self.local_attn_size = local_attn_size
        self.dit.set_stage2_local_attn_size(local_attn_size)
        if reset_training_cache:
            self.kv_cache1 = None
            self.crossattn_cache = None

    def enable_stage2(
        self,
        num_hub_tokens: int = 8,
        local_attn_size: int = 24,
        num_frame_per_block: int = 7,
        attention_mode: str = "hier_hub",
        num_groups: int = 1,
        num_local_hub_tokens: int = 0,
        num_global_hub_tokens: int = None,
        hier_direct_schedule: str = "none",
        hier_direct_frac: float = 0.0,
        hier_direct_layers=None,
        attn_impl: str = "sdpa",
        hier_no_local_hub: bool = False,
        hier_local_hub_mode: str = "shared",
    ):
        """Paper Stage 2: SHA + block-causal DF defaults (81 RGB → 21 latent, n=7)."""
        self.attention_mode = str("hier_hub")
        self.attn_impl = str(attn_impl)
        self.num_hub_tokens = int(num_hub_tokens)
        self.local_attn_size = int(local_attn_size)
        self.num_frame_per_block = int(num_frame_per_block)
        self.hier_no_local_hub = bool(hier_no_local_hub)
        self.hier_local_hub_mode = str(hier_local_hub_mode)
        self.independent_first_frame = False
        if self.dit is not None:
            self.dit.enable_stage2(
                num_hub_tokens=num_hub_tokens,
                local_attn_size=local_attn_size,
                num_frame_per_block=num_frame_per_block,
                attention_mode="hier_hub",
                num_groups=num_groups,
                num_local_hub_tokens=num_local_hub_tokens,
                num_global_hub_tokens=num_global_hub_tokens,
                hier_direct_schedule=hier_direct_schedule,
                hier_direct_frac=hier_direct_frac,
                hier_direct_layers=hier_direct_layers,
                attn_impl=attn_impl,
                hier_no_local_hub=hier_no_local_hub,
                hier_local_hub_mode=hier_local_hub_mode,
            )

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoNPlayersPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ]
        pipe = WanVideoNPlayersPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            pipe.initialize_usp()
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model(
            "wan_video_motion_controller"
        )
        vace = model_manager.fetch_model("wan_video_vace", index=2)
        pipe.vap = model_manager.fetch_model("wan_video_vap")
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor

            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(
                audio_processor_config.path
            )
        if use_usp:
            pipe.enable_usp()
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 16, width // 16, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 16, width // 16)
        msk = msk.transpose(1, 2)[0]
        vae_input = torch.concat(
            [
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width).to(image.device),
            ],
            dim=1,
        )
        y = self.vae.encode(
            [vae_input.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
        )[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(
        self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def _player_videos_from_latents(self, latents, tiler_kwargs):
        """Decode [P,C,T_lat,h,w] -> list length P, each item list[PIL] over T_rgb."""
        frames = self.decode_video(latents, **tiler_kwargs)
        return [self.tensor2video(frames[p]) for p in range(frames.shape[0])]

    def _return_player_videos(self, player_videos):
        if len(player_videos) == 2:
            return (player_videos[0], player_videos[1])
        return player_videos

    @torch.no_grad()
    def causal_inference(
        self,
        prompt,
        negative_prompt="",
        video1=None,
        video2=None,
        keyboard=None,
        mouse=None,
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=257,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        loss_type="bidirectional",
        start_frame_index=0,
        return_both_players: bool = False,
        agent_vertex_ids: Optional[torch.Tensor] = None,
        video_latents: Optional[torch.Tensor] = None,
        overlap_path: Optional[str] = None,
        camera_pose: Optional[torch.Tensor] = None,
        group_labels: Optional[torch.Tensor] = None,
        active_group: Optional[torch.Tensor] = None,
        membership: Optional[torch.Tensor] = None,
        global_only: Optional[torch.Tensor] = None,
    ) -> None:
        height, width = self.check_resize_height_width(height, width)
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )
        latent_frame = (num_frames - 1) // 4 + 1
        p_count = int(self.num_players)
        if video_latents is not None:
            p_count = int(video_latents.shape[0])
            if p_count != int(self.num_players):
                raise ValueError(
                    f"video_latents P={p_count} does not match pipe.num_players={self.num_players}"
                )
        noise = self.generate_noise(
            (p_count, 48, latent_frame, height // 16, width // 16),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        self.load_models_to_device(["vae"])
        if video_latents is not None:
            video_latents = video_latents.to(dtype=self.torch_dtype, device=self.device)
        else:
            if int(self.num_players) != 2:
                raise ValueError(
                    "N-player causal_inference requires precomputed video_latents; video1/video2 only covers P=2."
                )
            video1 = video1.to(dtype=self.torch_dtype, device=self.device)
            video2 = video2.to(dtype=self.torch_dtype, device=self.device)
            latents1 = self.encode_video(video1, **tiler_kwargs)
            latents2 = self.encode_video(video2, **tiler_kwargs)
            video_latents = torch.cat([latents1, latents2], dim=0)
        # Only the first latent frame conditions generation; the ground-truth
        # context cache below needs every frame.
        needs_all_frames = os.environ.get("STAGE2_GT_CONTEXT_CACHE") == "1"
        if video_latents.shape[0] != noise.shape[0] or (
            needs_all_frames and video_latents.shape[2] < latent_frame
        ):
            raise ValueError(
                f"video_latents [P,T]={tuple(video_latents.shape[0:3:2])} must cover noise P={noise.shape[0]}, latent_T={latent_frame}"
            )
        initial_latent = video_latents[:, :, 0:1, ...].clone().to(device=self.device)
        if mouse is not None:
            mouse = mouse.to(dtype=self.torch_dtype, device=self.device)
        if keyboard is not None:
            keyboard = keyboard.to(dtype=self.torch_dtype, device=self.device)
        camera_pose_full = None
        if camera_pose is not None:
            camera_pose_full = camera_pose.to(
                dtype=self.torch_dtype, device=self.device
            )
            if camera_pose_full.dim() == 3:
                cam_f = camera_pose_full.shape[1]
            elif camera_pose_full.dim() == 4:
                cam_f = camera_pose_full.shape[2]
            else:
                raise ValueError(
                    f"camera_pose must be [P,F,C] or [B,P,F,C], got {tuple(camera_pose_full.shape)}"
                )
            if cam_f < latent_frame:
                raise ValueError(
                    f"camera_pose latent frames {cam_f} must cover inference latent_frame {latent_frame}"
                )
        self.load_models_to_device(["text_encoder"])
        conditional_dict = self.encode_prompt(prompt, positive=True)
        do_cfg = float(cfg_scale) != 1.0
        unconditional_dict = (
            self.encode_prompt(negative_prompt, positive=False) if do_cfg else None
        )
        _cn_env = os.environ.get("STAGE2_CONTEXT_NOISE")
        if _cn_env is not None:
            self.context_noise = int(_cn_env)
            print(
                f"[diag] context_noise override = {self.context_noise}",
                flush=True,
            )
        if os.environ.get("STAGE2_ZERO_ACTIONS") == "1":
            if mouse is not None:
                mouse = torch.zeros_like(mouse)
            if keyboard is not None:
                keyboard = torch.zeros_like(keyboard)
            print("[diag] ZEROED mouse+keyboard", flush=True)
        use_gt_context_cache = os.environ.get("STAGE2_GT_CONTEXT_CACHE") == "1"
        if use_gt_context_cache:
            print("[diag] GT context cache enabled", flush=True)
        assert latent_frame % self.num_frame_per_block == 0, (
            f"latent_T={latent_frame} must be divisible by num_frame_per_block={self.num_frame_per_block}"
        )
        num_blocks = latent_frame // self.num_frame_per_block
        n_groups = int(self.dit.num_groups)
        n = self.num_frame_per_block
        if membership is not None:
            membership, global_only, active_group = _prepare_hier_membership_tensors(
                membership,
                global_only,
                active_group,
                p_count,
                latent_frame,
                n_groups,
                self.device,
            )
            membership = membership[:latent_frame].contiguous()
            global_only = global_only[:latent_frame].contiguous()
            active_group = active_group[:latent_frame].contiguous()
            if group_labels is None:
                group_labels = _debug_labels_from_membership(membership)
            else:
                group_labels = _squeeze_hier_batch(group_labels, 2, "group_labels")
                group_labels = group_labels.to(device=self.device, dtype=torch.long)
                if (
                    group_labels.shape[0] < latent_frame
                    or group_labels.shape[1] != p_count
                ):
                    raise ValueError(
                        f"group_labels must cover [T={latent_frame},P={p_count}], got {tuple(group_labels.shape)}"
                    )
                group_labels = group_labels[:latent_frame].contiguous()
            for block_start in range(0, latent_frame, n):
                mem_block = membership[block_start : block_start + n]
                go_block = global_only[block_start : block_start + n]
                if mem_block.numel() and (
                    not torch.equal(mem_block, mem_block[:1].expand_as(mem_block))
                ):
                    raise ValueError(
                        f"hier_hub cached inference requires membership constant inside each latent block; block=[{block_start},{block_start + n})"
                    )
                if go_block.numel() and (
                    not torch.equal(go_block, go_block[:1].expand_as(go_block))
                ):
                    raise ValueError(
                        f"hier_hub cached inference requires global_only constant inside each latent block; block=[{block_start},{block_start + n})"
                    )
        else:
            if group_labels is None:
                raise ValueError(
                    "hier_hub cached inference requires group_labels [T,P] or [1,T,P]"
                )
            group_labels = group_labels.to(device=self.device, dtype=torch.long)
            if group_labels.dim() == 3:
                if group_labels.shape[0] != 1:
                    raise ValueError(
                        f"hier_hub cached inference expects group_labels batch B=1, got {tuple(group_labels.shape)}"
                    )
                group_labels = group_labels[0]
            if group_labels.shape[0] < latent_frame or group_labels.shape[1] != p_count:
                raise ValueError(
                    f"group_labels must cover [T={latent_frame},P={p_count}], got {tuple(group_labels.shape)}"
                )
            group_labels = group_labels[:latent_frame].contiguous()
            if group_labels.numel() and (
                int(group_labels.min().item()) < 0
                or int(group_labels.max().item()) >= n_groups
            ):
                raise ValueError(
                    f"group_labels values must be in [0,{n_groups}), got min={int(group_labels.min().item())} max={int(group_labels.max().item())}"
                )
            if active_group is None:
                active_group = torch.zeros(
                    latent_frame, n_groups, dtype=torch.bool, device=self.device
                )
                active_group.scatter_(1, group_labels, True)
            else:
                active_group = active_group.to(device=self.device, dtype=torch.bool)
                if active_group.dim() == 3:
                    if active_group.shape[0] != 1:
                        raise ValueError(
                            f"hier_hub cached inference expects active_group batch B=1, got {tuple(active_group.shape)}"
                        )
                    active_group = active_group[0]
                if (
                    active_group.shape[0] < latent_frame
                    or active_group.shape[1] != n_groups
                ):
                    raise ValueError(
                        f"active_group must cover [T={latent_frame},N={n_groups}], got {tuple(active_group.shape)}"
                    )
                active_group = active_group[:latent_frame].contiguous()
            for block_start in range(0, latent_frame, n):
                block = group_labels[block_start : block_start + n]
                if block.numel() and (
                    not torch.equal(block, block[:1].expand_as(block))
                ):
                    raise ValueError(
                        f"hier_hub cached inference requires group_labels constant inside each latent block; block=[{block_start},{block_start + n})"
                    )
        batch_size, num_channels, _, height, width = noise.shape
        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = latent_frame
        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        kv_batch = batch_size // self.num_players

        def _new_cache_pair():
            prev_kv_cache, prev_crossattn_cache = (self.kv_cache1, self.crossattn_cache)
            self.kv_cache1, self.crossattn_cache = (None, None)
            self._initialize_kv_cache(
                batch_size=kv_batch,
                kv_cache_size=None,
                dtype=self.torch_dtype,
                device=self.device,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=self.torch_dtype, device=self.device
            )
            kv_cache, crossattn_cache = (self.kv_cache1, self.crossattn_cache)
            self.kv_cache1, self.crossattn_cache = (prev_kv_cache, prev_crossattn_cache)
            return (kv_cache, crossattn_cache)

        if do_cfg:
            self.kv_cache1, self.crossattn_cache = (None, None)
            kv_cache_pos, crossattn_cache_pos = _new_cache_pair()
            kv_cache_neg, crossattn_cache_neg = _new_cache_pair()
            print(f"[causal][cfg] enabled cfg_scale={cfg_scale}", flush=True)
        elif self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=kv_batch,
                kv_cache_size=None,
                dtype=self.torch_dtype,
                device=self.device,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=self.torch_dtype, device=self.device
            )
            kv_cache_pos, crossattn_cache_pos = (self.kv_cache1, self.crossattn_cache)
            kv_cache_neg, crossattn_cache_neg = (None, None)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            self._reset_kv_cache(noise.device)
            kv_cache_pos, crossattn_cache_pos = (self.kv_cache1, self.crossattn_cache)
            kv_cache_neg, crossattn_cache_neg = (None, None)
        token_stride = self.frame_seq_length * self.num_players
        cache_start_frame = 0
        if do_cfg:
            pass

        def _hier_group_slice(lo, hi):
            out = {
                "group_labels": group_labels[lo:hi],
                "active_group": active_group[lo:hi],
            }
            if membership is not None:
                out["membership"] = membership
                out["global_only"] = global_only
                out["active_group"] = active_group
            return out

        if initial_latent is not None:
            output[:, :, :1] = initial_latent[:, :, :1]
            warm_t = torch.zeros(
                [batch_size, 1], device=noise.device, dtype=torch.int64
            )
            model_fn_wan_video_causal(
                self.dit,
                initial_latent[:, :, :1],
                timestep=warm_t,
                mouse=mouse,
                keyboard=keyboard,
                kv_cache=kv_cache_pos,
                crossattn_cache=crossattn_cache_pos,
                current_start=0,
                cache_start=0,
                loss_type="diffusion_forcing",
                agent_vertex_ids=agent_vertex_ids,
                camera_pose_full=camera_pose_full,
                **_hier_group_slice(0, 1),
                **conditional_dict,
            )
            if do_cfg:
                model_fn_wan_video_causal(
                    self.dit,
                    initial_latent[:, :, :1],
                    timestep=warm_t,
                    mouse=mouse,
                    keyboard=keyboard,
                    kv_cache=kv_cache_neg,
                    crossattn_cache=crossattn_cache_neg,
                    current_start=0,
                    cache_start=0,
                    loss_type="diffusion_forcing",
                    agent_vertex_ids=agent_vertex_ids,
                    camera_pose_full=camera_pose_full,
                    **_hier_group_slice(0, 1),
                    **unconditional_dict,
                )
        n = self.num_frame_per_block
        for block_idx in range(num_blocks):
            block_start = block_idx * n
            block_end = block_start + n
            noisy_input = noise[:, :, block_start:block_end].clone()
            if block_idx == 0 and initial_latent is not None:
                noisy_input[:, :, 0:1] = initial_latent[:, :, 0:1]
            for index, current_timestep in enumerate(self.scheduler.timesteps):
                timestep = torch.full(
                    [batch_size, n],
                    current_timestep,
                    device=noise.device,
                    dtype=torch.int64,
                )
                if block_idx == 0 and initial_latent is not None:
                    timestep[:, 0] = 0
                    noisy_input[:, :, 0:1] = initial_latent[:, :, 0:1]
                noise_pred_posi = model_fn_wan_video_causal(
                    self.dit,
                    noisy_input,
                    timestep=timestep,
                    mouse=mouse,
                    keyboard=keyboard,
                    kv_cache=kv_cache_pos,
                    crossattn_cache=crossattn_cache_pos,
                    current_start=block_start * token_stride,
                    cache_start=block_start * token_stride,
                    loss_type="diffusion_forcing",
                    agent_vertex_ids=agent_vertex_ids,
                    camera_pose_full=camera_pose_full,
                    **_hier_group_slice(block_start, block_end),
                    **conditional_dict,
                )
                if do_cfg:
                    noise_pred_nega = model_fn_wan_video_causal(
                        self.dit,
                        noisy_input,
                        timestep=timestep,
                        mouse=mouse,
                        keyboard=keyboard,
                        kv_cache=kv_cache_neg,
                        crossattn_cache=crossattn_cache_neg,
                        current_start=block_start * token_stride,
                        cache_start=block_start * token_stride,
                        loss_type="diffusion_forcing",
                        agent_vertex_ids=agent_vertex_ids,
                        camera_pose_full=camera_pose_full,
                        **_hier_group_slice(block_start, block_end),
                        **unconditional_dict,
                    )
                    denoised_pred = noise_pred_nega + cfg_scale * (
                        noise_pred_posi - noise_pred_nega
                    )
                else:
                    denoised_pred = noise_pred_posi
                noisy_input = self.scheduler.step(
                    denoised_pred, self.scheduler.timesteps[index], noisy_input
                )
                if block_idx == 0 and initial_latent is not None:
                    noisy_input[:, :, 0:1] = initial_latent[:, :, 0:1]
            output[:, :, block_start:block_end] = noisy_input
            context_timestep = torch.full_like(timestep, self.context_noise)
            if block_idx == 0 and initial_latent is not None:
                context_timestep[:, 0] = 0
            if use_gt_context_cache:
                ctx_content = video_latents[:, :, block_start:block_end].to(
                    device=noisy_input.device, dtype=noisy_input.dtype
                )
            else:
                ctx_content = noisy_input
            if self.context_noise > 0:
                csig, _ = self.scheduler._sigma_from_timestep(
                    torch.tensor(float(self.context_noise), device=noisy_input.device),
                    device=noisy_input.device,
                    dtype=noisy_input.dtype,
                )
                ctx_content = (1.0 - csig) * noisy_input + csig * torch.randn_like(
                    noisy_input
                )
            if block_idx == 0 and initial_latent is not None:
                ctx_content[:, :, 0:1] = initial_latent[:, :, 0:1]
            model_fn_wan_video_causal(
                self.dit,
                ctx_content,
                timestep=context_timestep,
                mouse=mouse,
                keyboard=keyboard,
                kv_cache=kv_cache_pos,
                crossattn_cache=crossattn_cache_pos,
                current_start=block_start * token_stride,
                cache_start=block_start * token_stride,
                loss_type="diffusion_forcing",
                agent_vertex_ids=agent_vertex_ids,
                camera_pose_full=camera_pose_full,
                **_hier_group_slice(block_start, block_end),
                **conditional_dict,
            )
            if do_cfg:
                model_fn_wan_video_causal(
                    self.dit,
                    ctx_content,
                    timestep=context_timestep,
                    mouse=mouse,
                    keyboard=keyboard,
                    kv_cache=kv_cache_neg,
                    crossattn_cache=crossattn_cache_neg,
                    current_start=block_start * token_stride,
                    cache_start=block_start * token_stride,
                    loss_type="diffusion_forcing",
                    agent_vertex_ids=agent_vertex_ids,
                    camera_pose_full=camera_pose_full,
                    **_hier_group_slice(block_start, block_end),
                    **unconditional_dict,
                )
            cache_start_frame = block_end
        if os.environ.get("STAGE2_DIAG_MOTION") == "1":
            with torch.no_grad():
                o = output.float()
                fr_diff = (o[:, :, 1:] - o[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))
            n_blk = self.num_frame_per_block
            print(
                "[diag] per-latent-frame motion (|x_t - x_{t-1}| mean):",
                flush=True,
            )
            for t in range(fr_diff.shape[0]):
                tag = "  <block start" if (t + 1) % n_blk == 0 else ""
                print(f"    frame {t + 1:2d}: {fr_diff[t].item():.4f}{tag}", flush=True)
        if os.environ.get("STAGE2_LATENT_STATS") == "1":
            with torch.no_grad():
                o = output.detach().float()
                finite = torch.isfinite(o)
                print(
                    f"[diag][stats] latent shape={tuple(output.shape)} finite={bool(finite.all().item())} min={o.min().item():.4f} max={o.max().item():.4f} mean={o.mean().item():.4f} std={o.std().item():.4f}",
                    flush=True,
                )
                per_p = o.flatten(1).mean(dim=1)
                print(
                    "[diag][stats] latent per-player mean="
                    + ",".join((f"{v.item():.4f}" for v in per_p)),
                    flush=True,
                )
        self.load_models_to_device(["vae"])
        if return_both_players:
            frames = self.decode_video(output, **tiler_kwargs)
            if os.environ.get("STAGE2_LATENT_STATS") == "1":
                with torch.no_grad():
                    f32 = frames.detach().float()
                    print(
                        f"[diag][stats] decoded shape={tuple(frames.shape)} min={f32.min().item():.4f} max={f32.max().item():.4f} mean={f32.mean().item():.4f} std={f32.std().item():.4f}",
                        flush=True,
                    )
            player_videos = [
                self.tensor2video(frames[p]) for p in range(frames.shape[0])
            ]
            self.load_models_to_device([])
            return self._return_player_videos(player_videos)
        frames = self.decode_video(output[1:2, ...], **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        return frames

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size=None):
        """Initialize per-layer hierarchical KV caches."""
        self._initialize_hier_kv_cache(batch_size, dtype, device)
        return

    def _initialize_hier_kv_cache(self, batch_size, dtype, device):
        """Per-player agent cache + shared hub cache per DiT layer."""
        p_count = self.num_players
        n_groups = int(self.dit.num_groups)
        k_local = int(self.dit.num_local_hub_tokens)
        k_global = int(self.dit.num_global_hub_tokens)
        no_local = bool(getattr(self.dit, "hier_no_local_hub", False))
        k_len = n_groups * k_local + k_global
        l_len = self.frame_seq_length
        _attn_frames = self.local_attn_size if self.local_attn_size != -1 else 64
        _retention = _attn_frames + self.num_frame_per_block
        agent_cache_size = _retention * l_len
        hub_cache_size = _retention * k_len
        local_hub_cache_size = _retention * max(k_local, 1)
        global_hub_cache_size = _retention * max(k_global, 1)
        n_heads = self.dit.blocks[0].self_attn.num_heads
        head_dim = self.dit.blocks[0].self_attn.head_dim
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            agent_caches = []
            for _ in range(p_count):
                agent_caches.append(
                    {
                        "k": torch.zeros(
                            [batch_size, agent_cache_size, n_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, agent_cache_size, n_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "global_end_index": torch.tensor(
                            [0], dtype=torch.long, device=device
                        ),
                        "local_end_index": torch.tensor(
                            [0], dtype=torch.long, device=device
                        ),
                    }
                )
            local_hubs = []
            if not no_local:
                for _ in range(n_groups):
                    local_hubs.append(
                        {
                            "k": torch.zeros(
                                [batch_size, local_hub_cache_size, n_heads, head_dim],
                                dtype=dtype,
                                device=device,
                            ),
                            "v": torch.zeros(
                                [batch_size, local_hub_cache_size, n_heads, head_dim],
                                dtype=dtype,
                                device=device,
                            ),
                            "active": torch.zeros(
                                [local_hub_cache_size], dtype=torch.bool, device=device
                            ),
                            "global_end_index": torch.tensor(
                                [0], dtype=torch.long, device=device
                            ),
                            "local_end_index": torch.tensor(
                                [0], dtype=torch.long, device=device
                            ),
                        }
                    )
            kv_cache1.append(
                {
                    "sparse_hub": True,
                    "hier_hub": True,
                    "num_players": p_count,
                    "num_hub_tokens": k_len,
                    "num_groups": n_groups,
                    "num_local_hub_tokens": k_local,
                    "num_global_hub_tokens": k_global,
                    "hier_no_local_hub": no_local,
                    "frame_seq_length": l_len,
                    "agent_caches": agent_caches,
                    "local_hubs": local_hubs,
                    "global_hub": {
                        "k": torch.zeros(
                            [batch_size, global_hub_cache_size, n_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, global_hub_cache_size, n_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "global_end_index": torch.tensor(
                            [0], dtype=torch.long, device=device
                        ),
                        "local_end_index": torch.tensor(
                            [0], dtype=torch.long, device=device
                        ),
                    },
                }
            )
            continue
        self.kv_cache1 = kv_cache1

    def init_kv_cache_for_training(self, batch_size: int, device: torch.device):
        """Initialize (or reset) KV cache + crossattn cache for causal-forcing training.

        Mirrors the initialization logic in causal_inference so that the training
        forward passes use an identical cache structure to inference.

        Args:
            batch_size: full batch size (i.e. num_players; typically 2).
            device: target device.
        """
        kv_batch = batch_size // self.num_players
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=kv_batch, dtype=self.torch_dtype, device=device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=self.torch_dtype, device=device
            )
        else:
            for entry in self.kv_cache1:
                for ac in entry["agent_caches"]:
                    ac["k"] = ac["k"].detach()
                    ac["v"] = ac["v"].detach()
                for lc in entry["local_hubs"]:
                    lc["k"] = lc["k"].detach()
                    lc["v"] = lc["v"].detach()
                    lc["active"] = lc["active"].detach()
                entry["global_hub"]["k"] = entry["global_hub"]["k"].detach()
                entry["global_hub"]["v"] = entry["global_hub"]["v"].detach()
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            self._reset_kv_cache(device)

    def _reset_kv_cache(self, device):
        if self.kv_cache1 is None:
            return
        for block_index in range(len(self.kv_cache1)):
            entry = self.kv_cache1[block_index]
            for ac in entry["agent_caches"]:
                ac["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )
                ac["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )
            for lc in entry["local_hubs"]:
                lc["active"].zero_()
                lc["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )
                lc["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )
            entry["global_hub"]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device
            )
            entry["global_hub"]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device
            )
            for ac_name in ("action_temporal", "action_keyboard"):
                ac = entry.get(ac_name)
                if ac is not None:
                    ac["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=device
                    )
                    ac["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=device
                    )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache


def model_fn_wan_video_causal(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    mouse: Optional[torch.Tensor] = None,
    keyboard: Optional[torch.Tensor] = None,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    loss_type="bidirectional",
    cache_start=0,
    agent_vertex_ids: Optional[torch.Tensor] = None,
    camera_pose_full: Optional[torch.Tensor] = None,
    group_labels: Optional[torch.Tensor] = None,
    active_group: Optional[torch.Tensor] = None,
    membership: Optional[torch.Tensor] = None,
    global_only: Optional[torch.Tensor] = None,
    **kwargs,
):
    B, F = timestep.shape
    timestep = timestep.to(device=x.device, dtype=x.dtype)
    t = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, timestep.reshape(-1)).to(
            device=x.device, dtype=x.dtype
        )
    ).reshape(B, F, dit.dim)
    t_mod = dit.time_projection(t.reshape(-1, dit.dim)).reshape(B, F, 6, dit.dim)
    context = dit.text_embedding(context)
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    x, (f, h, w) = dit.patchify(x)
    if context.shape[0] == 1 and x.shape[0] != 1:
        context = context.expand(x.shape[0], -1, -1)
    elif context.shape[0] != x.shape[0]:
        raise ValueError(
            f"context batch {context.shape[0]} must be 1 or match latent BP={x.shape[0]}"
        )
    num_p = keyboard.shape[0] if keyboard is not None else 2
    if agent_vertex_ids is None:
        agent_vertex_ids = list(range(num_p))
    _tokens_per_frame_allp = h * w * num_p
    start_frame = (
        current_start // _tokens_per_frame_allp if _tokens_per_frame_allp > 0 else 0
    )
    camera_pose_block = None
    if camera_pose_full is not None:
        cam = camera_pose_full
        end_frame = start_frame + f
        if cam.dim() == 3:
            if cam.shape[1] < end_frame:
                raise ValueError(
                    f"camera_pose_full has {cam.shape[1]} latent frames, need up to {end_frame}"
                )
            camera_pose_block = cam[:, start_frame:end_frame, :]
        elif cam.dim() == 4:
            if cam.shape[2] < end_frame:
                raise ValueError(
                    f"camera_pose_full has {cam.shape[2]} latent frames, need up to {end_frame}"
                )
            camera_pose_block = cam[:, :, start_frame:end_frame, :]
        else:
            raise ValueError(
                f"camera_pose_full must be [P,F,C] or [B,P,F,C], got {tuple(cam.shape)}"
            )
        camera_pose_block = camera_pose_block.to(device=x.device, dtype=x.dtype)
    freqs = dit.build_self_attn_freqs_with_hub(
        f, h, w, x.device, agent_vertex_ids=agent_vertex_ids, start_frame=start_frame
    )
    grid_sizes = torch.stack(
        [torch.tensor((f, h, w), dtype=torch.long, device=x.device) for _ in range(B)],
        dim=0,
    )
    seq_lens = torch.full((B,), f * h * w, dtype=torch.long, device=x.device)
    keyboard = dit.keyboard_embedding(keyboard)
    n_heads = dit.blocks[0].self_attn.num_heads
    assert keyboard.shape[-1] % n_heads == 0
    head_dim = keyboard.shape[-1] // n_heads
    freqs_keyboard = precompute_freqs_cis(head_dim, end=keyboard.shape[1]).to(
        keyboard.device
    )
    freqs_keyboard = freqs_keyboard.unsqueeze(1)
    keyboard = rope_apply(keyboard, freqs_keyboard, n_heads)
    latent_frame_num = f
    keyboard_BPFD = torch.zeros(
        (keyboard.shape[0], latent_frame_num * 12, keyboard.shape[2]),
        device=keyboard.device,
        dtype=keyboard.dtype,
    )
    mouse_BPFD = torch.zeros(
        (mouse.shape[0], latent_frame_num * 12, mouse.shape[2]),
        device=mouse.device,
        dtype=mouse.dtype,
    )
    window_size = 3
    compression_ratio = 4
    for i in range(latent_frame_num):
        end_frame_id = (i + 1) * 12
        global_i = start_frame + i
        src_start = (global_i - window_size + 1) * compression_ratio
        src_end = (global_i + 1) * compression_ratio
        src_start = max(0, src_start)
        src_end = min(keyboard.shape[1], src_end)
        if src_end > src_start:
            src_len = src_end - src_start
            dst_start = end_frame_id - src_len
            keyboard_BPFD[:, dst_start:end_frame_id, :] = keyboard[
                :, src_start:src_end, :
            ]
            mouse_BPFD[:, dst_start:end_frame_id, :] = mouse[:, src_start:src_end, :]
    if os.environ.get("STAGE2_DEBUG_START") == "1":
        _mnz = mouse_BPFD.abs().sum().item()
        _knz = keyboard_BPFD.abs().sum().item()
        print(
            f"[DBG] current_start={current_start} h={h} w={w} num_p={num_p} start_frame={start_frame} f={f} raw_action_len={keyboard.shape[1]} mouse_BPFD_sum={_mnz:.3f} keyboard_BPFD_sum={_knz:.3f}",
            flush=True,
        )
    mouse_BPFD = rearrange(
        mouse_BPFD,
        "p (f r win) d -> p f (r win d)",
        f=f,
        r=compression_ratio,
        win=window_size,
    )
    mouse_BPFD = mouse_BPFD.unsqueeze(2).repeat(1, 1, h * w, 1)
    mouse_BPFD = rearrange(mouse_BPFD, "p f l d -> p (f l) d")
    keyboard_BPFD = rearrange(
        keyboard_BPFD,
        "p (f r win) d -> p f (r win) d",
        f=f,
        r=compression_ratio,
        win=window_size,
    )
    kwargs = dict(grid_sizes=grid_sizes, seq_lens=seq_lens)
    tea_cache = None
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        x_hub = None
        if group_labels is None:
            raise ValueError(
                "hier_hub cached causal inference requires group_labels for the current block"
            )
        K_local = int(dit.num_local_hub_tokens)
        K_global = int(dit.num_global_hub_tokens)
        global_seed = dit.global_hub_embed.to(dtype=x.dtype, device=x.device)
        if bool(getattr(dit, "hier_no_local_hub", False)):
            x_local_hub = x.new_zeros(1, 0, dit.dim)
        else:
            local_seed = dit.local_hub_embed.to(dtype=x.dtype, device=x.device)
            x_local_hub = expand_hier_local_hub_seed(
                local_seed,
                num_frames=f,
                num_groups=dit.num_groups,
                num_local_hub_tokens=K_local,
                model_dim=dit.dim,
                mode=dit.hier_local_hub_mode,
            )
        x_global_hub = (
            global_seed[None, None, :, :]
            .expand(1, f, K_global, dit.dim)
            .reshape(1, f * K_global, dit.dim)
        )
        x_hub = torch.cat([x_local_hub, x_global_hub], dim=1)
        for block_index, block in enumerate(dit.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                }
            )
            x, x_hub = block.causal_forward(
                x,
                context,
                mouse_BPFD,
                keyboard_BPFD,
                t_mod,
                freqs,
                "diffusion_forcing",
                x_hub=x_hub,
                camera_pose=camera_pose_block,
                group_labels=group_labels,
                active_group=active_group,
                membership=membership,
                global_only=global_only,
                **kwargs,
            )
        if tea_cache is not None:
            tea_cache.store(x)
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
