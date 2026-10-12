import torch, glob, os
from typing import Optional, Union
from dataclasses import dataclass
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download
from typing import Optional


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    local_model_path: str = None
    skip_download: bool = False

    def download_if_necessary(self, use_usp=False):
        if self.path is None:
            if self.model_id is None:
                raise ValueError(
                    f'No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.'
                )
            if use_usp:
                import torch.distributed as dist

                skip_download = self.skip_download or dist.get_rank() != 0
            else:
                skip_download = self.skip_download
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(
                self.origin_file_pattern, str
            ) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False
            if self.local_model_path is None:
                self.local_model_path = "./models"
            if not skip_download:
                downloaded_files = glob.glob(
                    self.origin_file_pattern,
                    root_dir=os.path.join(self.local_model_path, self.model_id),
                )
                if self.download_resource.lower() == "modelscope":
                    snapshot_download(
                        self.model_id,
                        local_dir=os.path.join(self.local_model_path, self.model_id),
                        allow_file_pattern=allow_file_pattern,
                        ignore_file_pattern=downloaded_files,
                        local_files_only=False,
                    )
                elif self.download_resource.lower() == "huggingface":
                    hf_snapshot_download(
                        self.model_id,
                        local_dir=os.path.join(self.local_model_path, self.model_id),
                        allow_patterns=allow_file_pattern,
                        ignore_patterns=downloaded_files,
                        local_files_only=False,
                    )
                else:
                    raise ValueError(
                        "`download_resource` should be `modelscope` or `huggingface`."
                    )
            if use_usp:
                import torch.distributed as dist

                dist.barrier(device_ids=[dist.get_rank()])
            if is_folder:
                self.path = os.path.join(
                    self.local_model_path, self.model_id, self.origin_file_pattern
                )
            else:
                self.path = glob.glob(
                    os.path.join(
                        self.local_model_path, self.model_id, self.origin_file_pattern
                    )
                )
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]
