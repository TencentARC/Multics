"""Shape-controlled BF16 GEMMs for inference.

Validated with CUDA 12.8 / PyTorch 2.8. This is not a guarantee
of bitwise reproducibility on other GPUs, shapes, or library versions.
Parameters keep nn.Linear's names and shapes for checkpoint compatibility.
"""

import torch
from torch.nn import functional as F


def _shape_linear(x, weight, bias, output_chunk_size=0, input_alignment=1):
    padding = (-weight.shape[1]) % input_alignment
    if padding:
        x = F.pad(x, (0, padding))
        # Do not detach/cache: gradients and newly loaded weights must propagate.
        weight = F.pad(weight, (0, padding))
    if output_chunk_size and weight.shape[0] > output_chunk_size:
        weights = weight.split(output_chunk_size, dim=0)
        biases = (
            bias.split(output_chunk_size, dim=0)
            if bias is not None else [None] * len(weights)
        )
        return torch.cat(
            [F.linear(x, w, b) for w, b in zip(weights, biases)], dim=-1
        )
    return F.linear(x, weight, bias)


def compatible_linear(x, weight, bias=None, output_chunk_size=0, input_alignment=1):
    # Match the validated patch's explicit CUDA BF16 path. Other dtypes retain
    # their original linear/autocast behavior; there is no GPU-model dispatch.
    if x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        return _shape_linear(x, weight, bias, output_chunk_size, input_alignment)
    return F.linear(x, weight, bias)


class CompatibleLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, *,
                 output_chunk_size=0, input_alignment=1, device=None, dtype=None):
        if output_chunk_size < 0 or input_alignment < 1:
            raise ValueError("output_chunk_size must be >= 0 and input_alignment >= 1")
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.output_chunk_size = output_chunk_size
        self.input_alignment = input_alignment

    def forward(self, x):
        return compatible_linear(
            x, self.weight, self.bias, self.output_chunk_size, self.input_alignment
        )
