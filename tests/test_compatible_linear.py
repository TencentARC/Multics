"""Checkpoint, gradient, offload, and CUDA parity regression checks."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from multics.models.compatible_linear import CompatibleLinear, _shape_linear
from multics.models.wan_video_dit import WanModel
from multics.vram_management import layers as vram_layers


class CompatibleLinearTests(unittest.TestCase):
    def test_checkpoint_and_cpu_fallback(self):
        for bias in (False, True):
            old = torch.nn.Linear(12, 35, bias=bias).double()
            new = CompatibleLinear(12, 35, bias=bias, input_alignment=16,
                                   output_chunk_size=16).double()
            new.load_state_dict(old.state_dict(), strict=True)
            old.load_state_dict(new.state_dict(), strict=True)
            x = torch.randn(2, 3, 12, dtype=torch.float64)
            torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)
            self.assertEqual(set(new.state_dict()), set(old.state_dict()))

    def test_shaped_forward_and_gradients(self):
        # Check padding, a partial final output chunk, no bias, and both together.
        for chunk, alignment, use_bias in ((16, 1, True), (0, 16, False), (16, 16, True)):
            x = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
            w = torch.randn(35, 12, dtype=torch.float64, requires_grad=True)
            b = torch.randn(35, dtype=torch.float64, requires_grad=True) if use_bias else None
            parameters = (x, w, b) if use_bias else (x, w)
            result = _shape_linear(x, w, b, chunk, alignment)
            reference = F.linear(x, w, b)
            torch.testing.assert_close(result, reference)
            grad = torch.randn_like(result)
            actual_grads = torch.autograd.grad(result, parameters, grad)
            expected_grads = torch.autograd.grad(reference, parameters, grad)
            for actual, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, expected)

    def test_weight_updates_are_not_cached(self):
        layer = CompatibleLinear(12, 35, bias=False, input_alignment=16).double()
        x = torch.ones(1, 12, dtype=torch.float64)
        before = _shape_linear(x, layer.weight, None, input_alignment=16)
        with torch.no_grad():
            layer.weight.add_(1)
        after = _shape_linear(x, layer.weight, None, input_alignment=16)
        torch.testing.assert_close(after - before, torch.full_like(after, 12))

    def test_model_wiring(self):
        model = WanModel(dim=32, in_dim=4, ffn_dim=64, out_dim=4, text_dim=16,
                         freq_dim=8, eps=1e-6, patch_size=(1, 2, 2),
                         num_heads=4, num_layers=1, has_image_input=False)
        model.enable_camera_pose_alignment()
        self.assertEqual(model.time_embedding[2].output_chunk_size, 1024)
        for name in ("cam_q", "cam_k"):
            layer = getattr(model.blocks[0].self_attn, name)
            self.assertEqual(layer.input_alignment, 16)
            self.assertEqual(tuple(layer.weight.shape), (32, 12))

    def test_offload_preserves_compatibility_settings(self):
        for settings in ({"output_chunk_size": 16}, {"input_alignment": 16}):
            original = CompatibleLinear(12, 35, **settings)
            wrapped = vram_layers.AutoWrappedLinear(
                original, offload_dtype=torch.float32, offload_device="cpu",
                onload_dtype=torch.float32, onload_device="cpu",
                computation_dtype=torch.float32, computation_device="cpu", vram_limit=None)
            x = torch.randn(2, 12)
            with patch.object(vram_layers, "compatible_linear",
                              wraps=vram_layers.compatible_linear) as call:
                torch.testing.assert_close(wrapped(x), original(x), rtol=0, atol=0)
                self.assertEqual(call.call_args.args[3:],
                                 (original.output_chunk_size, original.input_alignment))
            wrapped.load_state_dict(original.state_dict(), strict=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_cuda_matches_validated_operations(self):
        # Same operations as the tested runtime patch, now installed in the model.
        for in_dim, out_dim, settings in (
            (3072, 3072, {"output_chunk_size": 1024}),
            (12, 3072, {"input_alignment": 16}),
        ):
            layer = CompatibleLinear(in_dim, out_dim, bias=in_dim != 12, **settings).cuda().bfloat16()
            x = torch.randn(2, 7, in_dim, device="cuda", dtype=torch.bfloat16)
            if in_dim == 12:
                expected = F.linear(F.pad(x, (0, 4)), F.pad(layer.weight, (0, 4)), layer.bias)
            else:
                expected = torch.cat([
                    F.linear(x, w, b) for w, b in zip(
                        layer.weight.split(1024, 0), layer.bias.split(1024, 0))
                ], dim=-1)
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
