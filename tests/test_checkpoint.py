from types import SimpleNamespace

import pytest
import torch
from torch import nn

from multics.checkpoint import build_dit, load_complete_state_dict
from multics.models.utils import init_weights_on_device


def test_complete_checkpoint_materializes_exact_weights():
    source = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4)).eval()
    with init_weights_on_device():
        target = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    loaded = load_complete_state_dict(target, source.state_dict(), dtype=torch.float32)
    x = torch.randn(2, 3)
    assert torch.equal(source(x), loaded(x))
    assert not loaded.training
    assert all(not p.is_meta for p in loaded.parameters())


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "not_tensor"])
def test_incomplete_checkpoint_is_rejected_before_loading(problem):
    state = nn.Linear(3, 4).state_dict()
    with init_weights_on_device():
        target = nn.Linear(3, 4)
    if problem == "missing":
        state.pop("weight")
    elif problem == "unexpected":
        state["extra"] = torch.zeros(1)
    elif problem == "shape":
        state["weight"] = torch.zeros(4, 2)
    else:
        state["weight"] = "invalid"
    with pytest.raises(RuntimeError, match="complete Multics DiT checkpoint"):
        load_complete_state_dict(target, state)
    assert all(p.is_meta for p in target.parameters())


def test_full_architecture_needs_no_base_weights_and_scales():
    schemas = []
    for players in (4, 6, 8, 10):
        groups = (players - 1).bit_length()
        args = SimpleNamespace(
            simplex_pool_size=10, simplex_gamma=1.0,
            num_hub_tokens=groups * 5 + 6, num_groups=groups,
            num_local_hub_tokens=5, num_global_hub_tokens=6,
            local_attn_size=21, num_frame_per_block=3,
            attention_mode="hier_hub", hier_direct_schedule="uniform",
            hier_direct_frac=1.0, hier_direct_layers=None, attn_impl="sdpa",
            hier_no_local_hub=False, hier_local_hub_mode="shared",
            use_camera_pose=True, camera_dim=12, camera_dropout_p=0.1,
        )
        model = build_dit(args)
        assert all(p.is_meta for p in model.parameters())
        assert not model.simplex_vertices.is_meta
        assert torch.isfinite(model.simplex_vertices).all()
        assert model.blocks[0].self_attn.cam_q.weight.is_meta
        assert model.blocks[0].action_temporal_attn.cam_q is None
        schemas.append({k: tuple(v.shape) for k, v in model.state_dict().items()})
    assert all(schema == schemas[0] for schema in schemas)
