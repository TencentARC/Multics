import pytest
import torch

from multics.models.wan_video_dit import (
    WanModel,
    expand_hier_local_hub_seed,
    prepare_hier_local_hub_state_dict_for_load,
)


def _tiny_dit():
    return WanModel(
        dim=128,
        in_dim=16,
        ffn_dim=256,
        out_dim=16,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
    )


def test_shared_mode_keeps_legacy_shape_and_expansion():
    dit = _tiny_dit()
    dit.enable_stage2(
        attention_mode="hier_hub",
        num_hub_tokens=8,
        num_groups=2,
        num_local_hub_tokens=3,
        num_global_hub_tokens=2,
        hier_local_hub_mode="shared",
    )
    assert tuple(dit.local_hub_embed.shape) == (3, 128)

    seed = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128)
    actual = expand_hier_local_hub_seed(
        seed,
        num_frames=2,
        num_groups=2,
        num_local_hub_tokens=3,
        model_dim=128,
        mode="shared",
    )
    legacy = (
        seed[None, None, None, :, :].expand(1, 2, 2, 3, 128).reshape(1, 2 * 2 * 3, 128)
    )
    assert torch.equal(actual, legacy)


def test_per_group_mode_keeps_group_seeds_distinct():
    dit = _tiny_dit()
    dit.enable_stage2(
        attention_mode="hier_hub",
        num_hub_tokens=8,
        num_groups=2,
        num_local_hub_tokens=3,
        num_global_hub_tokens=2,
        hier_local_hub_mode="per_group",
    )
    assert tuple(dit.local_hub_embed.shape) == (2, 3, 128)

    seed = torch.stack((torch.zeros(3, 128), torch.ones(3, 128)))
    expanded = expand_hier_local_hub_seed(
        seed,
        num_frames=2,
        num_groups=2,
        num_local_hub_tokens=3,
        model_dim=128,
        mode="per_group",
    ).reshape(2, 2, 3, 128)
    assert torch.equal(expanded[0, 0], seed[0])
    assert torch.equal(expanded[0, 1], seed[1])
    assert torch.equal(expanded[1, 0], seed[0])
    assert torch.equal(expanded[1, 1], seed[1])


def test_shared_checkpoint_migrates_to_independent_group_copies():
    shared = torch.randn(3, 128)
    migrated, report = prepare_hier_local_hub_state_dict_for_load(
        {"local_hub_embed": shared},
        target_mode="per_group",
        target_num_groups=2,
        target_num_local_hub_tokens=3,
        model_dim=128,
    )
    local = migrated["local_hub_embed"]
    assert tuple(local.shape) == (2, 3, 128)
    assert torch.equal(local[0], shared)
    assert torch.equal(local[1], shared)
    assert local[0].data_ptr() != local[1].data_ptr()
    assert report["source_mode"] == "shared"
    assert report["target_mode"] == "per_group"


def test_per_group_checkpoint_cannot_silently_reduce_to_shared():
    with pytest.raises(RuntimeError, match="automatic group reduction"):
        prepare_hier_local_hub_state_dict_for_load(
            {"local_hub_embed": torch.randn(2, 3, 128)},
            target_mode="shared",
            target_num_groups=2,
            target_num_local_hub_tokens=3,
            model_dim=128,
        )


def test_no_local_rejects_per_group_mode():
    dit = _tiny_dit()
    with pytest.raises(ValueError, match="no local seed"):
        dit.enable_stage2(
            attention_mode="hier_hub",
            num_hub_tokens=2,
            num_groups=2,
            num_local_hub_tokens=0,
            num_global_hub_tokens=2,
            hier_no_local_hub=True,
            hier_local_hub_mode="per_group",
        )
