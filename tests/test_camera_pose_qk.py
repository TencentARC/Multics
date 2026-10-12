import torch

from multics.models.wan_video_dit import DiTBlock, SelfAttention


def test_camera_token_order_and_hub_zeros():
    block = DiTBlock(has_image_input=False, dim=16, num_heads=4, ffn_dim=32)
    block.num_hub_tokens = 2

    num_players, num_frames, num_patches, cam_dim = 2, 3, 4, 12
    camera_pose = torch.arange(num_players * num_frames * cam_dim, dtype=torch.float32)
    camera_pose = camera_pose.reshape(num_players, num_frames, cam_dim)
    ref_x = torch.zeros(1, num_players * num_frames * num_patches, 16)

    tokens = block._build_camera_tokens(
        camera_pose,
        num_frames=num_frames,
        num_patches=num_patches,
        num_players=num_players,
        ref_x=ref_x,
        include_hub=True,
    )

    expected_agent = (
        camera_pose[:, :, None, :]
        .expand(num_players, num_frames, num_patches, cam_dim)
        .reshape(1, num_players * num_frames * num_patches, cam_dim)
    )
    expected_hub = torch.zeros(1, num_frames * block.num_hub_tokens, cam_dim)

    assert tokens.shape == (1, expected_agent.shape[1] + expected_hub.shape[1], cam_dim)
    assert torch.equal(tokens[:, : expected_agent.shape[1]], expected_agent)
    assert torch.equal(tokens[:, expected_agent.shape[1] :], expected_hub)


def test_zero_init_camera_qk_is_noop():
    torch.manual_seed(0)
    attn = SelfAttention(dim=16, num_heads=4)
    x = torch.randn(1, 5, 16)
    freqs = torch.ones(5, 1, 2, dtype=torch.complex64)

    with torch.no_grad():
        out_before = attn(x, freqs)
        attn.enable_camera_pose_qk(cam_dim=12)
        cam_tokens = torch.randn(1, 5, 12)
        out_after = attn(x, freqs, cam_tokens=cam_tokens)

    assert torch.allclose(out_before, out_after, atol=0, rtol=0)
    assert torch.count_nonzero(attn.cam_q.weight) == 0
    assert torch.count_nonzero(attn.cam_k.weight) == 0
    assert attn.cam_q.bias is None
    assert attn.cam_k.bias is None
    assert attn.cam_dropout_p == 0.1


def test_camera_dropout_is_shared_and_disabled_in_eval():
    torch.manual_seed(0)
    dim = 12
    attn = SelfAttention(dim=dim, num_heads=3)
    attn.enable_camera_pose_qk(cam_dim=dim, dropout_p=0.1)
    with torch.no_grad():
        attn.cam_q.weight.copy_(torch.eye(dim))
        attn.cam_k.weight.copy_(torch.eye(dim))

    base_q = torch.zeros(1, 512, dim)
    base_k = torch.zeros_like(base_q)
    cam_tokens = torch.ones_like(base_q)

    attn.eval()
    eval_q, eval_k = attn._apply_camera_qk(base_q, base_k, cam_tokens)
    assert torch.equal(eval_q, cam_tokens)
    assert torch.equal(eval_k, cam_tokens)

    attn.train()
    torch.manual_seed(1)
    train_q, train_k = attn._apply_camera_qk(base_q, base_k, cam_tokens)
    # Drop camera tokens once before Q/K projection, so both branches see the
    # same dropout mask. A zero camera token remains a strict no-op.
    assert torch.equal(train_q, train_k)
    assert torch.count_nonzero(train_q == 0) > 0
    assert torch.count_nonzero(train_q) > 0
    zero_q, zero_k = attn._apply_camera_qk(base_q, base_k, torch.zeros_like(cam_tokens))
    assert torch.equal(zero_q, base_q)
    assert torch.equal(zero_k, base_k)


def test_camera_dropout_probability_validation():
    attn = SelfAttention(dim=16, num_heads=4)
    for invalid in (-0.1, 1.0):
        try:
            attn.enable_camera_pose_qk(cam_dim=12, dropout_p=invalid)
        except ValueError as exc:
            assert "dropout_p must be in [0,1)" in str(exc)
        else:
            raise AssertionError(f"invalid camera dropout must raise: {invalid}")


def test_legacy_camera_bias_is_the_only_unexpected_adapter_key():
    attn = SelfAttention(dim=16, num_heads=4)
    attn.enable_camera_pose_qk(cam_dim=12, dropout_p=0.1)
    legacy_state = dict(attn.state_dict())
    legacy_state["cam_q.bias"] = torch.randn(16)
    legacy_state["cam_k.bias"] = torch.randn(16)

    missing, unexpected = attn.load_state_dict(legacy_state, strict=False)
    assert missing == []
    assert set(unexpected) == {"cam_q.bias", "cam_k.bias"}


if __name__ == "__main__":
    test_camera_token_order_and_hub_zeros()
    test_zero_init_camera_qk_is_noop()
    test_camera_dropout_is_shared_and_disabled_in_eval()
    test_camera_dropout_probability_validation()
    test_legacy_camera_bias_is_the_only_unexpected_adapter_key()
    print("camera pose Q/K tests passed")
