"""Dense-mask tests for hier_group_format=membership (global-only + Anchor)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multics.models.wan_video_dit import (  # noqa: E402
    _slice_hier_active_group_for_cached_mask,
    build_stage2_hier_self_attn_mask,
)


def _tok(P, T, L, N, KL, KG, kind, *, p=0, g=0, t=0, k=0):
    """Token index in [P*T*L | T*N*K_L | T*K_G]."""
    agent = P * T * L
    local = T * N * KL
    if kind == "player":
        return p * T * L + t * L + k
    if kind == "local":
        return agent + t * N * KL + g * KL + k
    if kind == "global":
        return agent + local + t * KG + k
    raise ValueError(kind)


def _mask(
    P,
    T,
    L,
    N,
    KL,
    KG,
    n,
    window,
    labels=None,
    membership=None,
    global_only=None,
    direct=True,
):
    if membership is not None and global_only is None:
        global_only = torch.zeros(T, P, dtype=torch.bool)
    return build_stage2_hier_self_attn_mask(
        P,
        T,
        L,
        N,
        KL,
        KG,
        n,
        window,
        group_labels=labels
        if labels is not None
        else torch.zeros(T, P, dtype=torch.long),
        direct_player=direct,
        device=torch.device("cpu"),
        membership=membership,
        global_only=global_only,
    )


def _one_hot_from_labels(labels, N):
    """labels [T,P] in [0,N) -> membership [T,N,P], global_only all False."""
    T, P = labels.shape
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[torch.arange(T)[:, None], labels, torch.arange(P)[None, :]] = True
    go = torch.zeros(T, P, dtype=torch.bool)
    return mem, go


def test_cached_mask_slices_full_active_group_to_current_block():
    """Mask builder sees [n,N]; cached self-attention may retain full [T,N]."""
    T, N, start, n = 8, 2, 3, 2
    active = torch.tensor(
        [
            [False, False],
            [True, False],
            [True, True],
            [False, True],
            [True, False],
            [False, False],
            [True, True],
            [False, True],
        ],
        dtype=torch.bool,
    )

    current = _slice_hier_active_group_for_cached_mask(active, start, n)
    assert current.shape == (n, N)
    assert torch.equal(current, active[start : start + n])
    assert torch.equal(
        _slice_hier_active_group_for_cached_mask(active[None], start, n),
        current,
    )
    # Labels mode already supplies a current-block slice; do not slice it twice.
    assert torch.equal(
        _slice_hier_active_group_for_cached_mask(current, start, n),
        current,
    )

    try:
        _slice_hier_active_group_for_cached_mask(active, T - 1, n)
    except ValueError as exc:
        assert "does not cover cached block" in str(exc)
    else:
        raise AssertionError("out-of-range cached block must raise")

    # Reproduce the original failure at the real mask-builder boundary: a
    # full-clip active tensor is invalid for an n-frame membership slice, while
    # the helper's current slice is identical to deriving active from membership.
    P, L, KL, KG = 4, 2, 1, 1
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[:, 0, 0] = active[:, 0]
    mem[:, 1, 2] = active[:, 1]
    go = ~mem.any(dim=1)
    mem_cur = mem[start : start + n]
    go_cur = go[start : start + n]
    common = dict(
        num_players=P,
        num_frames=n,
        num_patches=L,
        num_groups=N,
        num_local_hub_tokens=KL,
        num_global_hub_tokens=KG,
        num_frame_per_block=n,
        local_attn_size=7,
        group_labels=torch.zeros(n, P, dtype=torch.long),
        direct_player=True,
        device=torch.device("cpu"),
        membership=mem_cur,
        global_only=go_cur,
    )
    fixed = build_stage2_hier_self_attn_mask(
        **common,
        active_group=current,
    )
    derived = build_stage2_hier_self_attn_mask(
        **common,
        active_group=None,
    )
    assert torch.equal(fixed, derived)
    try:
        build_stage2_hier_self_attn_mask(**common, active_group=active)
    except ValueError as exc:
        assert "active_group must be" in str(exc)
    else:
        raise AssertionError(
            "full active_group must not reach the current-block mask builder"
        )


def test_labels_partition_equals_membership_onehot():
    """A true partition must be bit-identical across the two formats (no player–global)."""
    P, T, L, N, KL, KG, n = 4, 3, 2, 2, 1, 1, 3
    labels = torch.tensor([[0, 0, 1, 1]] * T, dtype=torch.long)  # {p0,p1}+{p2,p3}
    mem, go = _one_hot_from_labels(labels, N)
    m_lab = _mask(P, T, L, N, KL, KG, n, 7, labels=labels)
    m_mem = _mask(P, T, L, N, KL, KG, n, 7, membership=mem, global_only=go)
    assert torch.equal(m_lab, m_mem)
    t = _tok
    # No player–global in a full partition.
    assert not bool(
        m_mem[
            t(P, T, L, N, KL, KG, "player", p=0, t=0),
            t(P, T, L, N, KL, KG, "global", t=0),
        ]
    )
    assert not bool(
        m_mem[
            t(P, T, L, N, KL, KG, "global", t=0),
            t(P, T, L, N, KL, KG, "player", p=0, t=0),
        ]
    )


def test_p4_mixed_global_only_and_empty_slot():
    """g0={p1,p2}, go={p0,p3}, N=2 → local1 inactive; no p0–p1 direct."""
    P, T, L, N, KL, KG, n = 4, 3, 2, 2, 1, 1, 3
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[:, 0, 1] = True
    mem[:, 0, 2] = True
    go = torch.zeros(T, P, dtype=torch.bool)
    go[:, 0] = True
    go[:, 3] = True
    m = _mask(P, T, L, N, KL, KG, n, 7, membership=mem, global_only=go)
    t = lambda *a, **k: _tok(P, T, L, N, KL, KG, *a, **k)

    # p1/p2 ↔ local0, not local1
    assert bool(m[t("player", p=1, t=0), t("local", g=0, t=0)])
    assert not bool(m[t("player", p=1, t=0), t("local", g=1, t=0)])
    assert bool(m[t("local", g=0, t=0), t("player", p=2, t=0)])

    # p0/p3 ↔ global bidirectional, not any local
    assert bool(m[t("player", p=0, t=0), t("global", t=0)])
    assert bool(m[t("global", t=0), t("player", p=0, t=0)])
    assert bool(m[t("player", p=3, t=0), t("global", t=0)])
    assert not bool(m[t("player", p=0, t=0), t("local", g=0, t=0)])
    assert not bool(m[t("player", p=0, t=0), t("local", g=1, t=0)])

    # ordinary members do not talk to global
    assert not bool(m[t("player", p=1, t=0), t("global", t=0)])
    assert not bool(m[t("global", t=0), t("player", p=1, t=0)])

    # no p0–p1 direct (orphan vs grouped)
    assert not bool(m[t("player", p=0, t=0), t("player", p=1, t=0)])
    # p1–p2 same group, same block, direct on
    assert bool(m[t("player", p=1, t=0), t("player", p=2, t=0)])

    # empty local1: no local↔global; inactive query has self-edge
    assert not bool(m[t("local", g=1, t=0), t("global", t=0)])
    assert not bool(m[t("global", t=0), t("local", g=1, t=0)])
    assert bool(m[t("local", g=1, t=0), t("local", g=1, t=0)])
    # local1 must not read players
    assert not bool(m[t("local", g=1, t=0), t("player", p=1, t=0)])


def test_p4_all_global_only_both_slots_inactive():
    P, T, L, N, KL, KG, n = 4, 3, 2, 2, 1, 1, 3
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    go = torch.ones(T, P, dtype=torch.bool)
    m = _mask(P, T, L, N, KL, KG, n, 7, membership=mem, global_only=go)
    t = lambda *a, **k: _tok(P, T, L, N, KL, KG, *a, **k)
    for p in range(P):
        assert bool(m[t("player", p=p, t=0), t("global", t=0)])
        assert bool(m[t("global", t=0), t("player", p=p, t=0)])
        assert not bool(m[t("player", p=p, t=0), t("local", g=0, t=0)])
        assert not bool(m[t("player", p=p, t=0), t("local", g=1, t=0)])
    for g in range(N):
        assert not bool(m[t("local", g=g, t=0), t("global", t=0)])
        assert not bool(m[t("global", t=0), t("local", g=g, t=0)])
        assert bool(m[t("local", g=g, t=0), t("local", g=g, t=0)])


def test_p10_anchor_shares_two_locals_no_cross_member_direct():
    """p0 ∈ {g0,g1}; p1 ∈ g0; p2 ∈ g1. Anchor directs to both members; p1–p2 do not."""
    P, T, L, N, KL, KG, n = 10, 3, 2, 4, 1, 1, 3
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[:, 0, 0] = True
    mem[:, 1, 0] = True
    mem[:, 0, 1] = True
    mem[:, 1, 2] = True
    go = torch.zeros(T, P, dtype=torch.bool)
    go[:, 3] = True
    m = _mask(P, T, L, N, KL, KG, n, 7, membership=mem, global_only=go)
    t = lambda *a, **k: _tok(P, T, L, N, KL, KG, *a, **k)

    assert bool(m[t("player", p=0, t=0), t("local", g=0, t=0)])
    assert bool(m[t("player", p=0, t=0), t("local", g=1, t=0)])
    assert not bool(m[t("player", p=0, t=0), t("local", g=2, t=0)])
    assert bool(m[t("player", p=0, t=0), t("player", p=1, t=0)])
    assert bool(m[t("player", p=0, t=0), t("player", p=2, t=0)])
    assert not bool(m[t("player", p=1, t=0), t("player", p=2, t=0)])
    assert not bool(m[t("player", p=0, t=0), t("global", t=0)])
    assert bool(m[t("player", p=3, t=0), t("global", t=0)])
    assert bool(m[t("global", t=0), t("player", p=3, t=0)])

    # labels fallback (argmax) would drop the g1 bridge for the anchor.
    labels = mem.long().argmax(dim=1)
    m_lab = _mask(P, T, L, N, KL, KG, n, 7, labels=labels)
    assert not bool(m_lab[t("player", p=0, t=0), t("local", g=1, t=0)])
    assert bool(m[t("player", p=0, t=0), t("local", g=1, t=0)])


def test_labels_mode_rejects_negative_ids():
    P, T, L, N = 4, 3, 2, 2
    labels = torch.tensor([[-1, 0, 0, 1]] * T, dtype=torch.long)
    try:
        _mask(P, T, L, N, 1, 1, 3, 7, labels=labels)
    except ValueError as e:
        assert "must be in [0,2)" in str(e)
    else:
        raise AssertionError("expected ValueError for labels containing -1")


def test_refuse_membership_on_scalar_kernel():
    from multics.models.wan_video_dit import _refuse_membership_on_scalar_kernel

    try:
        _refuse_membership_on_scalar_kernel(attn_impl="hier_tiled_ste")
    except ValueError as e:
        assert "hier_tiled_ste" in str(e)
    else:
        raise AssertionError("expected tiled_ste refusal")
    try:
        _refuse_membership_on_scalar_kernel(infer_kernel="tiled128")
    except ValueError as e:
        assert "tiled128" in str(e)
    else:
        raise AssertionError("expected tiled128 refusal")


def test_dataset_membership_loader_temp_pt(tmp_path=None):
    import tempfile

    from multics.data.dataset import PrecomputedLatentDataset

    T, N, P = 6, 2, 4
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[:, 0, 1] = True
    mem[:, 0, 2] = True
    go = torch.zeros(T, P, dtype=torch.bool)
    go[:, 0] = True
    go[:, 3] = True
    eligible = torch.ones(P, dtype=torch.bool)
    labels = torch.tensor([[-1, 0, 0, -1]] * T, dtype=torch.int8)
    td = Path(tempfile.mkdtemp()) if tmp_path is None else Path(tmp_path)
    pt = td / "f000000.pt"
    torch.save(
        {"membership": mem, "global_only": go, "eligible": eligible, "labels": labels},
        pt,
    )
    ds = PrecomputedLatentDataset.__new__(PrecomputedLatentDataset)
    ds.group_paths = [str(pt)]
    ds.hier_group_format = "membership"
    ds.num_groups = N
    ds.num_players = P
    ds.group_label_player_ids = [1, 2, 3, 4]
    ds.num_frame_per_block = 3
    out = ds._load_group_labels(0, T)
    assert "eligible" not in out
    assert out["membership"].shape == (T, N, P)
    assert torch.equal(out["active_group"], mem.any(-1))
    assert int(out["group_labels"][0, 0].item()) == -1
    assert bool(out["global_only"][0, 0].item())

    ds.hier_group_format = "labels"
    try:
        ds._load_group_labels(0, T)
    except ValueError as e:
        assert "out of range" in str(e) or "[0,2)" in str(e)
    else:
        raise AssertionError("labels mode must reject -1")


def test_own_history_always_on():
    P, T, L, N = 4, 3, 2, 2
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    go = torch.ones(T, P, dtype=torch.bool)
    m = _mask(P, T, L, N, 1, 1, 3, 7, membership=mem, global_only=go)
    t0 = _tok(P, T, L, N, 1, 1, "player", p=2, t=0)
    t1 = _tok(P, T, L, N, 1, 1, "player", p=2, t=1)
    assert bool(m[t1, t0])
    assert bool(m[t0, t0])


def test_prepare_membership_allow_full_t_block_or_full_active():
    """Cached path may pass [n,N] active with [T,N,P] membership; prepare must not crash."""
    from multics.models.wan_video_dit import _prepare_hier_membership_tensors

    T, N, P, n = 6, 2, 4, 3
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[:n, 0, 1] = True
    mem[:n, 0, 2] = True
    mem[n:, 1, 0] = True
    mem[n:, 1, 3] = True
    go = torch.zeros(T, P, dtype=torch.bool)
    go[:n, 0] = True
    go[:n, 3] = True
    device = torch.device("cpu")
    full = mem.any(dim=-1)
    mem_f, go_f, act_f = _prepare_hier_membership_tensors(
        mem,
        go,
        full,
        P,
        n,
        N,
        device,
        allow_full_t=True,
    )
    assert mem_f.shape == (T, N, P)
    assert act_f.shape == (T, N)
    assert torch.equal(act_f, full)
    mem_s, _, act_s = _prepare_hier_membership_tensors(
        mem,
        go,
        full[:n],
        P,
        n,
        N,
        device,
        allow_full_t=True,
    )
    assert mem_s.shape == (T, N, P)
    assert act_s.shape == (T, N)
    assert torch.equal(act_s, full)


def _tiny_hier_layer(P, N, L, KL, KG, n_frames, n_block, local_attn, device, dtype):
    from multics.models.wan_video_dit import SelfAttention

    heads, head_dim, dim = 4, 32, 128
    layer = (
        SelfAttention(dim=dim, num_heads=heads).to(device=device, dtype=dtype).eval()
    )
    layer.num_frame_per_block = n_block
    layer.local_attn_size = local_attn
    retention = local_attn + n_frames
    seq_len = P * n_frames * L + n_frames * (N * KL + KG)

    def stream(capacity, tokens_per_frame):
        return {
            "k": torch.zeros(1, capacity, heads, head_dim, dtype=dtype, device=device),
            "v": torch.zeros(1, capacity, heads, head_dim, dtype=dtype, device=device),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }

    agent = [stream(retention * L, L) for _ in range(P)]
    local = [stream(retention * KL, KL) for _ in range(N)]
    for cache in local:
        cache["active"] = torch.zeros(retention * KL, dtype=torch.bool, device=device)
    kv = {
        "hier_hub": True,
        "sparse_hub": True,
        "num_players": P,
        "num_groups": N,
        "num_local_hub_tokens": KL,
        "num_global_hub_tokens": KG,
        "agent_caches": agent,
        "local_hubs": local,
        "global_hub": stream(retention * max(KG, 1), max(KG, 1)),
    }
    freqs = torch.ones(seq_len, 1, head_dim // 2, dtype=torch.complex128, device=device)
    return layer, kv, freqs, seq_len, dim, dtype


def test_membership_segmented_still_refused():
    """Scalar kernels must still fail-fast on membership; labels path is not refused here."""
    import os
    from multics.models.wan_video_dit import SelfAttention  # noqa: F401

    P, N, L, KL, KG = 4, 2, 4, 1, 1
    n_frames, n_block, local_attn = 2, 2, 4
    layer, kv, freqs, seq_len, dim, dtype = _tiny_hier_layer(
        P,
        N,
        L,
        KL,
        KG,
        n_frames,
        n_block,
        local_attn,
        torch.device("cpu"),
        torch.float32,
    )
    T = 4
    mem = torch.zeros(T, N, P, dtype=torch.bool)
    mem[:, 0, 1] = True
    mem[:, 0, 2] = True
    go = torch.zeros(T, P, dtype=torch.bool)
    go[:, 0] = True
    go[:, 3] = True
    x = torch.randn(1, seq_len, dim, dtype=dtype)
    prev = os.environ.get("WAN_HIER_INFER_KERNEL")
    os.environ["WAN_HIER_INFER_KERNEL"] = "segmented"
    try:
        layer.causal_forward_hier_hub(
            x,
            freqs,
            kv,
            current_start=0,
            num_frames=n_frames,
            num_players=P,
            num_hub_tokens=N * KL + KG,
            membership=mem,
            global_only=go,
            active_group=mem.any(-1),
            direct_player=True,
        )
    except ValueError as e:
        assert "segmented" in str(e)
        assert "membership" in str(e)
    else:
        raise AssertionError("membership + segmented must raise")
    finally:
        if prev is None:
            os.environ.pop("WAN_HIER_INFER_KERNEL", None)
        else:
            os.environ["WAN_HIER_INFER_KERNEL"] = prev


def test_membership_sdpa_cached_two_blocks_gpu():
    """membership + sdpa cached: full-clip tensors, block 1 must see key-time history."""
    import os

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    P, N, L, KL, KG = 4, 2, 8, 2, 2
    n_frames, n_block, local_attn = 2, 2, 4
    device = torch.device("cuda")
    dtype = torch.bfloat16
    layer, kv, freqs, seq_len, dim, _ = _tiny_hier_layer(
        P,
        N,
        L,
        KL,
        KG,
        n_frames,
        n_block,
        local_attn,
        device,
        dtype,
    )
    T = 4
    mem = torch.zeros(T, N, P, dtype=torch.bool, device=device)
    mem[:2, 0, 1] = True
    mem[:2, 0, 2] = True
    mem[2:, 0, 1] = True
    mem[2:, 0, 2] = True
    go = torch.zeros(T, P, dtype=torch.bool, device=device)
    go[:, 0] = True
    go[:, 3] = True
    active = mem.any(dim=-1)
    prev = os.environ.get("WAN_HIER_INFER_KERNEL")
    os.environ["WAN_HIER_INFER_KERNEL"] = "sdpa"
    try:
        with torch.inference_mode():
            for block in range(2):
                x = torch.randn(1, seq_len, dim, dtype=dtype, device=device)
                out = layer.causal_forward_hier_hub(
                    x,
                    freqs,
                    kv,
                    current_start=block * n_frames * P * L,
                    num_frames=n_frames,
                    num_players=P,
                    num_hub_tokens=N * KL + KG,
                    membership=mem,
                    global_only=go,
                    active_group=active,
                    direct_player=True,
                )
                assert out.shape == x.shape
                assert torch.isfinite(out).all()
            # Same as the pipeline bug: sliced active_group for block 1.
            kv2 = _tiny_hier_layer(
                P,
                N,
                L,
                KL,
                KG,
                n_frames,
                n_block,
                local_attn,
                device,
                dtype,
            )[1]
            x = torch.randn(1, seq_len, dim, dtype=dtype, device=device)
            out = layer.causal_forward_hier_hub(
                x,
                freqs,
                kv2,
                current_start=n_frames * P * L,
                num_frames=n_frames,
                num_players=P,
                num_hub_tokens=N * KL + KG,
                membership=mem,
                global_only=go,
                active_group=active[2:4],
                direct_player=True,
            )
            assert torch.isfinite(out).all()
    finally:
        if prev is None:
            os.environ.pop("WAN_HIER_INFER_KERNEL", None)
        else:
            os.environ["WAN_HIER_INFER_KERNEL"] = prev


if __name__ == "__main__":
    test_cached_mask_slices_full_active_group_to_current_block()
    test_labels_partition_equals_membership_onehot()
    test_p4_mixed_global_only_and_empty_slot()
    test_p4_all_global_only_both_slots_inactive()
    test_p10_anchor_shares_two_locals_no_cross_member_direct()
    test_labels_mode_rejects_negative_ids()
    test_own_history_always_on()
    test_refuse_membership_on_scalar_kernel()
    test_dataset_membership_loader_temp_pt()
    test_prepare_membership_allow_full_t_block_or_full_active()
    test_membership_segmented_still_refused()
    test_membership_sdpa_cached_two_blocks_gpu()
    print("ok")
