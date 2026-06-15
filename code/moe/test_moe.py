"""Tests for the MoE layer and load-balancing utilities (CPU, no GPU needed)."""
import torch
import pytest

from moe_layer import MoELayer, Expert, compute_gates
from load_balancing import (
    switch_aux_loss, router_z_loss, apply_capacity, capacity_from_factor,
    update_router_bias, load_stats,
)

torch.manual_seed(0)


@pytest.mark.parametrize("gate", ["softmax", "sigmoid"])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_naive_equals_dispatch(gate, top_k):
    """The readable loop and the permute/dispatch form must compute the same thing."""
    d_model, d_ff, E, T = 32, 64, 8, 200
    torch.manual_seed(top_k + (0 if gate == "softmax" else 100))
    naive = MoELayer(d_model, d_ff, E, top_k, gate=gate, impl="naive")
    disp = MoELayer(d_model, d_ff, E, top_k, gate=gate, impl="dispatch")
    disp.load_state_dict(naive.state_dict())                # identical weights

    x = torch.randn(T, d_model)
    with torch.no_grad():
        yn, yd = naive(x), disp(x)
    assert torch.allclose(yn, yd, atol=1e-5), (yn - yd).abs().max().item()


def test_single_expert_reduces_to_ffn():
    """With E=1, top_k=1 and no renorm effect, MoE == its single expert FFN."""
    d_model, d_ff = 16, 32
    moe = MoELayer(d_model, d_ff, n_experts=1, top_k=1, gate="softmax", impl="dispatch")
    x = torch.randn(50, d_model)
    with torch.no_grad():
        y = moe(x)
        ref = moe.experts[0](x)                            # gate weight renormalizes to 1
    assert torch.allclose(y, ref, atol=1e-5)


def test_shared_expert_adds_path():
    d_model, d_ff = 16, 32
    moe = MoELayer(d_model, d_ff, 4, 2, shared=True, impl="dispatch")
    x = torch.randn(20, d_model)
    y = moe(x)
    assert y.shape == x.shape


def test_gates_normalize():
    logits = torch.randn(64, 8)
    for gate in ("softmax", "sigmoid"):
        topi, topw, scores = compute_gates(logits, top_k=2, gate=gate)
        assert torch.allclose(topw.sum(-1), torch.ones(64), atol=1e-5)
        assert topi.shape == (64, 2)


def test_aux_loss_minimized_at_uniform():
    """Uniform routing should give a lower aux loss than a collapsed router."""
    T, E = 256, 8
    uniform = torch.full((T, E), 1.0 / E)
    collapsed = torch.zeros(T, E); collapsed[:, 0] = 1.0
    idx_u = torch.randint(0, E, (T, 2))
    idx_c = torch.zeros(T, 2, dtype=torch.long)
    lu = switch_aux_loss(uniform, idx_u, E)
    lc = switch_aux_loss(collapsed, idx_c, E)
    assert lu < lc


def test_z_loss_penalizes_large_logits():
    small = torch.randn(100, 8) * 0.1
    large = torch.randn(100, 8) * 10.0
    assert router_z_loss(small) < router_z_loss(large)


def test_capacity_drops_overflow():
    # 10 tokens all routed to expert 0, capacity 4 -> 6 dropped.
    topk_idx = torch.zeros(10, 1, dtype=torch.long)
    keep = apply_capacity(topk_idx, n_experts=4, capacity=4)
    assert keep.sum().item() == 4
    assert capacity_from_factor(4096, top_k=2, n_experts=64, factor=1.25) == 160


def test_bias_controller_balances_skewed_router():
    """The aux-loss-free bias controller should drive a skewed router toward balance."""
    E = 8
    torch.manual_seed(0)
    # A deterministic router that strongly prefers expert 0 (skewed logits).
    base_logits = torch.zeros(E); base_logits[0] = 5.0
    bias = torch.zeros(E)
    cv_start = cv_end = None
    for step in range(400):
        # Each "token" scores experts as base + bias + small noise; pick top-1.
        logits = base_logits + bias + 0.5 * torch.randn(512, E)
        sel = logits.topk(1, dim=-1).indices
        counts = torch.bincount(sel.reshape(-1), minlength=E).float()
        if step == 0:
            cv_start = load_stats(counts)["load_cv"]
        update_router_bias(bias, counts, gamma=0.05)
        cv_end = load_stats(counts)["load_cv"]
    assert cv_end < cv_start * 0.6, f"cv {cv_start:.2f} -> {cv_end:.2f} (no improvement)"
