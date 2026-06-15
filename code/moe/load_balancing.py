"""MoE load-balancing utilities: aux loss, capacity/drop, aux-loss-free bias.

See: docs/moe/load-balancing.md and docs/moe/training-stability.md
"""
from __future__ import annotations
import torch


def switch_aux_loss(router_probs: torch.Tensor, topk_idx: torch.Tensor,
                    n_experts: int, alpha: float = 1e-2) -> torch.Tensor:
    """Switch-Transformer load-balancing loss: alpha * E * sum_e f_e * P_e.

    router_probs: [T, E] softmax probabilities; topk_idx: [T, k] selected experts.
    Minimized when both the selection fraction f and mean prob P are uniform.
    """
    T = router_probs.shape[0]
    P = router_probs.float().mean(dim=0)                       # [E]
    one_hot = torch.zeros(T, n_experts, device=router_probs.device)
    one_hot.scatter_(1, topk_idx, 1.0)
    f = one_hot.sum(dim=0) / T                                 # [E]
    return alpha * n_experts * torch.sum(f * P)


def router_z_loss(logits: torch.Tensor, beta: float = 1e-3) -> torch.Tensor:
    """Penalize large router logits for stability: beta * mean((logsumexp x)^2)."""
    lse = torch.logsumexp(logits.float(), dim=-1)             # [T]
    return beta * (lse ** 2).mean()


def apply_capacity(topk_idx: torch.Tensor, n_experts: int, capacity: int):
    """Boolean keep-mask [T, k]; drops assignments beyond `capacity` per expert."""
    keep = torch.ones_like(topk_idx, dtype=torch.bool)
    for e in range(n_experts):
        pos = (topk_idx == e).nonzero(as_tuple=False)         # [n_e, 2]
        if pos.shape[0] > capacity:
            drop = pos[capacity:]
            keep[drop[:, 0], drop[:, 1]] = False
    return keep


def capacity_from_factor(n_tokens: int, top_k: int, n_experts: int,
                         factor: float = 1.25) -> int:
    """C = ceil(factor * k * T / E)."""
    import math
    return math.ceil(factor * top_k * n_tokens / n_experts)


@torch.no_grad()
def update_router_bias(bias: torch.Tensor, counts: torch.Tensor,
                       gamma: float = 1e-2) -> torch.Tensor:
    """Aux-loss-free controller: nudge bias up for under-loaded experts.

    bias: [E] running bias (modified in place); counts: [E] tokens this step.
    """
    target = counts.float().mean()
    bias += gamma * torch.sign(target - counts.float())       # raise under-loaded
    return bias


def load_stats(counts: torch.Tensor) -> dict:
    """Balance diagnostics from per-expert token counts."""
    c = counts.float()
    mean = c.mean().clamp_min(1e-9)
    p = (c / c.sum().clamp_min(1e-9)).clamp_min(1e-12)
    entropy = -(p * p.log()).sum()
    return {
        "load_cv": (c.std(unbiased=False) / mean).item(),     # 0 = perfectly balanced
        "max_over_mean": (c.max() / mean).item(),
        "entropy": entropy.item(),
        "max_entropy": torch.log(torch.tensor(float(len(c)))).item(),
        "dead_experts": int((c == 0).sum().item()),
    }
