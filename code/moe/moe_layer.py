"""Mixture-of-Experts layer, from scratch, in two equivalent forms.

- `MoELayer` with `impl="naive"`: readable loop over experts with masking.
- `MoELayer` with `impl="dispatch"`: permute tokens into per-expert contiguous
  groups (the layout grouped-GEMM kernels and expert parallelism want), run, then
  scatter-add back.

Both compute the SAME function; test_moe.py asserts they agree. CPU-friendly.
See: docs/moe/moe-from-scratch.md
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """SwiGLU feed-forward network -- the standard MoE expert."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def compute_gates(logits: torch.Tensor, top_k: int, gate: str):
    """Return (topk_idx [T,k], topk_weight [T,k], full_scores [T,E]).

    Router math is done in fp32 for stability (see docs/moe/training-stability).
    """
    logits = logits.float()
    if gate == "softmax":
        scores = logits.softmax(dim=-1)
    elif gate == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError(f"unknown gate {gate!r}")
    topw, topi = scores.topk(top_k, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # renormalize
    return topi, topw.to(logits.dtype), scores


def _run_naive(x, experts, topi, topw):
    """Loop over experts, each running only on the tokens routed to it."""
    y = torch.zeros_like(x)
    n_experts = len(experts)
    for e in range(n_experts):
        mask = topi == e                          # [T, k]
        tok, slot = mask.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        out = experts[e](x[tok])                  # [n_e, d]
        y.index_add_(0, tok, out * topw[tok, slot, None])
    return y


def _run_dispatch(x, experts, topi, topw):
    """Permute tokens into per-expert contiguous blocks, run, scatter back."""
    T, k = topi.shape
    n_experts = len(experts)
    flat_expert = topi.reshape(-1)                          # [T*k]
    flat_weight = topw.reshape(-1, 1)                       # [T*k, 1]
    flat_token = torch.arange(T, device=x.device).repeat_interleave(k)

    order = torch.argsort(flat_expert, stable=True)        # group by expert
    sorted_expert = flat_expert[order]
    sorted_token = flat_token[order]
    counts = torch.bincount(sorted_expert, minlength=n_experts)

    x_sorted = x[sorted_token]                              # gather, contiguous
    out_sorted = torch.empty_like(x_sorted)
    start = 0
    for e in range(n_experts):                             # each block is contiguous
        n = int(counts[e])
        if n:
            out_sorted[start:start + n] = experts[e](x_sorted[start:start + n])
        start += n

    out_sorted = out_sorted * flat_weight[order]           # apply gate weight
    y = torch.zeros_like(x)
    y.index_add_(0, sorted_token, out_sorted)              # combine (scatter-add)
    return y


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2, gate="softmax",
                 shared=False, impl="dispatch"):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.shared = Expert(d_model, d_ff) if shared else None
        self.n_experts, self.top_k, self.gate, self.impl = n_experts, top_k, gate, impl
        # Aux-loss-free balancing bias (a running statistic, not a Parameter).
        self.register_buffer("router_bias", torch.zeros(n_experts))
        self.last_counts = None      # tokens/expert this step (for the controller)

    def forward(self, x):
        """x: [..., d_model]. Returns the same shape."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])                        # [T, d]
        logits = self.router(x) + self.router_bias          # bias affects selection
        topi, topw, scores = compute_gates(logits, self.top_k, self.gate)

        # Record per-expert load for the balancing controller / metrics.
        with torch.no_grad():
            self.last_counts = torch.bincount(
                topi.reshape(-1), minlength=self.n_experts).float()
        self.last_scores = scores                          # for aux-loss if used

        run = _run_naive if self.impl == "naive" else _run_dispatch
        y = run(x, self.experts, topi, topw)
        if self.shared is not None:
            y = y + self.shared(x)                          # always-on shared expert
        return y.reshape(shape)
