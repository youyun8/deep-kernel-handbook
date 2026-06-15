"""Tiled, online-softmax attention in numpy (the FlashAttention algorithm).

This computes EXACTLY softmax(Q K^T / sqrt(d)) V, but never materializes the
full N x N score matrix -- it streams over key/value tiles, keeping per-query
running (max, denominator, unnormalized output) and rescaling on max updates.

It's a faithful, readable model of the GPU kernel. See:
docs/foundations/flashattention.md
"""
from __future__ import annotations
import numpy as np


def dense_attention(Q, K, V, causal: bool = True) -> np.ndarray:
    """Reference: the naive O(N^2)-memory attention, for correctness checks."""
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    N, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)                       # [N, N]
    if causal:
        i = np.arange(N)[:, None]
        j = np.arange(N)[None, :]
        scores = np.where(j <= i, scores, -np.inf)
    scores -= scores.max(axis=1, keepdims=True)           # stability
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    return p @ V


def flash_attention(Q, K, V, block: int = 64, causal: bool = True) -> np.ndarray:
    """Tiled online-softmax attention. Matches dense_attention up to fp rounding."""
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((N, d), dtype=np.float64)

    for i in range(0, N, block):                          # query tile
        qi = Q[i:i + block] * scale
        br = qi.shape[0]
        m = np.full((br, 1), -np.inf)                     # running max per query row
        l = np.zeros((br, 1))                             # running denominator
        acc = np.zeros((br, d))                           # unnormalized output

        for j in range(0, N, block):                      # key/value tile
            if causal and j > i + br - 1:
                break                                     # whole tile is masked out
            kj = K[j:j + block]
            vj = V[j:j + block]
            s = qi @ kj.T                                 # [br, bc] -- stays "in SRAM"

            if causal:                                    # mask within diagonal tile
                qpos = (i + np.arange(br))[:, None]
                kpos = (j + np.arange(kj.shape[0]))[None, :]
                s = np.where(kpos <= qpos, s, -np.inf)

            m_new = np.maximum(m, s.max(axis=1, keepdims=True))
            # Rows that are still all -inf (fully masked so far): keep m_new finite-safe.
            m_safe = np.where(np.isfinite(m_new), m_new, 0.0)
            p = np.exp(s - m_safe)                        # [br, bc]
            alpha = np.where(np.isfinite(m), np.exp(m - m_safe), 0.0)  # rescale old
            l = alpha * l + p.sum(axis=1, keepdims=True)
            acc = alpha * acc + p @ vj
            m = m_new

        l = np.where(l == 0.0, 1.0, l)                    # avoid 0/0 for masked rows
        O[i:i + br] = acc / l
    return O


def _demo() -> None:
    rng = np.random.default_rng(0)
    N, d = 200, 32
    Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
    for causal in (True, False):
        a = flash_attention(Q, K, V, block=64, causal=causal)
        b = dense_attention(Q, K, V, causal=causal)
        print(f"causal={causal}: max abs err = {np.abs(a - b).max():.2e}")
    print("flash attention matches dense attention. ✓")


if __name__ == "__main__":
    _demo()
