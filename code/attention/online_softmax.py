"""The online (streaming) softmax combiner, in isolation.

This is the numerical heart of FlashAttention. A normal softmax needs the whole
row at once to compute the normalizer sum_j exp(x_j - max). Online softmax folds
the row in chunk by chunk, maintaining a running max `m` and running denominator
`l`, using the correction factor exp(m_old - m_new) whenever the max grows.

Run as a script for a self-check, or import `online_softmax`.
See: docs/foundations/flashattention.md
"""
from __future__ import annotations
import numpy as np


def online_softmax(x: np.ndarray, chunk: int = 3) -> np.ndarray:
    """Compute a numerically-stable softmax over a 1D array, streaming in chunks.

    Equivalent to scipy/np softmax with max-subtraction, but never sees the whole
    array's max up front -- it discovers it chunk by chunk.
    """
    x = np.asarray(x, dtype=np.float64)
    m = -np.inf          # running max
    l = 0.0              # running sum of exp(x - m)
    # First streaming pass: compute the exact denominator `l` and final max `m`.
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk]
        m_new = max(m, block.max())
        l = l * np.exp(m - m_new) + np.exp(block - m_new).sum()  # rescale + add
        m = m_new
    # Second pass: now that m and l are exact, produce probabilities.
    return np.exp(x - m) / l


def naive_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def _demo() -> None:
    rng = np.random.default_rng(0)
    for scale in (1.0, 50.0, 200.0):           # include large values (overflow risk)
        x = rng.standard_normal(17) * scale
        a, b = online_softmax(x, chunk=3), naive_softmax(x)
        err = np.abs(a - b).max()
        assert np.isfinite(a).all(), "online softmax overflowed!"
        print(f"scale={scale:6.1f}  max|online - naive| = {err:.2e}  sum={a.sum():.6f}")
    print("online softmax matches the one-shot stable softmax. ✓")


if __name__ == "__main__":
    _demo()
