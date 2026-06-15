"""Tests for the attention reference implementations (CPU, no GPU needed)."""
import numpy as np
import pytest

from online_softmax import online_softmax, naive_softmax
from flash_attention_numpy import flash_attention, dense_attention


@pytest.mark.parametrize("scale", [1.0, 10.0, 200.0])
@pytest.mark.parametrize("chunk", [1, 3, 8, 100])
def test_online_softmax_matches_naive(scale, chunk):
    rng = np.random.default_rng(int(scale) + chunk)
    x = rng.standard_normal(37) * scale
    a = online_softmax(x, chunk=chunk)
    b = naive_softmax(x)
    assert np.isfinite(a).all()                      # no overflow even at scale=200
    assert np.allclose(a, b, atol=1e-12)
    assert np.isclose(a.sum(), 1.0)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("block", [16, 64, 256])
@pytest.mark.parametrize("N,d", [(50, 16), (200, 32), (129, 8)])
def test_flash_matches_dense(causal, block, N, d):
    rng = np.random.default_rng(int(N * 7 + d + block))
    Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
    a = flash_attention(Q, K, V, block=block, causal=causal)
    b = dense_attention(Q, K, V, causal=causal)
    assert np.allclose(a, b, atol=1e-9), f"max err {np.abs(a-b).max():.2e}"


def test_flash_large_logits_stable():
    # Big magnitudes would overflow a non-stable softmax; flash must stay finite.
    rng = np.random.default_rng(1)
    N, d = 64, 16
    Q = rng.standard_normal((N, d)) * 30.0
    K = rng.standard_normal((N, d)) * 30.0
    V = rng.standard_normal((N, d))
    out = flash_attention(Q, K, V, block=16, causal=True)
    assert np.isfinite(out).all()
