"""GPU/Triton kernel tests. Skipped automatically without a GPU + Triton."""
import pytest
import torch

triton = pytest.importorskip("triton")          # skip whole module if no triton
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("no GPU available", allow_module_level=True)

from softmax_triton import softmax as triton_softmax
from triton_grouped_gemm import grouped_gemm, grouped_gemm_reference


@pytest.mark.parametrize("shape", [(64, 128), (1000, 781), (17, 2048)])
def test_triton_softmax(shape):
    x = torch.randn(*shape, device="cuda")
    ours = triton_softmax(x)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max().item()


@pytest.mark.parametrize("sizes", [
    [10, 0, 200, 33, 64, 5],
    [128, 128, 128],
    [1, 2, 3, 0, 256],
])
def test_grouped_gemm(sizes):
    torch.manual_seed(0)
    sizes_t = torch.tensor(sizes)
    E = len(sizes)
    K, N = 128, 96
    T = int(sizes_t.sum())
    x = torch.randn(T, K, device="cuda")
    w = torch.randn(E, K, N, device="cuda")
    y = grouped_gemm(x, sizes_t, w)
    ref = grouped_gemm_reference(x, sizes_t, w)
    assert torch.allclose(y, ref, atol=2e-2, rtol=1e-2), (y - ref).abs().max().item()
