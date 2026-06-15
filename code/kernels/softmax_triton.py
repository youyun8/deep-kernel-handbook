"""Fused row-softmax Triton kernel (one HBM read + one write per row).

Runs on NVIDIA (CUDA) or AMD (ROCm) GPUs. On AMD recall wavefront=64; the
autotuner/num_warps choice differs -- re-tune for best performance.

See: docs/performance/triton-track.md
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                       # allow import on CPU-only boxes
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        ptrs = x_ptr + row * row_stride + cols
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)                  # numerical stability
        num = tl.exp(x)
        out = num / tl.sum(num, axis=0)            # all on-chip
        tl.store(out_ptr + row * row_stride + cols, out, mask=mask)

    def softmax(x: torch.Tensor) -> torch.Tensor:
        """Row-wise softmax of a 2D tensor using the fused kernel."""
        assert x.dim() == 2 and x.is_cuda
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        block = triton.next_power_of_2(n_cols)
        _softmax_kernel[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK=block)
        return out


def _demo():
    if not (HAVE_TRITON and torch.cuda.is_available()):
        print("no GPU/triton; skipping demo")
        return
    x = torch.randn(1024, 781, device="cuda")
    ours = softmax(x)
    ref = torch.softmax(x, dim=-1)
    print("max abs err vs torch.softmax:", (ours - ref).abs().max().item())


if __name__ == "__main__":
    _demo()
