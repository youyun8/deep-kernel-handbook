"""Grouped GEMM in Triton: many per-expert matmuls under a single launch.

Given tokens already sorted into per-expert contiguous groups (the layout the
MoE dispatch produces), compute, for each expert e, Y_e = X_e @ W_e where all
W_e share shape [K, N]. No padding -- variable group sizes pack back-to-back.

This is the compute core of an efficient MoE FFN. Runs on CUDA or ROCm.
See: docs/moe/kernels.md
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _grouped_gemm_kernel(
        x_ptr, w_ptr, y_ptr,
        tile_expert_ptr,        # [num_m_tiles] expert id for each m-tile
        tile_row0_ptr,          # [num_m_tiles] global start row of the m-tile
        group_end_ptr,          # [E] one-past-last global row of each expert
        N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(tile_expert_ptr + pid_m)
        row0 = tl.load(tile_row0_ptr + pid_m)
        end = tl.load(group_end_ptr + e)

        rm = row0 + tl.arange(0, BM)            # global rows handled by this tile
        rn = pid_n * BN + tl.arange(0, BN)
        row_mask = rm < end                     # don't cross into the next expert
        col_mask = rn < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        w_base = w_ptr + e * K * N              # this expert's weight slab [K, N]
        for k in range(0, K, BK):
            rk = k + tl.arange(0, BK)
            a = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                        mask=row_mask[:, None] & (rk[None, :] < K), other=0.0)
            b = tl.load(w_base + rk[:, None] * N + rn[None, :],
                        mask=(rk[:, None] < K) & col_mask[None, :], other=0.0)
            acc += tl.dot(a, b)                 # Tensor Cores (NV) / MFMA (AMD)
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc,
                 mask=row_mask[:, None] & col_mask[None, :])

    def grouped_gemm(x, group_sizes, weights, BM=64, BN=64, BK=32):
        """x: [T, K] (sorted by expert); group_sizes: [E]; weights: [E, K, N]."""
        T, K = x.shape
        E, K2, N = weights.shape
        assert K == K2 and int(group_sizes.sum()) == T
        y = torch.empty((T, N), device=x.device, dtype=torch.float32)

        # Build the m-tile schedule on host: each expert contributes
        # ceil(size/BM) tiles; record its expert id and global start row.
        sizes = group_sizes.tolist()
        starts, s = [], 0
        for n in sizes:
            starts.append(s); s += n
        ends = [st + n for st, n in zip(starts, sizes)]
        tile_expert, tile_row0 = [], []
        for e, (st, n) in enumerate(zip(starts, sizes)):
            for off in range(0, n, BM):
                tile_expert.append(e)
                tile_row0.append(st + off)
        if not tile_expert:                    # all experts empty
            return y
        dev = x.device
        tile_expert_t = torch.tensor(tile_expert, device=dev, dtype=torch.int32)
        tile_row0_t = torch.tensor(tile_row0, device=dev, dtype=torch.int32)
        group_end_t = torch.tensor(ends, device=dev, dtype=torch.int32)

        grid = (len(tile_expert), triton.cdiv(N, BN))
        _grouped_gemm_kernel[grid](
            x, weights, y, tile_expert_t, tile_row0_t, group_end_t, N, K,
            BM=BM, BN=BN, BK=BK)
        return y


def grouped_gemm_reference(x, group_sizes, weights):
    """Plain PyTorch reference: loop of per-expert matmuls."""
    out = torch.empty((x.shape[0], weights.shape[2]),
                      device=x.device, dtype=torch.float32)
    start = 0
    for e, n in enumerate(group_sizes.tolist()):
        if n:
            out[start:start + n] = x[start:start + n].float() @ weights[e].float()
        start += n
    return out


def _demo():
    if not (HAVE_TRITON and torch.cuda.is_available()):
        print("no GPU/triton; skipping demo")
        return
    torch.manual_seed(0)
    E, K, N = 6, 128, 96
    sizes = torch.tensor([10, 0, 200, 33, 64, 5])
    T = int(sizes.sum())
    x = torch.randn(T, K, device="cuda")
    w = torch.randn(E, K, N, device="cuda")
    y = grouped_gemm(x, sizes, w)
    ref = grouped_gemm_reference(x, sizes, w)
    print("max abs err vs reference:", (y - ref).abs().max().item())


if __name__ == "__main__":
    _demo()
