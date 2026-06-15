# Triton track

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../gpu-programming/">GPU programming model</a></span>
  <span class="chip"><strong>Code:</strong> <code>code/kernels/</code> (needs a GPU)</span>
</div>

Triton lets you write GPU kernels in Python that compile to near-peak machine
code, while it handles the painful parts (coalescing, SMEM staging, vectorization,
much of the scheduling). You think in **tiles** (blocks of a tensor) rather than
individual threads. This track builds from a trivial kernel up to fused softmax
and matmul, then points at the attention/MoE kernels elsewhere in the handbook.

!!! info "Why Triton first"
    For most kernels Triton reaches 80–95% of hand-tuned CUDA performance at a
    fraction of the effort, and the *same source runs on AMD* (Triton has a ROCm
    backend mapping to wavefront-64 and MFMA). Drop to [CUDA/HIP](cuda-hip-track.md)
    only when you need control Triton doesn't expose.

## The Triton mental model

A Triton kernel is written from the perspective of **one program instance**
(roughly, one block). You:

1. Get your program id (`tl.program_id`) → which tile you own.
2. Compute the offsets of that tile (`tl.arange`).
3. `tl.load` the tile from HBM (with a mask for boundaries) into on-chip memory.
4. Compute on it (`tl.dot`, elementwise, `tl.max`/`tl.sum` reductions).
5. `tl.store` the result back.

Triton vectorizes within the tile and manages SMEM/registers for you.

## Level 1 — vector add

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # this program's slice
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    return out
```

This is memory-bound (3 bytes moved per 1 FLOP) — its only job is to teach the
load/compute/store pattern and masking.

## Level 2 — fused softmax (the first real win)

A naive softmax over rows reads the row, finds the max, reads again to exponentiate
and sum, reads again to divide — multiple HBM passes. A fused Triton kernel loads
each row **once** into SRAM and does max/exp/sum/divide there:

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)            # stability: subtract max (see numerics)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)      # all in SRAM, one HBM read + one write
    tl.store(out_ptr + row * row_stride + cols, out, mask=mask)
```

This is the [online-softmax / FlashAttention](../foundations/flashattention.md)
idea in miniature: collapse multiple memory passes into one by keeping the data
on chip. A runnable, PyTorch-checked version is in
[`code/kernels/softmax_triton.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/softmax_triton.py).

## Level 3 — matmul with tiling and autotuning

The canonical Triton matmul tiles $C=AB$ into `BM×BN` output blocks, each
accumulating over `BK` chunks of the K dimension in fp32:

```python
@triton.autotune(
    configs=[
        triton.Config({"BM":128,"BN":128,"BK":32}, num_warps=4, num_stages=3),
        triton.Config({"BM":128,"BN":256,"BK":32}, num_warps=8, num_stages=3),
        # add AMD-friendly configs (num_warps maps to wavefronts) ...
    ], key=["M","N","K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)       # fp32 accumulate
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:,None]*K + rk[None,:], mask=rm[:,None]<M)
        b = tl.load(b_ptr + rk[:,None]*N + rn[None,:], mask=rn[None,:]<N)
        acc += tl.dot(a, b)                           # maps to Tensor/Matrix cores
    tl.store(c_ptr + rm[:,None]*N + rn[None,:], acc, mask=(rm[:,None]<M)&(rn[None,:]<N))
```

`tl.dot` lowers to Tensor Cores on NVIDIA and **MFMA Matrix Cores on AMD**
automatically. `@triton.autotune` searches tile sizes and `num_warps` per shape —
**re-autotune on AMD**, since wavefront-64 shifts the optimal configs. This matmul
is the backbone of the [MoE grouped GEMM](../moe/kernels.md), which is "this
kernel, but each tile picks its expert's weight slab."

## Level 4 — attention and grouped GEMM

With softmax + matmul understood, the [FlashAttention](../foundations/flashattention.md)
kernel is "matmul $QK^\top$ in tiles, online-softmax the scores in SRAM, matmul by
$V$, never write the score matrix." And the [MoE grouped GEMM](../moe/kernels.md)
is "the Level-3 matmul with a per-tile expert lookup and packed variable-sized
row blocks, with the gather fused into the prologue." Those pages contain the full
kernels; you now have the vocabulary to read them.

## Practical tips

- **Always check against PyTorch** with `torch.allclose` before benchmarking — a
  fast wrong kernel is worthless. Our `code/kernels` tests do this.
- **`num_warps`/`num_stages`** are your main throughput knobs; autotune them.
- **Mask everything at boundaries** or you'll read/write out of bounds.
- **Benchmark with `triton.testing.do_bench`** (it handles warmup + CUDA events) —
  see [profiling](profiling.md).
- **On AMD**: confirm the ROCm Triton backend is installed; re-run autotuning;
  expect different best configs due to wavefront width and LDS sizing.

## Key takeaways

- Triton kernels are written per-**tile**: get program id → compute offsets →
  `tl.load` → compute (`tl.dot`, reductions) → `tl.store`, with masks.
- Fusing multiple HBM passes into one on-chip pass (softmax, attention) is the
  core win — the roofline playbook again.
- `tl.dot` targets Tensor Cores / MFMA automatically; **autotune per
  architecture** because wavefront-64 changes the optimal tiles on AMD.
- Triton gets you most of CUDA's performance portably; reach for
  [CUDA/HIP](cuda-hip-track.md) only when you must.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/performance.md). Try each exercise before expanding.

1. Run the vector-add and softmax kernels; verify against PyTorch and benchmark
   vs the native ops.
2. Add AMD-oriented autotune configs to the matmul (try `num_warps` 4/8 with
   wavefront-64 in mind) and compare best configs across GPUs you have.
3. Extend the softmax kernel to handle rows wider than one `BLOCK` using the
   online-softmax combiner.
4. Profile the fused softmax vs three-pass softmax and explain the bytes-moved
   difference.

## References

- Tillet, Kung, Cox. *Triton.* 2019; official Triton tutorials.
- Dao et al. *FlashAttention* (the kernel this builds toward). 2022.
- AMD ROCm Triton backend documentation.
