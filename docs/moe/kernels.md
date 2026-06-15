# MoE kernels (Triton / CUDA / HIP)

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> advanced</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../systems-ep/">systems & EP</a>, <a href="../../performance/triton-track/">Triton</a>, <a href="../../performance/cuda-hip-track/">CUDA/HIP</a></span>
  <span class="chip"><strong>Code:</strong> <code>code/kernels/</code> (GPU)</span>
</div>

The MoE FFN's runtime is dominated by two irregular operations the
[systems page](systems-ep.md) introduced: the **permutation** (scatter/gather
tokens into per-expert groups) and the **grouped GEMM** (many different-sized
matmuls). This page shows how to write them efficiently in **Triton**, **CUDA**,
and **ROCm/HIP** — treating AMD as a first-class target and flagging where
warp/wavefront width, occupancy, and APIs differ.

!!! info "What "fusing routing" means"
    Naively, MoE does: gather (kernel) → grouped GEMM (kernel) → scatter
    (kernel), each a full HBM round-trip. The wins come from **fusing**: gather
    *inside* the GEMM's prologue (read tokens through the permutation index, no
    separate gather pass) and scatter in the epilogue. We build up to that.

## The permutation (scatter/gather)

After routing we have, per token-slot, a destination expert. We need tokens
grouped contiguously by expert (the grouped-GEMM input), plus the inverse map to
scatter results back. The index math (from
[MoE-from-scratch](moe-from-scratch.md)): `argsort` the expert ids, gather rows
by the resulting order. On GPU the gather of $d$-wide rows is pure memory
traffic, so it's bandwidth-bound — the goal is coalesced loads/stores and,
ideally, fusing it away.

### Triton gather kernel

```python
import triton, triton.language as tl

@triton.jit
def gather_rows_kernel(src_ptr, dst_ptr, idx_ptr, n_rows, d,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # one program per output row
    src_row = tl.load(idx_ptr + row)           # which input row to fetch
    cols = tl.arange(0, BLOCK)
    for off in range(0, d, BLOCK):
        m = (off + cols) < d
        x = tl.load(src_ptr + src_row * d + off + cols, mask=m)
        tl.store(dst_ptr + row * d + off + cols, x, mask=m)
```

Each program copies one permuted row. `BLOCK` should be a multiple of the access
width so loads coalesce. This is the standalone gather; below we fuse it into the
GEMM prologue so the row is read straight into the matmul.

## Grouped GEMM

The core op: for experts $e=0..E{-}1$, compute $Y_e = X_e W_e$ where $X_e$ is the
(variable) block of tokens routed to expert $e$. A single kernel iterates over a
schedule of `(expert, tile)` work-items so all experts share one launch.

### Triton grouped GEMM (sketch)

A runnable, tested version is in
[`code/kernels/triton_grouped_gemm.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/triton_grouped_gemm.py).
The essential structure: precompute per-expert row offsets, launch a 1D grid over
flattened output tiles, and have each program look up which expert/tile it owns.

```python
@triton.jit
def grouped_gemm_kernel(
    x_ptr, w_ptr, y_ptr,
    group_off_ptr,            # [E+1] start row of each expert's tokens
    expert_of_tile_ptr,       # which expert each M-tile belongs to
    N, K,                     # W_e is [K, N], same for all experts here
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(expert_of_tile_ptr + pid_m)         # expert for this tile
    row0 = tl.load(group_off_ptr + e)               # first row of expert e
    # local row range within expert e's block:
    rm = row0 + (pid_m * BM - tl.load(group_off_ptr + e)) + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):                        # standard K-loop matmul
        a = tl.load(x_ptr + rm[:, None]*K + (k+tl.arange(0,BK))[None,:])
        b = tl.load(w_ptr + e*K*N + (k+tl.arange(0,BK))[:,None]*N + rn[None,:])
        acc += tl.dot(a, b)                          # tensor-core matmul, fp32 accum
    tl.store(y_ptr + rm[:,None]*N + rn[None,:], acc.to(y_ptr.dtype.element_ty))
```

The two ideas that make it "grouped": **(1)** the per-tile `expert_of_tile`
lookup routes each tile to the right weight slab `w_ptr + e*K*N`, and **(2)** the
row index uses the expert's `group_off` so variable-sized blocks pack
back-to-back with no padding. Autotune `BM/BN/BK` and `num_warps` per arch.

!!! tip "Fusing the gather (routing) into the GEMM"
    Replace the contiguous row load `x_ptr + rm*K` with an indirect load through
    the permutation index — `src = tl.load(perm_ptr + rm); x_ptr + src*K` — so the
    kernel reads *unpermuted* tokens directly into the matmul. That deletes the
    separate gather pass (one full HBM round-trip saved). Do the inverse in the
    epilogue to fuse the scatter. This fused dispatch is what production MoE
    kernels (e.g. in SGLang/vLLM, Megatron) do.

## CUDA and ROCm/HIP, side by side

For the lowest level, here's the permutation/scatter as a hand-written kernel in
both. HIP is intentionally **near-identical** to CUDA — that's the point of the
portability layer — but the *tuning* differs. Full files:
[`moe_permute.cu`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/moe_permute.cu)
and
[`moe_permute_hip.cpp`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/moe_permute_hip.cpp).

=== "CUDA"

    ```cpp
    // Gather token rows into expert-contiguous order. One block per output row.
    __global__ void gather_rows(const float* __restrict__ src,
                                float* __restrict__ dst,
                                const int* __restrict__ row_map, int d) {
        int out_row = blockIdx.x;
        int in_row  = row_map[out_row];
        for (int c = threadIdx.x; c < d; c += blockDim.x)      // coalesced
            dst[out_row * d + c] = src[in_row * d + c];
    }
    // launch: gather_rows<<<n_rows, 256>>>(...);  // warpSize == 32
    ```

=== "ROCm / HIP"

    ```cpp
    #include <hip/hip_runtime.h>
    // Identical logic; compile with hipcc. The source barely changes...
    __global__ void gather_rows(const float* __restrict__ src,
                                float* __restrict__ dst,
                                const int* __restrict__ row_map, int d) {
        int out_row = blockIdx.x;
        int in_row  = row_map[out_row];
        for (int c = threadIdx.x; c < d; c += blockDim.x)
            dst[out_row * d + c] = src[in_row * d + c];
    }
    // launch: hipLaunchKernelGGL(gather_rows, dim3(n_rows), dim3(256), 0, 0, ...);
    // ...but warpSize == 64 on CDNA (MI300). Block-size & LDS tuning must change.
    ```

### What actually differs (and bites)

| Aspect | NVIDIA (CUDA) | AMD (ROCm/HIP) |
|---|---|---|
| Execution unit | **warp = 32 lanes** | **wavefront = 64 lanes** (CDNA) |
| Shared memory name | shared memory (SMEM) | **LDS** (Local Data Share) |
| Warp shuffle | `__shfl_sync` (mask) | `__shfl` (no mask arg) |
| Matrix cores | Tensor Cores (`wmma`/`mma`) | **Matrix Cores** (`mfma` / rocWMMA) |
| Occupancy unit | registers/SMEM per SM | registers/LDS per **CU** |
| Launch | `<<<grid,block>>>` | `hipLaunchKernelGGL(...)` (or `<<<>>>` via hipcc) |
| Profiler | Nsight Compute/Systems | **rocprof / Omniperf** |

The **wavefront = 64** difference is the one that silently hurts: a reduction or
shuffle written assuming 32 lanes is wrong on AMD; a block size of 256 is 8 warps
on NVIDIA but 4 wavefronts on AMD, changing occupancy and the right tile size.
Always parameterize by `warpSize` (a built-in on both) rather than hardcoding 32.
For matmul, you map to `mma` (Tensor Cores) vs `mfma` (Matrix Cores) — Triton and
libraries (cuBLAS/hipBLASLt, CUTLASS/Composable Kernel) hide this, but a
hand-written GEMM must target each. See the
[CUDA/HIP track](../performance/cuda-hip-track.md) for the portability details.

## Measured impact (representative)

On an H100, for a single MoE FFN, $E{=}64$, $T{=}8192$ tokens, $d{=}4096$,
$d_{ff}{=}1408$ (fine-grained), bf16 — *representative numbers; reproduce with the
benchmark in `code/kernels/`*:

| Implementation | Time (ms) | Speedup | Notes |
|---|---:|---:|---|
| Python loop over experts (PyTorch) | ~12.0 | 1.0× | launch overhead, ragged batches |
| Batched GEMM + padding (cf=1.5) | ~3.1 | 3.9× | wastes ~33% FLOPs on padding |
| Triton grouped GEMM | ~1.9 | 6.3× | no padding |
| Grouped GEMM + fused gather/scatter | ~1.5 | 8.0× | one HBM round-trip saved |

The methodology (warmup, CUDA events, fixed clocks) is on the
[profiling](../performance/profiling.md) page — don't trust a speedup without it.

## Key takeaways

- MoE's hot ops are the **permutation** (bandwidth-bound gather/scatter) and the
  **grouped GEMM** (many variable-sized matmuls under one launch).
- A grouped GEMM routes each output *tile* to its expert's weight slab and packs
  variable blocks via per-expert row offsets — no padding.
- **Fusing the gather/scatter into the GEMM prologue/epilogue** removes whole HBM
  round-trips — the biggest practical win.
- CUDA and HIP source is nearly identical, but **tune for wavefront=64, LDS, and
  `mfma`** on AMD; parameterize by `warpSize` and let Triton/libraries map to the
  right matrix cores.

## Exercises

1. Extend the Triton gather kernel to also produce the inverse permutation for
   the scatter, and fuse it into a grouped-GEMM epilogue.
2. Take the CUDA `gather_rows` and make it wavefront-agnostic; benchmark block
   sizes 128/256/512 on whichever GPU you have and explain the occupancy curve.
3. Implement padded batched GEMM and grouped GEMM; plot time vs capacity factor
   and find the crossover.
4. Profile the fused vs unfused dispatch and confirm the HBM-bytes reduction with
   the profiler's memory counters.

## References

- Gale et al. *MegaBlocks.* 2022.
- Tillet et al. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* 2019.
- NVIDIA CUTLASS grouped GEMM; AMD Composable Kernel & hipBLASLt docs.
- AMD CDNA3 (MI300) ISA & ROCm programming guide (wavefront, LDS, MFMA).
- vLLM / SGLang fused MoE kernel implementations.
