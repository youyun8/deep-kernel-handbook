# CUDA / HIP track

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> advanced</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../gpu-programming/">GPU programming model</a>, C++</span>
  <span class="chip"><strong>Hardware:</strong> NVIDIA (CUDA) or AMD (ROCm/HIP) GPU + toolchain</span>
</div>

When you need control Triton doesn't expose — custom data layouts, specific
matrix-core instructions, fine-grained async pipelines — you drop to CUDA or HIP.
This track treats **portability across NVIDIA and AMD as a first-class concern**:
the source is nearly identical, but the tuning is not, and we make those
differences explicit throughout.

## HIP is "CUDA with a different prefix"

AMD's HIP mirrors the CUDA API almost name-for-name. Much code ports by running
`hipify` (a search-and-replace of `cuda*` → `hip*`) and recompiling with `hipcc`.
The same `.cpp` can target both via the `__HIP_PLATFORM_*` macros. So the
*portability* is easy; the *performance portability* is the work.

| Concept | CUDA | HIP / ROCm |
|---|---|---|
| Compiler | `nvcc` | `hipcc` |
| Launch | `kernel<<<g,b,sh,st>>>(...)` | `hipLaunchKernelGGL(kernel,g,b,sh,st,...)` or `<<<>>>` |
| Device malloc | `cudaMalloc` | `hipMalloc` |
| Memcpy | `cudaMemcpy` | `hipMemcpy` |
| Stream | `cudaStream_t` | `hipStream_t` |
| Lock-step width | warp = **32** | wavefront = **64** |
| On-chip scratch | `__shared__` (SMEM) | `__shared__` (LDS) |
| Warp shuffle | `__shfl_down_sync(mask,...)` | `__shfl_down(...)` (no mask) |
| Tensor matmul | Tensor Cores: `wmma` / `mma.sync` | Matrix Cores: `__builtin_amdgcn_mfma_*` / rocWMMA |
| BLAS | cuBLAS / cuBLASLt | hipBLAS / hipBLASLt |
| Templated GEMM | CUTLASS | Composable Kernel (CK) |
| Profiler | Nsight Compute / Systems | rocprof / Omniperf |

## A portable reduction (the wavefront trap)

The classic bug: a warp-level reduction hardcoded to 32 lanes silently drops half
the data on a 64-wide wavefront. Write it against `warpSize`:

```cpp
// Portable warp/wavefront sum reduction.
__device__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    #if defined(__HIP_PLATFORM_AMD__)
        v += __shfl_down(v, offset);            // HIP: no mask argument
    #else
        v += __shfl_down_sync(0xffffffff, v, offset);   // CUDA: full mask
    #endif
    }
    return v;                                    // lane 0 holds the sum
}
```

`warpSize` is 32 on NVIDIA and 64 on AMD, so the loop runs the right number of
steps automatically. Hardcoding `16` as the first offset would be a correctness
bug on AMD — the canonical portability mistake.

## Shared-memory tiled matmul (portable core)

The textbook tiled GEMM is identical source on both platforms; only the launch
and tuning differ. Each block stages `TILE×TILE` sub-blocks of A and B into
on-chip memory (SMEM/LDS), reused across the inner loop — the
[golden rule](gpu-programming.md) made concrete:

```cpp
#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];     // SMEM on NVIDIA, LDS on AMD
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
            ? A[row*K + t*TILE + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K)
            ? B[(t*TILE+threadIdx.y)*N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];   // reuse from on-chip
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = acc;
}
```

This portable version is for *understanding*. For production you'd use the matrix
cores: `mma`/`wmma`/CUTLASS on NVIDIA, `mfma`/rocWMMA/Composable-Kernel on AMD —
or just call cuBLASLt/hipBLASLt. Tuning that *does* differ:

- **TILE / block size**: a 16×16 block is 8 warps on NVIDIA, 4 wavefronts on AMD
  → different occupancy; AMD often prefers different tile shapes.
- **LDS sizing & bank conflicts** are tuned per architecture.
- **Async copy / pipelining**: NVIDIA `cp.async` (and Hopper TMA) vs AMD's async
  LDS loads — different intrinsics for the same idea.

## Async pipelines and the matrix cores (where they diverge most)

The highest-performance GEMM/attention kernels overlap global→shared copies with
matrix-core math in a software pipeline. The *concept* is shared; the *primitives*
differ:

- **NVIDIA**: `cp.async` to prefetch tiles, `mma.sync` / `wgmma` (Hopper) for
  matmul, TMA for bulk async copies; build with CUTLASS for the heavy lifting.
- **AMD (CDNA3/MI300)**: `mfma` instructions (e.g. 16×16×16, 32×32×8 shapes) for
  matmul, async LDS loads for prefetch; build with Composable Kernel.

A hand-written cross-vendor pipelined GEMM is a large undertaking — which is
exactly why most people use Triton (auto-maps to both) or the vendor BLASLt
libraries, dropping to raw CUDA/HIP only for the last few percent or for ops the
libraries don't cover.

## Building and integrating with PyTorch

Wrap a kernel as a PyTorch extension so it's callable from Python:

```python
# setup via torch.utils.cpp_extension; one source compiles for both backends.
from torch.utils.cpp_extension import load
mod = load(name="myk", sources=["myk.cu"],   # hipify handles ROCm builds
           extra_cuda_cflags=["-O3"])
```

On ROCm, PyTorch's build uses `hipcc` and `hipify_torch` transparently — the same
`.cu` typically compiles for both. The
[MoE permutation kernels](../moe/kernels.md) ship in both `.cu` and `_hip.cpp`
forms to show the (small) differences explicitly.

## Key takeaways

- **HIP ≈ CUDA with renamed APIs**; `hipify` + `hipcc` ports most source. The
  hard part is **performance portability**, not source portability.
- The recurring traps: **wavefront 64 vs warp 32** (use `warpSize`, never
  hardcode), **shuffle mask argument**, **LDS vs SMEM tuning**, and **MFMA vs
  Tensor Core** matmul paths.
- Use **CUTLASS / Composable Kernel** or **cuBLASLt / hipBLASLt** for production
  GEMMs; hand-write only what libraries/Triton can't give you.
- Profile with the right tool: **Nsight** (NVIDIA) vs **rocprof/Omniperf** (AMD).

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/performance.md). Try each exercise before expanding.

1. Port the tiled matmul to HIP, build with `hipcc` (or via PyTorch on ROCm), and
   verify against cuBLAS/hipBLAS.
2. Find the largest first-offset bug: take a 32-lane reduction and demonstrate it
   fails on a 64-wide wavefront; fix with `warpSize`.
3. Benchmark `TILE` ∈ {8,16,32} on your GPU; relate the best value to occupancy
   and warp/wavefront count.
4. Replace the inner product with a matrix-core call (`wmma` or rocWMMA) and
   measure the speedup over the scalar inner loop.

## References

- NVIDIA CUDA C++ Programming Guide; CUTLASS docs.
- AMD HIP Programming Guide; ROCm `hipify`; Composable Kernel; rocWMMA.
- NVIDIA *cp.async*/TMA and CDNA3 *MFMA* ISA references.
