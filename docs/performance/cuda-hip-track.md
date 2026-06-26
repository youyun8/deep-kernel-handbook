# CUDA / HIP 路線

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../gpu-programming/">GPU 程式設計模型</a>、C++</span>
  <span class="chip"><strong>硬體：</strong> NVIDIA (CUDA) 或 AMD (ROCm/HIP) GPU + 工具鏈</span>
</div>

當你需要 Triton 沒暴露出來的控制權——自訂資料佈局、特定的矩陣核心指令、細粒度的非同步管線——
就改用 CUDA 或 HIP。本路線把**跨 NVIDIA 與 AMD 的可移植性當成首要關注**：原始碼幾乎相同，但
調參不同，我們會一路把這些差異標出來。

## HIP 就是「換了前綴的 CUDA」

AMD 的 HIP 幾乎逐一對映 CUDA API。很多程式碼只要跑 `hipify`（把 `cuda*` 搜尋替換成 `hip*`）再用
`hipcc` 重編就能移植；同一份 `.cpp` 也能靠 `__HIP_PLATFORM_*` 巨集同時鎖定兩個平台。所以
*可移植性*很容易；真正的工作量在*效能可移植性*。

| 概念        | CUDA                          | HIP/ROCm                                               |
| ----------- | ----------------------------- | ------------------------------------------------------ |
| 編譯器      | `nvcc`                        | `hipcc`                                                |
| 啟動        | `kernel<<<g,b,sh,st>>>(...)`  | `hipLaunchKernelGGL(kernel,g,b,sh,st,...)` 或 `<<<>>>` |
| 裝置 malloc | `cudaMalloc`                  | `hipMalloc`                                            |
| 記憶體複製  | `cudaMemcpy`                  | `hipMemcpy`                                            |
| stream      | `cudaStream_t`                | `hipStream_t`                                          |
| 鎖步寬度    | warp = **32**                 | wavefront = **64**                                     |
| 晶片內暫存  | `__shared__`（SMEM）          | `__shared__`（LDS）                                    |
| warp shuffle | `__shfl_down_sync(mask,...)` | `__shfl_down(...)`（無 mask 參數）                     |
| 張量矩陣乘  | Tensor Core：`wmma` / `mma.sync` | 矩陣核心：`__builtin_amdgcn_mfma_*` / rocWMMA       |
| BLAS        | cuBLAS / cuBLASLt             | hipBLAS / hipBLASLt                                    |
| 模板化 GEMM | CUTLASS                       | Composable Kernel（CK）                                |
| profiler    | Nsight Compute/Systems        | rocprof / Omniperf                                     |

## 可移植的 reduction（wavefront 陷阱）

經典錯誤：把 warp 級 reduction 硬編碼成 32 個 lane，在 64 寬的 wavefront 上會悄悄漏掉一半資料。
一律針對 `warpSize` 寫：

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

`warpSize` 在 NVIDIA 上是 32、在 AMD 上是 64，所以這個迴圈會自動跑對的步數。把第一個 offset
硬寫成 `16` 在 AMD 上就是個 bug——典型的可移植性錯誤。

## 共享記憶體分塊 matmul（可移植核心）

教科書式的分塊 GEMM，在兩個平台上原始碼相同，只差在啟動方式與調參。每個 block 把 A、B 的
`TILE×TILE` 子區塊載進晶片內記憶體（SMEM/LDS），在內層迴圈反覆重用——把
[黃金法則](gpu-programming.md)具體化：

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

這個可移植版本是為了*理解*用的。生產上你會改用矩陣核心：NVIDIA 的 `mma`/`wmma`/CUTLASS、
AMD 的 `mfma`/rocWMMA/Composable Kernel——或直接呼叫 cuBLASLt/hipBLASLt。調參*確實*有差別：

- **TILE／block 大小**：16×16 block 在 NVIDIA 上是 8 個 warp、在 AMD 上是 4 個 wavefront → 占用
  率不同；AMD 通常偏好不同的 tile 形狀。
- **LDS 大小與 bank conflict** 要依架構調整。
- **非同步複製／管線**：NVIDIA 的 `cp.async`（與 Hopper TMA）對上 AMD 的非同步 LDS load——同一個
  想法、不同的 intrinsic。

## 非同步管線與矩陣核心（兩者差最多的地方）

效能最高的 GEMM/attention kernel，會在軟體管線中把 global → shared 的複製和矩陣核心數學重疊起來。
*概念*是共通的，但*原語*不同：

- **NVIDIA**：用 `cp.async` 預取 tile、`mma.sync` / `wgmma`（Hopper）做 matmul、TMA 做批量非同步
  複製；重活用 CUTLASS 來搭。
- **AMD（CDNA3/MI300）**：用 `mfma` 指令（例如 16×16×16、32×32×8 等形狀）做 matmul、非同步 LDS
  load 做預取；用 Composable Kernel 來搭。

手寫一個跨廠商的管線化 GEMM 是件硬差事——這正是為什麼多數人用 Triton（自動對映到兩者）或廠商
的 BLASLt 函式庫，只在追最後幾個百分點、或函式庫沒涵蓋的操作時，才下沉到原始 CUDA/HIP。

## 建置並與 PyTorch 整合

把 kernel 包成 PyTorch 擴充，就能從 Python 呼叫：

```python
# setup via torch.utils.cpp_extension; one source compiles for both backends.
from torch.utils.cpp_extension import load
mod = load(name="myk", sources=["myk.cu"],   # hipify handles ROCm builds
           extra_cuda_cflags=["-O3"])
```

在 ROCm 上，PyTorch 的建置會透明地使用 `hipcc` 與 `hipify_torch`——同一份 `.cu` 通常兩邊都能編。
[MoE permutation kernels](../moe/kernels.md) 同時提供 `.cu` 與 `_hip.cpp` 兩種形式，好把那些（細微的）
差異攤開來看。

## 要點

- **HIP ≈ 換名字的 CUDA**；`hipify` + `hipcc` 能移植大部分原始碼。難的是**效能可移植性**，不是
  原始碼可移植性。
- 反覆出現的陷阱：**wavefront 64 vs warp 32**（用 `warpSize`，別硬編碼）、**shuffle 的 mask 參數**、
  **LDS vs SMEM 調參**，以及 **MFMA vs Tensor Core** 的 matmul 路徑。
- 生產 GEMM 用 **CUTLASS / Composable Kernel** 或 **cuBLASLt / hipBLASLt**；只自己手寫函式庫／
  Triton 沒提供的部分。
- 用對的工具做 profiling：**Nsight**（NVIDIA）對 **rocprof/Omniperf**（AMD）。

## 練習

!!! tip "解答"
    參考解答在 [解答頁](../solutions/performance.md)。請先試做每一題，再展開對照。

1. 把分塊 matmul 移植到 HIP，用 `hipcc` 建置（或透過 ROCm 上的 PyTorch），並對照 cuBLAS/hipBLAS
   驗證。
2. 重現第一個 offset 的 bug：寫一個只 reduce 32 個 lane 的版本，示範它在 64 寬 wavefront 上會失敗，
   再用 `warpSize` 修好。
3. 在你的 GPU 上對 `TILE` ∈ {8,16,32} 做 benchmark；把最佳值連到占用率與 warp/wavefront 數。
4. 把內積換成矩陣核心呼叫（`wmma` 或 rocWMMA），量它相對純量內層迴圈的加速。

## 參考文獻

- NVIDIA CUDA C++ Programming Guide；CUTLASS 文件。
- AMD HIP Programming Guide；ROCm `hipify`；Composable Kernel；rocWMMA。
- NVIDIA _cp.async_/TMA 與 CDNA3 _MFMA_ ISA 參考。
