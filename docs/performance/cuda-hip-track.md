# CUDA/HIP 軌道

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 高階</span>
  <span class="chip"><strong>先決條件：</strong> <a href="../gpu-programming/">GPU程式設計模型</a>、C++</span>
  <span class="chip"><strong>硬體：</strong> NVIDIA (CUDA) 或 AMD (ROCm/HIP) GPU + 工具鏈</span>
</div>

當你需要控制時，Triton 不會公開 — 自訂資料佈局、特定的
矩陣核心指令、細粒度非同步管道—你可以使用 CUDA 或 HIP。
該賽道將**跨 NVIDIA 和 AMD 的可移植性視為首要關注點**：
來源幾乎相同，但調音不同，我們製作了這些
差異貫穿始終。

## HIP 是“具有不同前綴的 CUDA”

AMD 的 HIP 幾乎完全反映了 CUDA API。透過運行可以移植很多程式碼
`hipify`（`cuda*`→`hip*`的搜尋與取代）並以`hipcc`重新編譯。
相同的 `.cpp` 可以透過 `__HIP_PLATFORM_*` 巨集來定位兩者。所以
*便攜性*很容易； *效能可移植性*就是工作。

| 概念        | CUDA                          | HIP/ROCm                                               |
| ----------- | ----------------------------- | ------------------------------------------------------ |
| 編譯器      | `nvcc`                        | `hipcc`                                                |
| 發佈        | `kernel<<<g,b,sh,st>>>(...)`  | `hipLaunchkernelGGL(kernel,g,b,sh,st,...)` 或 `<<<>>>` |
| 設備 malloc | `cudaMalloc`                  | `hipMalloc`                                            |
| 記憶體複製  | `cudaMemcpy`                  | `hipMemcpy`                                            |
| 流          | `cudaStream_t`                | `hipStream_t`                                          |
| 鎖步寬度    | 扭曲 =**32**                  | 波前 =**64**                                           |
| 片上刮痕    | `__shared__` (SMEM)           | `__shared__` (LDS)                                     |
| 扭曲洗牌    | `__shfl_down_sync(mask,...)`  | `__shfl_down(...)`（無面罩）                           |
| 張量矩陣乘  | 張量核心：`wmma` / `mma.sync` | 矩陣核心：`__builtin_amdgcn_mfma_*` / rocWMMA          |
| 布拉斯      | cuBLAS / cuBLASLt             | 庫布拉斯 hipBLAS/hipBLASLt                             |
| 模板化 GEMM | 彎刀                          | 可組合 kernel (CK)                                     |
| 探查器      | Nsight 計算/系統              | rocprof/Omniperf                                       |

## 便攜式還原（波前陷阱）

經典錯誤：硬編碼為 32 通道的扭曲等級減少默默地下降了一半
64 寬波前的數據。針對 `warpSize` 編寫：

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

`warpSize` 在 NVIDIA 上為 32，在 AMD 上為 64，因此循環運行正確的數量
自動執行步驟。將 `16` 硬編碼為第一個偏移量是正確的
AMD 上的錯誤－典型的可移植性錯誤。

## 共享記憶體平鋪 matmul（可移植核心）

兩個平台上的教科書平鋪 GEMM 的來源相同；僅發射
和調音不同。每個區塊將 A 和 B 的 `TILE×TILE` 子區塊放入
片上記憶體 (SMEM/LDS)，在內循環中重複使用 —
[golden rule](gpu-programming.md)混凝土：

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

這個便攜式版本是為了*理解*。對於生產，你可以使用矩陣
核心：NVIDIA 上的 `mma`/`wmma`/CUTLASS、AMD 上的 `mfma`/rocWMMA/Composable-kernel —
或直接呼叫 cuBLASLt/hipBLASLt。調整*確實*有所不同：

-**TILE/區塊大小**：16×16 區塊在 NVIDIA 上有 8 個扭曲，在 AMD 上有 4 個波前
→ 不同的佔用； AMD 通常喜歡不同的瓷磚形狀。 -**LDS 大小和庫衝突**根據架構進行調整。 -**非同步複製/管線**：NVIDIA `cp.async`（和 Hopper TMA）與 AMD 的非同步
LDS 負載－同一想法不同的內在。

## 非同步管道和矩陣核心（它們分歧最大的地方）

最高效能的 GEMM/attention kernels 重疊全域 → 共用副本
軟體管道中的矩陣核心數學。 *概念*是共享的； _原語_
不同：

-**NVIDIA**：`cp.async` 用於預取圖塊，`mma.sync` / `wgmma`（料斗）用於
matmul、TMA 用於批量非同步複製；使用 CUTLASS 進行建置以應對繁重的工作。 -**AMD (CDNA3/MI300)**：`mfma` 指令（例如 16×16×16、32×32×8 形狀）
matmul，非同步 LDS 載入以進行預取；使用可組合 kernel 進行建置。

手寫的跨供應商管線 GEMM 是一項艱鉅的任務 - 這是
到底為什麼大多數人使用 Triton（自動映射到兩者）或供應商 BLASLt
庫，僅在最後百分之幾或操作中下降到原始 CUDA/HIP
圖書館不涵蓋。

## 建置並與 PyTorch 集成

將 kernel 包裝為 PyTorch 擴展，以便可以從 Python 呼叫：

```python
# setup via torch.utils.cpp_extension; one source compiles for both backends.
from torch.utils.cpp_extension import load
mod = load(name="myk", sources=["myk.cu"],   # hipify handles ROCm builds
           extra_cuda_cflags=["-O3"])
```

在 ROCm 上，PyTorch 的建造透明地使用 `hipcc` 和 `hipify_torch` — 相同
`.cu` 通常針對兩者進行編譯。的
[MoE permutation kernels](../moe/kernels.md) 以 `.cu` 和 `_hip.cpp` 形式出貨
形式來明確顯示（小的）差異。

## 要點

-**HIP ≈ CUDA 並重新命名 API**； `hipify` + `hipcc` 移植大部分來源。的
困難的部分是**效能可移植性**，而不是源可移植性。

- 重複出現的陷阱：**波前 64 與扭曲 32**（使用 `warpSize`，從不
  硬編碼），**洗牌掩碼參數**，**LDS 與 SMEM 調整**，以及**MFMA 與
  Tensor Core**matmul 路徑。
- 使用**CUTLASS / 可組合 kernel**或**cuBLASLt / hipBLASLt**進行生產
  GEMM；隻手寫圖書館/Triton 無法提供的內容。
- 使用正確的工具進行設定：**Nsight**(NVIDIA) 與**rocprof/Omniperf**(AMD)。

## 練習

!!! tip "解決方案"
參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 將平鋪 matmul 移植到 HIP，使用 `hipcc` 建置（或透過 ROCm 上的 PyTorch），以及
   針對 cuBLAS/hipBLAS 進行驗證。
2. 找出最大的第一個偏移 bug：減少 32 車道並進行示範
   在 64 寬波前上失敗；用 `warpSize` 修復。
3. 在你的 GPU 上對 `TILE` ∈ {8,16,32} 進行基準測試；將最佳價值與入住率連結起來
   和扭曲/波前計數。
4. 將內積替換為矩陣核心呼叫（`wmma` 或 rocWMMA）並
   測量標量內循環的加速比。

## 參考文獻

- NVIDIA CUDA C++ 程式設計指南；彎刀文件。
- AMD HIP 程式指南； ROCm `hipify`;可組合 kernel； rocWMMA。
- NVIDIA _cp.async_/TMA 和 CDNA3 _MFMA_ ISA 參考。
