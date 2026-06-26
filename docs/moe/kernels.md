# MoE kernels (Triton / CUDA / HIP)

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../systems-ep/">systems & EP</a>、<a href="../../performance/triton-track/">Triton</a>、<a href="../../performance/cuda-hip-track/">CUDA1ZZZZX
  <span class="chip"><strong>代碼：</strong> <code>code/kernels/</code> (GPU)</span>
</div>

MoE FFN 的運作時間主要由兩個不規則操作控制：
[systems page](systems-ep.md) 介紹：**排列**（分散/聚集
tokens 分為每個 expert 組）和**分組 GEMM**（許多不同大小的
馬特穆爾斯）。本頁展示如何在**Triton**、**CUDA**中有效地編寫它們
和**ROCm/HIP**— 將 AMD 視為一流目標並標記其中
扭曲/波前寬度、佔用率和 API 不同。

!!! info "「融合 routing」是什麼意思"
    天真地，MoE 的做法是：聚集 (kernel) → 分組 GEMM (kernel) → 分散
    (kernel)，每次都是完整的 HBM 往返。勝利來自**融合**：聚集 \*在 GEMM 的序言中（透過排列索引讀取 tokens，沒有
    單獨收集通行證）並在尾聲中分散。我們為此努力。

## 排列（分散/聚集）

在 routing 之後，每個 token 時隙都有一個目的地 expert。我們需要 tokens
按 expert（分組 GEMM 輸入）連續分組，加上逆映射
將結果分散回來。指數數學（來自
[MoE-from-scratch](moe-from-scratch.md))：`argsort` expert id，收集行
按最終的順序。在 GPU 上，$d$ 寬行的集合是純內存
流量，因此它受頻寬限制 - 目標是合併負載/存儲，並且，
理想情況下，將其熔斷。

###海衛聚集 kernel

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

每個程式複製一個排列後的行。 `BLOCK` 應該是存取權限的倍數
寬度使負載合併。這是獨立的聚集；下面我們將它融合到
GEMM 序言，因此該行被直接讀入 matmul。

## 分組 GEMM

核心操作：對於 experts $e=0..E{-}1$，計算$Y_e = X_e W_e$，其中$X_e$是
tokens 的（變數）區塊路由到 expert $e$。單一 kernel 迭代
`(expert, tile)` 工作項目的時間表，以便所有 experts 共享一次啟動。

### Triton 分組 GEMM（草圖）

一個可運行的、經過測試的版本位於
[`code/kernels/triton_grouped_gemm.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/triton_grouped_gemm.py)。
基本架構：預先計算每個 expert 行偏移量，啟動 1D 網格
展平輸出圖塊，並讓每個程式找出它擁有哪個 expert/圖塊。

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

使其「分組」的兩個想法：**(1)**每個圖塊 `expert_of_tile`
尋找將每個圖塊路由到正確的權重板 `w_ptr + e*K*N`，並且**(2)**
行索引使用 expert 的`group_off`所以可變大小的區塊包
背靠背，無填充。每個拱門自動調諧 `BM/BN/BK` 和 `num_warps`。

!!! tip "將聚集（routing）融合到 GEMM 中"
    將連續行加載 `x_ptr + rm*K` 替換為間接加載
    排列索引 — `src = tl.load(perm_ptr + rm); x_ptr + src*K` — 所以
    kernel 將*未排列的* tokens 直接讀取到 matmul 中。這會刪除
    單獨收集通行證（節省一次完整的 HBM 往返）。做相反的事情
    尾聲以融合分散。這種融合調度就是生產部
    kernels（例如在 SGLang/vLLM、Megatron 中）執行此操作。

## CUDA 和 ROCm/HIP，並排

對於最低級別，這裡是手寫的 kernel 的排列/分散
兩者都有。 HIP 有意與 CUDA**幾乎相同**— 這就是重點
可移植圖層－但*調整*有所不同。完整文件：
[`moe_permute.cu`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/moe_permute.cu)
和
[`moe_permute_hip.cpp`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/moe_permute_hip.cpp)。

=== "CUDA"
`cpp
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
    `
=== "ROCm/HIP"
`cpp
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
    `

### 實際上不同（和咬合）

| 面向           | 英偉達 (CUDA)           | AMD (ROCm/HIP)                                        |
| -------------- | ----------------------- | ----------------------------------------------------- |
| 執行單位       | **扭曲 = 32 車道**      | **波前 = 64 泳道**(CDNA)                              |
| 共享記憶體名稱 | 共享記憶體（SMEM）      | **LDS**（本地資料共享）                               |
| 扭曲洗牌       | `__shfl_sync`（面罩）   | `__shfl`（無掩碼參數）                                |
| 矩陣核心       | 張量核心 (`wmma`/`mma`) | **矩陣核心**(`mfma` / rocWMMA)                        |
| 入住單位       | 每個 SM 的暫存器/SMEM   | 每個**CU**的暫存器/LDS                                |
| 發佈           | `<<<grid,block>>>`      | `hipLaunchkernelGGL(...)`（或透過 hipcc 的 `<<<>>>`） |
| 探查器         | Nsight 計算/系統        | **rocprof / Omniperf**                                |

**波前 = 64**差異是無聲地傷害的差異：減少或
假設 32 通道的隨機寫入在 AMD 上是錯誤的；區塊大小 256 是 8 個扭曲
在 NVIDIA 上有 4 個波前，在 AMD 上有 4 個波前，改變佔用和正確的圖塊尺寸。
始終透過 `warpSize`（兩者內建）進行參數化，而不是硬編碼 32。
對於 matmul，你對應到 `mma`（張量核心）與 `mfma`（矩陣核心）— Triton 和
庫（cuBLAS/hipBLASLt、CUTLASS/Composable kernel）隱藏了這一點，但是
手寫 GEMM 必須針對每一個。請參閱
[CUDA/HIP track](../performance/cuda-hip-track.md) 了解可移植性詳細資訊。

## 測量影響（代表性）

在 H100 上，對於單一 MoE FFN、$E{=}64$、$T{=}8192$ tokens、$d{=}4096$、
$d_{ff}{=}1408$（細粒度），bf16 — _代表數字；重現與
`code/kernels/`_ 中的基準：

| 實作                               | 時間（毫秒） | 加速 | 筆記                        |
| ---------------------------------- | -----------: | ---: | --------------------------- |
| experts (PyTorch) 上的 Python 循環 |        ~12.0 | 1.0× | 發射開銷，參差不齊的批次    |
| 批量 GEMM + 填充 (cf=1.5)          |        〜3.1 | 3.9× | 填充上浪費了約 33% 的 FLOPs |
| Triton 分組 GEMM                   |        〜1.9 | 6.3× | 無填充                      |
| 分組 GEMM + 融合聚集/分散          |        〜1.5 | 8.0× | 節省了一次 HBM 往返行程     |

方法（預熱、CUDA 事件、固定時鐘）位於
[profiling](../performance/profiling.md) 頁面 — 不要相信沒有它的加速。

!!! tip "真實 decode 中的融合"
    [Anatomy of an MoE decode](decode-anatomy.md) 概述了這些選擇
    生產萬億參數模型：_unfused routing_（top-$k$ + 排序拆分
    跨 3 kernels 與 1) 是單一最大的跨堆疊差距，並且*融合了
    將 expert* 共用到路由分組 GEMM 中會刪除約 18% 的 decode latency。

## 要點

- MoE 的熱門操作是**排列**（頻寬限制聚集/分散）和
  **分組 GEMM**（一次啟動下有許多可變大小的 matmul）。
- 分組的 GEMM 將每個輸出 _tile_ 路由至其 expert 的配重板和包
  透過每 expert 行偏移的變數區塊 — 無填充。 -**將聚集/分散融合到 GEMM 序言/尾聲**刪除整個 HBM
  往返－最大的實際勝利。
- CUDA 和 HIP 源幾乎相同，但**調整為波前 = 64、LDS 和
  AMD 上的 `mfma`**；透過 `warpSize` 進行參數化並讓 Triton/庫映射到
  右矩陣核心。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 擴充 Triton 集合 kernel 也可以產生逆排列
   分散，並將其融合成分組 GEMM 尾聲。
2. 採用 CUDA `gather_rows` 並使其與波前無關；基準塊
   無論你擁有哪種 GPU，大小都是 128/256/512，並解釋佔用曲線。
3. 實作填充批量 GEMM 和分組 GEMM；繪圖時間與容量係數
   並找到交叉點。
4. 分析融合與非融合調度並確認 HBM 位元組減少
   探查器的記憶體計數器。

## 參考文獻

- 大風等人。 _巨型區塊。 _ 2022 年。
- 蒂萊特等。 _Triton：用於平鋪神經網路運算的中間語言和編譯器。 _ 2019。
- NVIDIA CUTLASS 分組 GEMM； AMD 可組合 kernel 和 hipBLASLt 文件。
- AMD CDNA3 (MI300) ISA 和 ROCm 程式指南（波前、LDS、MFMA）。
- vLLM / SGLang 融合了 MoE kernel 實作。
