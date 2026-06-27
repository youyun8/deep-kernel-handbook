# MoE kernels (Triton / CUDA / HIP)

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../systems-ep/">systems & EP</a>、<a href="../../performance/triton-track/">Triton</a>、<a href="../../performance/cuda-hip-track/">CUDA/HIP</a></span>
  <span class="chip"><strong>代碼：</strong> <code>code/kernels/</code> (GPU)</span>
</div>

MoE FFN 的執行時間主要由兩個不規則操作主導（[systems page](systems-ep.md) 有介紹）：**permute**（scatter/gather，把 tokens 依其目的 expert 分組） 與 **grouped GEMM**（一次處理許多大小不一的 matmul）。本頁展示如何 在 **Triton**、**CUDA** 與 **ROCm/HIP** 上有效地撰寫它們 — 將 AMD 視為 一流目標，並標記出 warp/wavefront 寬度、occupancy 與 API 的差異。

!!! Info "「融合 routing」是什麼意思"
    最樸素的 MoE 做法是：gather (kernel) → grouped GEMM (kernel) → scatter (kernel)，每一步都是一次完整的 HBM 往返。勝利來自 **fusion**：把 gather 併進 GEMM 的 prologue（透過 permutation 索引直接讀入 tokens，省去單獨的 gather pass），並把 scatter 併進 epilogue。本頁的目標即在此。

## Permute（scatter/gather）

Routing 之後，每個 token slot 都有一個目的 expert。我們需要把 tokens 依 expert 連續分組（grouped GEMM 的輸入），外加一個逆映射把結果 scatter 回去。 索引數學（來自 [MoE-from-scratch](moe-from-scratch.md)）：對 expert id 做 `argsort`，再依最終順序 gather 行。在 GPU 上，gather $d$-寬的行是純粹的 記憶體流量，因此它是 bandwidth-bound — 目標是讓 load/store coalesced，並在理想 情況下將其 fuse 掉。

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

每個 program 複製一個 permute 後的行。`BLOCK` 應為存取寬度的倍數，使 load 得以 coalesced。這是獨立的 gather；下面我們會把它 fuse 進 GEMM 的 prologue， 讓該行直接被讀進 matmul。

## Grouped GEMM

核心操作：對 experts $e=0..E{-}1$，計算 $Y_e = X_e W_e$，其中 $X_e$ 是 routing 到 expert $e$ 的 tokens 之（可變大小）區塊。單一 kernel 迭代一個 `(expert, tile)` 工作項目的排程表，讓所有 experts 共享一次 launch。

### Grouped GEMM 的 FLOPs

以 handbook 的 AITER 章節（真實的 Kimi K2.5 MXFP4 profile）作為貫穿全文的具體 範例。符號定義：$H$ 為 hidden size；$I$ 為每個 partition 的 MoE intermediate size，gate+up 的輸出寬度為 $2I$；$E$ 為 routed experts 數；$k$ 為 top-$k$； $m_e$ 為 expert $e$ 實際處理的 token-row 數。參考值：$H=7168$、$I=256$、 $2I=512$、$E=384$（另含 1 個 fused shared expert）、$k=8$；MXFP4 FP4 權重為 $0.5$ byte/element。

每個 expert 的 grouped GEMM 分兩個 stage。stage-1（gate/up，權重形狀 $H\times 2I$）對 $m_e$ 行的成本為

$$
\mathrm{FLOP}_1 = 2\,m_e\,H\,(2I),
$$

Stage-2（down，權重形狀 $I\times H$）的成本為

$$
\mathrm{FLOP}_2 = 2\,m_e\,I\,H .
$$

兩者比值

$$
\frac{\mathrm{FLOP}_1}{\mathrm{FLOP}_2} = \frac{2\,m_e\,H\,(2I)}{2\,m_e\,I\,H}
= \frac{2I}{I} = 2,
$$

即 stage-1 在結構上是 stage-2 的 $2\times$（與 profile 量測到的 ~2.05× 相符； 偏差來自 activation/scale 處理與 tile 量化）。代入數字（per row）：

$$
\mathrm{FLOP}_1/\text{row} = 2\cdot 7168 \cdot 512 = 7.34\ \text{MFLOP},\qquad
\mathrm{FLOP}_2/\text{row} = 2\cdot 256 \cdot 7168 = 3.67\ \text{MFLOP}.
$$

### 每個 expert 的權重 bytes（FP4）

以 FP4（$0.5$ byte/element），stage-1 與 stage-2 的權重佔用為

$$
W_{13} = 2I\cdot H \cdot 0.5 = 1.84\ \text{MB},\qquad
W_2 = H\cdot I \cdot 0.5 = 0.92\ \text{MB}.
$$

其中 $W_{13}$ 為 gate+up 合併權重的 bytes、$W_2$ 為 down 權重的 bytes。觸及全部 $E$ 個 routed experts，每個 MoE layer 需 stream

$$
(W_{13}+W_2)\,E \approx (1.84 + 0.92)\,\text{MB} \times 384 \approx 1.06\ \text{GB}
$$

的權重。這個數字是理解 decode 行為的關鍵：無論 batch 多小，都得把這 ~1 GB 讀進來一次。

!!! Example "數值例子：1 GB 權重讀取的時間下界"
    若一層需要 stream 約 $1.06$ GB FP4 expert 權重，在 HBM 有效頻寬 $3.5$ TB/s 時，純權重讀取下界約為 $1.06/3500\approx0.30$ ms。這還沒算 sort、activation、scale、combine 與 launch overhead。若用 BF16 權重，同一層權重 bytes 約 4 倍，頻寬下界也會接近 4 倍。

### Decode 是 weight-bandwidth-bound

考慮 stage-1 在每個 expert 處理 $m$ 行、權重只讀一次時的 arithmetic intensity $I_{\text{AI}}$（FLOP/byte）：

$$
I_{\text{AI}} = \frac{2\,m\,H\,(2I)}{2I\cdot H \cdot 0.5} = 4m\ \text{FLOP/byte}.
$$

平均每個 expert 的行數為

$$
M \approx \frac{\text{batch}\cdot k}{E}.
$$

例如 $\text{batch}=32$、$k=8$、$E=384$ 時 $m \approx 32\cdot 8/384 = 0.67$， 即 decode 時 $m \lesssim 1$，於是 $I_{\text{AI}} \approx 4$ FLOP/byte，遠低於現代 GPU 的 roofline ridge point（數百 FLOP/byte）⇒ **memory-bound**。這解釋了為何 decode 的 MoE GEMM 形同為 ~1 個 token 付出一次完整的權重讀取；而提高 batch 會 線性拉高 $I_{\text{AI}}$，因此提升效率，直到計算飽和（撞到 ridge point）為止。

!!! Example "數值例子：batch 要多大才接近 ridge"
    對 FP4 stage-1，$I_{\text{AI}}=4m$。若目標 GPU 的 ridge point 約 $250$ FLOP/byte，需要 $m\approx62.5$ rows/expert。以 $E=384,k=8$ 反推 batch：$m=\text{batch}\cdot k/E$，所以 batch 約 $62.5\cdot384/8\approx3000$。這遠高於一般低 latency decode 的並發量，因此單 token decode 的 routed GEMM 幾乎必然是 weight-bandwidth-bound。

### Sort/padding 開銷

Grouped GEMM 會把每個 expert 的行數向上 pad 到 tile $\text{block}_m$ 的整數倍 （matmul tile 沿 $M$ 維的高度）。若 expert 實際有 $m_e$ 行，實際算的是 $\lceil m_e / \text{block}_m\rceil \cdot \text{block}_m$ 行，padding 比例為

$$
\frac{\lceil m_e/\text{block}_m\rceil \cdot \text{block}_m - m_e}{m_e}.
$$

在小 $m_e$（decode，$m_e \lesssim 1$）時這個比例很大 — 加上 launch + sort + padding 的固定開銷，在低 batch 時主導整體時間；隨 batch 增大，這些開銷被攤平。

### Triton grouped GEMM（草圖）

一個可運行、經過測試的版本位於 [`code/kernels/triton_grouped_gemm.py`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/kernels/triton_grouped_gemm.py)。 基本架構：預先計算每個 expert 的行偏移，啟動一個攤平輸出 tile 的 1D grid，並讓 每個 program 找出自己負責哪個 expert/tile。

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

讓它成為「grouped」的兩個關鍵：**(1)** 每個 tile 用 `expert_of_tile` 把該 tile 導向正確的權重板 `w_ptr + e*K*N`；**(2)** 行索引使用該 expert 的 `group_off`， 讓可變大小的區塊背靠背地排在一起，無 padding。`BM/BN/BK` 與 `num_warps` 由每個 arch 各自 autotune。

!!! Tip "把 gather（routing）fuse 進 GEMM"
    把連續行的 load `x_ptr + rm*K` 換成透過 permutation 索引的 indirect load — `src = tl.load(perm_ptr + rm); x_ptr + src*K` — 如此 kernel 便把*未排序的* tokens 直接讀進 matmul。這會省去單獨的 gather pass（節省一次完整 HBM 往返）。 在 epilogue 做相反操作即可 fuse scatter。這種 fused scheduling 正是生產級 kernels（如 SGLang/vLLM、Megatron）的做法。

## CUDA 與 ROCm/HIP，並排對照

在最底層，這裡是兩者皆有的手寫 permute/scatter kernel。HIP 刻意與 CUDA **幾乎相同** — 這正是可攜層的重點 — 但 _tuning_ 有差異。完整檔案： [`moe_permute.cu`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/kernels/moe_permute.cu) 和 [`moe_permute_hip.cpp`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/kernels/moe_permute_hip.cpp)。

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

=== "ROCm/HIP"

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

### 真正的差異（與會咬人的坑）

| 面向 | NVIDIA (CUDA) | AMD (ROCm/HIP) |
| --- | --- | --- |
| 執行單位 | **warp = 32 lanes** | **wavefront = 64 lanes**（CDNA） |
| 共享記憶體名稱 | shared memory（SMEM） | **LDS**（Local Data Share） |
| warp shuffle | `__shfl_sync`（需 mask） | `__shfl`（無 mask 參數） |
| 矩陣單元 | Tensor Core（`wmma`/`mma`） | **Matrix Core**（`mfma` / rocWMMA） |
| occupancy 單位 | 每個 SM 的 register/SMEM | 每個 **CU** 的 register/LDS |
| launch | `<<<grid,block>>>` | `hipLaunchKernelGGL(...)`（或透過 hipcc 的 `<<<>>>`） |
| profiler | Nsight Compute/Systems | **rocprof / Omniperf** |

**wavefront = 64** 是會默默咬人的差異：在 AMD 上假設 32 lane 去做 reduction 或 shuffle 都是錯的；block size 256 在 NVIDIA 上是 8 個 warp、在 AMD 上是 4 個 wavefront，這會改變 occupancy 與正確的 tile 尺寸。永遠以 `warpSize`（兩者皆有 內建）來參數化，不要硬寫 32。對於 matmul，你要對應到 `mma`（Tensor Core）與 `mfma`（Matrix Core）— Triton 與函式庫（cuBLAS/hipBLASLt、CUTLASS/Composable Kernel）會隱藏這層，但手寫 GEMM 必須各別處理。可攜性細節見 [CUDA/HIP track](../performance/cuda-hip-track.md)。

## 測量影響（代表性）

在 H100 上，對於單一 MoE FFN、$E{=}64$、$T{=}8192$ tokens、$d{=}4096$、 $d_{ff}{=}1408$（fine-grained），BF16 — _代表性數字；以 `code/kernels/` 中的 基準重現_：

| 實作 | 時間（ms） | 加速 | 備註 |
| --- | --: | --: | --- |
| 在 experts 上的 Python 迴圈 (PyTorch) | ~12.0 | 1.0× | launch 開銷、ragged batch |
| batched GEMM + padding (cf=1.5) | ~3.1 | 3.9× | padding 浪費約 33% 的 FLOPs |
| Triton grouped GEMM | ~1.9 | 6.3× | 無 padding |
| grouped GEMM + fused gather/scatter | ~1.5 | 8.0× | 省下一次 HBM 往返 |

方法學（warmup、CUDA events、固定時鐘）見 [profiling](../performance/profiling.md) 頁 — 沒有它的加速數字別輕信。

!!! Tip "真實 decode 中的 fusion"
    [Anatomy of an MoE decode](decode-anatomy.md) 在生產級兆參數模型上對照這些 選擇：_unfused routing_（top-$k$ + sort 拆成 3 個 kernels）是單一最大的 跨堆疊缺口，而把 _fused shared expert_ 併進 routed grouped GEMM 可削去約 18% 的 decode latency。

## 要點

- MoE 的熱點操作是 **permute**（bandwidth-bound 的 scatter/gather）與 **grouped GEMM**（一次 launch 下的許多可變大小 matmul）。
- Grouped GEMM 把每個輸出 _tile_ 導向其 expert 的權重板，並透過 per-expert 行偏移 把可變大小的區塊封裝在一起 — 無 padding。
- **把 gather/scatter fuse 進 GEMM 的 prologue/epilogue** 可消除整次 HBM 往返 — 最大的實際勝利。
- Decode 的 MoE GEMM 是 **weight-bandwidth-bound**：arithmetic intensity $I_{\text{AI}}=4m$，而 decode 時 $m\approx\text{batch}\cdot k/E \lesssim 1$， 遠低於 ridge point；提高 batch 可線性拉高效率直到計算飽和。
- CUDA 與 HIP 原始碼幾乎相同，但要 **針對 wavefront = 64、LDS 與 AMD 上的 `mfma` 做 tuning**；以 `warpSize` 參數化，並讓 Triton/函式庫對應到正確的 Matrix Core。

## 練習

!!! Tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 擴充 Triton gather kernel，使其也能為 scatter 產生逆 permutation，並把它 fuse 進 grouped GEMM 的 epilogue。
2. 取 CUDA `gather_rows` 並使其與 wavefront 無關；在你手上的任一 GPU 上對 block size 128/256/512 做基準測試，並解釋 occupancy 曲線。
3. 實作 padded batched GEMM 與 grouped GEMM；繪出時間對 capacity factor 的曲線， 找出交叉點。
4. Profile fused 與 unfused 排程，並用 profiler 的記憶體計數器確認 HBM bytes 減少。

## 參考文獻

[1] T. Gale, D. Narayanan, C. Young, and M. Zaharia, "MegaBlocks: Efficient sparse training with mixture-of-experts," *arXiv:2211.15841*, 2022.

[2] P. Tillet, H. T. Kung, and D. Cox, "Triton: An intermediate language and compiler for tiled neural network computations," in *Proc. MAPL*, 2019.

[3] NVIDIA, "CUTLASS: CUDA templates for linear algebra subroutines," Documentation, 2024.

[4] AMD, "Composable Kernel and hipBLASLt," Documentation, 2024.

[5] AMD, "AMD CDNA3 instruction set architecture," Reference Manual, 2023.

[6] AMD, "ROCm HIP programming guide," Documentation, 2024.

[7] vLLM Project, "Fused MoE kernels," Documentation, 2024.

[8] SGLang Team, "SGLang serving system," Documentation, 2025.
