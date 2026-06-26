# 海衛一軌道

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../gpu-programming/">GPU程式設計模型</a></span>
  <span class="chip"><strong>代碼：</strong> <code>code/kernels/</code>（需GPU）</span>
</div>

Triton 可讓你用 Python 編寫 GPU kernels，編譯成接近峰值機器
程式碼，同時它處理痛苦的部分（合併、SMEM 分段、向量化、
大部分時間安排）。你認為**瓦片**（張量塊）而不是
單獨的線程。該軌道從簡單的 kernel 到融合的 softmax 構建
和 matmul，然後指向手冊中其他地方的 attention/MoE kernels。

!!! info "為什麼先選擇 Triton"
    對於大多數 kernels Triton 而言，可達到手動調整 CUDA 效能的 80-95%
    工作量的一小部分，並且*相同的來源在 AMD 上運行*（Triton 有一個 ROCm
    後端映射到 wavefront-64 和 MFMA）。掉落至[CUDA/HIP](cuda-hip-track.md)
    只有當你需要控制時 Triton 才不會暴露。

## Triton 心智模型

一個 Triton kernel 是從**一個程式實例**的角度寫的
（大約一個街區）。你：

1. 取得你的程式 ID (`tl.program_id`) → 你擁有哪個圖塊。
2. 計算該圖塊 (`tl.arange`) 的偏移量。
3. `tl.load` 將 HBM 中的圖塊（帶有邊界遮罩）放入片上記憶體中。
4. 對其進行計算（`tl.dot`，依元素，`tl.max`/`tl.sum` 歸約）。
5. `tl.store` 回傳結果。

Triton 在圖塊內進行向量化並為你管理 SMEM/暫存器。

## 等級 1 — 向量相加

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

這是記憶體限制的（每 1 FLOP 移動 3 個位元組）——它唯一的工作是教導
載入/計算/儲存模式和屏蔽。

## Level 2 — fused softmax（第一個真正的勝利）

行上的樸素 softmax 讀取行，找出最大值，再次讀取以求冪
和求和，再次讀取以除 - 多次 HBM 傳遞。裝有保險絲的 Triton kernel 負載
每行**一次**進入 SRAM 並在那裡執行 max/exp/sum/divide 操作：

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

這是 [online-softmax / Flashattention](../foundations/flashattention.md)
想法的縮影：透過保留資料將多個記憶體傳遞折疊為一個
在晶片上。可運作的、經過 PyTorch 檢查的版本位於
[`code/kernels/softmax_triton.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/kernels/softmax_triton.py)。

## Level 3 — 平鋪和自動調整的 matmul

規範的 Triton matmul 將 $C=AB$ 平鋪到 `BM×BN` 輸出區塊中，每個區塊
在 fp32 中累積 K 維度的 `BK` 區塊：

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

`tl.dot` 在 NVIDIA 上降低為 Tensor 核心，在 AMD 上降低為**MFMA 矩陣核心**
自動。 `@triton.autotune` 搜尋圖塊尺寸和每個形狀的 `num_warps` —
**在 AMD 上重新自動調整**，因為 wavefront-64 會改變最佳配置。這個矩陣相乘
是[MoE grouped GEMM](../moe/kernels.md)的主幹，即「這個
kernel，但每個圖塊都會選擇其 expert 的重量板。 」

## 4 級 — attention 和分組 GEMM

有了 softmax + matmul 的理解，[Flashattention](../foundations/flashattention.md)
kernel 是「tiles 中的 matmul $QK^\top$，online-softmax SRAM 中的分數，matmul 由
$V$，永遠不要寫分數矩陣。 」還有[MoE grouped GEMM](../moe/kernels.md)
是「具有每個圖塊 expert 尋找和打包可變大小的 3 級 matmul
排塊，將聚集融合到序言中。 」這些頁麵包含完整的
kernels；現在你已經有了閱讀它們的詞彙了。

## 實用技巧

-**在基準測試之前，請務必使用 `torch.allclose` 檢查 PyTorch**—
快錯了 kernel 毫無價值。我們的 `code/kernels` 測試就是這樣做的。 -**`num_warps`/`num_stages`**是你的主要 throughput 旋鈕；自動調整它們。 -**屏蔽邊界處的所有內容**，否則你將越界讀/寫。 -**使用 `triton.testing.do_bench`**進行基準測試（它處理預熱 + CUDA 事件）—
參見 [profiling](profiling.md)。 -**在 AMD 上**：確認已安裝 ROCm Triton 後端；重新執行自動調諧；
由於波前寬度和 LDS 尺寸的不同，預計會有不同的最佳配置。

## 要點

- Triton kernels 是按**tile**編寫的：取得程式 id → 計算偏移量 →
  `tl.load` → 計算（`tl.dot`，縮減）→ `tl.store`，含掩碼。
- 將多個 HBM 通道融合為一個片上通道（softmax、attention）是
  核心勝利——再次是 roofline 劇本。
- `tl.dot` 自動瞄準 Tensor Cores / MFMA；**自動調諧每
  架構**因為 wavefront-64 改變了 AMD 上的最佳圖塊。
- Triton 讓你可移植地獲得 CUDA 的大部分效能；伸手去拿
  [CUDA/HIP](cuda-hip-track.md) 僅在必要時使用。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 運行向量加法和 softmax kernels；對照 PyTorch 和基準進行驗證
   與本機操作相比。
2. 將面向 AMD 的自動調諧配置新增至 matmul（嘗試使用 `num_warps` 4/8
   記住 wavefront-64）並比較你擁有的 GPU 之間的最佳配置。
3. 使用下列指令擴充 softmax kernel 以處理比 1 `BLOCK` 寬的行
   線上 softmax 組合器。
4. 分析融合 softmax 與三聲道 softmax 並解釋位元組移動
   差異。

## 參考文獻

- 蒂萊、孔、考克斯。 *海衛一。 *2019； Triton 官方教學。
- 道等人。 _Flashattention_（這是針對 kernel 建置的）。 2022 年。
- AMD ROCm Triton 後端文件。
