# Triton 路線

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../gpu-programming/">GPU 程式設計模型</a></span>
  <span class="chip"><strong>程式碼：</strong> <code>code/kernels/</code>（需 GPU）</span>
</div>

Triton 讓你用 Python 寫 GPU kernel，編譯成接近峰值的機器碼，同時替你處理那些痛苦的細節 （coalescing、SMEM 分段、向量化、大部分排程）。你以 **tile（張量塊）** 的角度思考，而不是 單個 thread。本路線從最簡單的 kernel 一路堆到融合 softmax 與 matmul，再指向手冊其他地方的 attention / MoE kernel。

!!! info "為什麼先學 Triton"
    對大多數 kernel，Triton 能以一小部分的工作量達到手調 CUDA 80–95% 的效能，而且*同一份原始碼 也能在 AMD 上跑*（Triton 有 ROCm 後端，對映到 wavefront-64 與 MFMA）。只有當你需要 Triton 沒暴露出來的控制權時，才往下掉到 [CUDA/HIP](cuda-hip-track.md)。

## Triton 心智模型

一個 Triton kernel 是站在**單一 program 實例**（大致對應一個 block）的視角寫的。你會：

1. 取得自己的 program id（`tl.program_id`）→ 確定你負責哪個 tile。
2. 算出該 tile 的偏移（`tl.arange`）。
3. 用 `tl.load` 把 HBM 裡的 tile（帶邊界遮罩）載進晶片上記憶體。
4. 在上面計算（`tl.dot`、element-wise、`tl.max`/`tl.sum` 之類的 reduction）。
5. 用 `tl.store` 把結果寫回去。

Triton 會在 tile 內幫你做向量化、並替你管理 SMEM／暫存器。

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

這是 memory-bound 的（每 1 FLOP 搬 3 個 byte）——它唯一的用途就是示範 load／compute／store 的 模式與遮罩。

## 等級 2 — fused softmax（第一個真正的勝利）

對一列做樸素 softmax，要讀一次列找最大值、再讀一次做 exp 與求和、再讀一次做除法——多趟 HBM 往返。融合版的 Triton kernel 把每列**只載入一次**進 SRAM，然後在那裡做完 max/exp/sum/divide：

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

這是 [online softmax / FlashAttention](../foundations/flashattention.md) 想法的縮影：靠把資料 留在晶片上，把多趟記憶體往返摺成一趟。可執行、經 PyTorch 驗證的版本在 [`code/kernels/softmax_triton.py`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/kernels/softmax_triton.py)。

## 等級 3 — 分塊與自動調參的 matmul

標準的 Triton matmul 把 $C=AB$ 分塊成 `BM×BN` 的輸出區塊，每個區塊沿 K 維逐 `BK` 區塊、以 FP32 累積：

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

`tl.dot` 會自動在 NVIDIA 上降階成 Tensor Core、在 AMD 上降階成 **MFMA 矩陣核心**。 `@triton.autotune` 會為每個 shape 搜尋 tile 尺寸與 `num_warps`——**在 AMD 上要重新自動調參**， 因為 wavefront-64 會改變最佳配置。這個 matmul 就是 [MoE grouped GEMM](../moe/kernels.md) 的骨幹， 也就是「同一個 kernel，只是每個 tile 各自挑出它那個 expert 的權重板」。

## 等級 4 — attention 與 grouped GEMM

理解了 softmax + matmul 之後，[FlashAttention](../foundations/flashattention.md) kernel 就是 「分塊算 $QK^\top$、分數在 SRAM 裡做 online softmax、再乘上 $V$，全程不寫出分數矩陣」。而 [MoE grouped GEMM](../moe/kernels.md) 則是「等級 3 的 matmul，加上每個 tile 各自查它的 expert、 打包可變大小的 tile，並把 gather 融進 prologue」。那些頁面有完整 kernel；現在你已經有讀懂它們的 詞彙了。

## 實用要訣

- **在 benchmark 之前一定先用 `torch.allclose` 對照 PyTorch**——又快又錯的 kernel 毫無價值。 我們的 `code/kernels` 測試就是這麼做的。
- **`num_warps`／`num_stages`** 是你主要的 throughput 旋鈕；用 autotune 來調。
- **邊界處一律加遮罩**，否則會越界讀／寫。
- **用 `triton.testing.do_bench`** 來做 benchmark（它會處理 warmup + CUDA event）——見 [profiling](profiling.md)。
- **在 AMD 上**：確認裝了 ROCm Triton 後端；重跑 autotune；因為 wavefront 寬度與 LDS 大小不同， 最佳配置通常也不一樣。

## 要點

- Triton kernel 以 **tile** 為單位寫：取得 program id → 算偏移 → `tl.load` → 計算（`tl.dot`、 reduction）→ `tl.store`，全部帶遮罩。
- 把多趟 HBM 往返融成一趟晶片內運算（softmax、attention）是核心勝利——又是 roofline 劇本。
- `tl.dot` 自動瞄準 Tensor Core / MFMA；**每種架構都要重新 autotune**，因為 wavefront-64 改變了 AMD 上的最佳 tile。
- Triton 讓你可移植地拿到 CUDA 大部分的效能；只有必要時才伸手去碰 [CUDA/HIP](cuda-hip-track.md)。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 跑向量加法與 softmax kernel；對照 PyTorch 驗證，並與原生操作比 benchmark。
2. 為 matmul 加上面向 AMD 的 autotune 配置（試 `num_warps` 4/8，記得 wavefront-64），比較你手上 不同 GPU 的最佳配置。
3. 用 online softmax 組合器，擴充 softmax kernel 以處理比 1 個 `BLOCK` 還寬的列。
4. 對融合 softmax 與三趟式 softmax 做 profiling，並解釋兩者搬移 byte 數的差異。

## 參考文獻

- Tillet, Kung, Cox. _Triton._ 2019；以及 Triton 官方教學。
- Dao et al. _FlashAttention_（本頁的 kernel 就是為它鋪路）。2022。
- AMD ROCm Triton 後端文件。
