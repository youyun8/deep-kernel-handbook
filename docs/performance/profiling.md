# 分析和方法

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 全部</span>
  <span class="chip"><strong>先決條件：</strong> <a href="../../foundations/transformer-systems/">roofline</a></span>
  <span class="chip"><strong>硬體：</strong> GPU 用於運行分析器</span>
</div>

**儘早閱讀並經常重讀。**本手冊中的每項優化都是
透過測量來證明其合理性，並且大多數測量第一次都是錯誤的。這個
該頁面介紹如何正確對 GPU 進行基準測試，以及產生假冒產品的陷阱
加速，以及如何閱讀設定檔以找到真正的瓶頸。

## 針對目標而非氛圍進行衡量

始終從 [roofline](../foundations/transformer-systems.md) 開始：計算
你的操作的*理論*時間（如果受計算限制，則為 FLOPs/π；如果受計算限制，則為位元組/β）
內存限制）。這個數字就是你的目標。 「需要 2 毫秒」毫無意義；
「roofline 需要 2 毫秒，而 roofline 需要 1.1 毫秒 → 峰值的 50%」是可行的。沒有一個
目標是你無法區分 kernel 的好壞。

計算 training 的**MFU**（模型 FLOP 使用率）： $\text{MFU} =
\frac{6 P \cdot \text{tokens/s}}{\pi}$。健康大型號 training 大致是
40–55% MFU；如果你處於 15%，那麼溝通或失速（而不是 matmuls）才是最重要的
問題。

## 正確地對 GPU 進行基準測試

GPU kernels**非同步**啟動，因此天真的計時措施啟動 latency，
不執行。規則：

```python
import torch
def benchmark(fn, iters=100, warmup=20):
    for _ in range(warmup):           # 1. WARM UP: JIT, autotune, caches, clocks
        fn()
    torch.cuda.synchronize()          # 2. SYNC before starting the timer
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()          # 3. SYNC before reading the timer
    return start.elapsed_time(end) / iters    # ms per iter (use CUDA events!)
```

不可協商的事項：

1. **熱身。**第一次呼叫支付 JIT/autotune、分配器和 cuDNN/cuBLAS
   演算法選擇成本，且 GPU 的時鐘可能較低。丟棄它。
2. **使用 CUDA 事件**(`torch.cuda.Event`)，而非 `time.time()` — 事件測量
   設備上時間和括號異步工作正常。
   3、**`synchronize()`**計時前後；否則你會為發射計時，而不是
   kernel。
3. **重複並彙總。**報告中位數（對異常值穩健）和分佈。
4. ** 如果可以的話，鎖定時鐘**(`nvidia-smi -lgc` / `rocm-smi --setperflevel`)
   熱/升壓變化不會偽裝成回歸。

在 AMD 上，這也適用於 `torch.cuda`（HIP 支援）API；
`triton.testing.do_bench` 為你處理熱身+事件，並且是簡單的預設設定。

## 產生虛假加速的陷阱

這些至少咬每個人一次：

-**無需預熱**→ 你「優化掉」了首次呼叫開銷，而不是 kernel。 -**無同步**→ 你測量了啟動 latency（~微秒），報告了
荒謬的加速。 -**死程式碼消除**→ 編譯器刪除了你的 kernel 因為輸出
未使用。 _消耗輸出_（求和，返回）。 -**快取/常數折疊**→ 每個 iter 讓快取或相同的輸入
框架短路。改變輸入或清除快取（如果相關）。 -**在計算基準測試中包含 H2D/D2H 傳輸**→ 你測量了 PCIe。
將張量保留在設備上；如果重要的話，單獨計時。 -**小問題規模**→ 由啟動開銷主導；不代表
實際工作量。對真實形狀進行基準測試。 -**時鐘漂移/熱節流**→ 運行足夠長的時間或鎖定時鐘；一個「2%
迴歸”通常只是增加變異數。 -**比較蘋果和橘子**→ 不同的精確度、批次或序列長度
介於基線和最佳化之間。一次改變一件事。 -**挑選一個形狀**→ 報告掃蕩；一個 kernel 快速在一個尺寸可以
別人慢（這就是我們[autotune](triton-track.md)的原因）。

!!! warning "基本規則：首先驗證正確性"
    傳回錯誤數字的快速 kernel 的工作速度無限慢。每個
    `code/` kernel 這裡檢查 `torch.allclose` 與任何*之前*的參考
    時機。僅對經過驗證的程式碼進行基準測試。

## 讀取個人資料

掛鐘告訴你*有多慢*；分析器會告訴你*原因*。兩種觀點：

-**時間軸/系統視圖**（Nsight Systems；rocprof + Perfetto）：顯示 kernels，
memcpys，以及時間線上的間隙。尋找**差距**（CPU 限制的啟動開銷，
Python，同步點），**序列化通訊**（all-reduce/all-to-all 不是
重疊計算 — [MoE](../moe/systems-ep.md) 故障模式），以及
kernels 為主。 -**kernel 視圖**（Nsight Compute；Omniperf）：每個 kernel 計數器 — 已實現
佔用率、記憶體 throughput 與峰值、計算 throughput 與峰值、
翹曲失速原因。這告訴你**制度**：如果內存 throughput 接近
峰值和計算量較低，你會受到記憶體限制（熔斷/提高強度）；的
反向意味著計算限制（較低的精度/較少的 FLOP）。

PyTorch 分析器是簡單的入門工具（無需外部工具）：

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    model(x); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# prof.export_chrome_trace("trace.json")  # view in chrome://tracing / Perfetto
```

## 分析工作流程

1. **首先是 roofline**— 計算目標時間和預期狀態。
2. **時間軸視圖**— 時間是 kernels、間隙還是通訊？修復之前的間隙/通訊
   微優化 kernels（通常是更大的勝利）。
3. **kernel 頂部視圖 kernel**— 確認狀態，找到限制器
   （內存 throughput？佔用？卡頓？）。
4. **優化限制器**，這不是一件容易的事 - 如果內存有限，則提高強度，
   如果計算受限，則削減 FLOPs/精度；如果通訊受限，則重疊。
5. **正確重新測量**（預熱、事件、同步、掃描）並與目標進行比較。
6. **重複**，直到你靠近 roofline 或超出淨空高度。

| 探查器           | 英偉達                    | AMD                   |
| ---------------- | ------------------------- | --------------------- |
| 時間軸           | Nsight 系統               | rocprof (+ Perfetto)  |
| kernel 櫃檯      | Nsight 計算               | Omniperf/rocprof      |
| 框架內           | PyTorch 分析器            | PyTorch 分析器 (ROCm) |
| 快速 kernel 計時 | `triton.testing.do_bench` | 相同                  |

## 要點

-**先計算 roofline 目標**— 沒有測量時間就沒有任何意義；
追蹤 training 的**MFU**。

- 使用**預熱 + CUDA 事件 + 同步 + 重複 + 鎖定時鐘**進行基準測試，
  並且**消耗輸出**，因此它不會被最佳化掉。
- 大多數「加速」都是偽影：沒有同步、沒有預熱、死碼消除、
  包括轉移、微小形狀、時鐘漂移。一次更改一個變數。
- 使用**時間軸視圖**尋找間隙/序列化通訊和**kernel 視圖**
  找到每個 kernel 限制器；**最佳化限制器**。驗證正確性
  在計時之前。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 拿[Triton softmax](triton-track.md)，基準測試錯誤（沒有
   熱身/同步）然後對；量化差異。
2. 剖析小型 Transformer 的 decode 步驟；辨識是否為 attention、FFN、
   或啟動開銷占主導地位，並提出修復方案。
3. 給定 tokens/s 和 GPU 峰值，計算 training 運行的 MFU；診斷 15% MFU
   結果。
4. 建立一個基準測試，其中死程式碼消除隱藏了 kernel，然後修復它
   透過消耗輸出。

## 參考文獻

- 威廉斯等人。 _roofline。 _ 2009 年。
- NVIDIA Nsight 系統/Nsight 計算文件。
- AMD rocprof / Omniperf 文件。
- PyTorch Profiler 和 `triton.testing` 文件。
