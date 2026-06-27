# Profiling 與方法論

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 全部</span>
  <span class="chip"><strong>先決條件：</strong> <a href="../../foundations/transformer-systems/">roofline</a></span>
  <span class="chip"><strong>硬體：</strong> GPU 用於運行分析器</span>
</div>

**儘早閱讀並經常重讀。**本手冊中的每項優化都是 透過測量來證明其合理性，而大多數測量第一次都是錯誤的。本 頁面介紹如何正確地對 GPU 進行基準測試、會產生虛假加速的 陷阱，以及如何閱讀 profile 以找到真正的瓶頸。

## 針對目標而非感覺進行衡量

始終從 [roofline](../foundations/transformer-systems.md) 開始：計算 你的操作的*理論*時間（受計算限制時為 FLOP/π；受記憶體 限制時為 bytes/β）。這個數字就是你的目標。「需要 2 毫秒」毫無意義； 「需要 2 毫秒，而 roofline 是 1.1 毫秒 → 峰值的 55%」才是可行的。沒有 目標，你就無法區分 kernel 的好壞。

### Roofline 目標與峰值百分比

對於一個執行 $W$ FLOP、搬移 $Q$ bytes 的 kernel，定義：

- $W$：kernel 的浮點運算次數（FLOP）。
- $Q$：kernel 在裝置記憶體階層中搬移的位元組數（bytes）。
- $\pi$：硬體的峰值計算 throughput（FLOP/s）。
- $\beta$：峰值記憶體頻寬（bytes/s）。

計算時間與記憶體時間分別為

$$ t_{\text{compute}} = \frac{W}{\pi}, \qquad t_{\text{mem}} = \frac{Q}{\beta}. $$

兩者可在硬體上重疊，因此理想（roofline）下界為兩者的最大值：

$$ t_{\min} = \max\!\bigl(t_{\text{compute}},\, t_{\text{mem}}\bigr). $$

對於量測到的 wall-clock 時間 $t_{\text{measured}}$，效率（佔 roofline 的比例）為

$$ \text{Efficiency} = \frac{t_{\min}}{t_{\text{measured}}} \in (0, 1]. $$

當 $t_{\text{compute}} > t_{\text{mem}}$ 時 kernel 為 compute-bound，反之為 memory-bound；兩者相等處即為 roofline 的拐點（ridge point）。**一個沒有 附上此比值的測量是沒有意義的** —— 它無法告訴你距離硬體上限還有多遠。

!!! Example "數值例子：把 profiler 時間翻成效率"
    某 BF16 GEMM 有 $W=1.1\times10^{12}$ FLOP、$Q=0.4$ GB。以 H100 的 $\pi=990$ TFLOP/s、$\beta=3.35$ TB/s 估算，$t_{\text{compute}}\approx1.1$ ms、$t_{\text{mem}}\approx0.12$ ms，所以 roofline 下界是 1.1 ms、屬於 compute-bound。若 profiler 量到 1.8 ms，效率是 $1.1/1.8\approx61\%$。只報「1.8 ms」時，讀者看不出它離硬體上限有多遠。

### FLOP 計帳

正確的 FLOP 計帳是 roofline 與 MFU 的基礎。對於稠密矩陣乘法 $[m,k]\times[k,n]$，輸出共有 $mn$ 個元素，每個元素需 $k$ 次乘法與 $k$ 次加法，故

$$ W_{\text{matmul}} = 2\,m\,n\,k \quad \text{FLOP}, $$

其中因子 $2$ 來自每個 multiply-add（MAC）算作 2 FLOP。

令 $N$ 為模型參數量。由於前向傳遞每個參數約做一次 MAC（2 FLOP）， 而反向傳遞約為前向的兩倍成本（一次對輸入、一次對權重的梯度）， 每個 token 的 training 成本約為

$$ C_{\text{train/token}} \approx 2N \times 3 = 6N \quad \text{FLOP}, $$

其中 $2$ 為 multiply-add 因子，$3$ 為「一次 forward + 兩次 backward」。 僅 forward 的 inference 成本約為 $2N$ FLOP/token。

由此計算 training 的 **MFU**（Model FLOP Utilization，模型 FLOP 使用率）：

$$ \text{MFU} = \frac{6N \cdot (\text{tokens/s})}{\pi}, $$

- $N$：模型參數量。
- $\text{tokens/s}$：每秒處理的 token 數（量測得到的 throughput）。
- $\pi$：硬體峰值計算 throughput（FLOP/s）。

分子即上面 $6N$ 的每-token 成本，故 MFU 直接量測硬體峰值算力中 真正用於有用 matmul 的比例。健康的大型模型 training 大致是 **40–55% MFU**；若你只有 15%，那麼通訊或 stall（而非 matmuls）才是 主要問題。

!!! Example "數值例子：MFU 怎麼算"
    以 $N=7$B 的模型、量到 $20{,}000$ tokens/s、8 張 H100 訓練為例，硬體峰值 $\pi=8\cdot990$ TFLOP/s $=7.92$ PFLOP/s。模型 FLOP/s 約 $6\cdot7\times10^9\cdot20{,}000=8.4\times10^{14}$ FLOP/s，也就是 0.84 PFLOP/s。MFU 約 $0.84/7.92=10.6\%$。這通常不是 matmul kernel 太慢，而是 batch 太小、通訊未重疊、資料管線停頓或 activation 重算過多。

## 正確地對 GPU 進行基準測試

GPU kernels**非同步**啟動，因此天真的計時措施啟動 latency， 不執行。規則：

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

1. **熱身（warmup）。**第一次呼叫要支付 JIT/autotune、分配器與 cuDNN/cuBLAS 演算法選擇的成本，且 GPU 時鐘可能尚未升頻。丟棄它。
2. **使用 CUDA events**（`torch.cuda.Event`），而非 `time.time()` —— Events 量測的是裝置上的時間，且能正確框住非同步工作。
3. **計時前後都呼叫 `synchronize()`**；否則你計到的是啟動時間，而非 kernel 執行時間。
4. **重複並彙總。**報告中位數（對離群值穩健）與分佈。
5. **若可行，鎖定時鐘**（`nvidia-smi -lgc` / `rocm-smi --setperflevel`）， 讓熱/boost 變化不會偽裝成回歸。

在 AMD 上，這同樣適用於 `torch.cuda`（HIP 後端）API； `triton.testing.do_bench` 會為你處理 warmup + events，是簡單的預設選擇。

## 產生虛假加速的陷阱

這些陷阱至少會坑每個人一次：

- **沒有 warmup** → 你「優化掉」的是首次呼叫開銷，而非 kernel。
- **沒有 sync** → 你量到的是啟動 latency（~微秒），報告出荒謬的加速。
- **死碼消除（DCE）** → 編譯器刪掉了你的 kernel，因為輸出未被使用。 _消耗輸出_（求和後回傳）。
- **快取／常數折疊** → 每個 iter 用相同輸入讓快取或框架短路。 改變輸入，或在相關時清除快取。
- **在計算基準中包含 H2D/D2H 傳輸** → 你量的是 PCIe。 把張量留在裝置上；若重要則單獨計時。
- **問題規模太小** → 由啟動開銷主導，不代表真實工作量。對真實 shape 做基準。
- **時鐘漂移／熱節流** → 跑得夠久或鎖定時鐘；一個「2% 回歸」通常只是變異數變大。
- **拿蘋果比橘子** → 基線與優化之間的精度、batch 或序列長度不同。 一次只改一件事。
- **只挑一個 shape** → 報告掃描結果；一個 kernel 在某個尺寸快，在另一個可能慢 （這就是我們做 [autotune](triton-track.md) 的原因）。

!!! Warning "基本規則：首先驗證正確性"
    傳回錯誤數字的快速 kernel 的工作速度無限慢。每個 `code/` kernel 這裡檢查 `torch.allclose` 與任何*之前*的參考 時機。僅對經過驗證的程式碼進行基準測試。

## 讀取 profile

Wall-clock 告訴你*有多慢*；profiler 告訴你*為什麼*。兩種視角：

- **時間軸／系統視圖**（Nsight Systems；rocprof + Perfetto）：顯示 kernels、 memcpy 以及時間線上的空隙。尋找**空隙**（CPU 受限的啟動開銷、 Python、同步點）、**序列化的通訊**（all-reduce/all-to-all 未與計算 重疊 —— [MoE](../moe/systems-ep.md) 的典型失效模式），以及主導時間的 kernels。
- **kernel 視圖**（Nsight Compute；Omniperf）：每個 kernel 的計數器 —— 已達成 occupancy、記憶體 throughput 對峰值、計算 throughput 對峰值、 warp（AMD 上的 wavefront）的 stall 原因。這告訴你所處的**régime**：若記憶體 throughput 接近峰值而計算偏低，你是 memory-bound（fuse／提高算術強度）； 反之則是 compute-bound（降精度／減少 FLOP）。

### Amdahl 定律：為何優化非主導階段幫助甚微

設某階段佔總時間的比例為 $p$，且該階段被加速 $s$ 倍，其餘 $(1-p)$ 維持不變，則整體加速比為

$$ S = \frac{1}{(1-p) + p/s}, $$

- $p$：被優化階段所佔的原始時間比例，$0 \le p \le 1$。
- $s$：該階段的局部加速倍數（$s > 1$）。
- $S$：整體（end-to-end）加速倍數。

當 $s \to \infty$ 時，$S \to 1/(1-p)$。例如一個只佔 $p = 0.2$ 的階段，即使 無限加速也只能換來 $1.25\times$ 的整體提升 —— 這就是為何要先從時間軸找出 *主導*階段，再去微優化 kernel。

!!! Example "數值例子：局部 3 倍不等於整體 3 倍"
    若 routing 佔 decode wall-clock 的 12%，把它加速 $3\times$ 後，整體加速是 $1/(0.88+0.12/3)\approx1.09\times$。如果某個 GEMM 佔 35%，同樣 $3\times$ 則是 $1/(0.65+0.35/3)\approx1.30\times$。這就是 profile 要先看佔比的原因。

### Little 定律：serving 的在飛請求數

對於穩態的推論服務，平均在飛（in-flight）請求數等於 throughput 乘以 latency：

$$ \text{in-flight requests} = \text{throughput} \times \text{latency}, $$

- $\text{throughput}$：每秒完成的請求數（req/s）。
- $\text{latency}$：每個請求的平均端到端時間（s）。

這把 batch 大小（在飛請求）、QPS 與每請求延遲綁在一起：要在固定 latency 下提升 throughput，就必須提高並發度（更大的在飛批次）。

### 測量統計：判斷加速是真是噪聲

Kernel 計時通常右偏（偶發的長尾），因此報告**中位數與 IQR** （四分位距）比平均值更穩健。量化離散度用變異係數

$$ \mathrm{CV} = \frac{\sigma}{\mu}, $$

其中 $\mu$ 為樣本平均、$\sigma$ 為樣本標準差；$\mathrm{CV}$ 越小，量測越穩定。

對 $n$ 次量測的平均值 $\bar t$，其信賴區間為

$$ \bar t \pm t_{\alpha/2,\,n-1}\,\frac{\sigma}{\sqrt{n}}, $$

- $\bar t$：$n$ 次量測的樣本平均。
- $\sigma$：樣本標準差。
- $n$：量測次數。
- $t_{\alpha/2,\,n-1}$：自由度 $n-1$、信心水準 $1-\alpha$ 的 Student-$t$ 臨界值。

當兩個版本的信賴區間重疊時，所宣稱的「加速」很可能只是噪聲；只有當區間 分離（或差異的信賴區間不含 0）時，加速才算真實。

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

1. **先做 roofline** —— 計算目標時間與預期 régime。
2. **時間軸視圖** —— 時間花在 kernels、空隙還是通訊？先修空隙／通訊， 再去微優化 kernels（通常是更大的勝利）。
3. **kernel 視圖看最重的 kernel** —— 確認 régime，找出限制器 （記憶體 throughput？occupancy？stall？）。
4. **優化限制器**，而非舒適的東西 —— Memory-bound 就提高算術強度， compute-bound 就削減 FLOP／降精度，通訊受限就做重疊。
5. **正確地重新量測**（warmup、events、sync、掃描）並與目標比較。
6. **重複**，直到你逼近 roofline 或耗盡可用空間（headroom）。

| Profiler         | NVIDIA                    | AMD                     |
| ---------------- | ------------------------- | ----------------------- |
| 時間軸           | Nsight Systems            | rocprof (+ Perfetto)    |
| kernel 計數器    | Nsight Compute            | Omniperf/rocprof        |
| 框架內           | PyTorch Profiler          | PyTorch Profiler (ROCm) |
| 快速 kernel 計時 | `triton.testing.do_bench` | 相同                    |

## 要點

- **先算 roofline 目標** —— 沒有對照目標的量測毫無意義；training 要追蹤 **MFU**。
- 用 **warmup + CUDA events + sync + 重複 + 鎖定時鐘**做基準， 並**消耗輸出**，使其不會被優化掉。
- 大多數「加速」都是假象：沒有 sync、沒有 warmup、死碼消除、 包含傳輸、shape 太小、時鐘漂移。一次只改一個變數。
- 用**時間軸視圖**找空隙／序列化通訊，用 **kernel 視圖**找每個 kernel 的 限制器；**優化限制器**。計時之前先驗證正確性。

## 練習

!!! Tip "解決方案"
    參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 拿 [Triton softmax](triton-track.md)，先用錯誤方式做基準（沒有 warmup／sync），再用正確方式；量化兩者差異。
2. Profile 一個小型 Transformer 的 decode 步驟；辨識是 attention、FFN 還是啟動開銷占主導，並提出修復方案。
3. 給定 tokens/s 與 GPU 峰值，計算 training 的 MFU；診斷一個 15% MFU 的結果。
4. 建構一個被死碼消除藏掉 kernel 的基準，再透過消耗輸出修復它。

## 參考文獻

[1] S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Commun. ACM*, vol. 52, no. 4, pp. 65-76, 2009.

[2] NVIDIA, "Nsight Systems user guide," Documentation, 2024.

[3] NVIDIA, "Nsight Compute user guide," Documentation, 2024.

[4] AMD, "rocprof profiler," Documentation, 2024.

[5] AMD, "Omniperf profiler," Documentation, 2024.

[6] PyTorch Foundation, "PyTorch Profiler," Documentation, 2024.

[7] Triton Project, "`triton.testing` API," Documentation, 2024.
