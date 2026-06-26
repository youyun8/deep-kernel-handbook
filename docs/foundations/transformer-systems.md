# 作為系統的 Transformer

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 初學者</span>
  <span class="chip"><strong>先備知識：</strong> matmul、基本 Transformer</span>
  <span class="chip"><strong>硬體：</strong>無（筆和紙）</span>
</div>

這一頁教你看著模型組態和 GPU 規格表，就能*在跑任何東西之前*估出一層大概要跑多快、
以及**是什麼在限制它**。這是整本手冊裡最重要的一項技能：之後每一個優化，都是針對
**roofline** 的一次刻意出手。

## 兩個數字決定一切

每個 GPU 都有兩個招牌 throughput 數字：

- **算力（compute）**：每秒峰值浮點運算數 $\pi$（FLOP/s）。
- **記憶體頻寬（memory bandwidth）**：HBM 與晶片之間每秒可搬移的峰值位元組數 $\beta$（B/s）。

| 加速器                 | BF16 密集算力 (TFLOP/s) | HBM 頻寬 (TB/s) | 脊點 $\pi/\beta$ (FLOP/byte) |
| ---------------------- | ----------------------: | --------------: | ---------------------------: |
| NVIDIA A100 80GB (SXM) |                    ~312 |            ~2.0 |                         ~156 |
| NVIDIA H100 (SXM)      |                    ~990 |           ~3.35 |                         ~296 |
| AMD Instinct MI300X    |                   ~1300 |            ~5.3 |                         ~245 |

!!! note "這些是規格表上的峰值數字"
    真實 kernel 大概能拿到峰值算力的 50–80%、峰值頻寬的 70–90%。峰值只拿來算*比率、
    建立直覺*；要看真相請用 [profiler](../performance/profiling.md)。規格表上的稀疏／
    結構化 FLOP 數字通常是密集值的 2 倍——除非你真的在用稀疏性，否則直接忽略。

一個執行 $W$ 個 FLOP、搬移 $Q$ 個 bytes 的 kernel，其**算術強度（arithmetic intensity）**為

$$ I = \frac{W}{Q} \quad \text{[FLOP/byte]}. $$

**roofline** 給出可達到的效能：

$$ P = \min(\pi,\; \beta \cdot I). $$

如果 $I$ 落在**脊點（ridge point）** $\pi/\beta$ 左邊，你就是 **memory-bound**——
效能等於 $\beta \cdot I$，而且在撞到峰值之前，多加 FLOP 是「免費」的。落在右邊，你是
**compute-bound**——這時只有減少 FLOP 或用更快的精度才有用。

<figure class="roofline-figure">
<svg viewBox="0 0 760 430" role="img" aria-labelledby="roofline-title roofline-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="roofline-title">roofline效能型號</title>
  <desc id="roofline-desc">A 可實現的效能與 算術強度 的雙對數圖，隨 記憶體頻寬 上升直至達到脊線點，然後在峰值計算時變平。 </desc>
  <defs>
    <linearGradient id="roofline-line" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="#00bcd4" />
      <stop offset="55%" stop-color="#5e35b1" />
      <stop offset="100%" stop-color="#7c4dff" />
    </linearGradient>
    <marker id="axis-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
  </defs>
  <rect class="roofline-panel" x="24" y="20" width="712" height="360" rx="24" />
  <g class="roofline-grid">
    <path d="M130 320 H650" />
    <path d="M130 260 H650" />
    <path d="M130 200 H650" />
    <path d="M130 140 H650" />
    <path d="M210 80 V320" />
    <path d="M330 80 V320" />
    <path d="M450 80 V320" />
    <path d="M570 80 V320" />
  </g>
  <g class="roofline-axis">
    <path d="M110 320 H675" marker-end="url(#axis-arrow)" />
    <path d="M110 320 V64" marker-end="url(#axis-arrow)" />
  </g>
  <path class="roofline-slope" d="M135 305 L365 115" />
  <path class="roofline-cap" d="M365 115 H650" />
  <line class="roofline-ridge" x1="365" y1="115" x2="365" y2="320" />
  <circle class="roofline-ridge-dot" cx="365" cy="115" r="5" />
  <text class="roofline-label roofline-y" x="44" y="62">效能 (FLOP/s, log)</text>
  <text class="roofline-label roofline-x" x="462" y="365">算術強度 I (log)</text>
  <text class="roofline-tick" x="86" y="121">π</text>
  <text class="roofline-label" x="424" y="98">compute-bound：P = π</text>
  <text class="roofline-label" x="198" y="194">memory-bound：P = β · I</text>
  <text class="roofline-label roofline-slope-label" x="232" y="153">斜率 = β</text>
  <text class="roofline-label" x="323" y="350">脊點 = π/β</text>
</svg>
<figcaption>roofline：脊點左側受頻寬限制，右側受數學單元（算力）限制。</figcaption>
</figure>

ML 效能工程的整個遊戲就是：**(1) 判斷自己在哪一種機制；(2) 若 memory-bound，就提高 $I$
（融合操作、在 SRAM 內重用資料、量化）；(3) 若 compute-bound，就減少 FLOP 或改用更快的
精度。**

## 計算 Transformer 中的 FLOP 次數

取一個 decoder-only 的 Transformer：$L$ 層、隱藏維度 $d$、FFN 隱藏維度 $d_{ff}$
（通常為 $4d$）、序列長度 $N$、batch $B$、詞彙量 $V$。

一個 $(m\times k)\cdot(k\times n)$ 的 matmul 成本是 $2mkn$ FLOP（係數 2 = 每個內積項一次
乘法加一次加法）。

**每層、每 token**（先忽略 $O(N^2)$ 的 attention 分數項）：

| 子區塊        | Matmul 形狀    | FLOP/token                  |
| ------------- | -------------- | --------------------------- |
| QKV 投影      | $d \to 3d$     | $2 \cdot d \cdot 3d = 6d^2$ |
| attention 輸出投影 | $d \to d$ | $2d^2$                      |
| FFN up        | $d \to d_{ff}$ | $2 d\, d_{ff}$              |
| FFN down      | $d_{ff} \to d$ | $2 d\, d_{ff}$              |

當 $d_{ff}=4d$ 時，FFN 是 $16d^2$、attention 投影是 $8d^2$，所以**線性**部分每個 token
每層約 $24d^2$ FLOP。涵蓋 $L$ 層、$BN$ 個 token，前向傳播大致為

$$ W_{\text{fwd}} \approx 24\, L\, d^2 \cdot BN. $$

有一個經典捷徑：非嵌入參數約為 $P \approx 12 L d^2$，於是
$W_{\text{fwd}} \approx 2 P \cdot BN$——**每個參數、每個 token 約 2 個 FLOP**。反向傳播
成本大約是前向的兩倍，於是得到那條著名的公式

$$ \boxed{\;W_{\text{train}} \approx 6\, P \cdot (\text{tokens})\;} $$

用來估計算力預算（例如 Chinchilla）。把它記起來——這樣你幾秒鐘就能檢查一次 training
的 MFU（模型 FLOP 利用率）。

### $N^2$ attention 術語

分數矩陣 $QK^\top$ 是 $(N\times d)\cdot(d\times N)$，而 $\text{softmax}\cdot V$ 步驟是
$(N\times N)\cdot(N\times d)$，各約 $2N^2 d$ FLOP（合計所有頭），因此 attention 分數項
總共消耗 $\approx 4 L N^2 d \cdot B$。和線性項 $24 L d^2 BN$ 相比：

$$ \frac{\text{attention}}{\text{linear}} \approx \frac{4 N^2 d}{24 d^2 N} = \frac{N}{6d}. $$

所以 attention 占的 FLOP 比例會隨 $N/d$ 增長。在 $N=2048, d=4096$ 時約佔 8% 的 FLOP；
到 $N=128\text{k}$ 就反客為主。這正是*為什麼*長上下文的工作幾乎都圍著 attention 打轉，
也是 FlashAttention 為何重要的原因。

## 計算位元組：人們忘記的部分

FLOP 只是 roofline 的一半。考慮 **training** 時的單一 FFN-up matmul，batch × 序列
= $BN$ 個 token、權重 $W \in \mathbb{R}^{d\times d_{ff}}$、以 bf16（2 bytes）：

- 搬移的 bytes：讀 activation $BN\cdot d \cdot 2$、讀權重 $d\, d_{ff}\cdot 2$、寫輸出 $BN\, d_{ff}\cdot 2$。
- FLOP 數：$2\, BN\, d\, d_{ff}$。

當 $BN \gg d$（大 batch）時，讀權重的成本被攤平，強度趨近於
$I \approx \tfrac{2 BN d\, d_{ff}}{2(BN d + BN d_{ff})} \approx \tfrac{BN d\, d_{ff}}{BN(d+d_{ff})}$,
在 $d_{ff}=4d$ 時約為 $0.8\,d$——數百 FLOP/byte，輕鬆 **compute-bound**。大 batch 的
大型 matmul 是 GPU 的快樂天堂。

現在把同一個 matmul 換到 **decode**：batch $B=1$、只有一個新 token（$N=1$）。你得讀進
整個權重矩陣，只為了算一個 token 的激活。強度暴跌到 $\approx 1$ FLOP/byte，
變成 **memory-bound**。一樣的數學、相反的機制，純粹因為 batch 大小不同。

!!! important "LLM serving 的核心張力"
    **training / prefill** 一次處理很多 token → compute-bound → 你希望 kernel 逼近峰值
    FLOPs。**decode** 一次生成一個 token → memory-bound → 你希望搬移更少的 bytes
    （量化權重、把請求批在一起、快取 KV）。幾乎每個 serving 技巧——
    [Continuous Batching](../performance/inference-optimization.md)、
    [權重量化](../performance/quantization.md)、
    [speculative decoding](../performance/inference-optimization.md)——都是針對這個事實的
    出招。

## 一個有效的 roofline 範例

H100：$\pi = 990$ TFLOP/s、$\beta = 3.35$ TB/s、脊點 $= 296$ FLOP/byte。

一個 $m=n=k=8192$ 的 bf16 GEMM：$W = 2\cdot8192^3 \approx 1.1\times10^{12}$ FLOP；
bytes $= 3\cdot 8192^2 \cdot 2 \approx 4.0\times10^8$ B；強度
$I \approx 2730$ FLOP/byte $\gg 296$ → compute-bound。最佳情況時間
$\approx 1.1\times10^{12} / 9.9\times10^{14} \approx 1.1$ ms。如果你的 profiler 顯示 2.2 ms，
MFU 約 50%——現在你有一個*目標*，而不是憑感覺。

同一張量上的 bf16 element-wise GELU：約 $10\cdot8192^2$ FLOP，但要搬 $2\cdot 8192^2\cdot 2$
bytes → $I\approx 2.5$ → memory-bound，時間由頻寬決定。這正是我們把 GELU 融進 matmul
epilogue 的原因：單獨跑時，它純粹是記憶體流量。

## 要點

- 兩個 GPU 數字——峰值 FLOP/s $\pi$ 與頻寬 $\beta$——加上一個 kernel 數字——算術強度
  $I = W/Q$——就能透過 $P=\min(\pi, \beta I)$ 預測效能。
- Transformer training 每個 token 成本約 $6 P$ FLOP，前向約 $2P$；attention 的 FLOP 比例
  約為 $N/6d$。
- **大 batch 的 matmul 是 compute-bound；單 token 的 decode 是 memory-bound。** 本手冊
  大半在講怎麼把操作往 roofline 的右邊推。
- 在相信任何量測值之前，先用 roofline 算出「目標」時間。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/foundations.md) 上。請先嘗試每個練習，再展開解答。

1. 對一個 7B 參數模型（$P\approx7\times10^9$），估計一段 4096 token 序列的前向 FLOP。
   在 MI300X 上以 60% MFU 計算需要多久？
2. 在哪個序列長度 $N$ 下，attention 分數的 FLOP 會等於 $d=5120$ 時線性層的 FLOP？這對
   上下文長度的擴展意味著什麼？
3. 對 $[B{=}32, N{=}2048, d{=}4096]$ 張量做 bf16 LayerNorm：估計 FLOP 與 bytes、算出 $I$、
   判斷它在 A100 上屬於哪種機制。該不該融合？
4. 重新推導 $6P$ 規則，指出每個係數 2 來自何處（前向 vs 反向），以及係數 3 是怎麼來的
   （前向 + 2× 反向）。

## 參考文獻

- Williams, Waterman, Patterson. _Roofline: An Insightful Visual Performance Model for Multicore Architectures._ CACM 2009。
- Kaplan et al. _Scaling Laws for Neural Language Models._ 2020。
- Hoffmann et al. _Training Compute-Optimal Large Language Models（Chinchilla）._ 2022。
- Korthikanti et al. _Reducing Activation Recomputation in Large Transformer Models._ 2022。
- NVIDIA H100 與 AMD CDNA3（MI300）架構白皮書。
