# Transformer 作為一個系統

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 初學者</span>
  <span class="chip"><strong>先備知識：</strong> matmul，基本Transformer</span>
  <span class="chip"><strong>硬體：</strong>無（筆和紙）</span>
</div>

你將在此頁面中查看模型配置和 GPU 規格表，並
比如說，_在運行任何東西之前_，一個層應該運行的大致速度以及**什麼
正在限制它**。這是手冊中最重要的技能：每隔一段時間
優化是對**roofline**的刻意之舉。

## 兩個數字決定一切

每個 GPU 都有兩個標題 throughput：

-**計算**：每秒峰值浮點運算，$\pi$（FLOP/s）。 -**記憶體頻寬**：HBM 和晶片之間每秒移動的峰值位元組數，$\beta$ (B/s)。

| 加速器                 | BF16 密集 (TFLOP/s)    | HBM 頻寬（TB/秒）            | 脊點 $\pi/\beta$（FLOP/位元組） |
| ---------------------- | ---------------------- | ---------------------------- | ------------------------------- | ----- |
| NVIDIA A100 80GB (SXM) | NVIDIA A100 80GB (SXM) | NVIDIA A100 80GB (SXM) 〜312 | 〜2.0                           | 〜156 |
| NVIDIA H100 (SXM)      | ~990                   | 〜3.35                       | 〜296                           |
| AMD Instinct MI300X    | AMD Instinct MI300X    | AMD Instinct MI300X 〜1300   | 〜5.3                           | 〜245 |

!!! note "這些是行銷表的峰值數字"
    真正的 kernels 可能達到峰值計算的 50-80% 和峰值的 70-90%
    頻寬。使用峰值作為*比率和直覺*；使用
    [profiler](../performance/profiling.md) 為真理。稀疏/結構化 FLOP
    規格表上的數字通常是密集數字的 2 倍——忽略它們，除非
    你實際上正在使用稀疏性。

執行 $W$ FLOP 並移動 $Q$ 位元組的 kernel 具有**算術強度**

$$ I = \frac{W}{Q} \quad \text{[FLOP/byte]}. $$

**roofline**可實現的效能：

$$ P = \min(\pi,\; \beta \cdot I). $$

如果 $I$ 低於**山脊點**$\pi/\beta$，則你**受記憶體限制**—
效能為 $\beta \cdot I$，並且在達到峰值之前可以免費添加 FLOP。
在它上面，你是**計算限制**——只有更少的失敗或更快的數學幫助。

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
  <text class="roofline-label roofline-y" x="44" y="62">效能（FLOP/s，日誌）</text>
  <text class="roofline-label roofline-x" x="462" y="365">算術強度 I（日誌）</text>
  <text class="roofline-tick" x="86" y="121">π</text>
  <text class="roofline-label" x="424" y="98"> 計算綁定：P = π</text>
  <text class="roofline-label" x="198" y="194"> 記憶體限制：P = β · I</text>
  <text class="roofline-label roofline-slope-label" x="232" y="153"> 斜率 = β</text>
  <text class="roofline-label" x="323" y="350">脊 = π/β</text>
</svg>
<figcaption>roofline。在山脊的左側，你會受到頻寬的限制；在山脊的左側，你會受到頻寬的限制。它右邊的數學單位。 </figcaption>
</figure>

機器學習效能工程的整個遊戲是：**(1) 找出哪一種機制
(2) 如果記憶體受限，則提高 $I$（熔斷操作，重複使用 SRAM 中的數據，
量化），(3) 如果受計算限制，則減少 FLOP 或使用更快的精確度。**

## 計算 Transformer 中的 FLOP 次數

取一個僅 decoder 的 Transformer：$L$ 層，隱藏大小 $d$，FFN 隱藏
$d_{ff}$（通常是 $4d$），序列長度 $N$，批次 $B$，詞彙 $V$。

$(m\times k)\cdot(k\times n)$ 的 matmul 成本為 $2mkn$ FLOP（2 = 1）
乘法+每個內積項一次加法）。

**每層，每 token**（暫時忽略 $O(N^2)$ attention 分數術語）：

| 子區塊             | Matmul 形狀    | 失敗/token                  |
| ------------------ | -------------- | --------------------------- |
| QKV 投影           | $d \to 3d$     | $2 \cdot d \cdot 3d = 6d^2$ |
| attention 輸出專案 | $d \to d$      | $2d^2$                      |
| FFN 上             | $d \to d_{ff}$ | $2 d\, d_{ff}$              |
| FFN 下降           | $d_{ff} \to d$ | $2 d\, d_{ff}$              |

對於 $d_{ff}=4d$，FFN 是 $16d^2$，attention 投影是 $8d^2$，因此
對於**線性**部分，每個 token 層約為 $24d^2$ FLOP。超過 $L$ 層
和 $BN$ tokens 的前向傳播大致為

$$ W\_{\text{fwd}} \approx 24\, L\, d^2 \cdot BN. $$

有一個經典的快捷方式：使用 $P \approx 12 L d^2$ 非嵌入參數，
這是 $W_{\text{fwd}} \approx 2 P \cdot BN$ —**每個參數每個 2 次浮點運算
token**。向後傳球的成本大約是向前傳球的兩倍，給出了著名的

$$ \boxed{\;W\_{\text{train}} \approx 6\, P \cdot (\text{tokens})\;} $$

用於計算預算（例如 Chinchilla）。把這個記在心裡——就是這樣
你可以在幾秒鐘內檢查 training 運行的 MFU（模型 FLOP 使用率）。

### $N^2$ attention 術語

得分矩陣 $QK^\top$ 是 $(N\times d)\cdot(d\times N)$ 並且
$\text{softmax}\cdot V$步驟是$(N\times N)\cdot(N\times d)$，每個
每個頭組的 $2N^2 d$ FLOP 數，因此 attention 分數會消耗 $\approx 4 L N^2 d \cdot B$
總計。與線性$24 L d^2 BN$比較：

$$ \frac{\text{attention}}{\text{linear}} \approx \frac{4 N^2 d}{24 d^2 N} = \frac{N}{6d}. $$

因此，attention 的 FLOP 份額隨著 $N/d$ 的成長而成長。在 $N=2048, d=4096$ 中，約為 8%
失敗；在$N=128\text{k}$，它佔據主導地位。這就是*為什麼*對長情境工作的執著
關於 attention，以及為什麼 Flashattention 很重要。

## 計算位元組：人們忘記的部分

FLOP 僅為 roofline 的一半。考慮單一 FFN-up matmul
**training**批次 × 序列 = $BN$ tokens，重量 $W \in \mathbb{R}^{d\times d_{ff}}$
在 bf16（2 位元組）中：

- 移動的位元組：讀取啟動 $BN\cdot d \cdot 2$、讀取權重 $d\, d_{ff}\cdot 2$、寫入輸出 $BN\, d_{ff}\cdot 2$。
- 失敗次數：$2\, BN\, d\, d_{ff}$。

當$BN \gg d$（大批量）時，讀取的重量攤銷且強度接近
$I \approx \tfrac{2 BN d\, d_{ff}}{2(BN d + BN d_{ff})} \approx \tfrac{BN d\, d_{ff}}{BN(d+d_{ff})}$,
對於 $d_{ff}=4d$ 來說是 $\approx 0.8\,d$ — 數百個 FLOP/字節，輕鬆自如
**計算限制**。具有大批量的大型 matmul 是 GPU 的樂土。

現在在**decoding**期間使用批次 $B=1$ 執行相同的 matmul，一個新的 token
($N=1$)：你讀取整個權重矩陣以產生單一 token
激活。強度大幅下降至 $\approx 1$ FLOP/位元組
**記憶體限制**。相同的數學，相反的機制，純粹是因為批量大小。

!!! important "LLM serving 的中心張力"
    **training / prefill**一次處理多個 tokens → 受計算限制 → 你
    希望 kernels 能夠達到峰值 FLOPs。**decoding**一次產生一個 token
    → 記憶體限制 → 你想要移動更少的位元組（量化權重、批次處理）
    一起請求，快取 KV）。幾乎所有 serving 技巧
    （[continuous batching](../performance/inference-optimization.md)，
    [weight quantization](../performance/quantization.md),
    [speculative decoding](../performance/inference-optimization.md)）是一個
    對這一事實的攻擊。

## 一個有效的 roofline 範例

H100：$\pi = 990$ TFLOP/秒、$\beta = 3.35$ TB/秒、脊 $= 296$ FLOP/位元組。

帶有 $m=n=k=8192$ 的 bf16 GEMM：$W = 2\cdot8192^3 \approx 1.1\times10^{12}$ FLOP；
位元組 $= 3\cdot 8192^2 \cdot 2 \approx 4.0\times10^8$ B；強度
$I \approx 2730$ FLOP/位元組 $\gg 296$ → 計算密集型。最佳情況時間
$\approx 1.1\times10^{12} / 9.9\times10^{14} \approx 1.1$女士。如果你的探查器
表示 2.2 毫秒，你的 MFU 約為 50% — 現在你有一個*目標*，而不是氛圍。

相同張量上的 bf16 按元素 GELU：~$10\cdot8192^2$ FLOP 但是
$2\cdot 8192^2\cdot 2$ 位元組 → $I\approx 2.5$ → 記憶體限制，時間由
頻寬。這正是我們將 GELU 融合到 matmul 尾聲中的原因：
獨立的，它是純粹的記憶體流量。

## 要點

- 兩個 GPU 編號 — 峰值 FLOP/s $\pi$ 和頻寬 $\beta$ — 以及一個 kernel
  數字 — 算術強度 $I = W/Q$ — 透過預測效能
  $P=\min(\pi, \beta I)$。
- 每個 token Transformer training 的成本為 $\approx 6 P$ FLOP；向前是
  $\approx 2P$。 attention 的 FLOP 份額與 $N/6d$ 相當。 -**大批量 matmuls 受計算限制；單-token decoding 是
  內存限制。**本手冊的大部分內容是關於向右移動操作
  在 roofline 上。
- 在信任測量值之前，請務必根據 roofline 計算「目標」時間
  一。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/foundations.md) 上。請先嘗試每個練習，再展開解答。

1. 對於 7B 參數模型 ($P\approx7\times10^9$)，估計前向 FLOPs
   對於 4096 tokens 的一個序列。 MI300X 電量達到 60% 需要多長時間
   MFU？
2. $N$ 在什麼序列長度下，attention-score FLOPs 等於線性層
   $d=5120$ 的失敗？這對於上下文長度縮放意味著什麼？
3. $[B{=}32, N{=}2048, d{=}4096]$ 張量上的 bf16 LayerNorm：估計 FLOP
   和字節，計算 $I$，並確定 A100 上的狀態。應該融合嗎？
4. 重新推導$6P$規則並辨識每個地方的因子 2（fwd vs bwd）
   並輸入係數 3 (fwd + 2× bwd)。

## 參考文獻

- 威廉斯、沃特曼、帕特森。 _roofline：多核心架構的富有洞察力的視覺效能模型。 _ CACM 2009。
- 卡普蘭等。 _神經語言模型的縮放定律。 _ 2020。
- 霍夫曼等人。 _training 計算最佳大型語言模型_ (Chinchilla)。 2022 年。
- 科蒂坎蒂等人。 _減少大型 Transformer 模型中的啟動重新計算。 _ 2022。
- NVIDIA H100 和 AMD CDNA3 (MI300) 架構白皮書。
