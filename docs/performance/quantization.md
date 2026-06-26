# 量化和壓縮

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../../foundations/numerics-precision/">數值與精度</a>、<a href="../../foundations/attention-efficiency/">記憶體受限的 decoding</a></span>
  <span class="chip"><strong>硬體：</strong>理論無； GPU對標</span>
</div>

壓縮使模型的儲存成本更低、運行速度更快——尤其是在
[memory-bound decode](../foundations/attention-efficiency.md)，其中減半
重量位元組大約減半 latency。本頁涵蓋**量化**（大
槓桿：PTQ vs QAT，以及 GPTQ/AWQ 系列），然後**修剪**和
**蒸餾**。 [MoE serving page](../moe/inference-serving.md) 適用於所有
這個具體到 experts。

## 量化基礎知識

量化將高精度值映射為小整數（或低位浮點數）
網格。張量 $x$ 的標準仿射方案：

$$ q = \text{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s\,(q - z), $$

其中選擇 $s$（刻度）和 $z$（零點），以便整數網格覆蓋
張量的範圍。**粒度**非常重要：

-**每個張量**：整個張量使用一個 $(s,z)$ — 最便宜，最不準確。 -**每通道/每行**：每個輸出通道一個秤 — 重量標準。 -**每組**（例如 128 個元素）：每個小塊一個尺度 — 最佳點
對於 4 位元權重；與 [microscaling/MX formats](../foundations/numerics-precision.md) 配對。

你量化的「內容」的兩個軸：

-**僅重量**（W8/W4，激活保持在 bf16）：非常適合**decode**，它
受權重頻寬限制－你可以減少每個 token 讀取的位元組數。計算用途
反量化值，因此它不會大大加快計算密集型 prefill 的速度。 -**權重+啟動**（W8A8 / fp8）：也加速**matmul**（整數/fp8）
張量核），幫助計算密集型 prefill/training — 但啟動是
更難量化（異常值）。

## PTQ 與 QAT

-**training 後量化 (PTQ)**：使用小型量化訓練模型
**校準**設定為拾取刻度（對於 GPTQ/AWQ，可修正錯誤）。便宜，
快速，無需 retraining — LLM inference 的預設設定。 -**量化感知 training (QAT)**：在 training 期間模擬量化
（本輪的直通估計器）因此模型學會了魯棒性。
在非常低的位上恢復更高的精度，但需要 training 運作。

LLM PTQ 的困難是**啟動異常值**：有些管道有
巨大的量級，如果天真地量化，就會擴大規模並壓碎
其他一切。 GPTQ/AWQ 系列就是用來處理這個問題的。

### GPTQ — 糾錯權重量化

GPTQ 一次量化一列的權重，並在每一列捨入後，**更新
剩餘的未量化權重用於補償**引入的誤差（a
使用校準活化的二階/基於 Hessian 的校正）。的
結果：精確的 3-4 位元*僅權重*量化，精度損失很小。
僅重量 → decode 的理想選擇。

### AWQ — 激活感知權重量化

AWQ 觀察到並非所有權重都同等重要：權重相乘
高強度（顯著）活化通道是最重要的。它**可擴展
在量化**（和補償）之前將那些顯著的通道提升，保護它們
來自舍入誤差－不需要反向傳播。通常在 4 位上匹配或擊敗 GPTQ
並且簡單/快速。

### SmoothQuant — 使激活可量化

對於 W8A8，SmoothQuant**將啟動離群值遷移**到
透過數學上等效的重新縮放來調整權重（每個通道），因此兩者
激活和權重變得很容易量化為 int8 — 實現快速 int8
prefill/serving 的 matmuls。

```python
# The shared idea: choose per-channel scales so the *product* is unchanged
# but each operand is easier to quantize. (Conceptual sketch.)
# y = (x / s) @ (s * W)   # s absorbs outliers from x into W or vice-versa
```

!!! tip "使用哪一個"
    受記憶體限制**decode**，想要簡單+準確 →**AWQ 或 GPTQ**（W4，
    僅重量）。也想要更快的**prefill/serving 計算**→**SmoothQuant**
    或**fp8**(W8A8)。準確度低於 4 位元 → 考慮**QAT**。

## fp8 作為量化

fp8 (E4M3/E5M2) 是具有浮動網格的量化－更適合寬螢幕
啟動的動態範圍高於 int8，並且在 H100/MI300 上原生加速。
對於每張量/每塊尺度，它可用於 inference（W8A8 樣式）和，
越來越多地，**training**(DeepSeek-V3)。參見
格式為 [numerics & precision](../foundations/numerics-precision.md)
細節和縮放規則。

## 修剪

刪除權重/結構而不是降低其精確度：

-**非結構化**（將各小重量歸零）：高壓縮
理論上，但不規則稀疏性很難在 GPU 上加速。 -**結構化**（刪除整個頭/通道/層）：壓縮較少，但是
產生一個更小的“密集”模型，可以在庫存 kernels 上快速運行。 -**2:4 半結構化**（每 4 個中的 2 個權重為零）：硬體支持
中間立場 — NVIDIA 稀疏張量核心在這些方面提供了約 2 倍的效能。一個實用的
支援時的最佳位置。

修剪通常需要微調以恢復品質；當它最有吸引力的時候
你可以利用硬體稀疏性或想要更小的密集模型。

## 蒸餾

訓練一個小**學生**來模仿一個大**老師**（匹配輸出
分佈/軟邏輯，有時還有中間特徵）。不像
量化/修剪，它產生了一個真正更小的架構，並且可以
傳輸功能，以 training 運作和存取教師為代價。
通常與上述結合（蒸餾，然後量化學生）。

## 要點

- 量化透過比例/零點映射到低位網格；**粒度**
  （每個張量 → 每組）以準確度換取成本。 -**僅重量（GPTQ/AWQ、W4）**加速記憶體限制**decode**；
  **權重+啟動（SmoothQuant/fp8、W8A8）**還可以加快計算速度
  **prefill/training**，但必須馴服**啟動異常值**。 -**PTQ**（僅校準）是 LLM 預設值；**QAT**以非常低的價格購買準確性
  training 價格位。 -**修剪**（特別是 2:4 結構）和**蒸餾**是互補的
  壓縮工具；將它們結合起來。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 推導 int8 仿射量化/反量化與最大量化誤差
   每個張量與每個通道在具有一個離群通道的張量上進行縮放。
2. 在單一線性層上實現 AWQ 的顯著通道縮放並測量
   困惑與樸素的每聲道 int4。
3. 估計 13B 型號上 decode-latency 相對於僅 W4 重量的改進（使用
   [memory-bound](../foundations/attention-efficiency.md) 參數）。
4. 對於 MoE，爭論為什麼路由 experts 能夠容忍激進的 (int4) 量化
   比 router 或 attention 更好。 （連接到[MoE serving](../moe/inference-serving.md)。）

## 參考文獻

- 弗蘭塔等人。 _GPTQ。 _ 2022 年。
- 林等人。 _AWQ。 _ 2023 年。
- 蕭等人*SmoothQuant。 * 2022 年。
- 德特默斯等人。 _LLM.int8() / GPTQ 時代異常值分析。 _ 2022 年。
- 米甚拉等。 _2:4 結構化稀疏性。 _ 2021 年；辛頓等人。 _提煉神經網路中的知識。 _ 2015。
