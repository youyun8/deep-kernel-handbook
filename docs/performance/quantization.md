# 量化與壓縮

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../../foundations/numerics-precision/">數值與精度</a>、<a href="../../foundations/attention-efficiency/">記憶體受限的 decoding</a></span>
  <span class="chip"><strong>硬體：</strong>理論無； GPU對標</span>
</div>

壓縮讓模型的儲存成本更低、運行速度更快 —— 尤其是在 [memory-bound decode](../foundations/attention-efficiency.md) 階段，將權重 位元組減半大約能把 latency 減半。本頁涵蓋**量化**（最大的 槓桿：PTQ vs QAT，以及 GPTQ/AWQ 系列），接著是**剪枝（pruning）**和 **蒸餾（distillation）**。[MoE serving page](../moe/inference-serving.md) 把這些觀念具體套用到 experts 上。

## 量化基礎知識

量化把高精度值映射到一個小的整數（或低位元浮點數） 網格上。對張量 $x$ 而言，標準的**仿射（affine，非對稱）**方案把 $x$ 量化為 $b$ 位元：

$$ q = \mathrm{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s\,(q - z), $$

其中 $x$ 為原始的全精度值、$q$ 為量化後的整數碼、$\hat{x}$ 為 反量化（dequantize）後的近似值、$s$ 為 scale（步長）、$z$ 為 zero-point （把實數 $0$ 對應到的整數碼）。選擇 $s$ 與 $z$ 使整數網格剛好覆蓋 張量的數值範圍。

當數值分佈關於 $0$ 對稱時，可採用**對稱（symmetric）量化**：令 zero-point $z = 0$，並取

$$ s = \frac{\max|x|}{2^{\,b-1}-1}, $$

其中 $\max|x|$ 為張量中絕對值最大者，$2^{\,b-1}-1$ 為有號 $b$ 位元整數 的正側最大碼（例如 int8 時為 $127$）。此時 $q = \mathrm{round}(x/s)$、 $\hat{x} = s\,q$，省去存放 $z$ 的成本。

**粒度（granularity）**決定一個 scale 涵蓋多少元素，是準確度與成本的取捨：

- **per-tensor**：整個張量共用一個 $s$（與 $z$） —— 最便宜，最不準確。
- **per-channel / per-token**：每個輸出 channel（矩陣的列）或每個 token （矩陣的行）一個 $s$ —— 權重量化的標準做法。
- **per-block / group**：每 $g$ 個元素一組、各自一個 $s$（例如 $g=128$） —— 4 位元權重的甜蜜點；與 [microscaling / MX formats](../foundations/numerics-precision.md) 搭配。

形式上，令 $\mathcal{B}$ 為某個元素所屬的分組（per-tensor 時 $\mathcal{B}$ 為 整個張量、per-channel 時為一列、per-block 時為 $g$ 個元素），則該組共用 $s_{\mathcal{B}} = \dfrac{\max_{i\in\mathcal{B}}|x_i|}{2^{\,b-1}-1}$。

**MXFP4** 即是 per-block 的一個具體實例：取 $g = 32$，每個 block 共用一個 **E8M0**（以 `uint8` 表示、即 8 位元純指數）的 power-of-two scale，而每個元素 本身以 FP4（$b = 4$）儲存。其**有效位元數（effective bits）**為每元素位元數 加上攤提到每元素的 scale 成本：

$$ b_{\text{eff}} = b + \frac{\text{scale bits}}{g} = 4 + \frac{8}{32} = 4.25 \ \text{bits/element}. $$

「量化什麼」有兩個正交的軸：

- **weight-only**（W8/W4，activation 維持 BF16）：非常適合**decode**，因為 decode 受權重頻寬限制 —— 你能減少每個 token 需讀取的位元組數。計算時使用 反量化後的值，所以它**不會**明顯加速 compute-bound 的 prefill。
- **weight + activation**（W8A8 / FP8）：同時也加速 **matmul**（在整數 / FP8 的 Tensor Core / Matrix Core 上），有助於 compute-bound 的 prefill / training —— 但 activation 較難量化（異常值問題，見下）。

### 量化誤差與 SQNR

把 $x$ 量化再反量化會引入捨入誤差 $e = \hat{x} - x$。在步長 $\Delta = s$ 遠小於訊號變化的常見假設下，$e$ 近似為**均勻分佈**於 $[-\Delta/2,\ \Delta/2]$，其平均為 $0$、變異數為

$$ \mathrm{Var}(e) = \frac{\Delta^2}{12}, $$

其中 $\Delta = s$ 為量化步長。將訊號功率與此噪聲功率相比，得到 **訊號量化噪聲比（SQNR, signal-to-quantization-noise ratio）**。對滿量程 （full-scale）訊號，標準結果為

$$ \mathrm{SQNR} \approx 6.02\,b + 1.76 \ \text{dB}, $$

其中 $b$ 為位元數。重點是：**每多 1 個位元約增加 6 dB**（精度約增 4 倍）， 這也是 int8 → int4 會明顯掉精度的根本原因。

### 記憶體與頻寬的收益

權重所佔位元組數正比於位元寬度。因此 FP4 相對於 BF16 只佔

$$ \frac{4\ \text{bits}}{16\ \text{bits}} = \frac14 $$

的位元組數，對於 **memory-bound decode**（GEMM 在串流讀取權重、而非 compute-bound）最多可帶來 $4\times$ 的**權重頻寬**。這正是為什麼 低位元權重對 **decode** 的幫助遠大於對 **prefill** 的幫助：decode 受權重讀取頻寬限制，prefill 受算力限制。

!!! Example "數值例子：13B W4 decode 的下界"
    13B BF16 權重約 $13\times10^9\cdot2=26$ GB；W4 約 $13\times10^9\cdot0.5=6.5$ GB。若有效 HBM 頻寬是 2 TB/s，純權重讀取下界從 $26/2000=13$ ms/token 降到 $6.5/2000=3.25$ ms/token，理想上接近 4 倍。實際速度會被 KV cache、activation、dequant 與 launch overhead 稀釋，但一開始的量級就是這樣來的。

### 異常值（outliers）

由於量化範圍由 $\max|x|$ 決定（對稱量化中 $s = \max|x| / (2^{\,b-1}-1)$）， **單一異常值**就會撐大 $s$，使得整個分組的步長 $\Delta = s$ 變大，把解析度 浪費在那一個大值上，而其餘所有「正常」值都只落在少數幾個碼上、損失精度。 這正是 **per-channel / per-group scaling**、**clipping（截斷）**、以及 outlier-aware（如 SmoothQuant、AWQ）方案存在的動機。

## PTQ 與 QAT

- **訓練後量化（PTQ, post-training quantization）**：用一個小的 **校準（calibration）**集來挑選 scale（對 GPTQ/AWQ 還會做誤差修正）。 便宜、快速、不需要 retraining —— 是 LLM inference 的預設做法。
- **量化感知訓練（QAT, quantization-aware training）**：在 training 期間 模擬量化（捨入處用 straight-through estimator 反傳梯度），使模型學會對 量化的魯棒性。能在很低的位元數下恢復較高精度，但需要一次 training 流程。

LLM PTQ 的難點是 **activation 異常值**：某些 channel 的量級極大，若天真地 量化就會撐大 scale、壓垮其餘所有值（見上方〈異常值〉）。GPTQ/AWQ 系列 正是為了處理這個問題而生。

### GPTQ — 糾錯權重量化

GPTQ 一次量化權重的一個 column，並在每個 column 捨入後**更新尚未量化的 剩餘權重以補償**所引入的誤差（一種使用校準 activation 的二階 / Hessian-based 修正）。結果是精度損失很小的 3–4 位元 _weight-only_ 量化， 屬於 weight-only → decode 的理想選擇。

### AWQ — 激活感知權重量化

AWQ 觀察到並非所有權重同等重要：與高量級（顯著）activation channel 相乘的 權重最重要。它在量化前**放大（scale up）**那些顯著的 channel（再對應地補償）， 保護它們不受捨入誤差影響 —— 而且不需要反向傳播。在 4 位元上通常可匹敵或勝過 GPTQ，且簡單、快速。

### SmoothQuant — 使激活可量化

對於 W8A8，SmoothQuant 透過一個數學上等價的 per-channel 重新縮放，把 **activation 的異常值「搬移」到權重上**，使 activation 與權重都變得容易 量化為 int8 —— 從而實現 prefill / serving 用的快速 int8 matmul。

```python
# The shared idea: choose per-channel scales so the *product* is unchanged
# but each operand is easier to quantize. (Conceptual sketch.)
# y = (x / s) @ (s * W)   # s absorbs outliers from x into W or vice-versa
```

!!! Tip "該用哪一個"
    Memory-bound 的 **decode**、想要簡單 + 準確 → **AWQ 或 GPTQ** （W4，weight-only）。也想要更快的 **prefill / serving 計算** → **SmoothQuant** 或 **FP8**（W8A8）。要在 4 位元以下仍維持準確度 → 考慮 **QAT**。

## FP8 作為量化

FP8（E4M3 / E5M2）是一種帶有浮點網格的量化 —— 它的動態範圍比 int8 更寬， 更適合 activation，且在 H100 / MI300 上有原生硬體加速。搭配 per-tensor / per-block scale，它可用於 inference（W8A8 風格），並且越來越常用於 **training**（DeepSeek-V3）。格式細節與縮放規則見 [numerics & precision](../foundations/numerics-precision.md)。

## 剪枝（Pruning）

剪枝是直接移除權重 / 結構，而非降低其精度：

- **非結構化（unstructured）**（把個別小權重歸零）：理論上壓縮率高， 但不規則的稀疏性很難在 GPU 上加速。
- **結構化（structured）**（移除整個 head / channel / layer）：壓縮較少， 但產生一個更小的「dense」模型，能在現成 kernels 上快速運行。
- **2:4 半結構化（semi-structured）**（每 4 個權重中有 2 個為零）：有硬體 支援的折衷方案 —— NVIDIA 的 sparse Tensor Core 對此提供約 $2\times$ 的效能。 在有支援時是實務上的甜蜜點。

剪枝通常需要 fine-tuning 才能恢復品質；當你能利用硬體稀疏性、或想要更小的 dense 模型時，它最有吸引力。

## 蒸餾

訓練一個小的 **student** 去模仿一個大的 **teacher**（匹配輸出分佈 / soft logits，有時還包含中間層特徵）。不同於量化 / 剪枝，它產生的是一個 真正更小的架構，並能轉移能力，代價是需要一次 training 流程與存取 teacher。 常與上述方法結合（先蒸餾、再量化 student）。

## 要點

- 量化透過 scale / zero-point 把值映射到低位元網格；**粒度** （per-tensor → per-group）以準確度換取成本。每多 1 位元約 +6 dB SQNR。
- **weight-only（GPTQ/AWQ、W4）**加速 memory-bound 的 **decode**； **weight + activation（SmoothQuant/FP8、W8A8）**還能加速 compute-bound 的 **prefill / training**，但必須馴服 **activation 異常值**。
- **PTQ**（僅校準）是 LLM 的預設；**QAT** 以一次 training 流程的代價換取 在極低位元下的準確度。
- **剪枝**（尤其是 2:4 結構化）與**蒸餾**是互補的壓縮工具；可彼此結合。

## 練習

!!! Tip "解決方案"
    參考解答位於 [解答頁](../solutions/performance.md) 上。請先嘗試每個練習，再展開解答。

1. 對一個含有單一異常 channel 的張量，推導 int8 仿射量化 / 反量化，並比較 per-tensor 與 per-channel scaling 下的最大量化誤差。
2. 在單一線性層上實作 AWQ 的顯著 channel 縮放，並與樸素的 per-channel int4 比較 perplexity。
3. 用 [memory-bound](../foundations/attention-efficiency.md) 參數，估計在 13B 模型上採用 W4 weight-only 對 decode-latency 的改善。
4. 對 MoE，論證為什麼被路由的 experts 比 router 或 attention 更能容忍激進的 （int4）量化。（連接到 [MoE serving](../moe/inference-serving.md)。）

## 參考文獻

[1] E. Frantar and D. Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers," in *Proc. ICLR*, 2023.

[2] J. Lin *et al.*, "AWQ: Activation-aware weight quantization for LLM compression and acceleration," in *Proc. MLSys*, 2024.

[3] G. Xiao *et al.*, "SmoothQuant: Accurate and efficient post-training quantization for large language models," in *Proc. ICML*, 2023.

[4] T. Dettmers *et al.*, "LLM.int8(): 8-bit matrix multiplication for transformers at scale," in *Proc. NeurIPS*, 2022.

[5] A. Mishra *et al.*, "Accelerating sparse deep neural networks," *arXiv:2104.08378*, 2021.

[6] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv:1503.02531*, 2015.
