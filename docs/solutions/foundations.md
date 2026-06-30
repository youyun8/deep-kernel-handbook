# 解答 — 基礎

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 作為系統的 Transformer、attention 效率、FlashAttention、數值</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

解答[基礎篇](../foundations/index.md)的練習。數字使用簡化的硬體規格（A100 ≈ 312 TFLOP/s BF16 / 2.0 TB/s；H100 ≈ 990 TFLOP/s BF16 / 3.35 TB/s；MI300X ≈ 1.3 PFLOP/s BF16 / 5.3 TB/s）；你算出來的確切數字會隨假設的晶片而變化，但重要的是*機制*（compute-bound 還是 memory-bound）。

## 從零實作 Transformer

??? Success "1 — 描繪形狀"
    從 token id $[N]$ 開始：

    | 步驟之後   | 形狀                |
    | ---------- | ------------------- |
    | embedding  | $[N, d]$            |
    | $QK^\top$  | $[N, N]$ ← **對 $N$ 二次** |
    | softmax    | $[N, N]$            |
    | $\times V$ | $[N, d_h]$（每個頭）|
    | $W_O$      | $[N, d]$            |
    | LM head    | $[N, V]$            |

    只有 $QK^\top$ 分數矩陣（及其 softmax）對序列長度是二次的 —— 也就是 FlashAttention 攻打的 那個成本。其餘對 $N$ 都是線性的。

??? Success "2 — head dim 與 GQA 快取節省"
    $d_h = d/h = 4096/32 = 128$。完整多頭 attention 的 32 個頭各自保留自己的 K、V 快取；8 頭的 GQA 只快取 8 組 → 每個 token 的 KV 快取縮小 $32/8 = \mathbf{4\times}$。query 頭數仍是 32，只是每 4 個 query 頭共用一組 K、V。

??? Success "3 — 為什麼需要殘差連接和 layer norm"
    **residual connection**（$x+\text{sublayer}(x)$）：給梯度一條恆等路徑，讓它們不會隨著深堆疊而消失，也讓每一層做的是「修正」表徵而非「重建」表徵。沒有它，深層模型幾乎無法訓練（梯度會隨深度幾何級數收縮或爆炸）。**layer norm**：讓每個子層的輸入維持在穩定的數值尺度，這樣 activation 不會隨深度漂移、讓非線性飽和。沒有它，training 會不穩定，且 learning rate 必須調得很保守。

??? Success "4 — 為什麼快取 K/V 而不是 Q"
    causal mask 讓 attention 變成下三角結構：token $t$ 只關注 key/value $1..t$，而*這些向量在生成過程中永遠不會改變*，所以可以被快取重複使用。相比之下，**query** 只需要*目前*正在生成的那個 token；過去的 query 早就產生過輸出，不會被重複使用。所以 K/V 會在快取裡持續累積；Q 則是每一步針對新 token 重新計算。

??? Success "5 — FFN 與 attention 參數；MoE"
    每層 attention 投影（$W_Q,W_K,W_V,W_O$）≈ $4d^2$；FFN（up $4d^2$ + down $4d^2$）≈ $8d^2$。所以**FFN 主導**（約 2× attention）—— 在 $d=4096$ 時分別約是 $1.3\times10^8$ 與 $6.7\times10^7$ 參數/層。[MoE](../moe/index.md) 層則用許多 expert，每個 token 只路由到其中 $k$ 個：總 FFN 參數量隨 expert 數成長，但每個 token 的 *active* 參數量只計入那 $k$ 個 expert —— 把總參數量與每 token 計算量解耦。

## 作為系統的 Transformer

??? Success "1 — 7B 模型、4096 tokens 的 forward FLOP"
    前向傳播每個 token 約 $2P$ FLOP（每個參數一次乘加）：

    $$ 2 \times 7\times10^9 \times 4096 \approx 5.7\times10^{13}\ \text{FLOP}. $$

    在 60% MFU 的 MI300X 上，可用算力約 $0.6 \times 1.3\times10^{15} \approx 7.8\times10^{14}$ FLOP/s，所以

    $$ t \approx \frac{5.7\times10^{13}}{7.8\times10^{14}} \approx 73\ \text{ms}. $$

    Attention 分數項（$\propto N^2$）在這裡很小 —— 在 $N=4096$ 時只多了百分之幾 —— 這正是為什麼 $2P$ 法則是很好的第一近似，直到上下文變長才需要修正。

??? Success "2 — attention 與線性 FLOP 相等的序列長度"
    每層參數量 ≈ $12d^2$（$W_{q,k,v,o}$ 的 attention 投影 $4d^2$ + FFN $8d^2$），因此**線性**部分的前向 FLOP ≈ $2\cdot 12d^2 \cdot N = 24Nd^2$。**attention 分數**的 FLOP（$QK^\top$ + $AV$）≈ $4N^2 d$。令兩者相等：

    $$ 4N^2 d = 24 N d^2 \;\Rightarrow\; N = 6d. $$

    對 $d=5120$，$N \approx 3.1\times10^{4}$ 個 token。在這之下，線性 matmul 主導 FLOP 帳單；超過它，attention 的二次項就接管 —— 這就是為什麼長上下文的成敗取決於 attention 效率、而非 FFN。

??? Success "3 — BF16 LayerNorm：FLOP、位元組、機制"
    元素數 $= 32 \times 2048 \times 4096 \approx 2.68\times10^{8}$。

    - **FLOP：**平均值、變異數、normalize、縮放＋平移 ≈ 10 FLOP/elem → $\approx 2.7\times10^{9}$。
    - **位元組：**讀輸入 + 寫 BF16 輸出 = $2+2=4$ 位元組/elem → $\approx 1.07\times10^{9}$ 位元組（$\gamma,\beta$ 向量可忽略）。
    - **算術強度：**$I \approx 2.7\times10^9 / 1.07\times10^9 \approx 2.5$ FLOP/byte。

    A100 的脊點 $= 312\text{T}/2.0\text{T} \approx 156$ FLOP/byte。因為 $2.5 \ll 156$，LayerNorm **嚴重 memory-bound**——幾乎全是純數據搬動。沒錯：把它融合進相鄰的 matmul/residual，這樣 tensor 就不必為了 normalize 而往返一次 HBM。

??? Success "4 — 重新推導 6P 規則"
    按每個 token 計算：

    - **forward：**每個參數都用於一次乘加 = **2 FLOP** → $2P$。*（係數 2 來自 MAC：一次乘法 + 一次加法。）*
    - **backward：**你要算梯度 w.r.t. 該層的**輸入**（$2P$），*也*要算 w.r.t. **權重**（$2P$）→ $4P$。

    總計 $2P + 4P = 6P$。*（係數 3 = 一次 forward pass + 兩次 backward pass；$2P \times 3 = 6P$。）*每個出現過 matmul 的地方，backward 都會用到它的轉置兩次 —— 這就是係數 3 的來源。

## Attention 效率

??? Success "1 — GQA 模型的 KV 快取大小"
    每個 token、每層：$2\ (\text{K,V}) \times n_{kv} \times d_h \times 2\ \text{bytes} = 2 \times 8 \times 128 \times 2 = 4096\ \text{B} = 4\ \text{KB}$。

    $$ 4\,\text{KB} \times L(32) \times N(8192) \times B(16) \approx 1.7\times10^{10}\ \text{B} \approx 17\ \text{GB}. $$

    BF16 下的 7B 權重約 14 GB。所以在這個 batch/長度下，**KV cache 已經超過權重大小**——這正是為什麼需要 GQA/MLA，也是 decode 受記憶體限制的原因。

??? Success "2 — 單一 decode 步的算術強度"
    一個 query 對 $t$ 個快取 key 做 attention。FLOP $\approx \underbrace{2td_h}_{QK^T} + \underbrace{2td_h}_{AV} = 4td_h$。讀 K、V 的 bytes $\approx 2\cdot t d_h \cdot 2 = 4td_h$。因此

    $$ I = \frac{4td_h}{4td_h} = O(1)\ \text{FLOP/byte}, $$

    與 $t$ 無關。Decode 要讀進*整個* KV cache 才能吐出**一個** token —— 是典型的 memory-bound 操作，這也是為什麼把許多請求一起 batch 是拉高 throughput 的主要槓桿。

??? Success "3 — 16-token block 的碎片浪費"
    長度為 $\ell$ 的序列用 $\lceil \ell/16\rceil$ 個 block；浪費的 slot 數 $= 16\lceil\ell/16\rceil - \ell$。若 $\ell$ 均勻分布，$\ell \bmod 16$ 也大約均勻分布在 $\{0..15\}$ 上，平均浪費 $\approx 7.5$ slot/序列。換成已用記憶體的比例：$7.5/\overline{\ell} = 7.5/2048 \approx 0.37\%$——可以忽略不計。這正是 paged KV 用小 block、而不是預先保留最大長度（那樣會浪費約 50%）的原因。

??? Success "4 — MLA 快取比率與 GQA"
    GQA 每個 token、每層快取 $2\,n_{kv}d_h$；MLA 只快取一個 latent 向量、維度 $d_c$（K 和 V 在使用時透過 up-projection 重建）：

    $$ \frac{\text{MLA}}{\text{GQA}} \approx \frac{d_c}{2\,n_{kv}d_h}. $$

    對 DeepSeek 風格的 $d_c \approx 512$，這個比值小了一個數量級。**取捨**：用較少的記憶體與頻寬換 decode 時的限制條件，但每一步都要多花一點 FLOP 把 latent 向量投影回 K/V——這划算是因為 decode 本來就是 memory-bound，多出來的計算幾乎是免費的。

## 從零實作 FlashAttention

??? Success "1 — online softmax 的合併運算是精確的"
    設兩個區塊各有局部最大值 $m_1,m_2$、分母 $\ell_1,\ell_2$ 與部分輸出 $O_1,O_2$，令 $m=\max(m_1,m_2)$，則

    $$
    \ell = \ell_1 e^{m_1-m} + \ell_2 e^{m_2-m},\qquad
       O = \frac{O_1\,\ell_1 e^{m_1-m} + O_2\,\ell_2 e^{m_2-m}}{\ell}.
    $$

    因為 $e^{x_i-m_1}\cdot e^{m_1-m} = e^{x_i-m}$，每一項其實都被重新表達成相對於*全局*最大值，因此合併後 $\ell$ 就等於 $\sum_i e^{x_i-m}$、$O$ 就等於 $\sum_i \mathrm{softmax}(x)_i v_i$——和一次算完的 softmax 完全相同。這個合併運算具結合律，所以不管切成多少塊都是精確的。

??? Success "2 — 為什麼要減去 running max"
    分數為 $+100$ 時，樸素算法直接算 $e^{100}$，會溢出（FP16 最大值 $= 65504 \ll e^{100}$）→ `inf` → normalize 後變成 `nan`。穩定寫法先減掉最大值：$e^{100-100}=1$，之後所有項都落在 $[0,1]$，不會溢出。減去最大值在數學上沒有任何影響（分子分母同時消掉同一個比例因子），但數值上差很多。

??? Success "3 — 在因果關係下跳過完全被遮罩的 tile"
    causal mask 代表 query tile $i$ 只需要 key tile $j \le i$。在一個 $n\times n$ 的 tile 網格裡，你只算下三角部分，共 $n(n+1)/2$ 個 tile。對 $N=4096$、tile 大小 $128$（$n=32$）：只需算 $32\cdot33/2 = 528$ 個 tile，而不是 $1024$ 個 → **省下約 48% 的 FLOP**，接近漸近值 $\tfrac{n-1}{2n}\to 50\%$。

??? Success "4 — HBM 位元組：N=8192、d=128（每頭）時的 naive vs FlashAttention"
    naive 實現要具現化整個 $N\times N$ 分數矩陣（寫入後再讀回來做 softmax）：$\approx 2 \cdot N^2 \cdot 2\,\text{B} = 4N^2 \approx 2.7\times10^8$ B ≈ **270 MB**。FlashAttention 從不寫出 $S$；它把 Q、K、V 各串流一次，只寫出 O：$\approx (3{\cdot}Nd + Nd)\cdot 2 = 8Nd \approx 8\times10^6$ B ≈ **8 MB**——流量大約少了 **30 倍**。在 H100 上（脊點 ≈ $990\text{T}/3.35\text{T} \approx 295$ FLOP/byte），naive 版本落在脊點左側（往返 $S$ 讓它 memory-bound）；FlashAttention 把 $I$ 拉過脊點，進入 compute-bound 區域。

## 數值與精度

??? Success "1 — 最大有限 `exp` logit：FP16 與 BF16"
    `exp(x)` 在 $x < \ln(\text{max normal})$ 時才是有限值。

    - **FP16**（5 個指數位，最大 $65504$）：$x < \ln 65504 \approx 11.1$。
    - **BF16**（8 個指數位，最大 $\approx 3.4\times10^{38}$）：$x < \ln(3.4\times10^{38}) \approx 88.7$。

    8 個指數位相對 5 個指數位，讓 BF16 擁有 FP32 等級的*範圍*，因此會讓 FP16 立即溢出的 softmax logit，在 BF16 裡完全安全——這正是 BF16 成為 training 預設資料型別的核心原因之一。

??? Success "2 — BF16 在 $10^6 \times$ `1e-3` 求和中遺失精度；FP32 能恢復"
    真實總和 $= 1000$。BF16 只有 8 個尾數位（約 2–3 位十進制有效數字）。一旦累加值跑到約 256，ULP 就超過 $10^{-3}$，後面每個新加數都會被**捨入掉**，總和會遠低於 1000。FP32 累加器（23 個尾數位，約 7 位有效數字）即使總和超過 1000，仍能精確表示 $10^{-3}$，所以能還原正確答案。教訓：**即使輸入是 BF16，也要用 FP32 累加**。

??? Success "3 — 動態 loss scaling；為什麼 BF16 很少需要減半"
    維護一個縮放因子 $S$：在 `backward` 之前把 loss 乘上 $S$，在 `step` 之前把梯度除回去取消縮放。若任何梯度出現 `inf`/`nan`，**跳過這一步並把 $S$ 減半**；連續 $N$ 個乾淨步驟之後再把 $S$ 加倍。減半只會在溢位時發生，而這本質上是 FP16 的問題（5 個指數位，動態範圍很小）。BF16 共享 FP32 的指數範圍，梯度基本不會上溢/下溢——loss scaling 是 FP16 的修補方案，在 BF16 上通常沒有必要。

??? Success "4 — FP8 E4M3，per-tensor scale，最大值 = 1000"
    E4M3 最大可表示值 ≈ 448。選擇 scale $s = 448/1000 = 0.448$，讓張量最大值落在範圍頂端附近。**quantize**：$q = \text{cast}_{E4M3}(x \cdot s)$；**dequantize**：$\hat x = q / s$。3 個尾數位帶來的半 ULP 相對誤差為 $\le 2^{-(3+1)} = 6.25\%$（典型值是百分之幾）。誤差主要來自**粗糙的尾數**，而不是 scale 選擇本身——這也是為什麼 FP8 需要細粒度（per-tensor/per-block）的縮放，而且通常用在容錯度較高的張量上（例如被路由的 expert 權重），而不是用在 router 上。
