# 解答 — 第一部 · 基礎

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 作為系統的 Transformer、attention 效率、FlashAttention、數值</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

解答[第一部](../foundations/index.md)的練習。數字
使用圓形硬體規格（A100 ≈ 312 TFLOP/s bf16 / 2.0 TB/s；H100 ≈ 990 TFLOP/s
bf16 / 3.35 TB/秒； MI300X ≈ 1.3 PFLOP/s bf16 / 5.3 TB/s)；你的確切增量將
隨你假設的晶片而變化，但*制度*（記憶體與計算限制）是
重要的是。

## 從頭開始 Transformer

??? success "1 — 描繪形狀"
    從 token id $[N]$ 開始：

    | 步驟之後   | 形狀                |
    | ---------- | ------------------- |
    | 嵌入       | $[N, d]$            |
    | $QK^\top$  | $[N, N]$ ← **對 $N$ 二次** |
    | softmax    | $[N, N]$            |
    | $\times V$ | $[N, d_h]$（每個頭）|
    | $W_O$      | $[N, d]$            |
    | LM head    | $[N, V]$            |

    只有 $QK^\top$ 分數矩陣（及其 softmax）對序列長度是二次的——也就是 FlashAttention 攻打的
    那個成本。其餘對 $N$ 都是線性的。

??? success "2 — 頭部暗淡和 GQA 快取節省"
    $d_h = d/h = 4096/32 = 128$。所有 32 個磁頭均具有完整的多磁頭快取 K、V；
    8 頭 GQA 快取僅為 8 → 每個 token KV 快取縮小
    $32/8 = \mathbf{4\times}$。查詢頭保持在 32；僅共用 K,V
    跨 4 個查詢頭組。

??? success "3 - 為什麼殘差和層範數"
    **殘差**($x+\text{sublayer}(x)$)：為漸變提供身份路徑，以便它們
    不要透過深堆疊消失，並讓每一層“細化”
    代表而不是重建它。沒有它，深度模型幾乎無法訓練
    （梯度隨深度呈幾何收縮/爆炸）。**圖層規範**：保留每個
    子層的輸入處於穩定的規模，因此活化不會隨著深度的變化而漂移
    使非線性飽和。沒有它，training 不穩定並且
    學習率脆弱。

??? success "4 — 為什麼快取 K/V 而不是 Q"
    因果掩碼使 attention 成為下三角： token $t$ 關注鍵/
    值 $1..t$，並且*這些向量永遠不會隨著生成的進行而改變* - 所以
    它們可以被快取和重複使用。相比之下，**查詢**僅需要
    **目前**token 正在產生；過去的查詢已經產生了它們的
    輸出並且永遠不會重複使用。所以 K/V 在快取中累積；Q 是新計算的
    每一步都有一個新的 token。

??? success "5 — FFN 與 attention 參數；MoE"
    每層 attention 投影($W_Q,W_K,W_V,W_O$) ≈ $4d^2$；FFN
    （向上 $4d^2$ + 向下 $4d^2$） ≈ $8d^2$。所以**FFN 主導**(~2× attention)
    — 對於 $d=4096$、~$1.3\times10^8$ 與~$6.7\times10^7$ 參數/層。安
    [MoE](../moe/index.md)層用許多 experts 和
    將每個 token 僅路由到其中的 $k$：總 FFN 參數依 expert 成長
    在 _active_ params-per-token 保持 ~$k$ experts' 值時進行計數 — 解耦
    計算能力。

## Transformer 作為一個系統

??? success "1 — 7B 模型的正向觸發器，4096 tokens"
    前向是每個 token 的 $\approx 2P$ FLOP（每個參數一次乘加）：

    $$ 2 \times 7\times10^9 \times 4096 \approx 5.7\times10^{13}\ \text{FLOPs}. $$

    在 60% MFU 的 MI300X 上，持續費率為 $0.6 \times 1.3\times10^{15}
    \約 7.8\times10^{14}$ FLOP/s，所以

    $$ t \approx \frac{5.7\times10^{13}}{7.8\times10^{14}} \approx 73\ \text{ms}. $$

    attention 分數項目 ($\propto N^2$) 在這裡很小 — 在 $N=4096$ 處它添加了
    只有百分之幾——這正是為什麼 $2P$ 法則是個好的第一規則
    估計直到上下文變長。

??? success "2 — 序列長度，其中 attention = 線性 FLOP"
    每層，參數 $\approx 12d^2$（$W_{q,k,v,o}$ + attention $4d^2$）
    FFN $8d^2$)，因此**線性**向前 FLOPs $\approx 2\cdot 12d^2 \cdot N = 24Nd^2$。
    **attention 分數**的 FLOP（QKᵀ + AV）$\approx 4N^2 d$。令兩者相等：

    $$ 4N^2 d = 24 N d^2 \;\Rightarrow\; N = 6d. $$

    對 $d=5120$，$N \approx 3.1\times10^{4}$ 個 token。在這之下，線性 matmul 主導 FLOP 帳單；
    超過它，attention 的二次項就接管——這就是為什麼長上下文的成敗取決於 attention 效率、而非 FFN。

??? success "3 — bf16 LayerNorm：FLOPs、位元組、機制"
    元素 $= 32 \times 2048 \times 4096 \approx 2.68\times10^{8}$。

    -**FLOPs：**平均值、變異數、歸一化、縮放+移位 ≈ ~10 FLOPs/elem →
      $\approx 2.7\times10^{9}$。
    -**位元組：**讀取輸入 + 寫入 bf16 中的輸出 = $2+2 = 4$ 位元組/elem →
      $\approx 1.07\times10^{9}$ 位元組（$\gamma,\beta$ 向量可以忽略不計）。
    -**強度：**$I \approx 2.7\times10^9 / 1.07\times10^9 \approx 2.5$ FLOP/位元組。

    A100 脊 $= 312\text{T}/2.0\text{T} \approx 156$ FLOP/位元組。自從
    $2.5 \ll 156$，LayerNorm**深受記憶體限制**— 幾乎是純數據
    運動。**是的，將其融合到相鄰的矩陣乘/殘差中，因此張量為
    永遠不會為了標準化而往返 HBM。

??? success "4——重新推導 6P 規則"
    根據 token：

    -**正向：**每個參數都用於一次乘加 =**2 FLOPs**→
      $2P$。 *（2 的因數是 MAC：一次乘法 + 一次加法。）*
    -**向後：**你計算梯度 w.r.t。層**輸入**($2P$)
      *和* w.r.t。**權重**($2P$) → $4P$。

    總計 $2P + 4P = 6P$。 *（係數 3 = 一次向前傳球 + 兩次向後傳球；
    $2P \times 3 = 6P$.)* 每個 matmul 出現的地方都會出現它的轉置
    向後兩次——這就是 3 的由來。

## Attention 效率

??? success "1 - GQA 模型的 KV 快取大小"
    每層 token：$2\ (\text{K,V}) \times n_{kv} \times d_h \times 2\
    \text{bytes} = 2 \times 8 \times 128 \times 2 = 4096$ B $= 4$ KB。

    $$ 4\,\text{KB} \times L(32) \times N(8192) \times B(16) \approx 1.7\times10^{10}\ \text{B} \approx 17\ \text{GB}. $$

    bf16中的7B權重為$\approx 14$ GB。所以在這個批次/長度下**KV
    快取已經超過了權重**——GQA/MLA 和
    為什麼 decode 受記憶體限制。

??? success "2 — 單一 decode 步的算術強度"
    一個 query 對 $t$ 個快取 key 做 attention。FLOP $\approx \underbrace{2td_h}_{QK^T}
    + \underbrace{2td_h}_{AV} = 4td_h$。讀 K、V 的 bytes $\approx 2\cdot t d_h \cdot 2 = 4td_h$。因此

    $$ I = \frac{4td_h}{4td_h} = O(1)\ \text{FLOP/byte}, $$

    獨立於$t$。 decode 讀取*整個* KV 快取以發出**一個**token
    — 規範的記憶體綁定操作，批次許多請求的原因是
    主 throughput 控制桿。

??? success "3 — 具有 16-token 塊的碎片廢物"
    長度為 $\ell$ 的序列使用 $\lceil \ell/16\rceil$ 區塊；浪費的插槽
    $= 16\lceil\ell/16\rceil - \ell$。與$\ell$統一，$\ell \bmod 16$是
    ~在 $\{0..15\}$ 上統一，因此意味著浪費 $\approx 7.5$ 插槽/序列。作為一個
    已用記憶體的比例：$7.5 / \overline{\ell} = 7.5/2048 \approx 0.37\%$。
    可以忽略不計——這正是分頁 KV 使用小塊而不是
    pre-reserving 最大長度（浪費約 50%）。

??? success "4 — MLA 快取比率與 GQA"
    GQA 在每個 token 層快取 $2\,n_{kv}d_h$； MLA 快取單一潛在的暗淡
    $d_c$（K 和 V 在計算時透過上投影重建）：

    $$ \frac{\text{MLA}}{\text{GQA}} \approx \frac{d_c}{2\,n_{kv}d_h}. $$

對於 DeepSeek 風格的 $d_c \approx 512$，這是
數量級較小的快取。**交易**：更少的記憶體和頻寬
decode（綁定約束）以換取額外的 FLOP 來向上投影
每一步都潛在地回到 K/V——這很重要，因為 decode 是
受記憶體限制，因此增加的計算幾乎是免費的。

## Flashattention

??? success "1 — Online-softmax 組合器是精確的"
    對於兩個具有局部最大值 $m_1,m_2$、分母 $\ell_1,\ell_2$ 的區塊，以及
    部分輸出 $O_1,O_2$，讓 $m=\max(m_1,m_2)$ 和

    $$ \ell = \ell_1 e^{m_1-m} + \ell_2 e^{m_2-m},\qquad
       O = \frac{O_1\,\ell_1 e^{m_1-m} + O_2\,\ell_2 e^{m_2-m}}{\ell}. $$

    因為$e^{x_i-m_1}\cdot e^{m_1-m} = e^{x_i-m}$，每一項都被重新表達
    反對*全局*最大值，因此 $\ell$ 變成 $\sum_i e^{x_i-m}$，$O$ 變成
    $\sum_i \mathrm{softmax}(x)_i v_i$ 透過聯合 —**與一次性相同**
    softmax。摺疊運算具結合律，所以不論切成幾塊都是精確的。

??? success "2 — 為什麼減去運行最大值"
    分數為 $+100$，樸素的 fp16 計算出 $e^{100}$，結果溢出
    (fp16 最大 $= 65504 \ll e^{100}$) → `inf` → 歸一化後的 `nan`。的
    穩定形式先減去最大值：$e^{100-100}=1$，$[0,1]$中的所有項，
    沒有溢出。減去最大值在數學上沒有任何變化（它取消了
    比例）但一切都是數字。

??? success "3 — 在因果關係下跳過完全屏蔽的圖塊"
    因果屏蔽意味著查詢圖塊 $i$ 僅需要關鍵圖塊 $j \le i$。的一個
    $n\times n$ 平鋪網格你計算下三角形，$n(n+1)/2$ 平鋪。對於
    $N=4096$ 與瓦片 $128$、$n=32$：計算出 $1024$ 的 $= 32\cdot33/2 = 528$ →
    **~48% 的 FLOP 被消除**，接近 $\tfrac{n-1}{2n}\to
    50\%$ 漸近線。

??? success "4 — HBM 位元組：N=8192、d=128（每頭）時的樸素 vs 快閃記憶體"
    Naive 實現了 $N\times N$ 分數矩陣（寫入然後重新讀取
    Softmax）：$\approx 2 \cdot N^2 \cdot 2\,\text{B} = 4N^2 \approx 2.7\times10^8$
    B ≈**270 MB**。 Flash 從不寫入$S$；它串流 Q、K、V 一次並寫入 O：
    $\approx (3{\cdot}Nd + Nd)\cdot 2 = 8Nd \approx 8\times10^6$ B ≈**8 MB**—
    大約**流量減少 30 倍**。在 H100 上（脊 ≈ $990\text{T}/3.35\text{T}
    \大約 295$ FLOP/位元組）天真的版本位於山脊左側
    （$S$ 往返內存受限）；閃光燈將 $I$ 升過山脊進入
    受計算限制的領域。

## 數字和精度

??? success "1 — 最大有限 `exp` logit：fp16 與 bf16"
    `exp(x)` 是有限的，而 $x < \ln(\text{max normal})$ 是有限的。

    -**fp16**（5 個指數位，最大 $65504$）：$x < \ln 65504 \approx 11.1$。
    -**bf16**（8 個指數位，最大 $\approx 3.4\times10^{38}$）：$x < \ln(3.4\times10^{38}) \approx 88.7$。

    8 vs 5 指數位給出 bf16 fp32 等級 *範圍*，因此 softmax logits
    立即溢出 fp16 在 bf16 中非常安全——bf16 的一個核心原因是
    預設 training 資料類型。

??? success "2 — bf16 遺失 $10^6 \times$ `1e-3` 總和；fp32 恢復它"
    真和$= 1000$。 bf16 有 8 個尾數位（約 2-3 位十進制數字）。一旦
    運行總次數約 256，ULP 超過 $10^{-3}$，因此每個新加數
    四捨五入（**沼澤**）並且總和遠低於 1000。 fp32
    累加器（23 個尾數位，約 7 位元）可保持 $10^{-3}$ 有效
    超過 1000，恢復正確答案。教訓：**即使在下列情況下也要減少 fp32**
    輸入為 bf16。

??? success "3 — 動態損失縮放；為什麼 bf16 很少減半"
    保持縮放$S$：在`backward`之前將損失乘以$S$，取消縮放梯度
    在`step`之前。若任意等級為`inf`/`nan`，**跳過此步驟，將$S$減半**；
    在 $N$ 清潔步驟之後，**雙 $S$**。半分支僅在
    溢出，這是一個 fp16 問題（5 個指數位，很小的範圍）。 BF16 股
    fp32 的指數範圍，因此梯度基本上不會上溢/下溢 —
    損失縮放是 fp16 的修復，在 bf16 中通常是不必要的。

??? success "4 — fp8 E4M3，每個張量尺度，最大值 = 1000"
    E4M3 最大可代表$\approx 448$。挑選秤 $s = 448/1000 = 0.448$ 所以
    張量最大值接近範圍的頂端。**量化**$q = \text{cast}_{E4M3}(x
    \cdot s)$;**dequantize**$\hat x = q / s$。具有 3 個尾數位
    半 ULP 相對誤差為 $\le 2^{-(3+1)} = 6.25\%$（典型值為幾個百分點）。
    誤差主要由**粗尾數**決定，而非比例選擇 -
    這就是為什麼 fp8 需要細粒度（每個張量/每個區塊）縮放並且是
    應用於容錯張量（例如路由 expert 權重），而不是 router。
