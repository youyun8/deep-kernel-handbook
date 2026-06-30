# 詞彙表

手冊中用到的術語的簡明定義。連結指向各自詳述的頁面。技術類詞彙維持原文（英文），不刻意翻譯成中文；只有少數已經有通行中文說法的概念（例如「算術強度」「占用率」）才以中文為主、英文附在括號裡。若詞彙有對應的數學表達式，會列出公式並說明每個符號（notation）的意思。

## 數學基礎

**vector（向量）** ：$n$ 個數的有序排列，記為 $v\in\mathbb{R}^n$；幾何上是 $n$ 維空間裡從原點出發的箭頭。一個 token 的 embedding 就是 $\mathbb{R}^d$ 裡的一個向量，$d$ 為 embedding 維度。見 [向量、矩陣與線性映射](math/linear-algebra.md)。

**matrix（矩陣）** ：把向量映射成向量的線性變換，記為 $A\in\mathbb{R}^{m\times n}$（$m$ 列、$n$ 行）；Transformer 裡幾乎所有操作都是矩陣乘法加上少量非線性。

**rank（秩）** ：矩陣 $A$ 的行（或列）空間的維度，記為 $\operatorname{rank}(A)$；上界是 $\min(m,n)$。見 [低秩矩陣與矩陣分解](math/low-rank.md)。

**low-rank（低秩） / low-rank approximation（低秩近似）** ：當 $\operatorname{rank}(A)=r\ll\min(m,n)$ 時，$A$ 可用遠少於 $mn$ 個數值（約 $r(m+n)$ 個）表示；保留一個矩陣前 $r$ 個最大的奇異值方向，就是該矩陣最佳的 rank-$r$ 近似，是 LoRA 等技術的基礎。

**SVD（singular value decomposition，奇異值分解）** ：把矩陣分解成 $M=U\Sigma V^\top$。$U,V$ 為正交矩陣，行分別是左、右奇異向量；$\Sigma$ 為對角矩陣，對角線上是依大小排序的奇異值 $\sigma_1\ge\sigma_2\ge\dots\ge0$。

**LoRA（Low-Rank Adaptation）** ：用兩個低秩矩陣的乘積 $\Delta W=BA$（$B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$、$r\ll\min(m,n)$）去近似權重的更新量，大幅減少微調時需要訓練的參數。

**gradient（梯度）** ：損失函數 $\mathcal{L}$ 對參數向量 $\theta$ 的偏微分向量 $\nabla_\theta\mathcal{L}$，指向損失上升最快的方向；gradient descent 沿其反方向更新參數 $\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}$，其中 $\eta$ 為 learning rate。見 [梯度、反向傳播與自動微分](math/calculus.md)。

**chain rule（鏈式法則）** ：若 $y=f(g(x))$，則 $\dfrac{dy}{dx}=\dfrac{dy}{dg}\cdot\dfrac{dg}{dx}$——複合函數的導數等於各層導數的乘積；backpropagation 的數學基礎。

**backpropagation（反向傳播）** ：從輸出往輸入反向套用 chain rule，把梯度沿著計算圖傳回去，算出損失對每個參數的梯度。

**autograd（自動微分）** ：backpropagation 的自動化實作——框架在前向傳播時記錄計算圖，反向傳播時自動套用 chain rule，不需要手動推導每個梯度公式。

**Jacobian** ：向量值函數 $f:\mathbb{R}^n\to\mathbb{R}^m$ 對輸入的偏導數矩陣，$J_{ij}=\partial f_i/\partial x_j$；例如 softmax 的 Jacobian 描述了每個輸出對每個輸入 logit 的敏感度。

**logits** ：模型在 softmax/sigmoid 之前算出的原始分數，常記為向量 $x$ 或 $z$；router、LM head 的輸出在歸一化前都叫 logits。

**softmax** ：把任意實數向量 $x\in\mathbb{R}^n$ 轉成機率分佈：$\text{softmax}(x)_i=\dfrac{e^{x_i}}{\sum_j e^{x_j}}$，其中 $x_i$ 為第 $i$ 個 logit。

**temperature（溫度）** ：softmax 前先把 logits 除以溫度 $\tau$：$\text{softmax}(x/\tau)_i=\dfrac{e^{x_i/\tau}}{\sum_j e^{x_j/\tau}}$；$\tau$ 越小分佈越接近 one-hot，越大越接近均勻。

**cross-entropy（交叉熵）** ：$H(p,q)=-\sum_i p_i\log q_i$，其中 $p$ 為真實標籤分佈（通常是 one-hot）、$q$ 為模型預測分佈；是語言模型訓練最常用的損失函數。

**entropy（熵）** ：$H(p)=-\sum_i p_i\log p_i$，其中 $p_i$ 為分佈裡第 $i$ 個結果的機率；衡量一個分佈的不確定程度——越集中（接近 one-hot）越低，越均勻越高。

**KL divergence（KL 散度）** ：$D_{KL}(p\|q)=\sum_i p_i\log\dfrac{p_i}{q_i}$，衡量分佈 $p$ 相對分佈 $q$ 的差距；恆為非負，兩分佈相同時為 0。

**one-hot** ：只有一個位置是 1、其餘是 0 的向量，$y\in\{0,1\}^n,\ \sum_i y_i=1$；常用來表示離散標籤或理想化的 routing 決策。

## Transformer 基礎

**tokenizer / token** ：tokenizer 把文字切成固定詞彙表裡的子詞片段（token），每個 token 對應一個整數 id $\in\{0,\dots,V-1\}$，$V$ 為詞彙表大小。見 [從零實作 Transformer](foundations/transformer-from-scratch.md)。

**embedding** ：用矩陣 $E\in\mathbb{R}^{V\times d}$ 把 token id 查表轉成 $d$ 維向量；模型其餘所有運算都建立在這個向量表示上。

**positional encoding（位置編碼）** ：把詞序資訊注入模型；傳統做法是把一個位置向量加到 embedding 上：$x_i \leftarrow x_i + p_i$，其中 $p_i$ 為第 $i$ 個位置的編碼向量。

**RoPE（Rotary Position Embedding）** ：positional encoding 的一種實作，在 attention 內部依位置旋轉 query/key 向量——把每一對維度 $(x_{2k},x_{2k+1})$ 用角度 $i\theta_k$ 的旋轉矩陣轉動（$i$ 為位置、$k$ 為維度配對索引），讓 $q_i^\top k_j$ 的內積只依賴相對位置 $i-j$。

**query / key / value（Q/K/V）** ：attention 把每個 token 的向量 $x$ 投影成這三個角色：$q=xW_Q$、$k=xW_K$、$v=xW_V$（$W_Q,W_K,W_V\in\mathbb{R}^{d\times d_h}$，$d_h$ 為每頭維度）；query 問「我在找什麼」，key 答「我能提供什麼」，value 是匹配後實際傳出的內容。

**causal mask（因果遮罩）** ：把 attention 分數矩陣 $S\in\mathbb{R}^{N\times N}$ 裡 $j>i$ 的位置設為 $-\infty$，強制每個 token $i$ 只能看到自己與更早的 token，是 decode 可以快取 K/V 的前提。

**multi-head attention（多頭 attention）** ：並行跑 $h$ 個各自有獨立 $W_Q,W_K,W_V\in\mathbb{R}^{d\times d_h}$（$d_h=d/h$）的小 attention「頭」，再把 $h$ 個輸出接起來、乘上輸出投影 $W_O\in\mathbb{R}^{d\times d}$，讓模型同時追蹤多種關係。

**FFN（feed-forward network） / MLP** ：對每個 token 獨立處理的兩層網路：$\text{FFN}(x)=W_{\text{down}}\,\sigma(W_{\text{up}}\,x)$，其中 $W_{\text{up}}\in\mathbb{R}^{d_{ff}\times d}$、$W_{\text{down}}\in\mathbb{R}^{d\times d_{ff}}$、$\sigma$ 為非線性 activation function、通常 $d_{ff}\approx4d$；握有模型大部分參數，是 MoE 要稀疏化的對象。

**GELU / SiLU / SwiGLU** ：FFN 裡常用的非線性 activation function；SwiGLU 額外引入一個 gate 投影：$\text{SwiGLU}(x)=\text{SiLU}(W_{\text{gate}}x)\odot(W_{\text{up}}x)$（$\odot$ 為逐元素相乘），是現代 LLM（以及 MoE expert）最常用的選擇。

**residual connection（殘差連接）** ：把每個子層的輸入加回它的輸出：$x'=x+\text{sublayer}(x)$，給梯度一條恆等路徑，讓深層網路可以訓練。

**LM head** ：模型最後用矩陣 $W_{\text{LM}}\in\mathbb{R}^{V\times d}$ 把每個 token 的向量投影回詞彙表大小的分數（logits），用來預測下一個 token。

**autoregressive generation（自回歸生成）** ：把預測出的 token 接到序列尾端、再丟回模型重跑一次的回饋迴圈——一次只生成 $P(x_{t+1}\mid x_1,\dots,x_t)$ 對應的一個新 token；解釋了為何 decode 一次只能生成一個 token。

## 系統和效能

**算術強度（arithmetic intensity，$I$）** ：每從記憶體搬一個 byte 所執行的 FLOP 數：$I=W/Q$，其中 $W$ 為執行的浮點運算數（FLOP）、$Q$ 為搬移的位元組數（bytes）。決定你落在 roofline 的哪個機制。見 [作為系統的 Transformer](foundations/transformer-systems.md)。

**roofline 模型** ：效能上限 $P=\min(\pi,\ \beta I)$，其中 $\pi$ 為峰值算力（FLOP/s）、$\beta$ 為峰值頻寬（byte/s）、$I$ 為算術強度。整本手冊的組織主軸。

**compute-bound / memory-bound** ：roofline 公式裡 $P=\beta I$（脊點左側，被頻寬卡住）vs $P=\pi$（脊點右側，被算力卡住）。decode 受記憶體限制；大 batch 的 matmul 受計算限制。

**MFU（模型 FLOP 利用率，Model FLOP Utilization）** ：達到的模型 FLOP ÷ 峰值 FLOP；training 時 $\text{MFU}\approx 6P\cdot\text{tok/s}/\pi$，其中 $P$ 為參數量、$\text{tok/s}$ 為訓練吞吐量、$\pi$ 為峰值 FLOP/s。招牌效率指標。見 [profiling](performance/profiling.md)。

**HBM** ：高頻寬記憶體 —— GPU 的主 DRAM；就是 roofline 公式裡的頻寬 $\beta$。

**SRAM / 共享記憶體 / LDS** ：快速的晶片內暫存。「共享記憶體」（NVIDIA）=「LDS」（AMD）。把 tile 暫存在這裡，正是 kernel 提高算術強度的手段。

**warp / wavefront** ：鎖步的 SIMT 執行群組 —— NVIDIA 上是 **32 個 thread**（warp），AMD CDNA 上是 **64**（wavefront）。 常見的可移植性陷阱。見 [GPU 程式設計](performance/gpu-programming.md)。

**占用率（occupancy）** ：每個 SM/CU 上常駐的 warp/wavefront 數與該 SM/CU 可容納上限的比值：$\text{occupancy}=\dfrac{\text{常駐 warp 數}}{\text{每 SM 最大 warp 數}}$；它是藏 latency 的手段，不是目標。

**coalescing（合併存取）** ：相鄰 lane 存取相鄰位址，使記憶體交易能被合併。

**算子融合** ：把多個操作合在一起以避免 HBM 往返（例如 FlashAttention、融合 MLP） —— 提高算術強度。

**FLOP** ：浮點運算次數；matmul $(m\times k)(k\times n)$ 的成本為 $2mkn$，其中 $m,n$ 為輸出矩陣的兩個維度、$k$ 為縮減（內積）維度，係數 2 來自每個輸出元素的一次乘法加一次加法。

**batch / batch size** ：一次餵進模型的序列數；training 時影響梯度雜訊與硬體利用率，inference 時是 throughput 與 latency 之間的主要槓桿（continuous batching、batch-1 decode 等概念都建立在這個量上）。

## GPU 程式設計

**thread（執行緒）** ：kernel 裡最小的執行單元，每個 thread 處理一份純量資料。見 [GPU 程式設計模型](performance/gpu-programming.md)。

**block / workgroup** ：一組能用 shared memory 互相同步的 thread（NVIDIA 稱 block，AMD 稱 workgroup）；block 內的 thread 數常記為 `blockDim`。

**grid** ：一次 kernel launch 涵蓋的所有 block；grid 內的 block 數常記為 `gridDim`。

**SM / CU** ：streaming multiprocessor（NVIDIA）/ compute unit（AMD）——容納多個 warp/wavefront 常駐執行的硬體單位。

**register（暫存器）** ：每個 thread 專屬、速度最快的儲存空間；用量過高會造成 **register pressure**，壓低 occupancy。

**divergence（分支發散）** ：同一個 warp/wavefront 裡的 lane 走上不同分支時，硬體必須依序執行兩條路徑：$T_{\text{branch}}\approx T_{\text{if}}+T_{\text{else}}$，浪費掉一半算力。

**bank conflict** ：同一個 warp 裡 $c$ 條 lane 同時存取 shared memory/LDS 裡同一個 bank 的不同位址時，存取會被序列化成 $c$ 路：$T_{\text{access}}\approx c\cdot T_{\text{bank}}$。

**tile** ：把矩陣或張量切成的小塊，是 kernel 在 shared memory/register 裡重複使用資料、提高算術強度的基本單位。

**Tensor Core / Matrix Core** ：NVIDIA / AMD 用來加速矩陣乘加的專用硬體單位，透過 `wmma`/`mma`（CUTLASS）或 `mfma`（rocWMMA / Composable Kernel）等介面呼叫，每執行一次完整 tile-MMA 指令的 FLOP/s 遠高於純量 FMA 路徑。

**persistent kernel** ：只 launch 一次、在內部用迴圈持續處理多個工作項目的 kernel，用來省掉重複 launch 的開銷。

**launch overhead** ：CPU 端發出一次 kernel launch 的固定開銷；decode 等小 kernel 密集的場景裡常是主要瓶頸，CUDA Graph 是常見的解法。

**CUDA Graph / HIP Graph** ：把一串 kernel launch 預先錄製成一個圖、之後重放整個圖而不必逐一重新 launch，用來消除 launch overhead，常與形狀固定的 decode 步驟搭配使用。

**program id（`tl.program_id`）** ：Triton kernel 內用來辨識目前這個實例該處理哪個 tile/block 的索引。

**mask（遮罩）** ：Triton/CUDA 在 tile 邊界用條件遮罩擋掉超出範圍的存取，讓 kernel 可以處理不是 tile 大小整數倍的形狀。

**autotuning** ：對 block size、`num_warps`、`num_stages` 等設定做網格搜尋，挑出在目標硬體上最快的組合；換一張 GPU 通常要重新 autotune。

## 精度

**BF16 / FP16 / FP8** ：16/16/8 位元浮點。BF16 保留 FP32 的指數範圍（8 位）因而在 training 上勝出；FP16 mantissa 較多 但範圍較窄；FP8（E4M3/E5M2）是前沿。見 [數值與精度](foundations/numerics-precision.md)。

**mixed precision** ：低精度儲存/matmul+FP32 累加+FP32 主權重。

**loss scaling（損失縮放）** ：把損失乘上一個因子，讓 FP16 梯度落在可表示範圍內；BF16 基本上不需要。

**exponent（指數）** ：浮點數格式裡決定動態範圍的欄位；指數位數越多，能表示的數值範圍越大。BF16 用 8 個指數位（與 FP32 相同），FP16 只有 5 個。

**mantissa（尾數）** ：浮點數格式裡決定精度的欄位；尾數位數越多，能分辨的相對誤差越小。FP16 有 10 個尾數位，BF16 只有 7 個。

**overflow（溢位）** ：數值超出格式能表示的最大值，結果變成 `inf`。

**underflow（下溢）** ：數值小到無法表示成正規數，結果變成 0（或極小的非正規數）。

**dynamic range（動態範圍）** ：一個數值格式能表示的最大值與最小正規值之比，由指數位數決定，量級約為 $2^{2^{e}}$（$e$ 為指數位數）。

**machine epsilon / unit roundoff** ：浮點格式能分辨的最小相對誤差，決定累加長序列數字時的精度上限。

**Kahan summation（補償求和）** ：用額外的誤差累積項修正浮點加法的捨入誤差，讓長序列求和更精確。

## Attention

**KV cache** ：儲存過去 token 的 key/value，使 decode 對序列長度 $N$ 的成本從 $O(N^2)$ 降為 $O(N)$；常常是 inference 記憶體的主角。見 [Attention 效率](foundations/attention-efficiency.md)。

**MQA / GQA / MLA** ：Multi-Query / Grouped-Query / Multi-head Latent Attention —— 從架構層面縮小 KV cache（減少或壓縮 KV 頭）。

**absorb（weight absorption）** ：MLA 的技巧——把 up-projection 矩陣吸收進 query/output 側的投影，這樣 decode 時就不必把完整的 K/V 還原出來，只需保留並快取 latent 向量。

**FlashAttention** ：IO-aware 的 attention，把 $Q,K,V$ 分塊、用 online softmax 避免具現化 $N{\times}N$ 分數矩陣。見 [FlashAttention](foundations/flashattention.md)。

**online softmax** ：靠 running max $m$ 與校正因子 $e^{m_{\text{old}}-m_{\text{new}}}$ 重新縮放前面累積的部分結果，單趟傳遞就算出數值穩定的 softmax。

**PagedAttention** ：以區塊為單位配置 KV cache（如虛擬記憶體分頁），消除碎片並支援共享。

## MoE (Mixture of Experts)

**MoE (Mixture of Experts)** ：含許多 expert FFN 與一個 router 的層，每個 token 只啟動其中幾個，把總參數和每 token FLOP 解耦。 見[MoE 篇](moe/index.md)。

**expert** ：MoE 層裡並列的其中一個 FFN（通常是 SwiGLU）。

**router / gate** ：產生 per-expert 分數的小網路；由 top-$k$ 選出 expert。**softmax gating** 讓 expert 互相競爭； **sigmoid gating** 則獨立為它們計分。

**top-$k$ routing** ：每個 token 從 router 算出的 logits $x\in\mathbb{R}^E$（$E$ 為 expert 總數）裡，選出分數最高的 $k$ 個 expert。

**token-choice vs expert-choice** ：token 挑 expert（保證覆蓋、不保證平衡）vs expert 挑自己的 top-$C$ token（保證平衡、不保證覆蓋）。 見 [Routing 變體](moe/routing-variants.md)。

**共享 expert** ：在路由 expert 之外再加一個總是啟用的 FFN，用來吸收常識並穩定 training。

**細粒度 expert** ：用許多小 expert 取代少數大 expert，在固定活躍計算下放大 expert 組合空間。

**auxiliary（負載平衡）損失** ：懲罰項 $\mathcal{L}_{\text{aux}}=\alpha E\sum_e f_e P_e$，其中 $\alpha$ 為權重係數、$E$ 為 expert 總數、$f_e$ 為路由到 expert $e$ 的 token 比例、$P_e$ 為 expert $e$ 的平均 gate 機率；鼓勵均勻 routing。見 [負載平衡](moe/load-balancing.md)。

**aux-loss-free 平衡** ：靠控制器更新的 per-expert **選擇偏差**（不是 gate 權重）來平衡，避免扭曲梯度（DeepSeek 風格）。

**expert capacity** ：每個 batch 每個 expert 能處理的 token 數上限，記為 $C$。

**capacity factor（容量係數）** ：expert capacity 相對平均負載的乘數：$C=\lceil\text{cf}\cdot Tk/E\rceil$，其中 $T$ 為每個 batch 的 token 數、$k$ 為 top-$k$、$E$ 為 expert 數、cf 即容量係數；係數為 1.0 不浪費記憶體但容易丟 token，係數越大越不容易丟、但 padding/buffer 也越大。

**token drop / overflow** ：超出 capacity 的 token 略過 MoE 層（由殘差帶過去）。

**dead expert** ：連續多步幾乎沒有 token 路由過去、形同退出訓練的 expert，是 router collapse 的徵兆之一。

**router z-loss** ：$\mathcal{L}_z=\beta\Big(\log\sum_e e^{x_e}\Big)^2$，其中 $\beta$ 為權重係數、$x_e$ 為 expert $e$ 的 router logit，懲罰過大的 router logit 以求穩定。見 [訓練穩定性](moe/training-stability.md)。

**expert parallelism（EP）** ：把 expert 跨 GPU 切分；token 經由 all-to-all 抵達它的 expert。見 [系統與 EP](moe/systems-ep.md)。

**gather** ：MoE 依 permutation 索引把屬於同一個 expert 的 token 收集成連續區塊，讓 grouped GEMM 可以一次處理一個 expert 的所有 token。

**scatter** ：把每個 expert 算完的輸出依原始 token 位置寫回去——方向與 gather 相反的記憶體搬移。

**permute / unpermute** ：把 token 依 expert 重新排序成連續區塊（permute），讓 grouped GEMM 可以跑；算完之後再用反向映射把輸出排回原始順序（unpermute）。

**tokens-per-expert** ：路由到某個 expert 的平均 token 數，約為 $Tk/E$（$T$ 為 batch token 數、$k$ 為 top-$k$、$E$ 為 expert 數）；decode 時這個數字通常 $\ll 1$，是 MoE GEMM 落在 memory-bound 區域的根本原因。

**grouped GEMM** ：單一 kernel 執行許多個不同大小的矩陣乘法（每個 expert 一個），無需 padding。

**split-K GEMM** ：把 GEMM 的 K（縮減）維度切開、分給多個 block 平行計算後再用一個 reduce kernel 合併，能在單一 GEMM 太小、無法餵滿裝置時提高平行度；對逐層重複的小 GEMM（例如 decode 的 routed GEMM）通常不值得，因為額外的 reduce launch 次數會被層數放大。

**MegaBlocks / 區塊稀疏 MoE** ：把 MoE FFN 改寫成區塊稀疏 matmul，藉此避免 token drop 與 padding。

**MTP（Multi-Token Prediction）** ：訓練時額外加一個頭，讓模型一次預測多個未來 token；可以當作 speculative decoding 內建的 draft，也能加強訓練訊號。

## 分散式 training

**collective（集合通訊）** ：all-reduce、all-gather、reduce-scatter、all-to-all、broadcast/P2P（NCCL/RCCL）。見 [分散式訓練](performance/distributed-training.md)。

**data parallelism（DP）** ：在每個 GPU 上複製整個模型、把 batch 切分到各 GPU，每步對梯度做 all-reduce 讓所有副本保持一致；通訊量與模型參數量成正比，跟 batch size 無關。

**tensor parallelism（TP）** ：把單個 matmul 切到多個 GPU 上算（例如 FFN 的 up-projection 按行切、down-projection 按列切），每層 forward/backward 各需要一次 all-reduce 合併結果；通訊密集，通常限制在節點內。

**pipeline parallelism（PP）** ：按層把模型切成放在不同 GPU 上的 stage，microbatch 依序流過各個 stage；通訊量低（只在 stage 邊界傳遞 activation），但會產生 pipeline bubble。

**sequence parallelism（SP）** ：把序列維度切到多個 GPU 上，讓每個 GPU 只處理一部分 token，分攤隨序列長度增長的 activation 與 attention 計算量；expert parallelism（EP）則見上方 MoE 一節。

**rank** ：分散式 training 裡每個 process/GPU 的編號，collective 通訊以 rank 為單位定義誰跟誰交換資料。

**device mesh** ：把一組 GPU 排成多維陣列、明確標出哪一維對應 DP、哪一維對應 TP/PP/EP，方便組合多種並行策略。

**ZeRO / FSDP** ：在 DP 群組間切分 optimizer 狀態（ZeRO-1）、梯度（ZeRO-2）與參數（ZeRO-3 / FSDP）以省記憶體；分片後每 GPU 狀態量約為 $16P/N$，其中 $P$ 為參數量、$N$ 為 DP 裝置數、16 為 Adam 混合精度下每參數的位元組數。

**all-to-all** ：每個 rank 向其他每個 rank 各送一塊不同資料的 collective —— MoE dispatch/combine 的原語。

**node-limited routing** ：限制一個 token 的 expert 能跨多少節點，以約束跨節點 all-to-all。

**microbatch** ：pipeline parallelism 裡把一個 batch 再切成的小份，讓不同 stage 可以同時處理不同的 microbatch、減少 pipeline bubble。

**pipeline bubble** ：pipeline parallelism 裡因為要填充與排空管線而閒置的時間比例：$\text{bubble}\approx(P{-}1)/(m{+}P{-}1)$，其中 $P$ 為 pipeline stage 數、$m$ 為 microbatch 數。

**1F1B（one-forward-one-backward）** ：一種 pipeline 排程，讓每個 stage 交替做一次 forward、一次 backward，比樸素排程占用更少的 activation 記憶體。

**strong scaling / weak scaling** ：固定總問題規模增加 GPU 數看加速比（strong），或讓問題規模跟 GPU 數一起變大看每 GPU 效率是否維持（weak）。

## Inference & 壓縮

**prefill / decode** ：處理 prompt（很多 token、compute-bound）vs 一次生成一個 token（memory-bound）。

**continuous（in-flight）batching** ：以迭代為單位排程，隨時把跑完的序列換成等待中的序列，讓 GPU 保持滿載。見 [推論最佳化](performance/inference-optimization.md)。

**speculative decoding** ：用一個便宜的 draft 模型提出 token，再由目標模型一次驗證；無損，利用 decode 閒置的算力。

**draft model / target model** ：speculative decoding 裡分別負責提出候選 token（便宜的 draft）與一次性驗證（昂貴的 target）的兩個模型。

**acceptance rate（接受率，$\alpha$）** ：target 模型驗證時接受 draft 提案的比例；提案長度為 $\gamma$ 時，每次驗證步驟預期被接受的 token 數為 $\mathbb{E}[\text{tokens}]=\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$，決定 speculative decoding 實際能拿到多少加速。

**chunked prefill** ：把很長的 prompt 切成小塊，跟其他請求的 decode 步驟交錯執行，避免長 prefill 獨佔 GPU、餓死其他請求。

**disaggregated serving / disaggregation** ：把 compute-bound 的 prefill 與 memory-bound 的 decode 拆到各自獨立調校的 GPU 池上跑；省下的爭用要超過池間搬 KV cache 的成本才划算。

**prefix caching** ：快取共用 prompt 前綴的 KV，讓多個請求共用同一份前綴計算結果，省下重複的 prefill。

**expert offload** ：把不常用的 expert 權重放在 CPU/NVMe，用到時才透過 PCIe 串流進 HBM，用頻寬換顯存；只有 tokens-per-expert 夠大、GEMM 時間蓋過傳輸時間時才能把延遲藏住。

**PTQ / QAT** ：訓練後量化（只做校準）vs 量化感知訓練。見 [量化](performance/quantization.md)。

**affine quantization** ：用 $q=\text{round}(x/s)+z$ 把浮點值映射到整數網格、$\hat x=s(q-z)$ 還原（**dequantize**），其中 $x$ 為原始浮點值、$s$ 為 scale、$z$ 為 zero-point、$q$ 為量化後的整數。

**calibration（校準）** ：用一小批代表性資料跑過模型，統計 activation/權重的數值範圍，決定量化用的 scale。

**outlier（離群值）** ：數值遠大於同一 channel/tensor 其他元素的元素；少數 outlier 會撐大量化用的 scale，拖累其餘元素的精度，AWQ/SmoothQuant 等方法都是為了應付它。

**GPTQ / AWQ / SmoothQuant** ：PTQ 方法 —— 誤差修正的權重量化／activation-aware 權重縮放／把 activation 離群值搬進權重。

**straight-through estimator** ：在反向傳播時，把量化這種不可微的 round 操作的梯度直接當作恆等函數傳過去，讓量化感知訓練（QAT）能算梯度。

**2:4 半結構化稀疏** ：每 4 個權重裡固定有 2 個為零的稀疏模式，硬體（sparse Tensor Core）可以直接利用這個結構跳過零值計算。

**perplexity** ：語言模型品質的標準量測：$\text{PPL}=\exp\Big(-\dfrac{1}{N}\sum_{i=1}^N\log p(x_i\mid x_{<i})\Big)$，其中 $N$ 為 token 數、$p(x_i\mid x_{<i})$ 為模型給真實第 $i$ 個 token 的機率；是平均 negative log-likelihood 取指數，常用來比較量化後模型的品質損失。

**剪枝（pruning）** ：移除權重或結構（非結構化、結構化，或 2:4 半結構化）以壓縮模型。

**蒸餾（distillation）** ：訓練一個小的 student 模型去模仿大的 teacher。

**TTFT（Time To First Token）** ：產生第一個 token 的時間，主要由 prefill latency 決定。

**TPOT（Time Per Output Token） / ITL（Inter-Token Latency）** ：每個輸出 token 之間的時間，主要由 decode latency 決定。

## Profiling 與方法論

**critical path** ：決定整體 wall-clock 時間的那一條依賴鏈；只有在 critical path 上的 kernel 才會直接影響端到端 latency，被完全重疊掉的 kernel 即使再快也不會縮短它。

**self-time** ：把每個 kernel 的耗時獨立加總；若兩個 kernel 在不同 stream 上同時執行，self-time 加總會超過實際經過的時間。

**wall-clock** ：實際經過的時間。self-time 明顯超過 wall-clock 代表 kernel 之間有重疊；明顯低於則代表中間有空閒間隙。

**warmup** ：在正式計時前先執行幾次迴圈，讓 JIT 編譯、autotune、clock boost 等一次性成本先發生，避免污染量測結果。

**Amdahl's law** ：把一段程式加速 $s$ 倍時，整體加速被那段程式佔總時間的比例上限封頂：$\dfrac{1}{(1-p)+p/s}$，其中 $p$ 為該段佔總時間的比例；提醒你優化前先看佔比，而不是只看單次呼叫的快慢。

**Little's law** ：穩態系統裡，平均在飛（in-flight）的工作量 $L$ 等於到達率 $\lambda$ 乘以平均停留時間 $W$：$L=\lambda W$；常用來把 serving 的並發度、throughput、latency 三者綁在一起。

**confidence interval（信賴區間）** ：描述一個量測值的不確定範圍，通常以「點估計 ± 誤差範圍」表示；誤差範圍隨重複量測次數增加而縮小。

**coefficient of variation（變異係數，CV）** ：標準差除以平均值：$\text{CV}=\sigma/\mu$；描述一組數字的相對離散程度，常用來量化負載平衡的好壞。

**dead code elimination（DCE）** ：編譯器發現某段計算的結果從未被使用就直接刪除；micro-benchmark 若沒有「消費」kernel 的輸出，常會被 DCE 整個刪掉，導致量到的時間趨近於 0。

## 硬體快速參考

用來做粗略估計的整數（練習與 [解答](solutions/index.md) 都用這些）。峰值 FLOP/s 為密集 BF16；真實 kernel 只達到其中一部分（MFU）。頻寬指 HBM。**脊點** $\pi/\beta$ 是晶片由 memory-bound 轉成 compute-bound 的那個算術強度，$\pi$ 為峰值算力、$\beta$ 為峰值頻寬。

| GPU | BF16 峰值 ($\pi$) | HBM BW ($\beta$) | HBM 大小 | 脊點 $\pi/\beta$ |
| --- | --- | --- | --- | --- |
| A100（80 GB） | ~312 TFLOP/s | ~2.0 TB/s | 80 GB | ~156 FLOP/byte |
| H100 (SXM) | ~990 TFLOP/s | ~3.35 TB/s | 80 GB | ~295 FLOP/byte |
| H200 | ~990 TFLOP/s | ~4.8 TB/s | 141 GB | ~206 FLOP/byte |
| MI300X | ~1.3 PFLOP/s | ~5.3 TB/s | 192 GB | ~245 FLOP/byte |

互連（用於 [分散式訓練](performance/distributed-training.md)）：節點內 **NVLink** ~0.9 TB/s/GPU（NVLink 4）或 MI300 上的 **Infinity Fabric**；跨節點 **InfiniBand/RoCE** ~25–50 GB/s/GPU；主機 **PCIe Gen5** ~64 GB/s。從 HBM → NVLink → IB → PCIe 每一級掉 1–2 個數量級 —— 這正是為什麼要把並行小心對映，好讓最多話的 collective 走最快的連結。

!!! Note "怎麼用這些數字"
    數字會隨 SKU、時脈與稀疏性宣稱而變（廠商常引用 2× 結構稀疏 —— 對密集要減半）。你的估計裡該穩健的是**機制**與**數量級**，而不是第三位有效數字。
