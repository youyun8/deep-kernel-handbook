# 解答 — MoE (Mixture of Experts)

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 全部九個 MoE 頁面</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

解答[MoE 篇](../moe/index.md)的練習。部分題目會參照 [`code/`](https://github.com/youyun8/deep-kernel-handbook/tree/main/code) 裡的玩具模型；如果推導有乾淨的封閉解，我們直接給出；否則給出方法與預期的定性結果。

## 為什麼需要稀疏化

??? Success "1 — 總參數與 active 參數、FLOP 比率（$E{=}128, k{=}2, d{=}4096$）"
    1 個 FFN ≈ $8d^2$ 參數（up $4d^2$ + down $4d^2$，$d_{ff}=4d$）。

    - **expert 總計：**$128 \times 8d^2 = 1024d^2 \approx 1.72\times10^{10}$。
    - **每個 token 的 active 參數：**$2 \times 8d^2 = 16d^2 \approx 2.68\times10^{8}$。

    FLOP 只隨 *active* 參數量縮放，所以每 token FLOP 相對密集模型（$E{=}1$，即 $8d^2$ active）的比率是 $16d^2/8d^2 = \mathbf{2\times}$——你只為 $k=2$ 個 expert 付出計算代價。但你要**儲存** $128\times$ 的 FFN 參數：巨大的容量、固定的計算量。這個落差正是稀疏化的全部價值所在。

??? Success "2 — DeepSeek-V3 稀疏度與 Mixtral"
    DeepSeek-V3：$37/671 \approx 5.5\%$ active（≈**18× 稀疏**）。Mixtral 8×7B：8 個 expert，top-2 約 $13/47 \approx 28\%$ active（≈3.6×）。V3 明顯更稀疏——許多細粒度 expert、每個 active 比例都很低——這正是近期的趨勢：更多、更小的 expert，在相同 active 成本下換來更大的組合空間（見下一題）。

??? Success "3 — 何時偏好密集 37B 而非 671B 總/37B active 的稀疏模型"
    active FLOP 相同，但稀疏模型需要 **~18× 的記憶體**才能把所有 expert 都放進顯存。優先選**密集**模型的情況：(a) 記憶體吃緊（單張 GPU / 邊緣裝置）；(b) **batch-1 latency** 很重要、無法把 expert 載入成本攤銷掉——稀疏模型的 decode 可能要為少數幾個 token 觸及很多 expert，傷害 locality；(c) 在小資料集上**微調**，這時大多數 expert 只拿到很少梯度，容易過時。反過來，稀疏模型在高 throughput serving 與 pretraining 的每-FLOP 品質上勝出。

??? Success "4 — expert offload：卡在 roofline 的哪一軸？"
    透過 PCIe 從 CPU/NVMe 把 expert 權重串流進來（約數十 GB/s），相比 HBM（TB/s）差了 1–2 個數量級的頻寬。所以限制因素變成**頻寬**（expert 權重的 PCIe/NVMe 傳輸），不是算力。只有當 tokens-per-expert 夠大、GEMM 時間超過傳輸時間時，streaming 才能被藏起來——確切條件見[推論與 serving](#inference-serving) 的練習 2。

## 從零實作 MoE layer

??? Success "1 — `MoELayerNaive` 裡的 sigmoid gating"
    把 top-$k$ gate 的 `softmax(logits)` 換成 `sigmoid(logits)`。和 softmax 不同，sigmoid gate 是**獨立的**（總和不等於 1），所以每個 expert 的權重是一個絕對的「這個 expert 該不該被觸發」分數。選出 top-$k$ 之後，如果想要凸組合，要把選中的 gate 重新 normalize。用測試檢查輸出形狀/數值有限性；預期 loss 量級相近，但平衡動態不同（sigmoid + bias controller 正是 DeepSeek-V3 的配方）。

??? Success "2 — $k{=}1$（Switch），不重新 normalize"
    $k=1$ 且不重新 normalize 時，輸出是 $g\cdot \text{expert}(h)$，其中 $g=\text{softmax}(\cdot)_{\text{top1}} \in (0,1)$。所以 expert 輸出會**被 $g<1$ 縮小**，縮小了 activation 的幅度（也讓殘差尺度跟 gate 的置信度耦合在一起）。Switch 故意保留這個乘法的 gate——它給 router 一個可微訊號——但你得在 init/learning rate 上考慮這個縮小，或者重新 normalize 讓保留的 gate 等於 1。

??? Success "3 — naive loop 與 permute 形式（$T{=}8192, E{=}64$，CPU）"
    **naive loop** 對每個 expert 都掃過全部 $T$ 個 token、用 mask 蓋掉不屬於它的部分 → 等於把所有 $T$ 個 token 碰了 $E$ 次（浪費大量被 mask 掉的運算，但容易向量化）。**permute 形式**先把 token 依 expert 重新排列成連續區塊，再對每個 expert 只跑屬於它的那一段 → 浪費的計算少得多，但要付 gather/scatter 的成本。在 CPU 上，一旦 $E$ 變大，permute 形式通常會贏（naive 的 $O(TE)$ mask 成本會主導）。在**GPU**上差距更明顯：permute 形式產出的連續 grouped GEMM 正是硬體喜歡的形狀，而 naive loop 會啟動很多個極小、被 mask 蓋住大半的 matmul（啟動開銷 + 低 occupancy）。

??? Success "4 — 加入 shared expert；逐步確認梯度"
    加入 $y = \text{shared}(h) + \sum_{e\in\text{TopK}} g_e\,\text{expert}_e(h)$。因為 `shared(h)` 在每個 token 的路徑上、不受 routing 影響，所以 `shared.weight.grad` 永遠不是 `None`、每一步都非零（`backward` 之後檢查即可）。路由 expert 只有在被選中時才拿到梯度；shared expert 則是一條永遠在線的密集路徑，有助於穩定冷啟動（見 [訓練穩定性](#training-stability) 的練習 4）。

## 負載平衡

??? Success "1 — 均勻分佈最小化 $\sum_e f_e P_e$"
    在 $\sum f_e = k$、$\sum P_e = 1$ 的限制下最小化 $\sum_e f_e P_e$。Switch 的 auxiliary loss 把路由到 expert $e$ 的**token 比例**（$f_e$）乘上 $e$ 的平均**gate 機率**（$P_e$）；當負載平均分散時這個乘積最小。形式上，由排序不等式／Cauchy–Schwarz 可知，兩個向量都「拉平」時乘積和最小：對所有 $e$，$f_e = k/E$、$P_e = 1/E$，得到 $\sum_e f_e P_e = E\cdot\frac{k}{E}\frac{1}{E} = k/E$。任何集中現象（某些 $f_e,P_e$ 同時偏大）都會推高乘積——所以最小化這個 loss 會把 routing 推向均勻分佈。*（auxiliary loss 用 $f$ = 硬計數、$P$ = 可微機率，所以梯度只流經 $P$。）*

??? Success "2 — capacity 與 drop rate（$T{=}4096, E{=}64, k{=}2$，cf=1.25）"
    預期 tokens/expert $= Tk/E = 4096\cdot2/64 = 128$。capacity $C = \lceil \text{cf}\cdot Tk/E\rceil = \lceil 1.25\times128\rceil = 160$。若某個 expert 拿到全部 $Tk = 8192$ 次分配中的 5%，即 410 個 token，就會溢出 $410-160 = 250$ 個；這些都會被**丟棄**。佔全部分配的 drop rate ≈ $250/8192 \approx 3.0\%$（假設其餘 expert 都在 capacity 之內）。把 cf 提高到 2.0（$C=256$）還是會丟 $410-256=154$ 個——說明 capacity 沒辦法修補嚴重傾斜的 router；你需要的是更好的負載平衡。

??? Success "3 — 調整 bias controller 的 $\gamma$"
    aux-loss-free 的控制器逐步微調每個 expert 的 bias：$b_e \leftarrow b_e + \gamma\,\text{sign}(\text{target} - \text{load}_e)$。**$\gamma$ 較大**→收斂更快，但會在平衡點附近震盪/超調；**$\gamma$ 較小**→平滑但慢。**sigmoid** gate 的響應比 **softmax** 更快，因為 sigmoid 分數是獨立的——調整一個 expert 的 bias 不會透過共享的 normalizer 牽動其他 expert，所以控制器對各 expert 的調整不會互相干擾。調好 $\gamma$ 之後，預期負載 CV 只要幾百步就能降到 ~0.1–0.2。

??? Success "4 — 玩具 MoE 上的 auxiliary loss 與 aux-loss-free"
    用兩種模式各跑一次 `train_tiny_moe.py`。預期：**aux-loss-free** 在相同負載 CV 下可以打平、甚至略微**降低最終 LM loss**，因為它是用 bias 去調整 routing 的*計數*，而不是加一個和 LM 目標互相競爭的梯度項（auxiliary loss 多少會扭曲 loss surface）。兩種模式的負載 CV 都應該降到 ~0.1–0.2；aux-loss-free 的優勢是省掉了平衡與品質之間的拉鋸。把兩次跑的最終 loss 和 CV 並排報告——這個對照本身就是重點。

## Routing 變體

??? Success "1 — expert 組合數與細粒度的好處"
    組合數 $= \binom{E}{k}$：

    - $(8,2)$：$\binom{8}{2} = 28$
    - $(64,8)$：$\binom{64}{8} \approx 4.4\times10^{9}$
    - $(256,8)$：$\binom{256}{8} \approx 4.1\times10^{14}$

    每個 token 都在挑一種組合，所以 expert 數越多、越細粒度，可能的「expert 混合」數量呈指數級增長——這正是細粒度 expert（DeepSeekMoE）的論點：在**相同的 active $k$** 下換來遠大得多的組合空間，計算量卻不變。

??? Success "2 — 為什麼 expert-choice 會破壞 autoregressive decode"
    expert-choice 讓每個 expert 從整個 batch 裡挑出自己的 top-$C$ token——這假設所有 token 同時存在（training/prefill 確實如此）。但在 **autoregressive decode** 裡，你一次只生成一個 token；expert 不可能在還不存在的未來 token 之中挑出 top-$C$，「整批」這個概念本身就不成立。token-choice 則是每個新 token 獨立決定自己的路徑，所以它才是 inference 時自然的方案。

??? Success "3 — shared expert 對穩定性的影響"
    把 shared expert 加進玩具 MoE，比較兩次跑的結果。預期：**loss 變異數降低**（更少尖峰），最終 loss 持平或更好，因為 shared 路徑每一步都提供密集的梯度，早期的 routing 失誤不會讓模型完全沒有訊號可學。這個效果在 training 早期（冷啟動）最明顯，隨著路由 expert 逐漸分化會縮小。

??? Success "4 — $E$ 增加會讓 all-to-all 訊息變小"
    固定總參數量下，expert 數變多 ⇒ 每個 expert 變小 ⇒ 每個 token 送到某個 expert-GPU 的 payload 不變，但 token 被分散到**更多目的地**，所以每筆 all-to-all 訊息會變得*更小、數量更多*。大量微小訊息會傷害網路效率（變成 latency/開銷限制，連結利用率變差）。**node-limited routing** 限制一個 token 的 expert 能跨多少個節點，讓訊息保持夠大，繼續是頻寬限制而不是 latency 限制。

## 訓練穩定性 { #training-stability }

??? Success "1 — z-loss 如何縮小 $\|x\|$、讓 softmax 遠離 one-hot"
    $\mathcal{L}_z = \beta(\log\sum_e e^{x_e})^2$。log-sum-exp 會隨 logit 的大小增長，懲罰它的平方會把 logit 拉向較小值 → $\|x\|$ 縮小。logit 變小 ⇒ softmax 更接近均勻分佈 ⇒ **routing 熵更高**、機率遠離 one-hot（0/1）。具體來說，這讓 gate 維持**可塑性**：接近飽和的 softmax 梯度會消失、一旦分配錯誤就逃不出來；z-loss 能防止這種凍結狀態。

??? Success "2 — BF16 會翻轉 argmax，FP32 不會"
    BF16 有 8 個尾數位 → 1.0 附近的 ULP 是 $2^{-7}\approx0.0078$。取 logit $x_1 = 1.0000,\ x_2 = 1.0039$（真實 argmax 應該是 2）。兩個值在 BF16 下都會四捨五入成**同一個**值 $1.0$，所以 BF16 的 argmax/top-k 平局是任意決定的（而且會跟 data parallel 裡的 rank 相關）；FP32（23 個尾數位）則能保留兩者的差異，正確選出 expert 2。這就是一種「無聲錯誤」：不同 replica 在 routing 上的決定會不一致，破壞共享的負載平衡計數。

??? Success "3 — router 用大初始化 scale，搭配/不搭配 z-loss"
    在玩具 MoE 上，故意用較大的 scale 初始化 router，分別訓練兩個版本。預期：**沒有 z-loss** 時，logit 很快就會爆炸 → 頻繁的 loss 尖峰/NaN，還會出現幾個**dead expert**（gate 飽和、routing 被鎖死）。**加上 z-loss** 後，logit 維持在有界範圍內 → 尖峰很少、dead expert 數量接近零、loss 曲線更平滑。這讓 z-loss 的作用變得具體：它是對抗 logit 膨脹的廉價保險。

??? Success "4 — 為什麼 shared expert 能簡化冷啟動"
    在第 0 步，所有路由 expert 幾乎完全相同（還沒分化），所以路由梯度帶雜訊、近乎對稱——沒有任何明確的訊號可以專業化。shared expert 在 top-$k$ 選擇之**外**，所以 $\partial \mathcal{L}/\partial\,\text{shared}$ 從第 0 步就是對**每個** token 都密集的梯度，給模型一條可靠的學習路徑，讓 routing 慢慢自行收斂。追蹤一下：$y = \text{shared}(h)+\sum g_e e(h)$ → $\nabla_{\text{shared}}$ 完全不依賴（一開始是隨機的）routing 決策。

## 系統與 expert parallelism

??? Success "1 — all-to-all 位元組數與 expert GEMM 時間（$T{=}4096/\text{GPU}, d{=}4096$）"
    對 all-to-all 來說，每個 GPU 要傳輸 ≈ 自己的 tokens × $d$ × 2 B $= 4096\times4096\times2 \approx 3.4\times10^{7}$ B = 34 MB；每層有**兩次**（dispatch + combine）→ ~67 MB/GPU/層。乘上 60 層 ≈ **4 GB/GPU** 的流量。走 NVLink（節點內 ~300 GB/s）約 13 ms；跨節點走 IB（~50 GB/s）約 80 ms。expert GEMM 時間：每 token $2\cdot k\cdot 8d^2$ FLOP；$k=2$、$T=4096$ 時約 $4096\cdot2\cdot8\cdot4096^2 \approx 1.1\times10^{12}$ FLOP/層 → 在 H100（~990 TFLOP/s）上約 1.1 ms/層，60 層約 66 ms。所以**跨節點 EP 是通訊受限的**（80 ms 通訊 vs 66 ms 計算），除非 all-to-all 能重疊或限制在節點內；節點內則大致平衡——這正是通訊重疊與 node-limited routing 重要的原因。

??? Success "2 — node-limited routing 如何限制跨節點流量"
    如果一個 token 的 $k$ 個 expert 可能落在 $k$ 個不同節點上，跨節點頻寬成本最多會到單節點的 $k$ 倍。把每個 token 的 expert 限制在 $\le M$ 個節點內，就能把每個 token 最壞情況下的跨節點訊息數量固定在 $M$（DeepSeek-V3 用 $M=4$、$k=8$）。**代價：**如果全域最好的 $k$ 個 expert 散落在超過 $M$ 個節點上，router 就不能再選它們——用一點品質損失換取頻寬上限。

??? Success "3 — padding 浪費（batched GEMM，cf=2.0）與 grouped GEMM（CV=0.5）"
    capacity factor 2.0 的 batched GEMM 會把**每個** expert 都 pad 到 $2\times$ 平均負載，所以就連完美平衡的 expert 也會浪費約 50% 的 slot；而且 pad 出來的 tensor 大小是針對最壞情況 → 浪費大量 FLOP。grouped GEMM 只跑每個 expert 實際的 token 數（不 pad），所以不管 CV 是多少，浪費都 $\approx 0$。在 CV = 0.5 時，batched 形式浪費的比例大致等於 capacity 與實際負載的差距（百分之幾十）；grouped 的優勢非常明顯——這正是現代 MoE kernel 都採用 grouped/可變長度 GEMM 的原因。

??? Success "4 — 用分塊排程讓 shared expert 計算與 dispatch 重疊"
    把 token batch 切成幾個 chunk；當 chunk $i$ 的 dispatch all-to-all 正在飛行時，去算 chunk $i-1$ 不需要 routing 結果的**shared expert**（和/或 attention）計算。對應的 pipeline 圖見 [系統與 expert parallelism](../moe/systems-ep.md)。**重疊量受限於**（通訊時間，可獨立計算的時間）兩者中較小的那個：如果 shared expert/attention 的工作量比 all-to-all 短，通訊就會暴露出來；分塊本身的開銷與並發 kernel 能拿到的 SM/queue 資源也會限制重疊效果。

## MoE kernels

??? Success "1 — Triton gather + unpermute 融合進 epilogue"
    在算正向 permute `perm[i]`（排序後位置 → 原始位置）的同一個 kernel 裡，順手用一次 scatter 寫出 `inv_perm[perm[i]] = i`。這樣 grouped GEMM 的 **epilogue** 就能用 `inv_perm` 把每個 expert 輸出的那一列直接 scatter 回原始 token 位置，省掉一次額外經過 HBM 的 gather/scatter——一次融合寫入，而不是先算完再額外做一次 permute。

??? Success "2 — wavefront-agnostic 的 CUDA `gather_rows`；block size 掃描"
    把所有硬編碼的 32 換成 `warpSize`（NVIDIA 上是 32，CDNA 上是 64），用在 warp 內部的邏輯，並把 block size 參數化。對 128/256/512 做 benchmark：throughput 先上升、再持平/下降——block 太小無法餵滿 SM（occupancy 低、啟動開銷占比更高），block 太大則撞上 register/SMEM 上限、降低常駐 block 數。峰值出現在 occupancy 剛好把記憶體頻寬餵飽、但還沒溢位的那一點——對 memory-bound 的 gather 來說通常是 256。

??? Success "3 — padded batched GEMM 與 grouped GEMM：時間 vs capacity factor"
    把 wall-clock 時間對 cf 畫出來。**batched** 的時間大致**隨 cf 線性增長**（你確實在對 pad 出來的零做計算）。**grouped** 幾乎是平的（跟 cf 無關——它只處理真正的 token）。兩條線大約在 cf ≈ 1 加一個小常數附近交叉；在交叉點以下，較簡單的 batched kernel 可能因為實現簡單而略勝；超過交叉點之後 grouped 全面領先。實際的 MoE 通常跑在 cf ≥ 1.25，所以 grouped 是標準選擇。

??? Success "4 — profile fused 與 unfused dispatch（HBM 位元組）"
    融合 gather → GEMM 可以避免把排列後的 token tensor 具現化到 HBM 上。在 profiler 的記憶體計數器（例如 `dram__bytes` 或等效項）裡，fused 路徑應該會顯示**更低的 HBM 讀+寫位元組**，差距大致是那個 tensor 的大小（$T\times d\times 2$ B），這證實省下來的是*流量*而不是 FLOP——正符合 permute 本身是 memory-bound 的預期。

## 推論與 serving { #inference-serving }

??? Success "1 — DeepSeek-V3 權重的 HBM 占用：BF16 / FP8 / int4"
    671B 參數：**BF16** $= 671\times2 = 1342$ GB；**FP8** $= 671$ GB；**int4** $\approx 336$ GB。用 80 GB 的 GPU（只算權重，還沒算 KV）：BF16 需要 $\lceil 1342/80\rceil = 17$ 張；FP8 需要 9 張；int4 需要 5 張。量化直接幫你省下 GPU 數量——這正是 MoE serving 高度依賴低精度的主要原因。

??? Success "2 — expert weight streaming 被藏起來的條件"
    當 expert 的**GEMM 時間 ≥ 傳輸時間**時，weight streaming 就能被藏起來：

    $$ \frac{2\,n_e\,(8d^2)}{\text{FLOP/s}} \;\ge\; \frac{8d^2 \cdot \text{bytes}}{\text{PCIe BW}}, $$

    其中 $n_e$ = 路由到該 expert 的 token 數。約掉 $8d^2$ 之後得到 **tokens-per-expert** 的門檻：$n_e \ge \tfrac{1}{2}\cdot \tfrac{\text{FLOP/s}}{\text{PCIe BW/byte}}$。大 batch（每個 expert 的 token 數多）能把傳輸藏起來；batch-1 的 decode 永遠做不到——所以 expert offload 對高 throughput serving 有幫助，但救不了低 batch 的 latency。

??? Success "3 — 一個 batch 會碰到多少不同的 expert（batch 256，$E{=}256, k{=}8$）"
    $E=256$ 個 expert 總共要分配 $256\times8 = 2048$ 次。用 balls-into-bins 估計被碰到的不同 expert 數：$E(1-(1-1/E)^{2048}) = 256(1-(255/256)^{2048}) \approx 256(1-e^{-8}) \approx 256\times0.99966 \approx \mathbf{256}$——也就是**幾乎所有 expert 都會被用到**。換算下來每個 expert 平均 tokens/expert $= 2048/256 = 8$，呈 Poisson 分佈（std ≈ √8 ≈ 2.8）。意義在於：在 serving 的 batch size 下，你幾乎無法避免載入大部分 expert，所以常駐權重的策略會贏過 cache。

??? Success "4 — 依熱度做 expert cache eviction；失效模式"
    用觀測到的 routing 頻率做 LFU/LRU：熱的 expert 留在 HBM，冷的就 stream。**失效模式：**如果 routing 分佈在執行期間**漂移**（例如請求流的領域變了），cache 裡塞滿的會是以前熱、現在冷的 expert → miss rate 與 latency 都會在新領域重新變熱時飆高。緩解方式是用自適應視窗/衰減去追蹤最近的流量，並設定一個常駐 expert 的下限。

## 案例研究

??? Success "1 — 每個模型的 $\binom{E}{k}$ 與細粒度參數"
    計算每個案例研究模型的 $\binom{E}{k}$（例如 Mixtral $\binom{8}{2}=28$；DeepSeek-V3 的路由 expert 是 $\binom{256}{8}\approx4\times10^{14}$）。expert 數越多、越細粒度 ⇒ 在相同 active $k$ 下，每個 token 能組出的「expert 混合」數量天文數字般增加——這正是「細粒度 expert」這個論點的量化核心，也是各代模型的 $E$ 持續演進的原因。

??? Success "2 — 每 1k tokens 的 KV 快取：MLA、GQA 與 MHA"
    每個 token、每層：**MHA** 快取 $2\,n_h d_h$；**GQA** 快取 $2\,n_{kv}d_h$（$n_{kv}\ll n_h$）；**MLA** 只存一個 latent 向量 $d_c$。對 $n_h=128$、$d_h=128$、$n_{kv}=8$、$d_c\approx 512$ 的模型：MHA $=2\cdot128\cdot128=32768$ B；GQA $=2\cdot8\cdot128=2048$ B；MLA 每 token 每層約 $512\cdot2=1024$ B。乘上 $L$ 和 1000 個 token 即可得到總量。MLA 比 GQA 省約 2 倍，比 MHA 省約 30 倍——這正是讓長上下文的 DeepSeek serving 划算的關鍵槓桿。

??? Success "3 — DeepSeek-V3 與 Mixtral：active/total 比例與 serving 取捨"
    V3 的 active 比例 ≈ 5.5%（37B/671B），Mixtral ≈ 28%（13B/47B）。V3 較低的 active 比例意味著**每個 token 背後的記憶體更多，但計算量更少**——對高 throughput、記憶體充裕的 serving（很多 GPU、大 batch）更有利。Mixtral 的比例較高，記憶體佔用較小，更適合較小規模的部署與 batch-1 latency。active/total 比例本質上就是在記憶體成本與計算成本之間做取捨。

??? Success "4 — 把一個模型對應回 MoE 篇各頁"
    例如**DeepSeek-V3**：sigmoid gate + bias controller → [負載平衡](../moe/load-balancing.md)；細粒度＋shared expert → [Routing 變體](../moe/routing-variants.md)；z-loss / FP32 router → [訓練穩定性](../moe/training-stability.md)；node-limited routing + DualPipe → [系統與 expert parallelism](../moe/systems-ep.md)；MLA → [Attention 效率](../foundations/attention-efficiency.md)；FP8 → [數值與精度](../foundations/numerics-precision.md)。每一項生產技巧都能對應回某一頁——這正是這本手冊想呈現的主題。

## MoE decode 剖析

??? Success "1 — 完全重疊的 12% 階段，最大加速是多少"
    如果某個 stage 的 kernel 完全被**另一個 stream 重疊掉**，它就不在 _critical path_ 上——wall-clock 不會把它算進去。即使把它變成無限快，省下的端對端 latency 也是**~0%**（你只能省下沒被藏住的那一小部分，如果還有的話）。這正是 Track A 與 Track B 的差別：12% 是 Track A（kernel 效率）的數字，但它對 *latency* 的貢獻早就被 Track B（並發）抵銷掉了。同一段時間不能被省兩次——只有**暴露（未被重疊）**的 kernel 時間，縮小後才會直接轉換成 latency 的節省。

??? Success "2 — 把 GEMM1（15%, ×60）或 LM head（1%, ×1）速度提升一倍？"
    把路由 expert 的 GEMM1 速度提升一倍，可以省下 decode 總時間的 $\sim 0.5\times15\% = 7.5\%$；把 LM head 速度提升一倍只省 $\sim 0.5\times1\% = 0.5\%$。**GEMM1 贏了約 15 倍。**教訓：要優化的是**總成本**（= 每次呼叫成本 × 呼叫次數），不是單次呼叫成本。每次呼叫便宜但每層都會觸發的 kernel，加起來可能比每次呼叫昂貴但只觸發一次的 kernel 更重要——呼叫次數是你不能忽略的乘數。

??? Success "3 — shared-expert fusion 何時划算？"
    fusion 去掉了 $n$ 個原本獨立的 kernel（shared GEMM、activation、quant、residual——每層各一次），省下 $S = L\sum_i t_i^{\text{removed}}$，同時在路由路徑上多花 $A$（sort 變大 + grouped GEMM 因為多了 shared token 而稍微變大）。fusion 划算的條件是

    $$ S > A \quad\Longleftrightarrow\quad L\sum_i t_i^{\text{removed}} \;>\; \Delta t_{\text{sort}} + \Delta t_{\text{GEMM}}. $$

    在實測的 decode trace 裡，$S \approx 19.8\%$、$A \approx 0.7\%$，淨收益約 18%。這很划算，因為被去掉的工作是*每層都重複、而且冗餘*的，而多出來的工作只是讓本來就在跑的 GEMM 稍微變大一點——這個不等式因此非常寬鬆。

??? Success "4 — self-time 111% vs wall-clock 96%"
    **self-time** 是把每個 kernel 的時間獨立加總。如果兩個 kernel 在**不同 stream 上同時跑**，它們各自的時間都會被計入，但兩者共用同一段 wall-clock，所以 self-time 總和可以**超過** wall-clock——111% 代表確實有重疊（被藏起來的時間 = self − busy，是正數）。數值**低於 100%**（96%）則代表 kernel 大致是**串行**執行、而且中間有**空閒間隙**：wall-clock = busy + idle，串行時 busy ≈ self，所以 self/wall < 100% 意味著有 ~4% 的時間 GPU 完全閒著、什麼都沒在跑（啟動 latency、同步等待）。

??? Success "5 — 每層 GEMM 用 split-K 值得嗎？"
    split-K 把 1 個 GEMM 變成（計算 kernel + reduce kernel）= **2 次 launch**。對 $L=60$ 層、每層都做的 GEMM 來說，每個 decode 步驟就多了**+60 次 reduce launch**；如果只用在每個 decode 步驟跑一次的 LM head 上，成本只是 **+1**。split-K 換來的並行性（更多 block 同時忙在 K 維度上）能在單一 GEMM 太小、無法餵滿整個裝置時有幫助——但在**batch-1 decode**裡，GEMM 本來就已經是 memory-bound、規模又小，額外的 60 個 reduce kernel（每個都要往返一次 HBM 的部分結果）大多只是開銷。**in-kernel 的 K 累積通常對每層的 decode GEMM 更好**；split-K 該留給規模較大、每步只做一次的投影（例如 LM head），那裡的並行性才真正划算。
