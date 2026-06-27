# 解答 — Mixture-of-Experts

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 全部九個 MoE 頁面</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

解答[MoE 篇](../moe/index.md)的練習。部分參考 到[`code/`](https://github.com/youyun8/deep-kernel-handbook/tree/main/code)中的玩具模型； 如果推導具有乾淨的封閉形式，我們給出它，否則我們給出 方法和預期的定性結果。

## 為什麼稀疏

??? success "1 — 總參數與活動參數、FLOP 比率 ($E{=}128, k{=}2, d{=}4096$)"
    1 FFN ≈ $8d^2$ 參數（向上 $4d^2$ + 向下 $4d^2$，帶有 $d_{ff}=4d$）。

    - **experts 總計：**$128 \times 8d^2 = 1024d^2 \approx 1.72\times10^{10}$。
    - **根據 token 有效：**$2 \times 8d^2 = 16d^2 \approx 2.68\times10^{8}$。

    FLOPs 透過 *active* 參數進行縮放，因此 FLOPs-per-token 比率與密集 （$E{=}1$，即 $8d^2$ 有效）是 $16d^2/8d^2 = \mathbf{2\times}$ — 你付費 $k=2$ experts。但你**儲存**$128\times$ FFN 參數：巨大的容量， 常數計算。這種差距就是稀疏性的全部價值主張。

??? success "2 — DeepSeek-V3 稀疏性與 Mixtral"
    DeepSeek-V3：$37/671 \approx 5.5\%$ 活躍（≈**18× 稀疏性**）。混合 8×7B：8 個 experts、$\approx 13/47 \approx 28\%$ 的前 2 個有效（≈ 3.6×）。 V3 是 更稀疏——許多細粒度的 experts 具有低激活——這是 現代趨勢：更多、更小的 experts 提供更多組合（下一個練習） 在相同的主動成本下。

??? success "3 — 何時更喜歡密集 37B 而不是 671B/37B-主動稀疏"
    相同的活動 FLOP，但稀疏模型需要**~18× 記憶體**來保存所有 experts 居民。在以下情況下首選**密集**：(a) 記憶體張力（單一 GPU / 邊緣）； (b)**batch-1 latency**很重要，你不能攤銷 expert 裝載 — 稀疏的 decode 可能會為少數 tokens 觸及許多 experts，從而損害局部性； (c) 在小資料集上進行**微調**，其中大多數 experts 得到的資訊很少 梯度和風險陳舊性。對於高 throughput serving 和 pretraining 每 FLOP 品質。

??? success "4 — 卸載 experts：哪個 roofline 軸綁定？"
    透過 PCIe 從 CPU/NVMe 串流 experts（約數十 GB/秒）與 HBM（TB/秒） — 1-2 數量級的頻寬懸崖。限制器變成**頻寬** （expert 權重的 PCIe/NVMe 傳輸），不計算。串流媒體僅隱藏 如果 tokens-per-expert 夠大以至於 GEMM 時間超過傳輸 時間 — [inference & serving](#inference-serving) 中匯出的確切條件 練習 2.

## MoE 層從頭開始

??? success "1 — `MoELayerNaive` 中的 S 型門控"
    對於頂部 $k$ 閘值，將 `softmax(logits)` 替換為 `sigmoid(logits)`； 與 Softmax 不同，Sigmoid 閘是**獨立的**（它們的總和不等於 1），所以 每個 expert 的重量是一個絕對的「應該這個火」的分數。選擇後 top-$k$，如果你想要凸組合，請重新規範化所選的閘。驗證 針對測試的輸出形狀/有限性；預計損失可比但不同 平衡動態（Sigmoid + 偏移控制器是 DeepSeek-V3 的配方）。

??? success "2 — $k{=}1$（開關），無需重整化"
    使用 $k=1$ 且不進行重規範，輸出為 $g\cdot \text{expert}(h)$，其中 $g=\text{softmax}(\cdot)_{\text{top1}} \in (0,1)$。所以 expert 輸出是 **按 $g<1$ 縮小**，縮小活化幅度（並耦合 剩餘尺度來決定門控置信度）。開關保持門作為乘法器打開 目的 - 它為 router 提供可微訊號 - 但你必須 考慮縮小的規模（init/LR），或重新歸一化，使保留的門為 1。

??? success "3 — 樸素循環與調度形式（$T{=}8192, E{=}64$，CPU）"
    **樸素循環**迭代 experts，每次都會屏蔽 $T$ tokens → 它接觸 所有 $T$ tokens $E$ 次（大量浪費蒙版工作，但微不足道 向量化）。**調度表**將 tokens 排列成每 expert 連續 僅在其 tokens 上分組並運行每個 expert → 浪費的計算量要少得多，但是 支付聚集/分散費用。在 CPU 上，一旦 $E$ 很大，調度形式通常會獲勝 （樸素的 $O(TE)$ 掩蔽占主導地位）。在**GPU**上差距擴大：調度 形式的連續分組 GEMM 是硬體想要的，而天真的 循環啟動許多微小的屏蔽 matmul（啟動 + 低佔用率邊界）。

??? success "4 — 新增共享 expert；每一步確認梯度"
    加入$y = \text{shared}(h) + \sum_{e\in\text{TopK}} g_e\,\text{expert}_e(h)$。 因為 `shared(h)` 位於**每個**token 的路徑上，無論 routing 為何， `shared.weight.grad` 是非 `None` 並且在每個步驟上都非零（檢查後 `backward`）。路由 experts 僅在選取時才獲得漸層；共用 expert 是穩定冷啟動的始終在線密集路徑（請參閱 training 穩定性例如 4).

## 負載平衡

??? success "1 — 均勻分佈最小化 $\sum_e f_e P_e$"
    根據$\sum f_e = k$、$\sum P_e = 1$最小化$\sum_e f_e P_e$。的 開關輔助損耗將路由到 $e$ ($f_e$) 的**tokens**的分數乘以 $e$ ($P_e$) 的平均**閘機率**；當負載分散時，它被最小化。 形式上，透過重排/柯西-施瓦茨，耦合和最小時 兩個向量都是平的：對於所有 $e$，$f_e = k/E$ 和 $P_e = 1/E$，給出 $\sum_e f_e P_e = E\cdot\frac{k}{E}\frac{1}{E} = k/E$。任意濃度 （一些 $f_e,P_e$ 大在一起）提高了產品 - 因此降低了這種損失 推動 routing 走向統一。 _（輔助損耗使用 $f$ = 硬計數，$P$ = 可微的機率，因此梯度流經 $P$。 ）_

??? success "2 — 容量和掉落率（$T{=}4096, E{=}64, k{=}2$，參見 1.25）"
    預期為 tokens/expert $= Tk/E = 4096\cdot2/64 = 128$。容量 $C = \lceil \text{cf}\cdot Tk/E\rceil = \lceil 1.25\times128\rceil = 160$。 如果一個 expert 獲得所有 $Tk = 8192$ 分配 $= 410$ tokens 的 5%，則它 $410-160 = 250$ 溢位；那些被**丟棄**。那個的掉率 所有分配的 expert $\approx 250/8192 \approx 3.0\%$（其他假設 能力範圍內）。將 cf 提高到 2.0 ($C=256$) 仍然會降低 $410-256=154$ — 僅顯示容量無法修復嚴重傾斜的 router；你需要平衡。

??? success "3 — 調整偏移控制器"
    輔助無損控制器微調每個 expert 偏差： $b_e \leftarrow b_e + \gamma\,\text{sign}(\text{target} - \text{load}_e)$. **Larger $\gamma$**→ 更快的收斂，但圍繞平衡振盪/超調；**較小 $\gamma$**→ 平滑但緩慢。**Sigmoid**門的響應比 **softmax**因為 sigmoid 分數是獨立的－改變一個偏差不會 透過共享標準化器重新調整其他參數，因此控制器的 per-expert 調整不會互相衝突。預計負載 CV 會下降到 ~0.1–0.2 在經過精心調校的 $\gamma$ 上只需幾百步。

??? success "4 — 玩具 MoE 上的輔助損耗與無輔助損耗"
    雙向運行 `train_tiny_moe.py`。預期：**aux-loss-free**達到類似水平 或在相同負載 CV 下略微**降低最終 LM 損耗**，因為它平衡了 透過 routing _計數_ 上的偏差，而不添加競爭的梯度項 使用 LM 物鏡（輔助損耗稍微扭曲了損耗表面）。兩者都 負載 CV 應降低至 ~0.1–0.2；輔助無損耗運轉避免了 平衡與質量之間的拉鋸戰。並列報告最終損失和 CV 側面——配對就是重點。

## Routing 變體

??? success "1 — expert 組合和細粒度增益"
    組合 $= \binom{E}{k}$：

    - $(8,2): \binom{8}{2} = 28$
    - $(64,8): \binom{64}{8} \approx 4.4\times10^{9}$
    - $(256,8): \binom{256}{8} \approx 4.1\times10^{14}$

    每個 token 選擇一種組合，因此更多、更精細的 experts 呈指數級增長

    **相同活性 $k$**上更專業的「混合」—細粒度 expert 參數 (DeepSeekMoE)。組合能力，恆定計算。

??? success "2 — 為什麼 expert-選擇會破壞自回歸 decode"
    expert-選擇讓每個 expert 從整批中挑選其頂級 $C$ tokens** — 假設所有 tokens 都同時存在（training/prefill 中為真）。在 自回歸**decode\*\*你一次產生一個 token；expert 不能 在尚不存在的未來 tokens 和頂級 $C$ 中“選擇” 一批是沒有意義的。 token-選擇每個新 token 獨立的路線， 所以這是自然的 inference 時間方案。

??? success "3 — 共享 expert 對穩定性的影響"
    將共享 expert 加入玩具 MoE 並比較運行。預期：**損失降低 方差**（更少的尖峰）和等於或更好的最終損失，因為共享 路徑每一步都提供密集的梯度，因此早期的 routing 錯誤不會 使模型缺乏訊號。 training 早期效果最大（冷 開始）並隨著路由 experts 的區分而縮小。

??? success "4 - 成長的 $E$ 縮小了 all-to-all 訊息"
    在固定的總參數下，更多的 experts ⇒ 每個 expert 更小 ⇒ 每個 token 給定 expert-GPU 的有效負載不變，但 tokens**分佈在更多 目的地**，因此每個 all-to-all 訊息都會變得*更小，數量更多*。 許多微小的訊息會損害網路效率（latency-和開銷限制，較差 連結利用率）。**節點限制 routing**限制 token 的節點數量 experts 跨度，保持訊息足夠大以保持頻寬限制 比 latency 綁定。

## Training 穩定性

??? success "1 — $\mathcal{L}_z$ 縮小了 $\|x\|$ 並將 softmax 限制為 one-hot"
    $\mathcal{L}_z = \beta(\log\sum_e e^{x_e})^2$。 log-sum-exp 隨 logits 的大小，因此懲罰其平方會將 logits 拉向較小的值 → $\|x\|$ 縮小。較小的 logits ⇒ softmax 更接近均勻 ⇒**更高 routing 熵**和機率遠離 one-hot (0/1)。具體來說 它使門保持**可塑性**：接近飽和的 softmax 具有消失梯度 且無法逃避糟糕的任務； z 損失可以防止凍結狀態。

??? success "2 — BF16 翻轉 argmax，FP32 不翻轉"
    BF16 有 8 個尾數位 → 1.0 附近的 ULP 是 $2^{-7}\approx0.0078$。取邏輯值 $x_1 = 1.0000,\ x_2 = 1.0039$（真實 argmax = 2）。兩者都四捨五入到**相同** BF16 值 $1.0$，因此 BF16 argmax/topk 平局是任意的（並且 資料並行性下的秩相關），而 FP32（23 尾數位）則保持 它們截然不同，並選擇 expert 2。這是「無聲錯誤」：複製品不同意 在 routing 上並破壞共享平衡計數。

??? success "3 — 大型 router 初始化，帶/不帶 z 損失"
    在玩具 MoE 上，用故意大的比例初始化 router， 雙向訓練。預期：**沒有 z 損失**，logits 會提前爆炸 → 頻繁 損失尖峰/NaN 和幾個**死 experts**（飽和門鎖 routing）。 **使用 z 損失**，logits 保持有界 → 峰值很少，死亡 expert 計數接近 零，更平滑的損失。這使得 z-loss 的作用變得具體：它是便宜的保險 反對邏輯膨脹。

??? success "4 — 為什麼共享 expert 可以簡化冷啟動"
    在步驟 0 中，路由的 experts 幾乎相同（無差異），因此 路由梯度有雜訊並且幾乎對稱——沒有什麼需要專門研究的訊號。 共享的 expert 位於頂部 $k$ 選擇的**外部**，因此 $\partial \mathcal{L}/\partial\,\text{shared}$ 是**每個**上的全密集梯度 token 從第 0 步開始，為模型提供可靠的學習路徑，而 routing 自行解決。追蹤：$y = \text{shared}(h)+\sum g_e e(h)$ → $\nabla_{\text{shared}}$ 從不依賴（隨機）routing 的決定。

## 系統和 expert 並行性

??? success "1 — all-to-all 位元組與 expert GEMM 時間 ($T{=}4096/\text{GPU}, d{=}4096$)"
    根據 all-to-all，每個 GPU 移動 ≈ 其 tokens × $d$ × 2 B $= 4096\times4096\times2 \approx 3.4\times10^{7}$ B = 34 MB；每層**兩個**（調度+組合）→ ~67 MB/GPU/層。超過 60 層 ≈**4 GB/GPU**流量。在 NVLink 上（~300 GB/s 節點內）約 13 毫秒；跨節點 IB (~50 GB/s) ~80 ms。 expert GEMM 時間：每 token $2\cdot k\cdot 8d^2$ FLOPs；在$k=2$， $T=4096$，就是$\approx 4096\cdot2\cdot8\cdot4096^2 \approx 1.1\times10^{12}$ FLOPs/層 → 在 H100 (~990 TFLOP/s) 上~1.1 ms/層，超過 60 層~66 ms。 因此，**跨節點 EP 是受通訊限制的**（80 毫秒通訊 vs 66 毫秒計算），除非 all-to-all 是重疊/節點限制的；節點內大致平衡 - 這正是重疊和節點限制 routing 很重要的原因。

??? success "2 — 節點限制 routing 限制跨節點流量"
    如果每個 token 的$k$ experts 可能登陸$k$不同節點你支付跨節點 每個 token 的頻寬高達 $k$ 倍。將 experts 限制為 $\le M$ 節點邊界 $M$ 處每個 token 最壞情況的跨節點訊息（DeepSeek-V3 使用 $M=4$ $k=8$）。**成本：**router 無法再選擇全球最好的 $k$ experts 如果它們分散在 $>M$ 節點上——一個小的質量打擊會換來一個嚴重的打擊 頻寬上限。

??? success "3 - 填充浪費（批量，cf 2.0）與分組 GEMM（CV 0.5）"
    批量 GEMM 容量因子 2.0 焊盤**每個**expert 至 $2\times$ 均值 負載，因此即使是完美平衡的 expert 也會浪費約 50% 的插槽；和 實際負載填充張量的大小是針對最壞情況 → 大量浪費的 FLOP。 分組 GEMM 恰好在其 token 計數上運行每個 expert（無填充），因此 無論 CV 為何，都會浪費 $\approx 0$。對於 CV = 0.5，批量表單浪費 大致容量與實際的差距（百分之幾十）；分組獲勝明顯－ 現代 MoE kernels 使用分組/可變長度 GEMM 的原因。

??? success "4 — 分塊計劃與共享 expert 計算重疊調度"
    將 token 批次拆分為區塊；當區塊 $i$ 的調度 all-to-all 處於 飛行，計算塊 $i-1$ 的**共享 expert**（和/或 attention）， 不需要 routing。請參閱轉換後的管道圖 [Systems & EP](../moe/systems-ep.md)。**重疊受到**以下較小者的限制 （通訊時間，獨立計算時間）：如果共享 expert/attention 工作是 比 all-to-all 短，通訊暴露；也透過分塊開銷和 並發 kernels 的可用 SM/佇列。

## MoE kernels

??? success "1 — Triton 聚集 + 逆排列，融合到尾聲"
    沿著前向排列 `perm[i]`（排序位置 → 原始），發出 `inv_perm[perm[i]] = i` 與 kernel 相同（一次分散寫入）。然後 grouped-GEMM**epilogue**可以將每個 expert 的輸出行直接分散到 他們原來的 token 位置使用`inv_perm`，避免了單獨的 透過 HBM 進行分散讀取 — 一次融合寫入，而不是計算然後排列。

??? success "2 — 波前不可知的 CUDA `gather_rows`；塊大小掃描"
    將任何硬編碼的 32 替換為 `warpSize`（NVIDIA 上為 32，CDNA 上為 64） 扭曲內邏輯，並對區塊大小進行參數化。基準測試 128/256/512： throughput 上升然後穩定/下降 — 小塊未充分利用 SM（低 佔用，更多的啟動開銷），非常大的區塊達到暫存器/SMEM 限制 並減少居民街區。峰值是佔用率使記憶體飽和的地方 不溢出的頻寬 — 對於受記憶體限制的收集，通常為 256。

??? success "3 — 填充批次與分組 GEMM：時間與容量因子"
    對照 cf 繪製牆上時間。**批量**時間大致**與 cf 線性增長** （你從字面上計算填充零）。**分組**是〜平坦的（獨立於 cf — 它只處理真實的 tokens）。它們在 cf ≈ 1 + 一個小常數附近交叉； 在交叉下方，更簡單的批量 kernel 可以在啟動簡單性方面獲勝， 在它上面分組占主導地位。真實 MoE 運行於 cf ≥ 1.25，因此分組為 標準選擇。

??? success "4 — 設定檔融合與未融合調度（HBM 位元組）"
    融合聚集 →GEMM 避免了 HBM 中排列的 token 張量的具體化。 在分析器的記憶體計數器（例如 `dram__bytes` / 等效項）中，融合 路徑應顯示**較低的 HBM 讀+寫位元組**，大致為該大小 中（$T\times d\times 2$ B），確認獲勝是*流量*，而不是 FLOPs — 正是你根據排列的記憶體限制性質所預測的。

## Inference 和 serving

??? success "1 — DeepSeek-V3 權重的 HBM：BF16 / FP8 / int4"
    671B 參數：**BF16** $= 671\times2 = 1342$ GB; **FP8** $= 671$ GB; **int4** $\approx 336$ GB。在 80 GB GPU 上（僅權重，KV 之前）：BF16 → $\lceil 1342/80\rceil = 17$; FP8 → 9; int4 → 5。量化直接買給你更少 GPU——MoE serving 嚴重依賴低精度的主要原因。

??? success "2 — 隱藏 expert 流的條件"
    當 expert 的**GEMM 時間 ≥ 傳輸時間**時，串流媒體將被隱藏：

    $$ \frac{2\,n_e\,(8d^2)}{\text{FLOP/s}} \;\ge\; \frac{8d^2 \cdot \text{bytes}}{\text{PCIe BW}}, $$

    其中 $n_e$ = tokens 路由到 expert。 $8d^2$ 取消，給出

    **tokens-per-expert**的閾值：$n_e \ge \tfrac{1}{2}\cdot \tfrac{\text{FLOP/s}}{\text{PCIe BW/byte}}$。大批量（很多 tokens/expert） 隱藏轉帳；批次 1 decode 從來不會這樣做 - 所以卸載有助於 throughput serving，不是低階 latency 單碼串流。

??? success "3 — 接觸過的不同 experts（第 256 批，$E{=}256, k{=}8$）"
    透過 $E=256$ experts 分配 $= 256\times8 = 2048$。預期明顯 experts（球箱）：$E(1-(1-1/E)^{2048}) = 256(1-(255/256)^{2048}) \approx 256(1-e^{-8}) \approx 256\times0.99966 \approx \mathbf{256}$ — 即 **基本上所有 experts**都被觸及。意思是 tokens/expert $= 2048/256 = 8$， 泊松分佈（std ≈ √8 ≈ 2.8）。意義：在 serving 批量大小下，你 無法避免載入大多數 experts，因此常駐權重策略擊敗了快取。

??? success "4 — expert-快取依受歡迎程度逐出；故障模式"
    使用基於觀察到的 routing 頻率的 LFU/LRU：在 HBM 中保持熱 experts， 流冷的。**故障模式：** 如果 routing 分佈**轉變為 運行時**（例如請求流中的域更改），快取現在 充滿了以前熱、現在冷的 experts → 高失誤率和 latency 當天氣重新變暖時懸崖。透過自適應視窗/衰減來緩解如此受歡迎 追蹤最近的交通，並始終駐留 experts 的樓層。

## 案例研究

??? success "1 — 每個模型的 $\binom{E}{k}$ 和細粒度參數"
    計算每個研究模型的 $\binom{E}{k}$（例如 Mixtral $\binom{8}{2}=28$； DeepSeek-V3 $\binom{256}{8}\approx4\times10^{14}$ 用於路由 experts）。 更多/更精細的 experts ⇒ 天文數字上更多的 token 特定混合物 活躍的$k$——「細粒度 experts」聲明的定量核心和 $E$ 在各代車型中不斷發展的原因。

??? success "2 — 每 1k tokens 的 KV 快取：MLA、GQA 與 MHA"
    每個 token、每層：**MHA** 快取 $2\,n_h d_h$；**GQA** $2\,n_{kv}d_h$（$n_{kv}\ll n_h$）； **MLA** 只存 latent $d_c$。對 $n_h=128$、$d_h=128$、$n_{kv}=8$、$d_c\approx 512$ 的模型： MHA $=2\cdot128\cdot128=32768$ B；GQA $=2\cdot8\cdot128=2048$ B；MLA 每 token 每層 $\approx512\cdot2=1024$ B。 乘以 $L$ 和 1000 tokens。 MLA 比 GQA 節省約 2 倍，比 MHA 節省約 30 倍 — 使長上下文 DeepSeek serving 價格實惠的槓桿。

??? success "3 — DeepSeek-V3 與 Mixtral：活躍/總計和 serving 交易"
    V3 ≈ 5.5% 活性 (671B/37B)，混合 ≈ 28% (47B/13B)。 V3 激活率低 比率意味著**每個 token 擁有更多內存，但計算量更少**— 有利於 高 throughput，記憶體豐富的 serving（很多 GPU，大批量）。混合的 比率越高，記憶體佔用越少，對較小的部署更友好，並且 第 1 批 latency。有效/總比率是介於 記憶體成本和計算成本。

??? success "4 — 將一個模型映射回MoE 篇分頁面"
    例如**DeepSeek-V3**：S 形門 + 偏移控制器 → [負載平衡](../moe/load-balancing.md)；細粒度+共享 experts → [routing variants](../moe/routing-variants.md)； z 損失/FP32 router → [training stability](../moe/training-stability.md)；節點限制 routing + DualPipe → [系統與 EP](../moe/systems-ep.md)；MLA → [attention efficiency](../foundations/attention-efficiency.md)； FP8 → [numerics](../foundations/numerics-precision.md)。每一項生產技巧都能在 trace 上留下痕跡 到一頁——這就是手冊的主題。

## MoE decode 剖析

??? success "1 — 完全重疊 12% 階段的最大加速"
    如果舞台的 kernels**完全重疊**另一個串流，則它們 _不在 關鍵路徑_—掛鐘不包括它們。讓他們無限 快速保存端對端 latency 的**~0%**（你只需保存該部分，如果有的話， 這並沒有被隱藏）。這就是 Track A 與 Track B 的差別：12% 是 Track-A（kernel-效率）編號，但其*latency*貢獻已經 由 Track B（並發）支付。你不能將同一時間存入銀行兩次——只能 暴露（非重疊）kernel 時間在縮小時轉換為 latency。

??? success "2 — 將 GEMM1 (15%, ×60) 或 LM 頭減半 (1%, ×1)？"
    將路由的 GEMM1 減半，保存 decode 的$\sim 0.5\times15\% = 7.5\%$； 將 LM 頭減半可節省 $\sim 0.5\times1\% = 0.5\%$。**GEMM1 以 ~15× 獲勝。** 教訓：最佳化**總**（= 呼叫 × 每次呼叫）成本，而不是每次呼叫成本。 每次調用便宜的 kernel 會觸發每一層，而每次調用昂貴的 kernel 則占主導地位 一次觸發－呼叫次數是你絕對不能放棄的乘數。

??? success "3 — 共享 experts 融合何時獲利？"
    讓融合去除$n$冗餘的 kernels（單獨共享 GEMM，激活， Quant,殘差-每個 ×$L$層）保存$S = L\sum_i t_i^{\text{removed}}$， 同時將 $A$ 新增到路由路徑（更大的排序 + 稍大的分組 來自附加共用 tokens 的 GEMM）。當融合是有利可圖的

    $$ S > A \quad\Longleftrightarrow\quad L\sum_i t_i^{\text{removed}} \;>\; \Delta t_{\text{sort}} + \Delta t_{\text{GEMM}}. $$

    在decode的測量跡線$S \approx 19.8\%$和$A \approx 0.7\%$中，所以 淨$\approx 18\%$。這是值得的，因為刪除的工作是*每層並且 冗餘*，而增加的工作對 GEMM 來說只是「小幅邊際」增加， 已經運作了——這裡的不平等非常緩和。

??? success "4 — 自拍 111% vs 掛鐘 96%"
    **自拍時間**獨立地匯總每個 kernel 的持續時間。如果兩台 kernels 運行 **同時**（不同的流），它們的持續時間都重要，但它們共享 掛鐘，因此自拍時間可以**超過**掛鐘 - 111% 意味著實質性 重疊（時間隱藏 = self − busy 為正）。值**低於 100%** (96%) 表示 kernels 是 ~串列 _且_ 有**空閒間隙**：掛鐘 = busy +idle，且串列時 busy ≈ self，因此 self/wall < 100% 意味著 缺失 ~4% 處於空閒狀態（啟動 latency，同步停止），GPU 不運作任何內容。

??? success "5 — 對於每層 GEMM 來說 split-K 值得嗎？"
    Split-K 將 1 個 GEMM 變成（計算 kernel + 減少 kernel）= **2 次發射**。 對於 $L=60$ 層的每層 GEMM，每層**+60 次減少發射** decode；僅在每個 decode LM 頭上執行一次，成本為**+1**。 Split-K 購買 並行性（更多的區塊在 K 維度上忙碌）在單一 GEMM 時會有所幫助 無法填充設備 - 但在**batch-1 decode**中，GEMM 已經 受記憶體限制且較小，因此額外的 60 個減少 kernels（每個 HBM 部分的往返）主要是開銷。**kernel 中的 K 累積為 通常對於每層 decode GEMM 更好**；為大型預留 split-K， 每步一次的投影（LM head），其中並行性實際上是值得的。
