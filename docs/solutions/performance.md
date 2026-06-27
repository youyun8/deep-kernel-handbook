# 解答 — 效能工程

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 全部七個效能頁面</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

解答[效能工程篇](../performance/index.md)的練習。kernel 練習是開放式的（「運行它，對其進行基準測試」）；我們給出了預期的結果 以及推理，以便你可以檢查你的數字。

## GPU 程式設計模型

??? Success "1 — 為什麼 32 通道減少在 CDNA 上是錯誤的"
    使用 `offset = 16,8,4,2,1` 和 32 頻道遮罩硬編碼減少扭曲 假設 32 寬扭曲 (NVIDIA)。 AMD CDNA 波前是**64 通道**，因此 32 通道隨機播放僅減少波前的「一半」 —— 上面的 32 通道是 被忽略，給出錯誤的（部分）總和。修復：開始隨機播放循環 `warpSize/2` 並且在各處使用 `warpSize` 而非文字 32，因此 相同的代碼可以正確減少 32 或 64 個 lane。

??? Success "2 — 佔用限制器（64 個暫存器/線程，48 KB SMEM/區塊）"
    每個 SM：64K 暫存器，100 KB SMEM。取一個 256 線程塊。

    - **暫存器：**$64\times256 = 16384$ 暫存器/區塊 → $65536/16384 = 4$ 區塊。
    - **SMEM：**$100/48 = 2.08$ → **2 區塊**。

    SMEM 是更嚴格的約束 → 2 個常駐區塊（512 個執行緒）。**共享 記憶體是佔用限制器**；按區塊（或區塊大小）切割 SMEM 會提高占用率。

??? Success "3 — 合併用於行優先，而不是其轉置"
    對於行主 $[M,N]$ 張量，扭曲中的執行緒由**列**索引 讀取 `A[row, col0+lane]` — 連續位址 → 一個合併事務。 讀取**轉置**（線程沿著列走）給出 stride-$N$ 位址 → $N$ 單獨事務，約 32 倍記憶體流量。這是 正是 MoE**聚集**：分散的 token 指數打破合併，即 為什麼聚集 kernel 受記憶體限制並且值得融合。

??? Success "4 — 當降低佔用率時會上升 throughput"
    大量暫存器的 matmul 區塊將更多的工作集保留在**暫存器**中 （最快的記憶體），減少 SMEM/HBM 流量和指令數 輸出。這提高了套準壓力 → 更少的常駐扭曲 → 更低 佔用率，但**更高的 throughput**因為每個扭曲確實更有用 工作，並且 kernel 受計算限制，不受 latency 限制。最大占用率只會有幫助 當你需要很多扭曲來隱藏記憶體 latency；鋪得好的 GEMM 則不然。

## Triton 路線

??? Success "1 — 向量加法與 softmax 與 PyTorch"
    兩個 Triton kernels 都應將 `torch` 輸出與 BF16/FP32 容差相符。 向量相加受記憶體限制 → 預計 throughput 接近 HBM 頻寬，與 本機操作。 Fused Softmax 應該**擊敗**簡單的三通道火炬 Softmax 頻寬（一次讀 + 一次寫 vs 三次）並且大致平局 `torch.softmax`（本身已熔斷）。

??? Success "2 — 面向 AMD 的自動調整配置"
    加入不同 `num_warps` (4/8) 和 `BLOCK` 的配置，其中**wavefront-64** 請注意 - 在 CDNA 上，`num_warps=4` 已經意味著 256 個通道/區塊，因此 最佳點塊大小與 NVIDIA 的 32 通道扭曲不同。最好的配置是 GPU 特定；教訓是，在一個供應商上自動調整的配置很少 另一方面是最佳的 —— 總是根據目標重新自動調整。
??? Success "3 — Softmax，用於比 1 `BLOCK` 寬的行"
    循環遍歷 `BLOCK` 大小的圖塊中的行，保持運行最大值 $m$ 和總和 $\ell$ 通過**online-softmax 組合器**（與 FlashAttention 相同） 例如 1)：對於每個圖塊 $m' = \max(m, \max_{\text{tile}})$，將 $\ell$ 重新縮放 $e^{m-m'}$，增加圖塊的貢獻。第二遍（或緩存的分子） 正常化。這消除了“行必須適合一個塊”的限制。

??? Success "4 — 融合 vs 三聲道 softmax 位元組"
    三次讀取行**3×**（最大值、exp-sum、歸一化）並寫入一次。 Fused 讀一次，寫一次。對於 $[R,C]$ 張量，位元組比率為 ≈ $(3+1)/(1+1) = 2\times$ 減少了融合 kernel 的流量 — 並且因為 softmax 受記憶體限制，這相當於 2 倍加速，你的基準測試應該如此 確認。

## CUDA / HIP 軌道

??? Success "1 — 埠平鋪 matmul 到 HIP"
    `hipify`或手口：`__shared__`撐、`cudaMalloc→hipMalloc`、 `<<<>>>` 啟動與 `hipcc` 下的語法相同。使用 `hipcc` 建置（或透過 ROCm 上的 PyTorch）並根據 cuBLAS/hipBLAS 驗證 fp 容差。要點： HIP 是來源相容的 —— 相同的 kernel 在兩個供應商上運行，是唯一真正的 可移植性錯誤是波前寬度（下一個練習）。

??? Success "2 — 32 lane 的 reduction 在 64 寬 wavefront 上會失敗"
    做一個只用 `offset = 16…1` shuffle 的 reduction。在 CDNA 上，64 通道 wavefront 意味著 lane 32–63 從不參與 → 結果只加總了下半部 的一半。用一個長度 64 的全 1 向量做 reduction 來示範：你會得到 32，而不是 64。用 `for (offset = warpSize/2; offset>0; offset>>=1)` 修復。

??? Success "3 — `TILE` ∈ {8,16,32} 掃描"
    更大的圖塊 → 每個全域負載從 SMEM 重複使用更多的資料（更高的算術 強度），但每個區塊有更多 SMEM/寄存器 → 佔用率較低。通常 `TILE=16` 或 `32` 獲勝：8 太小（重複使用性差，受記憶體限制），32 可能溢出 或減少較小 GPU 的佔用。將最佳值與扭曲/波前連結起來 你的特定卡上的區塊和 SMEM 預算。

??? Success "4 — 以矩陣核心取代內部積 (`wmma`/rocWMMA)"
    將標量內循環交換為張量核心/矩陣核心 MMA 片段給出 大幅加速（通常為 4-10 倍），因為矩陣核心每執行一次完整的 tile-MMA 指令的 FLOP/s 比標量 FMA 路徑高很多。要點： 片段需要特定的圖塊形狀/dtypes（例如 16×16×16 BF16）並小心 SMEM 佈局 — 測量相對於標量基線的正確性和加速比。

## 分散式 training { #distributed-training }

??? Success "1 — all-reduce = 減少-分散 + 全聚集；ZeRO-2 體積"
    **恆等式：**減少分散總和，並為每個排名留下一個結果分片； all-gather 然後分配所有碎片 → 每個等級都有完整的減少 張量 = all-reduce。每一步的環成本 ≈ $S(G{-}1)/G$ 位元組/rank，所以 兩者合計 ≈ $2S(G{-}1)/G$ = all-reduce 成本。 **ZeRO-2**對梯度進行分片，因此而不是完整梯度的 all-reduce 它執行**reduce-scatter**（每個排名保留其 grad 碎片，更新其 優化器分片）和**所有參數的集合**— 總體積與 DDP 的 all-reduce ($\approx 2S$)，但它從未實現完整的漸變 或優化器狀態，在同等通訊下節省記憶體。
??? Success "2 — 每 GPU 內存，70B BF16 + Adam，8 個 GPU"
    每個參數的混合精度 Adam 狀態：2 B (BF16 權重) + 2 B (grad) + 4 B (FP32 主) + 4 + 4 B (FP32 m, v) = **16 B/參數**。對於 70B 來說是 $70\text{B}\times16 = 1120$ GB 總計。

    - **DDP：**每個 GPU 儲存所有 16 B/param →**~1120 GB/GPU**（在 80 GB 上不可能 - 需要分片）。
    - **ZeRO-1**（分片優化器，16 B 中的 12）：2+2 + 12/8 = 4 + 1.5 = **~5.5 B/param × 70B ≈ 385 GB**…仍然分裂？每個 GPU：未分片 4 B/參數（權重+梯度，280 GB）+ 分片 12/8 = 1.5 B (105 GB) ≈**385 GB/GPU**。
    - **ZeRO-2**（也是分片等級）：重量 2 B (140 GB) + (2+12)/8 = 1.75 B (122 GB) ≈**262 GB/GPU**。
    - **ZeRO-3**（對所有內容進行分片）：16/8 = 2 B/param × 70B ≈**140 GB/GPU**。

    趨勢就是重點：ZeRO-3 削減每個 GPU 狀態 ~$G\times$ 與 DDP，轉變為 不可能的模型變成了一個合適的模型（具有更多的全聚集流量）。

??? Success "3 — 管道氣泡分數"
    對於 $P$ 階段和 $m$ 微批次，氣泡分數為

    $$ \text{bubble} = \frac{P-1}{m + P - 1}. $$

    保留$<10\%$：$\frac{P-1}{m+P-1} < 0.1 \Rightarrow m > 9(P-1)$。所以對於 $P=8$ 你需要 $m > 63$ 微批次；適用於 $P=4$、$m>27$。更多階段⇒許多 需要更多的微批次來攤提填充/排出 —— 核心 PP 張力。

??? Success 《4－為什麼 TP 可以節點內，EP 可以跨節點》
    **TP**在每層**內執行 all-reduce（兩次：fwd + bwd） 啟動 — 每步都有巨大的、latency 敏感的體積 → 它必須騎在 最快的連結（節點內 NVLink/Infinity Fabric）。**EP**有兩台 all-to-all 每個 MoE 層，但每個 token 有效負載較小，最重要的是，**重疊 具有計算**並且可以是**節點限制\*\*；它可以容忍較慢的跨節點 頻寬。把最多話的 collective（TP）對映到最快的連結，把 可重疊的（EP）放到較慢的網路上。

## 量化和壓縮 { #quantization-compression }

??? Success "1 — int8 仿射量化/反量化和最大誤差"
    仿射：$q = \text{round}(x/s) + z$、$\hat x = s(q - z)$，附 $s = (\max-\min)/255$ 為 int8。每個元素的最大誤差是**半步**， $s/2$。**每個張量**對整個張量使用一個 $s$，因此有一個離群值通道 膨脹 $\max$ → 大 $s$ → 所有*小*通道上的大誤差。 **每個通道**為每個通道提供自己的 $s$，因此異常值較大的 $s$ 不污染其他 → 誤差要低很多。這就是為什麼每個通道（和 AWQ）存在。

??? Success "2 — AWQ 顯著頻道縮放"
    AWQ 擴大了**顯著**（高激活幅度）權重通道 在量化之前透過縮放相應的激活進行補償 下降，因此重要通道實際上獲得更多位元。在一台上實施 線性層：透過啟動統計資料識別顯著通道，應用 每通道縮放、量化為 int4、反量化、測量困惑度。期待 **同時明顯低於簡單的每通道 int4**的困惑度 位寬。

??? Success "3 — decode-latency 在 13B 型號上從 W4 獲得增益"
    Decode 是**記憶體限制**：latency ≈ 權重位元組 / HBM-BW。 BF16 權重 = $13\text{B}\times2 = 26$ GB；int4 ≈ $13\text{B}\times0.5 = 6.5$ GB。位元組數下降 ~4×，因此每個 token decode latency 下降到**~4×**（減 去量化和非量化活化/KV）。勝利純粹來自於移動 每個 token 的權重位元組更少 — [memory-bound](../foundations/attention-efficiency.md) 論證是定量的。

??? Success "4 — 為什麼路由 experts 比 router/attention 更好地容忍 int4"
    路由的 experts 是**冗餘和平均**— 每個 token 只能看到 $k$ 許多 experts，並且許多 experts 的量化噪音在 加權總和。**router**對**小 logit 做出離散決策 差異**（精度至關重要 - 請參閱 MoE 穩定性）和**attention** 向 KV 快取提供數據，其中錯誤**在序列上複合**。如此咄咄逼人 int4 繼續 experts（大多數參數，最寬容），而 router 和 attention 保持更高的精度 — 標準 MoE serving 配方。

## Inference 最佳化

??? Success "1 — 推測 decoding 加速"
    根據草案接受率 $\alpha$ 和提案長度 $\gamma$，預期 每個驗證步驟接受的 tokens 數量為

    $$ \mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}. $$

    加速比 ≈ 除以每步成本比（一個大模型驗證 + $\gamma$ 便宜草稿）。高 $\alpha$ 和適度的 $\gamma$ 提供最好的 返回；作為 $\alpha\to1$，你每次驗證都會接近 $\gamma+1$ tokens。驗收 主導的是頻寬速率，不是裸頻寬。

??? Success "2 — continuous batching vs 靜態 batching，長度在 [64,1024] 均勻"
    靜態批次將每個請求填入**批次中最長的**並等待 對於完成速度最慢的，因此短請求會浪費計算/插槽；有長度 在 [64,1024] 中均勻，平均長度 ≈ 544，但批次運行速度為 ≈ 1024 → ~40–50% 浪費了。**連續批次**驅逐已完成的序列並接納新的序列 每一步，保持批次滿 → throughput 增益的順序 填充廢物分數（對於這種傳播，大約**1.5–2×**，越高越好 方差）。

??? Success "3 — 前綴快取 KV 已儲存，100 個請求，2k 共用提示"
    不共享時，每個請求都會儲存 2k-token 系統提示的 KV 單獨 → $100\times$ 副本。使用前綴緩存，共享前綴是 儲存**一次**並重複使用，儲存 $99\times$ 前綴 KV。如果一層 token KV 的大小為$b$字節，有$L$層，儲存$= 99\times2000\times L\times b$ — 通常為幾 GB。使用通用系統為任何工作負載帶來純粹的勝利 提示。

??? Success "4 — prefill/decode 分解：幫助與傷害"
    分解將計算綁定**prefill**和記憶體綁定**decode**放在一起 單獨的池，每個池都調整到其瓶頸（prefill：大批量，高 MFU； decode：高記憶體頻寬）。當兩個階段發生時它**有幫助** 否則競爭（突發長提示飢餓 decode）。當 **池之間的 KV 快取傳輸**（prefill→decode 切換）成本超過 它避免的爭用－即短提示/小 KV，或慢 互連。關於 KV 位元組、鏈路頻寬、爭用的原因已保存。

## 分析和方法

??? Success "1 — 對 Triton softmax 進行基準測試錯誤，然後正確"
    **錯誤：**在沒有預熱且沒有 `torch.cuda.synchronize()` 的情況下對第一次呼叫進行計時 - 你測量 kernel-啟動+編譯（JIT）latency 和 CPU 端非同步返回， 不是 GPU 時間，通常會減少 10-100 倍。**右：**預熱幾次迭代 （觸發自動調諧/編譯），然後使用 `synchronize()` 進行多次迭代 將循環括起來。修正後的數字是真實的每次呼叫 GPU 時間； 量化差距。

??? Success "2 — 分析 decode 步驟：誰占主導地位？"
    簡介一 decode 小型 Transformer 的一步。**第 1 批**decode，預計 **啟動開銷和記憶體限制的 attention/FFN**占主導地位 - 許多微小的 kernels，每次從 HBM 讀取權重/KV，GPU 未充分利用。 常見修復：**CUDA 圖表**（消除啟動開銷）+ 批次（提高 算術強度）。如果 attention 占主導地位，KV 版面/Flash-decoding 會有所幫助； 如果是 FFN，權重量化會有所幫助。

??? Success "3 — 計算 MFU；診斷 15%"
    $\text{MFU} = \frac{6P \cdot \text{tokens/s}}{\text{GPU peak FLOP/s}}$（training， $6P$ FLOP/token）。15% 代表你只用到約六分之一的數學單元。 依序診斷：**輸入管道停頓**（GPU 匱乏）、**小批量/ 低佔用率**、**通訊不重疊**（DP/TP/EP 暴露）、**記憶體限制 ops**（未融合的規範/激活），**重新計算**開銷。尋找個人資料 然後修復最大貢獻者 — MFU 是最好的 training-health 數量。

??? Success "4 — 死代碼消除隱藏了 kernel"
    如果基準測試計算程式從未讀取的結果，則編譯器可能會 **完全消除**kernel → 你「測量」約 0 次。構造它通過 丟棄輸出，觀察到荒謬的速度，然後透過消耗來修復 輸出\*\*（例如累積成你列印/傳回的值，或新增數據 依賴性）。始終使結果可觀察，這樣工作就無法優化 遠離－一個經典的微基準陷阱。
