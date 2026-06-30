# 解答 — 效能工程

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 全部七個效能頁面</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

解答[效能工程篇](../performance/index.md)的練習。kernel 練習大多是開放式的（「跑起來，幫它 benchmark」）；我們給出預期結果與推理過程，方便你核對自己算出來的數字。

## GPU 程式設計模型

??? Success "1 — 為什麼 32-lane reduction 在 CDNA 上是錯的"
    用 `offset = 16,8,4,2,1` 硬編碼的 warp shuffle reduction，等於假設 32-lane 的 warp（NVIDIA）。AMD CDNA 的 wavefront 是 **64 個 lane**，所以 32-lane 的 shuffle 只 reduce 了 wavefront 的「一半」——上半的 32 個 lane 被忽略，得到錯誤的（部分）總和。修法：把 shuffle 迴圈從 `warpSize/2` 開始，並全部用 `warpSize` 取代寫死的 32，這樣同一份程式碼在 32 或 64 個 lane 上都能正確 reduce。

??? Success "2 — occupancy limiter（每個 thread 64 個 register，每個 block 48 KB SMEM）"
    每個 SM：64K 個 register，100 KB SMEM。取一個 256-thread 的 block。

    - **register：**$64\times256 = 16384$ register/block → $65536/16384 = 4$ 個 block。
    - **SMEM：**$100/48 = 2.08$ → **2 個 block**。

    SMEM 是更緊的限制 → 2 個常駐 block（512 個 thread）。**shared memory 是 occupancy limiter**；把每個 block 用的 SMEM 切小（或調整 block size）能提高 occupancy。

??? Success "3 — coalescing 適用於 row-major，轉置後就不行"
    對 row-major 的 $[M,N]$ tensor，warp 裡的 thread 用 **column** 索引讀取 `A[row, col0+lane]`——位址連續 → 合併成一筆 transaction。讀**轉置**版本（thread 沿著 row 走）會得到 stride-$N$ 的位址 → 變成 $N$ 筆獨立 transaction，記憶體流量大約多 32 倍。這正是 MoE 的 **gather** 在發生的事：散落的 token index 破壞了 coalescing，這也是為什麼 gather kernel 是 memory-bound、值得拿去融合。

??? Success "4 — 為什麼降低 occupancy 反而能提升 throughput"
    用大量 register 的 matmul block 會把更多工作集留在**register**裡（最快的記憶體），減少 SMEM/HBM 流量和輸出的指令數。這會提高 register pressure → 常駐 warp 變少 → occupancy 變低，但**throughput 反而更高**，因為每個 warp 做的都是真正有用的工作，而且這個 kernel 是 compute-bound、不是 latency-bound。最大化 occupancy 只在你需要大量 warp 來藏住記憶體 latency 時才有幫助；tile 排得好的 GEMM 不需要。

## Triton 路線

??? Success "1 — 比對 vector add 與 softmax 對 PyTorch"
    兩個 Triton kernel 的輸出都應該在 BF16/FP32 誤差範圍內對上 `torch` 的結果。vector add 是 memory-bound → throughput 應該接近 HBM 頻寬，和 native 操作差不多。fused softmax 應該**贏過**單純的三段式 torch softmax（一次讀 + 一次寫，相對三次），大致打平 `torch.softmax`（本身也已經是 fused 的）。

??? Success "2 — 為 AMD 寫的 autotune config"
    加入不同 `num_warps`（4/8）和 `BLOCK` 的組合，要注意**wavefront 是 64**這件事——在 CDNA 上 `num_warps=4` 已經等於每個 block 256 個 lane，所以最佳 block size 跟 NVIDIA 的 32-lane warp 不一樣。最佳設定是因 GPU 而異的；教訓是：在一個廠商上 autotune 出來的設定，幾乎不會是另一個廠商上的最佳解——換目標一定要重新 autotune。

??? Success "3 — softmax：當一行寬度超過 1 個 `BLOCK`"
    把每一行切成 `BLOCK` 大小的 tile 逐個迭代，用**online softmax 的合併運算**（和 FlashAttention 練習 1 一樣）維護 running max $m$ 與分母 $\ell$：每個 tile 算 $m' = \max(m, \max_{\text{tile}})$，把 $\ell$ 用 $e^{m-m'}$ 重新縮放，再加上這個 tile 的貢獻。第二輪（或快取下來的分子）再做 normalize。這樣就消除了「一行必須塞進一個 block」的限制。

??? Success "4 — fused vs 三段式 softmax 的位元組數"
    三段式版本要把每一行讀**三次**（max、exp-sum、normalize）再寫一次。fused 版本只讀一次、寫一次。對 $[R,C]$ 的 tensor，位元組比率約 $(3+1)/(1+1) = 2\times$——fused kernel 的流量少了一半，而因為 softmax 是 memory-bound，這就直接對應到 2 倍加速，你的 benchmark 應該能驗證這一點。

## CUDA / HIP 路線

??? Success "1 — 把 tiled matmul 移植到 HIP"
    用 `hipify` 自動轉換，或手動移植：`__shared__` 維持原樣，`cudaMalloc→hipMalloc`，`<<<>>>` 的 launch 語法在 `hipcc` 下完全相同。用 `hipcc` 編譯（或透過 ROCm 上的 PyTorch），再對 cuBLAS/hipBLAS 驗證浮點誤差範圍。重點：HIP 是原始碼相容的——同一份 kernel 在兩個廠商上都能跑，唯一真正會出錯的可移植性陷阱是 wavefront 寬度（見下一題）。

??? Success "2 — 32-lane 的 reduction 在 64 寬 wavefront 上會失敗"
    做一個只用 `offset = 16…1` shuffle 的 reduction。在 CDNA 上，64-lane 的 wavefront 代表 lane 32–63 完全沒被用到 → 結果只加總了下半部一半的值。示範方法：對一個長度 64、全部是 1 的向量做 reduction，你會得到 32 而不是 64。修法是用 `for (offset = warpSize/2; offset>0; offset>>=1)`。

??? Success "3 — `TILE` ∈ {8,16,32} 掃描"
    tile 越大 → 每筆從 global memory 載入的資料在 SMEM 裡被重複使用的次數越多（算術強度更高），但每個 block 用的 SMEM/register 也更多 → occupancy 較低。通常 `TILE=16` 或 `32` 會贏：8 太小（重複使用率差、memory-bound），32 在較小的 GPU 上可能會撞到 register/SMEM 上限、壓低 occupancy。把最佳值跟你那張卡的 warp/wavefront 大小與 SMEM 預算對起來看。

??? Success "4 — 用 Tensor Core/Matrix Core 取代內積迴圈（`wmma`/rocWMMA）"
    把標量的內積迴圈換成 Tensor Core/Matrix Core 的 MMA fragment，通常能拿到 4–10 倍的大幅加速，因為 Matrix Core 執行一次完整 tile-MMA 指令的 FLOP/s 遠高於標量 FMA 路徑。重點：fragment 需要特定的 tile 形狀/dtype（例如 16×16×16 BF16），SMEM 佈局也要小心對齊——記得對照標量版本量測正確性與加速比。

## 分散式 training { #distributed-training }

??? Success "1 — all-reduce = reduce-scatter + all-gather；ZeRO-2 的流量"
    **恆等式：**reduce-scatter 把總和算出來，每個 rank 留下一份結果分片；再 all-gather 把所有分片發給所有人 → 每個 rank 都拿到完整的 reduce 結果張量 = all-reduce。每一步的 ring 成本 ≈ $S(G{-}1)/G$ 位元組/rank，兩步合計 ≈ $2S(G{-}1)/G$ = all-reduce 的成本。**ZeRO-2** 把梯度分片，所以它對完整梯度做的不是 all-reduce，而是 **reduce-scatter**（每個 rank 只留自己的 grad 分片、更新自己的 optimizer 分片）加上**對所有參數的 all-gather**——總流量跟 DDP 的 all-reduce（$\approx 2S$）相當，但它從未具現化完整的梯度或 optimizer 狀態，用相同的通訊量換到記憶體節省。

??? Success "2 — 每 GPU 記憶體，70B BF16 + Adam，8 個 GPU"
    每個參數的混合精度 Adam 狀態：2 B（BF16 權重）+ 2 B（grad）+ 4 B（FP32 master）+ 4 + 4 B（FP32 的一階、二階動量）= **16 B/參數**。70B 模型總計 $70\text{B}\times16 = 1120$ GB。

    - **DDP：**每個 GPU 都存全部 16 B/param →**~1120 GB/GPU**（80 GB 顯卡上不可能——必須分片）。
    - **ZeRO-1**（只分片 optimizer 狀態，16 B 裡的 12 B）：每個 GPU 未分片的部分是權重+梯度 4 B/param（280 GB），分片的 12/8 = 1.5 B/param（105 GB）≈ **385 GB/GPU**。
    - **ZeRO-2**（也分片梯度）：權重 2 B（140 GB）+ (2+12)/8 = 1.75 B（122 GB）≈ **262 GB/GPU**。
    - **ZeRO-3**（全部分片）：16/8 = 2 B/param × 70B ≈ **140 GB/GPU**。

    重點是這個趨勢：ZeRO-3 把每 GPU 的狀態量砍到約 DDP 的 $1/G$，把原本塞不進去的模型變成裝得進去的模型（代價是更多 all-gather 流量）。

??? Success "3 — pipeline bubble 比例"
    對 $P$ 個 stage、$m$ 個 microbatch，bubble 比例為

    $$ \text{bubble} = \frac{P-1}{m + P - 1}. $$

    要讓它 $<10\%$：$\frac{P-1}{m+P-1} < 0.1 \Rightarrow m > 9(P-1)$。所以 $P=8$ 時你需要 $m > 63$ 個 microbatch；$P=4$ 時需要 $m>27$。stage 越多 ⇒ 需要越多 microbatch 才能把填充/排空的時間攤掉——這正是 pipeline parallelism 的核心張力。

??? Success "4 — 為什麼 TP 適合節點內、EP 適合跨節點"
    **TP** 在每一層**內部**做 all-reduce（forward 一次、backward 一次），每一步都要傳很大、對 latency 很敏感的流量 → 必須走最快的連結（節點內的 NVLink/Infinity Fabric）。**EP** 每個 MoE 層只有兩次 all-to-all，但每個 token 的 payload 較小，最重要的是它能**和計算重疊**、也可以做成**node-limited**；它能容忍較慢的跨節點頻寬。把話最多的 collective（TP）放到最快的連結上，把可以重疊的（EP）放到較慢的網路上。

## 量化與壓縮 { #quantization-compression }

??? Success "1 — int8 affine quantize/dequantize 與最大誤差"
    affine：$q = \text{round}(x/s) + z$，$\hat x = s(q - z)$，int8 的 $s = (\max-\min)/255$。每個元素最大誤差是**半個 step**，即 $s/2$。**per-tensor** 對整個 tensor 用同一個 $s$，所以只要有一個 outlier channel 撐大了 $\max$ → $s$ 變大 → 所有*小*的 channel 上誤差都跟著變大。**per-channel** 讓每個 channel 有自己的 $s$，outlier 那個 channel 的大 $s$ 不會污染其他 channel → 誤差低很多。這就是 per-channel（以及 AWQ）存在的理由。

??? Success "2 — AWQ 的 salient channel scaling"
    AWQ 在量化前先放大**salient**（activation 幅度大）的權重 channel，並把對應的 activation 縮小回去補償，這樣重要的 channel 實質上拿到更多有效位元。在一個 linear 層上實作：用 activation 統計找出 salient channel，套用 per-channel scale，量化成 int4，再 dequantize、量測 perplexity。預期在**相同位寬**下，perplexity 會明顯低於單純的 per-channel int4。

??? Success "3 — 13B 模型 decode latency 從 W4 得到的增益"
    decode 是**memory-bound** 的：latency ≈ 權重位元組數 / HBM 頻寬。BF16 權重 = $13\text{B}\times2 = 26$ GB；int4 ≈ $13\text{B}\times0.5 = 6.5$ GB。位元組數降到約 1/4，所以每個 token 的 decode latency 也降到**約 1/4**（還要扣掉沒被量化的 activation/KV 那部分）。這個收益純粹來自每個 token 要搬的權重位元組變少——這正是 [memory-bound](../foundations/attention-efficiency.md) 論證的量化版本。

??? Success "4 — 為什麼路由 expert 比 router/attention 更耐得住 int4"
    路由 expert 本身就是**冗餘且會被平均**的——每個 token 只看 $k$ 個 expert，許多 expert 的量化噪音會在加權和裡互相抵消。**router** 是根據**很小的 logit 差異做離散決策**（精度至關重要，見訓練穩定性篇），**attention** 則把資料寫進 KV cache，誤差會**在序列上累積放大**。所以可以放心對 expert 用激進的 int4（佔大多數參數、也最能容錯），而 router 與 attention 維持較高精度——這正是標準的 MoE serving 配方。

## Inference 優化

??? Success "1 — speculative decoding 的加速"
    給定 draft 的 acceptance rate $\alpha$ 與提案長度 $\gamma$，每次驗證步驟預期能接受的 token 數是

    $$ \mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}. $$

    加速比 ≈ 這個期望值除以每步成本比（一次 target 模型驗證 + $\gamma$ 次便宜的 draft）。高 $\alpha$ 配上適中的 $\gamma$ 回報最好；當 $\alpha\to1$ 時，每次驗證能接受接近 $\gamma+1$ 個 token。acceptance rate 主導的是有效加速，而不是 draft 本身的裸算力。

??? Success "2 — continuous batching vs 靜態 batching，長度在 [64,1024] 均勻"
    static batching 會把每個請求 pad 到**這個 batch 裡最長的那個**，並等最慢完成的那個跑完，所以短請求會浪費計算/slot；長度在 [64,1024] 均勻分布時，平均長度 ≈ 544，但整個 batch 會跑到 ≈ 1024 長度 → 浪費約 40–50%。**continuous batching** 每一步都把跑完的序列換掉、塞進新序列，讓 batch 一直保持滿載 → throughput 提升的量級大致等於這個 padding 浪費的比例（對這種長度分布大約是 **1.5–2×**，分布的變異數越大，提升越多）。

??? Success "3 — prefix caching 省下的 KV，100 個請求、2k 共用 prompt"
    不共享的話，每個請求都會各自存一份 2k-token system prompt 的 KV → 重複 $100\times$。用了 **prefix caching**，共用的前綴只存**一次**並重複使用，省下 $99\times$ 份的前綴 KV。若每個 token 每層 KV 大小為 $b$ bytes、共 $L$ 層，省下的量 $= 99\times2000\times L\times b$——通常是好幾 GB。只要工作負載有共用 prompt，這就是純粹的免費收益。

??? Success "4 — prefill/decode disaggregation：什麼時候有幫助，什麼時候有害"
    disaggregation 把 compute-bound 的 **prefill** 和 memory-bound 的 **decode** 拆到各自的 GPU 池，各自針對自己的瓶頸調校（prefill：大 batch、高 MFU；decode：高記憶體頻寬）。當兩個階段原本會互相搶資源時（例如長 prompt 的尖峰會讓 decode 餓著），disaggregation **有幫助**。但當**池之間搬 KV cache**（prefill→decode 切換）的成本超過它省下的爭用時——也就是 prompt 短/KV 小，或是互連速度慢——disaggregation 反而**有害**。判斷依據是：KV 位元組數、連結頻寬、原本能省下多少爭用。

## Profiling 與方法論

??? Success "1 — 先用錯誤方式 benchmark Triton softmax，再修正"
    **錯誤做法：**沒有 warmup、也沒呼叫 `torch.cuda.synchronize()` 就直接幫第一次呼叫計時——你量到的是 kernel 啟動 + JIT 編譯的 latency，加上 CPU 端非同步提前返回的時間，不是真正的 GPU 時間，通常會比真實值小 10–100 倍。**正確做法：**先跑幾次 warmup 迭代（觸發 autotune/編譯），再用 `synchronize()` 包住一段多次迭代的迴圈計時。修正後的數字才是真實的每次呼叫 GPU 時間；記得量化兩者的差距。

??? Success "2 — profile 一個 decode 步驟：誰主導？"
    profile 一個小型 Transformer 的一個 decode 步驟。**batch-1** decode 的情況下，預期是**啟動開銷與 memory-bound 的 attention/FFN** 主導——很多微小的 kernel，每次都要從 HBM 重新讀權重/KV，GPU 沒被餵飽。常見修法：**CUDA Graph**（消除啟動開銷）+ batching（提高算術強度）。如果是 attention 主導，KV layout/Flash-decoding 有幫助；如果是 FFN 主導，權重量化有幫助。

??? Success "3 — 算 MFU；診斷為什麼只有 15%"
    $\text{MFU} = \frac{6P \cdot \text{tokens/s}}{\text{GPU peak FLOP/s}}$（training 時每 token 約 $6P$ FLOP）。15% 代表你大概只用到了六分之一的數學單元算力。依序排查：**data pipeline 停頓**（GPU 餓著沒東西做）、**batch 太小/occupancy 太低**、**通訊沒有重疊**（DP/TP/EP 的通訊暴露在外）、**memory-bound 的 op**（沒融合的 norm/activation）、**重算（recompute）**的額外開銷。先 profile 再修最大的那一項——MFU 是最好的 training 健康度指標。

??? Success "4 — dead code elimination 把 kernel 藏起來了"
    如果你 benchmark 的程式算出的結果從來沒被讀取，編譯器可能會直接**把這個 kernel 整個消除**→ 你「量到」的時間約等於 0。先重現這個現象（丟掉輸出、觀察到離譜的速度），再透過消費這個輸出來修正（例如累加進你會印出來或回傳的值，或者加一個資料依賴）。永遠讓結果是可觀察的，這樣編譯器就無法把整個工作優化掉——這是個經典的 micro-benchmark 陷阱。
