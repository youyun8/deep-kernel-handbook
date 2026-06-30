# 案例研究：真實的 MoE 架構

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階→高階</span>
  <span class="chip"><strong>先備知識：</strong> MoE 篇前面所有章節</span>
  <span class="chip"><strong>硬體：</strong> 無</span>
</div>

把每個元件都做過一遍之後，現在可以來讀真正的前沿 MoE，看看它們做了*哪些*設計選擇、又是 _為什麼_。我們拆解四個：**Mixtral**（乾淨的經典）、**DeepSeek-V3**（系統協同設計的旗艦）、 **Qwen-MoE**（務實的量產者），以及 **Kimi K2/K2.5**（兆級規模下的極端稀疏）。

!!! Warning "確切數字請以一手來源為準"
    以下組態反映截至 2026 年初公布的技術報告，目的是說明*設計模式*，不是當規格表用。較新的 小改版（例如 K2.5 刷新）可能調整過某些數字。引用前請務必在對應論文／模型卡上確認。

## 跨模型比較

| 模型 | 總/活躍參數 | expert（路由 + 共享） | top-$k$ | gate | 平衡 | attention | 標誌性系統 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Mixtral 8×7B** | ~47B / ~13B | 8、無共享 | 2 | softmax | auxiliary loss | GQA | 乾淨的 SMoE、密集 |
| **DeepSeek-V3** | 671B / 37B | 256 + 1 共享 | 8 | sigmoid | aux-loss-free 偏差 | **MLA** | FP8 訓練、node-limited routing、DualPipe/DeepEP、MTP |
| **Qwen3-MoE (235B)** | ~235B / ~22B | 128 + 0 共享\* | 8 | softmax | auxiliary loss | GQA | 量產化、工具生態廣 |
| **Kimi K2** | ~1T / ~32B | 384 + 1 共享 | 8 | sigmoid | aux-loss-free 風格 | MLA | 極端稀疏、MuonClip optimizer |

<small>\*Qwen 變體隨版本不同；請查對應的模型卡。</small>

趨勢非常清楚：從少數幾個大 expert（Mixtral），走向**許多細粒度 expert + 共享 expert + sigmoid gating + aux-loss-free 平衡**（DeepSeek、Kimi），並用 attention 壓縮（[MLA](../foundations/attention-efficiency.md)）對抗 KV cache 成本。

## Mixtral 8×7B — 乾淨的經典

這個模型把開放權重的 SMoE 帶進主流。它幾乎就是 [從零實作](moe-from-scratch.md) 的逐字翻版：

- **每層 8 個 expert、top-2**，softmax gating，把選中的兩個 gate 重新歸一化。沒有共享 expert。
- **GQA** attention，壓住 KV cache。
- 用標準的 **auxiliary 負載平衡損失**訓練。

為什麼它在教學上重要：它是能在規模上跑起來的最小 SMoE，因此把核心觀念（稀疏 FFN routing） 孤立出來，不疊加後來的複雜性。約 47B 參數、但每個 token 只用約 13B —— [為什麼需要稀疏化](why-sparsity.md)最單純形式的解耦。它的侷限（只有 8 個粗 expert → 只有 28 種 expert 組合）正是細粒度設計要改進的地方。

## DeepSeek-V3 — 端對端系統協同設計

本篇*每一項*技術協同運作的旗艦範例。它值得細讀，因為建模和系統是一起設計的。

**架構**

- **DeepSeekMoE**：256 個細粒度路由 expert + **1 個共享 expert**、top-8 routing —— 數十億種 expert 組合，對比 Mixtral 的 28 種（[Routing 變體](routing-variants.md)）。
- **Sigmoid gating** 搭配 **aux-loss-free** 平衡：per-expert 的 [偏差控制器](load-balancing.md)調整選擇，而不會像大 auxiliary loss 那樣扭曲目標 —— 平衡*與* 品質都更好。
- **Multi-head Latent Attention（MLA）**：把 K/V 壓成低秩 latent，大幅縮小 [KV cache](../foundations/attention-efficiency.md) —— 對長上下文與便宜的 decode 至關重要。
- **Multi-Token Prediction（MTP）**：額外的頭一次預測多個未來 token，提升資料效率，並能做類似 speculative 的 inference。

**系統**（這裡最相關的部分）

- **FP8 training**：GEMM 走 FP8、累積走高精度，敏感部件留 BF16/FP32 —— 前沿等級的 [數值](../foundations/numerics-precision.md)配方。
- **Node-limited routing**：一個 token 的 expert 最多跨 ≤4 個節點，壓低跨節點的 [all-to-all](systems-ep.md) 流量。- **DualPipe + DeepEP**：管線排程加通訊函式庫，把 **all-to-all 與計算幾乎完全重疊** —— 量產化的 最大單一 EP 優化。

總共 671B／**37B 活躍**：你付的是約 37B 模型的 inference 計算，換到的卻是更大模型的品質 —— *因為*系統工作把 EP 開銷壓得夠小。

## Qwen-MoE — 務實的生產者

Qwen MoE 系列（Qwen1.5-MoE-A2.7B → Qwen2-57B-A14B → Qwen3-235B-A22B）展示了當**廣泛部署與 工具生態**很重要時，團隊會做的設計選擇：

- **細粒度 expert**（例如 128 個路由、top-8） —— 採納「許多小 expert」的路線。
- **GQA** attention（到處都有良好支援），而非更奇特的 MLA。
- 標準的 **auxiliary loss** 平衡 —— 穩健、好推理，把可靠性看得比榨出最後一點平衡更重。
- 強大的**生態系支援**（量化變體、serving 整合），這本身就是一個系統決策：架構再好，也只能 和跑它的 kernel 與 server 一樣好用。

重點：存在一條光譜，一端是「奇特的研究前沿」（DeepSeek/Kimi），另一端是「久經考驗、可移植」（Qwen）；該選哪一端，取決於你是否掌控整個堆疊。

## Kimi K2 / K2.5 — 兆級的極度稀疏

Moonshot 的 Kimi K2 把稀疏性推到極致：總參數約 **~1T、僅約 32B 活躍**，靠一個非常大的細粒度 expert 池（約 384 個路由 + 共享 expert、top-8）與 MLA attention。有兩點特別突出：

- **極高的稀疏率**（活躍/總數 ≈ 3%） —— 比 DeepSeek-V3 更進一步沿著 [為什麼需要稀疏化](why-sparsity.md)的軸走，重押「便宜的記憶體容量能買到品質」。
- **這種規模的 training 穩定性工程**，包括 **MuonClip** optimizer／裁剪法，據報能壓制困擾兆級 參數 MoE training 的損失尖峰與 attention-logit 膨脹 —— 直接對應我們在 [穩定性病態](training-stability.md)講的那些問題。

K2.5 是同系列的後續改進；把具體數字當成版本相關，請依 Moonshot 當前的模型卡確認。但*架構上 的教訓*是穩定的：極端稀疏是可行的 —— 當（且唯當）平衡、穩定性與 serving 記憶體問題一起被解決。

!!! Tip "逐 kernel 看它怎麼跑"
    [MoE decode 剖析](decode-anatomy.md) 描述的正是這一類模型（MLA + 細粒度 MoE + 共享 expert）的 decode 步驟，顯示時間實際花在哪 —— Routing、grouped expert GEMM、共享 expert、每層 all-reduce —— 以及會移動這些比例的融合與並發槓桿。

## 拿走什麼

把這些並排讀完後，現代 MoE 的共識是：

1. **許多細粒度 expert + 共享 expert** 勝過少數粗 expert。
2. **sigmoid gating + aux-loss-free 偏差**比重 auxiliary loss 平衡得更好。
3. **壓縮 attention**（MLA/GQA），免得 KV cache 把省下來的空間又吃回去。
4. **系統工作（all-to-all 重疊、node-limited routing、FP8）不是可選項** —— 正是它讓 FLOP 解耦在 碰到真實硬體時還能成立。
5. **穩定性工程隨規模放大** —— Z-loss、FP32 routing、謹慎的 optimizer（MuonClip）會隨 $E$ 與總參數 成長而越來越重要。

## 要點

- Mixtral = 最小的乾淨 SMoE（8 expert、top-2、softmax、aux loss）。
- DeepSeek-V3 = 協同設計的旗艦（細粒度 + 共享、sigmoid + aux-loss-free、MLA、FP8、node-limited routing、重疊 all-to-all）。
- Qwen-MoE = 務實、可移植、生態系優先。
- Kimi K2/K2.5 = ~1T 參數下的極端稀疏，配上重度穩定性工程。
- 整個領域已收斂到：細粒度 + 共享 expert、sigmoid + 偏差平衡、壓縮 attention，以及嚴謹的 通訊／重疊系統工作。

## 練習

!!! Tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 為每個模型計算 expert 組合數 $\binom{E}{k}$，並把它連到細粒度的品質論證。
2. 估計每個模型每 1k token 的 KV cache 大小；量化 MLA 相對 GQA、再相對普通 MHA 各省多少。
3. DeepSeek-V3 vs Mixtral：比較活躍/總比，討論它如何改變 serving 記憶體與 latency。
4. 挑一個模型，把它的每個元件對映回MoE 篇各章（gate、平衡、routing 變體、EP 策略、attention、 精度）。

## 參考文獻

[1] A. Q. Jiang *et al.*, "Mixtral of experts," *arXiv:2401.04088*, 2024.

[2] DeepSeek-AI, "DeepSeek-V3 technical report," *arXiv:2412.19437*, 2024.

[3] D. Dai *et al.*, "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," *arXiv:2401.06066*, 2024.

[4] Qwen Team, "Qwen2 technical report," *arXiv:2407.10671*, 2024.

[5] Qwen Team, "Qwen3 technical report," Technical Report, 2025.

[6] Moonshot AI, "Kimi K2 technical report," Technical Report, 2025.
