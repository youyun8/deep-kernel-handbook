# 案例研究：真實的 MoE 架構

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階→高階</span>
  <span class="chip"><strong>先備知識：</strong> 所有先前第二部分頁</span>
  <span class="chip"><strong>硬體：</strong> 無</span>
</div>

建造完每個組件後，我們現在可以閱讀真正的前沿 MoE 並查看*哪個*
他們所做的設計選擇以及*原因*。我們剖析四個：**Mixtral**（乾淨的
經典）、**DeepSeek-V3**（系統共同設計的旗艦產品）、**Qwen-MoE**（
務實的生產者）和**Kimi K2/K2.5**（萬億級的極度稀疏
規模）。

!!! warning "根據主要來源驗證確切的數字"
    以下配置反映了截至 2026 年初發布的技術報告
    旨在說明*設計模式*，而不是作為規格表。較新
    點版本（例如 K2.5 刷新）可能會調整計數。始終確認準確
    在引用連結的論文/模型卡中的數字之前。

## 跨模型比較

| 型號                 | 總/活動參數    | experts（路由+共享） | 頂部-$k$ | 門       | 平衡               | attention    | 著名系統                                         |
| -------------------- | -------------- | -------------------- | -------- | -------- | ------------------ | ------------ | ------------------------------------------------ |
| **混合 8×7B**        | 〜47B / 〜13B  | 8、無共享            | 2        | 軟最大   | 輔助損失           | 品質保證     | 乾淨的 SMoE，密集                                |
| **DeepSeek-V3**      | 671B/37B       | 256 + 1 共享         | 8        | 乙狀結腸 | aux-無損耗偏置     | **司法協助** | fp8 列車，節點限制 routing，DualPipe/DeepEP，MTP |
| **Qwen3-MoE (235B)** | 〜235B / 〜22B | 128 + 0——共享\*      | 8        | 軟最大   | 輔助損失           | 品質保證     | 生產化、廣泛的工具                               |
| **基米 K2**          | 〜1T / 〜32B   | 384 + 1 共享         | 8        | 乙狀結腸 | aux-loss-free 風格 | 司法協助     | 極度稀疏，MuonClip 優化器                        |

<small>\*Qwen 變體因版本而異；查看特定型號卡</small>

趨勢是明確無誤的：從幾個大的 experts (Mixtral) 到**許多
細粒度 experts + 共享 expert + sigmoid 門控 + aux-loss-free
平衡**（DeepSeek、Kimi），透過 attention 壓縮
[MLA](../foundations/attention-efficiency.md) 對抗 KV 快取成本。

## Mixtral 8×7B — 乾淨的經典

該模型使開放式 SMoE 成為主流。這是
[from-scratch design](moe-from-scratch.md) 幾乎逐字逐句：

-**每層 8 個 experts，top-2**，softmax 閘控，將兩者重新歸一化
選定的門。無共用 expert。 -**GQA**attention 綁定 KV 快取。

- 使用標準**輔助負載平衡損耗**進行訓練。

為什麼它在教學上很重要：它是大規模運作的最小 SMoE，所以它
隔離核心思想（稀疏 FFN routing），而不會增加後續的複雜性。
~47B 參數，但每個 token 使用 ~13B — [why-sparsity](why-sparsity.md)
以最簡單的形式解耦。它的限制－只有 8 個粗 experts→ 只有 28 個
expert 組合 — 正是細粒度設計的改進之處。

## DeepSeek-V3 — 端對端系統協同設計

這部分中*每種*技術協同工作的旗艦範例。它是
值得研究，因為建模和系統是共同設計的。

**架構**

-**DeepSeekMoE**：256 個細粒度路由 experts +**1 個共用 expert**，top-8
routing — 數十億個 expert 組合與 Mixtral 的 28 個組合
（[routing variants](routing-variants.md)）。 -**Sigmoid 門控**具有**aux-loss-free**平衡：每 expert
[bias controller](load-balancing.md) 調整選擇而不會造成嚴重的輔助損失
扭曲目標－更好地平衡*和*品質。 -**多頭潛在 attention (MLA)**：將 K/V 壓縮為低階潛在，
大幅縮減 [KV cache](../foundations/attention-efficiency.md) —
對於長上下文和廉價的 decode 至關重要。 -**多 token 預測 (MTP)**：額外的頭預測多個未來的 tokens，
提高資料效率並啟用類似 decoding 的推測性 inference。

**系統**（此處最相關的部分）

-**fp8 training**的 GEMM 具有高精度累加和 bf16/fp32
敏感零件 — [numerics](../foundations/numerics-precision.md) 配方位於
邊境。 -**節點限制的 routing**：token 的 experts 跨度 ≤4 個節點，邊界跨節點
[all-to-all](systems-ep.md) 流量。 -**DualPipe + DeepEP**：管道調度和通訊庫構建
**all-to-all 與計算幾乎完全重疊 – 最大的單一
EP 優化，量產化。

總計 671B /**37B 活動**：你支付 ~37B 型號 inference 計算以獲得更大的-
模型質量，*因為*系統工作使 EP 開銷較小。

## Qwen-MoE — 務實的生產者

Qwen MoE 系列（Qwen1.5-MoE-A2.7B → Qwen2-57B-A14B → Qwen3-235B-A22B）顯示
當**廣泛部署和工具**很重要時，團隊所做的設計選擇：

-**細粒度 experts**（例如 128 個路由，top-8） — 採用
許多小 experts 課程。 -**GQA**attention（到處都得到良好支援）而不是更奇特的 MLA。

- 標準**輔助損耗**平衡 - 穩健且易於推理、優先排序
  可靠性勝過擠壓最後一點平衡。
- 強大的**生態系統支援**（量化變體、serving 整合），
  本身就是一個系統決策：架構僅與 kernels 一樣有用
  以及運行它的伺服器。

重點：有一個來自「異國研究前沿」的光譜（DeepSeek/Kimi）
到「久經考驗且便攜」（Qwen），正確的觀點取決於你是否
控制整個堆疊。

## Kimi K2 / K2.5 — 兆級的極度稀疏

Moonshot 的 Kimi K2 大力推動稀疏性：總參數約**~1T
僅約 32B 活動**，通過非常大的細粒度 expert 池（約 384 個路由 +
共享 expert，前 8 名）和 MLA attention。有兩點很突出：

-**非常高的稀疏率**（活躍/總數 ≈ 3%） - 進一步沿著
[why-sparsity](why-sparsity.md)成交量甚至比 DeepSeek-V3 還要大，下注力道大
廉價的內存容量可以買到高品質的內存。 -**training-這種規模的穩定性工程**，包括**MuonClip**
據報道，優化器/裁剪方法可以抑制損失峰值和 attention-logit
困擾萬億參數 MoE training 的成長—對
我們涵蓋了 [stability pathologies](training-stability.md)。

K2.5 是同一系列的後續改進；將特定計數視為
版本相關，並根據 Moonshot 的當前型號卡進行確認。的
*建築課程*是穩定的：極端稀疏是可行的，當（且僅當）
平衡性、穩定性和 serving-記憶體問題一起解決。

!!! tip "查看它運行，kernel by kernel"
    [Anatomy of an MoE decode](decode-anatomy.md) 頁面描述了 decode 步驟
    正是此類中的模型（MLA + 細粒度 MoE + 共享 expert）以及
    顯示時間實際去向 — routing、分組 expert GEMM、共享
    expert 和每層 all-reduce — 加上融合和並發槓桿
    那個移動它。

## 拿走什麼

並排閱讀這些內容後，現代 MoE 的共識是：

1. **許多細粒度的 experts + 共享的 expert**擊敗了一些粗粒度的 experts。
2. **S 形門控+無輔助損耗偏壓**比嚴重輔助損耗更好地平衡。
3. **壓縮 attention**(MLA/GQA)，這樣 KV 快取就不會佔用節省的空間。
4. **系統工作（all-to-all 重疊，節點限制 routing，fp8）不
   可選**——這就是使 FLOP 解耦能夠在與真實接觸的情況下倖存下來的原因
   硬體。
5. **穩定性工程尺度與尺寸**— z 損失，fp32 routing，小心
   隨著 $E$ 和總參數的成長，優化器 (MuonClip) 變得更加重要。

## 要點

- Mixtral = 最小的乾淨 SMoE（8 experts、top-2、softmax、aux 損失）。
- DeepSeek-V3 = 共同設計的旗艦產品（細粒度+共享、sigmoid+
  aux-loss-free、MLA、fp8、節點限制 routing、重疊 all-to-all）。
- Qwen-MoE = 務實、便攜、生態系統優先。
- Kimi K2/K2.5 = ~1T 參數下的極端稀疏性，具有高穩定性工程。
- 該領域已收斂於細粒度+共享 experts、sigmoid+bias
  平衡、壓縮 attention 和嚴格的通訊/重疊系統工作。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 對於每個型號，計算 expert 組合 $\binom{E}{k}$ 和
   將其與細粒度的品質論證聯繫起來。
2. 估算每個模型每 1k tokens 的 KV 快取大小；量化 MLA 節省了多少
   超過 GQA 超過普通 MHA。
3. DeepSeek-V3 與 Mixtral：比較活躍/總比率並討論其如何變化
   serving 記憶體與 latency 的比較。
4. 選擇一個模型並將每個元件映射回第二部分頁面（門、平衡、
   routing 變種，EP 策略，attention，精度）。

## 參考文獻

- 江等人。 _experts 的混合。 _ 2024 年。
- DeepSeek-AI。 _DeepSeek-V3 技術報告。 _ 2024（MLA、aux-loss-free、fp8、DualPipe、MTP）。
- 戴等。 _DeepSeekMoE。 _ 2024 年。
- Qwen 團隊。 _Qwen2 / Qwen3 技術報告。 _ 2024–2025。
- 登月人工智慧。 _Kimi K2 技術報告_（MuonClip，大型 MoE）。 2025 年。
