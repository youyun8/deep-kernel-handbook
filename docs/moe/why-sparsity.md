# 為什麼需要稀疏化

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../../foundations/transformer-systems/">作為系統的 Transformer</a></span>
  <span class="chip"><strong>硬體：</strong> 無</span>
</div>

在動手做 MoE 之前，先精確搞清楚稀疏化*解決了什麼問題*、又*付出什麼代價*。一句話：MoE
把**參數量和每個 token 的 FLOP 數解耦**。這一頁把這句話量化、把取捨講清楚，好讓後續
其餘章節有個明確的目標。

## 密集瓶頸

在密集 Transformer 裡，FFN 主導了參數量與 FLOP。由[基礎篇](../foundations/transformer-systems.md)：
前向成本約 $\approx 2P$ FLOP/token，其中 $P$ 是參數量——*每個參數對每個 token 都要動一次*。
想讓模型「懂更多」，你得增加 $P$，而計算開銷會跟著等比例成長。縮放定律告訴你損失隨參數與
資料下降，但你實際要付的帳是 $\approx 6 P D$（參數 × token）的計算量。

於是 MoE 問了一個問題：**我們能不能加參數，卻不等比例地加 FLOP？**

## 條件計算

可以——只要每個 token 只用到參數的一個*子集*。把 1 個 FFN 換成 $E$ 個 expert FFN 加一個
router，每個 token 只啟動其中 $k$ 個（$k \ll E$，通常 $k=1$ 或 $2$）。於是：

- **總參數**隨 $E$ 成長（所有 expert 都存在，用來儲存知識）。
- **每個 token 的有效參數**隨 $k$ 成長（只有 $k$ 個 expert 真的跑）。
- **每個 token 的 FLOP** 追的是*活躍*參數，而不是總參數。

定義**稀疏率** $k/E$。一個 $E=64$ 個 expert、$k=2$ 的模型，FFN 參數量約是其等效活躍計算
密集模型的 32 倍。真實例子：Mixtral 8×7B 總共 47B、活躍約 13B；DeepSeek-V3 總共 671B、
但每個 token 只有 **37B 活躍**。你付的是約 37B 模型的 FLOP，拿到的品質卻更接近 671B 模型。

$$ \underbrace{P_{\text{total}}}_{\text{capacity / memory}} \;\propto\; E, \qquad \underbrace{P_{\text{active}}}_{\text{FLOPs, speed}} \;\propto\; k. $$

## 縮放參數

經驗上（Switch Transformer、GShard 及其後續），在**固定 training FLOP 預算**下，稀疏模型
比密集模型更快達到給定損失；在**固定活躍參數預算**下，增加 expert 數能持續提升品質，而
只付出次線性的額外計算。*為什麼*的直覺：

- **專業化。** 不同 expert 可以各自分工（粗略地說——依 token 類型、主題或語法角色），整體
  容量超過同樣活躍大小的單一 FFN。
- **更多參數 = 記住更多知識**，而不需要更多 token 上的數學；router 扮演一個學出來的稀疏查表。
- **容量很便宜。** 參數的儲存成本低（HBM／offload）；*執行* FLOP 才貴。MoE 是用便宜的貨幣
  去買容量。

!!! note "這不是免費的品質"
    稀疏模型的「每參數效率」低於密集模型——一個 671B 的稀疏模型，比不上一個假想的 671B 密集
    模型。它的勝利在於**每 FLOP 的品質**與**每元 inference 成本的品質**，而不是每參數的品質。
    你是在用充裕的記憶體，去換稀缺的計算。

## 稀疏性的代價是什麼（本章其餘部分）

條件計算不是免費的午餐；它帶進一整批密集模型永遠不會碰到的系統問題：

| 代價                                                | 痛在哪裡                | 對應章節                                               |
| --------------------------------------------------- | ----------------------- | ------------------------------------------------------ |
| **負載不平衡**——router 崩塌到少數熱門 expert        | 浪費 expert、出現掉隊者 | [負載平衡](load-balancing.md)                          |
| **離散 routing**——top-k 不可微、不穩定              | training 發散           | [訓練穩定性](training-stability.md)                    |
| **all-to-all 通訊**——token 必須送到 expert 所在 GPU | 受網路綁定的一層        | [系統與 EP](systems-ep.md)                             |
| **記憶體足跡**——所有 expert 都得存／載入            | 龐大的 HBM／offload     | [推論與 serving](inference-serving.md)                 |
| **不規則計算**——可變 tokens-per-expert 破壞密集 GEMM | kernel 效率低下        | [kernels](kernels.md)                                  |
| **capacity 與 padding**——固定緩衝區浪費或丟 token   | 品質／throughput 取捨   | [負載平衡](load-balancing.md)、[系統](systems-ep.md)   |

MoE 工程的重點，就在於把這些代價付得夠便宜，讓「FLOP 解耦」帶來的優勢能撐得住。MoE 後續章節
講的全是這件事。

## 粗略比較

在相同「活躍」計算下，比較密集 FFN 與 MoE FFN（隱藏維度 $d$、$d_{ff}=4d$、$E$ 個 expert、
top-$k$）：

- 密集 FFN：參數 $\approx 8 d^2$（up + down）；FLOP/token $\approx 16 d^2$。
- MoE：參數 $\approx 8 d^2 E$；FLOP/token $\approx 16 d^2 k$（外加一個極小的 router $d\times E$）。
  在 $k=1$、$E$× 參數時，FLOP 與密集相同。

所以在 $k=1$ 時，你用*相同*的 FLOP 換到 $E$ 倍的 FFN 容量，外加可忽略的 router 成本——再扣掉
上面那些系統開銷。整個工程問題，就是你能把這些開銷壓到多低。

## 要點

- MoE **把總參數（容量）和活躍參數（FLOP）解耦**：容量隨 $E$ 擴展，計算隨 $k$ 擴展。
- 它的優勢是**每 FLOP 的品質／每元 inference 成本的品質**，靠的是買便宜的記憶體、而非昂貴的
  算力——這*不是*更好的每參數品質。
- 稀疏化引入負載平衡、通訊、記憶體與 kernel 不規則性等代價——這就是 MoE 後續章節的主題。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 對 $E=128$、$k=2$、$d=4096$，計算總 FFN 參數、有效 FFN 參數，以及每 token FLOP 相對於
   密集 $E=1$ 基線的比值。
2. DeepSeek-V3：總共 671B、活躍 37B。有效稀疏率是多少？和 Mixtral 8×7B 相比如何？
3. 為兩邊各辯護一次：什麼時候你會選密集的 37B 模型、而不是 671B/37B-active 的稀疏模型？
   把記憶體、batch-1 latency 與微調都考慮進去。
4. 如果把 expert offload 到 CPU/NVMe、按需串流進來，roofline 的哪一軸（算力或頻寬）會變成
   新的限制因素？（這是 [推論與 serving](inference-serving.md) 的伏筆。）

## 參考文獻

- Shazeer et al. _Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer._ 2017。
- Lepikhin et al. _GShard._ 2020。
- Fedus, Zoph, Shazeer. _Switch Transformer._ 2021。
- Clark et al. _Unified Scaling Laws for Routed Language Models._ 2022。
- DeepSeek-AI. _DeepSeek-V3 Technical Report._ 2024。
