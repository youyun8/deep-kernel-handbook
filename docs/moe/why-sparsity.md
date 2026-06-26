# 為什麼稀疏

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../../foundations/transformer-systems/">Transformer作為系統</a></span>
  <span class="chip"><strong>硬體：</strong> 無</span>
</div>

在建構 MoE 之前，有必要準確了解稀疏性問題是什麼
解決*和*成本\*。標題：MoE**將參數計數與
每個 token**的失敗次數。此頁面使該聲明定量且誠實
權衡，所以第二部分的其餘部分有一個明確的目標。

## 密集瓶頸

在密集 Transformer 中，FFN 主導參數和 FLOP。來自
[Part I](../foundations/transformer-systems.md)：遠期成本為$\approx 2P$
每個 token 的 FLOPs，其中 $P$ 是參數計數 — _每個參數都涉及
每個 token_。為了使模型“了解更多”，你需要增加 $P$ 和你的計算費用
步調一致地成長。縮放定律表示損失隨著參數和
數據，但計算 $\approx 6 P D$ (params × tokens) 是你實際的預算
付錢。

MoE 提出的問題：**我們可以在不加入比例的情況下加入參數嗎
失敗？**

## 條件計算

是 - 如果每個 token 僅使用參數的*子集*。將 1 個 FFN 替換為
$E$ expert FFN 和一個 router，每個 token 啟動其中的 $k$（$k \ll E$，
通常為 $k=1$ 或 $2$）。然後：

-**總參數**與$E$縮放（所有 experts 都存在，儲存知識）。 -**每個 token**的有效參數與 $k$ 一起縮放（僅 $k$ experts 運行）。 -**每個 token**追蹤 _活動_ 參數的 FLOP，而不是總數。

定義**稀疏率**$k/E$。具有$E=64$、experts、$k=2$的型號有
$\sim$ 比其主動計算等效密集模型多 32 倍 FFN 參數。真實
例：Mixtral 8×7B 總共有 47B，但活躍的約有 13B； DeepSeek-V3 總共有 671B
但每個 token 僅有**37B 活動**。你支付了大約 37B 模型的 FLOPs，並且更接近
671B 型的品質。

$$ \underbrace{P*{\text{total}}}*{\text{capacity / memory}} \;\propto\; E, \qquad \underbrace{P*{\text{active}}}*{\text{FLOPs, speed}} \;\propto\; k. $$

## 縮放參數

根據經驗（開關 Transformer、GShard 和後續產品），**固定 training
FLOP 預算**，稀疏模型比密集模型更快達到給定的損失，並且在
**固定活動參數預算**，增加 experts 可以持續提高質量
次線性額外計算。 *為什麼*的直覺：

-**專業化。**不同的 experts 可以專業化（寬鬆地 - 透過 token 類型，
主題，或句法角色），因此有效功能類比
相同活動大小的單一 FFN。 -**更多參數 = 更多記憶知識**，無需更多 token 數學；的
router 充當學習稀疏查找。 -**容量便宜。**參數的儲存成本低廉（HBM/卸載）；
_運行_ FLOP 的成本很高。MoE 以廉價貨幣購買產能。

!!! note "這不是免費品質"
    稀疏模型的「參數效率」低於密集模型—671B 稀疏模型
    模型不如假設的 671B 密集模型。勝利是
    **每 FLOP 的質量**和**inference 的每美元質量**，而不是每美元的質量
    參數。你正在用充足的記憶體換取稀缺的計算。

## 稀疏性的代價是什麼（第二部分的其餘部分）

條件計算不是免費的午餐；它導入了一堆系統
密集模型永遠不會遇到的問題：

| 成本                                                     | 被咬的地方             | 涵蓋於                                                  |
| -------------------------------------------------------- | ---------------------- | ------------------------------------------------------- |
| **負載不平衡**— router 崩潰為一些流行的 experts          | 浪費了 experts，掉隊者 | [負載平衡](load-balancing.md)                           |
| **離散 routing**— top-k 不可微，不穩定                   | training 背離          | [training stability](training-stability.md)             |
| **all-to-all 通訊**— tokens 必須前往其 expert 的 GPU     | 網路綁定層             | [systems & EP](systems-ep.md)                           |
| **記憶體佔用**— 所有 experts 都必須儲存/載入             | 巨大的 HBM/卸載        | [inference & serving](inference-serving.md)             |
| **不規則計算**— 變數 tokens-per-expert 破壞了密集的 GEMM | kernel 效率低          | [kernels](kernels.md)                                   |
| **容量與填充**— 修復緩衝區浪費或下降 tokens              | 品質/throughput 貿易   | [負載平衡](load-balancing.md)、[systems](systems-ep.md) |

MoE 的藝術在於足夠有效地支付這些成本，以便
失敗——解耦的勝利得以延續。這部分的其他內容都是關於這個的。

## 粗略比較

在相同的「活動」計算下比較密集 FFN 與 MoE FFN，隱藏 $d$，
$d_{ff}=4d$、$E$ experts、上-$k$：

- 密集 FFN 參數：$\approx 8 d^2$（上+下）。失敗次數/token：$\approx 16 d^2$。
- MoE：參數 $\approx 8 d^2 E$； FLOPs/token $\approx 16 d^2 k$（加上一個微小的
  router $d\times E$）。當 $k=1$、$E$× 參數時，FLOPs 與密集相同。

因此，在 $k=1$，你將獲得 $E$× _相同_ FLOP 的 FFN 容量，加上可忽略的
router 成本 — 減去上述系統開銷。整個工程問題
就是你可以將這些管理費用減少到多少。

## 要點

- MoE**將總參數（容量）與活動參數 (FLOP) 解耦**。
  容量可隨 $E$ 擴充；使用 $k$ 計算秤。
- 獲勝是**每 FLOP 的品質/每 inference 美元**，透過購買實現
  便宜記憶體而不是昂貴的計算能力。這*不是*更好
  每個參數的品質。
- 稀疏性導入負載平衡、通訊、記憶體和 kernel-不規則性
  成本——第二部分其餘部分的主題。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 對於 $E=128$、$k=2$、$d=4096$，計算總 FFN 參數與有效 FFN 參數及
   每 token 的 FLOPs 與密集 $E=1$ 基線的比率。
2. DeepSeek-V3：總共 671B，活躍 37B。那有效稀疏比是多少？
   它與 Mixtral 8×7B 相比如何？
3. 爭論雙方：什麼時候你喜歡密集的 37B 型號而不是 671B/37B-active
   稀疏的一個？考慮記憶體、第 1 批的 latency 以及微調。
4. 如果 experts 被卸載到 CPU/NVMe 並流入，roofline 軸
   （計算或頻寬）成為新的限制因素？ （預示
   [inference & serving](inference-serving.md)。 ）

## 參考文獻

- 沙澤爾等人。 _極為龐大的神經網路：experts 層的稀疏門控混合。 _ 2017。
- 萊皮欣等人。 _GShard。 _ 2020。
- 費杜斯、佐夫、沙吉爾。 _開關 Transformer。 _ 2021 年。
- 克拉克等人。 _路由語言模型的統一縮放法則。 _ 2022。
- DeepSeek-AI。 _DeepSeek-V3 技術報告。 _ 2024 年。
