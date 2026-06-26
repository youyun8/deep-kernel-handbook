# Capstone：擴大規模

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../build-moe/">建立小型 MoE LM</a>、<a href="../../performance/distributed-training/">分散式 training</a></span>
  <span class="chip"><strong>硬體：</strong>多GPU全面執行</span>
</div>

[previous capstone](build-moe.md) 建置並優化了單 GPU MoE LM。
這是一份用於多 GPU 的**規劃和實施指南**，
應用 [parallelism techniques](../performance/distributed-training.md) 和
[expert-parallel all-to-all](../moe/systems-ep.md)。它的結構為
你可以針對任何模型/叢集遵循決策流程。

!!! warning "部分實踐"
    <span class="status-badge wip">SCAFFOLDED</span> 完全執行此操作需要
    多 GPU（最好是多節點）叢集。 *推理、規劃和程式碼
    下面的結構*是完整的並且可以以骨架形式運行；測量的
    多節點編號留給你在硬體上填寫。這裡什麼都沒有
    隱藏在「TODO」後面——但是叢集規模的基準表是你的
    填充。

## 步驟 1 — 決定分片內容以及原因

遍歷記憶體預算。對於每個 GPU，估計 (bf16 + Adam)：參數 (2
位元組）、梯度 (2)、最佳化器狀態（fp32 矩 + master ≈ 12）和峰值
激活。然後根據首先溢出的尺寸來選擇尺寸：

```text
fits on 1 GPU?                      → just DP/DDP for throughput
optimizer/grad state too big?       → ZeRO-1/2 (or FSDP)
parameters too big?                 → ZeRO-3/FSDP, or TP for the big matmuls
a single layer too big?             → TP (intra-node, NVLink/IF)
too many layers for memory?         → PP (cross-node)
experts dominate parameters? (MoE)  → EP (shard experts, all-to-all)
context too long?                   → SP/CP (shard the sequence)
```

對我們 MoE 來說，experts 是大部分參數，所以**EP 是標題
尺寸**，由外部的 DP/ZeRO 和（大尺寸）TP 組成
attention 和 PP 跨階段。

## 步驟 2 — 對應到拓樸上

將最健談的群體放在最快的連結上
（[distributed training](../performance/distributed-training.md)）：

- 節點內的**TP**（每層 all-reduce 需要 NVLink/Infinity Fabric）。 -** 如果\*你重疊 all-to-all 並綁定它，跨節點的**EP**是可接受的
  與 [node-limited routing](../moe/routing-variants.md)；將 EP 組保留為
  expert 計數允許的本地。 -**PP**跨節點（僅啟動跨階段邊界）。 -**DP/ZeRO**最外層（梯度 all-reduce/reduce-scatter 與向後重疊）。

明確寫下裝置網格，例如 16 個 GPU（2 個節點 × 8）：
`DP=2 × PP=2 × TP=2 × EP=4`（度數乘以右側的設備計數
軸； EP 通常共享資料平行軸）。

## 步驟 3 — 為 MoE 層實作 EP

單進程[dispatch form](../moe/moe-from-scratch.md)成為真正的
all-to-all（來自 [Systems & EP](../moe/systems-ep.md)）。骨骼：

```python
# Per MoE layer, with an expert-parallel process group `ep_group`:
# 1. router -> top-k -> per-token destination expert (and thus dest rank)
# 2. sort local tokens by destination rank; compute send_counts
# 3. all_to_all_single(send_counts) -> recv_counts
# 4. all_to_all_single(recv_buf, send_buf, recv_counts, send_counts)  # dispatch
# 5. local grouped GEMM over resident experts on recv_buf
# 6. reverse all_to_all                                               # combine
# 7. unpermute + weighted sum into the residual
```

盡可能使用現有函式庫（Megatron-LM、DeepSpeed-MoE 或 DeepEP）
優化的 all-to-all 重疊）而不是手動滾動通訊 - 但是
了解這七個步驟可以讓你調試不平衡和失速。

## 步驟 4 — 與計算重疊通信

這是 MFU 獲勝或失敗的地方 ([Systems & EP](../moe/systems-ep.md))：

- 將 token 批次和管道調度與前一個區塊的 expert 進行分塊
  GEMM。
- 將**shared-expert**FFN（密集，無通訊）與路由-expert 重疊
  all-to-all。
- 與後向傳遞重疊 DP 梯度減少散射。
- 簡介 [timeline](../performance/profiling.md)：序列化 all-to-all
  似乎是關鍵路徑上與通訊之間的差距——這是需要解決的第一件事。

## 步驟 5 — 測量縮放比例

報告**強**和**弱**縮放並觀察常見的懸崖：

| GPU | 並行設定          | tokens/s | MFU | 筆記                      |
| --- | ----------------- | -------: | --: | ------------------------- |
| 1   | 單身              |   _基線_ |   — | 從之前的 Capstone         |
| 8   | EP=8（1 個節點）  |        — |   — | 節點內 all-to-all（快速） |
| 16  | 16 EP=8 × DP=2    |        — |   — | + 跨節點 DP               |
| 16  | 16 PP=2×TP=2×EP=4 |        — |   — | 全 3D+EP                  |

（_填寫你的測量結果；說明硬體、形狀和方法。_）
隨著通訊的成長，次線性擴展；曲線和線性之間的差距
是你尚未隱藏的通訊。使用分析器進行診斷，而不是猜測。

## 步驟 6 — 大規模驗證正確性

分佈式錯誤很微妙（[training stability](../moe/training-stability.md)）：

-**偏差控制器計數必須在 DP 等級或副本之間同步**
平衡不同的目標。 -**損失/梯度規格應該匹配**單 GPU 運行幾個步驟並固定
種子（梯度累積作為健全性檢查）。 -**fp32 router 數學**無所不在，以避免跨等級 routing 分歧
BF16 關係。

## 要點

- 透過**什麼首先溢出記憶體**來選擇並行性；對於 MoE 來說，**EP**是
  標題維度，由 DP/ZeRO/TP/PP 組成。 -**將集合對應到連結**：TP 節點內、EP/PP 跨節點、DP 最外層；
  明確地寫入設備網格。
- EP MoE 層是[seven-step](../moe/systems-ep.md)排列 →all-to-all→
  分組-GEMM→all-to-all→ 取消排列；**與 all-to-all 重疊**是 MFU 的位置
  被贏了。 -**正確測量強/弱縮放**並**驗證正確性**（已同步
  偏差計數、fp32 routing、匹配損失與單 GPU）。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/capstones.md) 上。請先嘗試每個練習，再展開解答。

1. 對於你的模型，計算每個 GPU 記憶體並選擇 8 和 8 的並行配置
   64 個 GPU；證明每個維度的合理性。
2. 為 MoE 層實作 EP（或連接 DeepSpeed-MoE/Megatron）並驗證
   損失與單 GPU 運行 50 步相符。
3. 分析並量化 all-to-all 重疊；啟用前/後報告 MFU
   分塊流水線。
4. 製作強標度表並解釋線性偏差。

## 參考文獻

- [Distributed training](../performance/distributed-training.md) 和
  [Systems & EP](../moe/systems-ep.md)（此 Capstone 的基礎）。
- 拉傑班達裡等人*DeepSpeed-MoE。 * 2022 年；舒伊比等人。 _威震天-LM。 _ 2019。
- DeepSeek-AI。 _DeepSeek-V3 / DeepEP / DualPipe。 _ 2024 年。
