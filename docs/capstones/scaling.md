# 實戰專案：擴展到更大規模

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../build-moe/">建立小型 MoE LM</a>、<a href="../../performance/distributed-training/">分散式訓練</a></span>
  <span class="chip"><strong>硬體：</strong> 多 GPU 才能完整執行</span>
</div>

[上一個實戰專案](build-moe.md) 建好並優化了單 GPU 的 MoE LM。這一篇是把
[並行技術](../performance/distributed-training.md)與
[expert-parallel all-to-all](../moe/systems-ep.md)套到多 GPU 的**規劃與實作指南**，結構安排成一條
你可以套用到任何模型/叢集的決策流程。

!!! warning "部分實作 — 歡迎貢獻"
    要完整跑這套，需要一個多 GPU（最好多節點）的
    叢集。下面的*推理、規劃與程式碼骨架*是完整的、可以骨架形式執行；量測到的多節點數字留給你在
    硬體上填。這裡沒有任何東西藏在「TODO」後面——只有叢集規模的 benchmark 表要由你填上。

## 步驟 1 — 決定分片內容以及原因

把記憶體預算走一遍。對每張 GPU，估計（bf16 + Adam）：參數（2 bytes）、梯度（2）、optimizer 狀態
（fp32 動量 + master ≈ 12）與峰值 activation。然後依「哪一項先溢位」來決定切分方式：

```text
fits on 1 GPU?                      → just DP/DDP for throughput
optimizer/grad state too big?       → ZeRO-1/2 (or FSDP)
parameters too big?                 → ZeRO-3/FSDP, or TP for the big matmuls
a single layer too big?             → TP (intra-node, NVLink/IF)
too many layers for memory?         → PP (cross-node)
experts dominate parameters? (MoE)  → EP (shard experts, all-to-all)
context too long?                   → SP/CP (shard the sequence)
```

對我們的 MoE 來說，expert 佔了大部分參數，所以 **EP 是主軸維度**，外面再包上 DP/ZeRO、（為大
matmul 用的）TP 與跨階段的 PP。

## 第 2 步 — 對映到拓樸

把最「多話」的 collective 放到最快的連結上
（[分散式訓練](../performance/distributed-training.md)）：

- **TP** 放節點內（每層 all-reduce 需要 NVLink/Infinity Fabric）。
- **EP** 跨節點是可接受的，*只要*你把 all-to-all 重疊掉、並用
  [node-limited routing](../moe/routing-variants.md) 約束它；在 expert 數允許的範圍內，盡量讓 EP
  群組保持在本地。
- **PP** 跨節點（只在階段邊界做啟動）。
- **DP/ZeRO** 放最外層（梯度 all-reduce/reduce-scatter 與反向重疊）。

把 device mesh 明確寫出來，例如 16 個 GPU（2 節點 × 8）：`DP=2 × PP=2 × TP=2 × EP=4`（各軸度數
相乘要等於 device 數；EP 通常與 data-parallel 軸共享）。

## 步驟 3 — 為 MoE 層實作 EP

單行程的 [dispatch 形式](../moe/moe-from-scratch.md)在這裡變成真正的 all-to-all（來自
[系統與 EP](../moe/systems-ep.md)）。骨架：

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

能用現成函式庫就用（Megatron-LM、DeepSpeed-MoE，或為 all-to-all 重疊優化過的 DeepEP），不要手刻
通訊——但搞懂這七步，才有辦法 debug 不平衡與停頓。

## 第 4 步 — 讓通訊與計算重疊

MFU 的成敗就在這裡（[系統與 EP](../moe/systems-ep.md)）：

- 把 token batch 分塊、用管線把它和前一塊的 expert grouped GEMM 排在一起。
- 把 **shared-expert** FFN（密集、無通訊）和路由 expert 的 all-to-all 重疊。
- 把 DP 的 gradient reduce-scatter 與反向傳播重疊。
- 看 [timeline](../performance/profiling.md)：序列化的 all-to-all 會在關鍵路徑上顯示成一段空隙——
  那是第一個該解決的東西。

## 步驟 5 — 測量縮放比例

回報**強擴展**與**弱擴展**，並留意常見的懸崖：

| GPU | 並行設定              | tokens/s | MFU | 備註                       |
| --- | --------------------- | -------: | --: | -------------------------- |
| 1   | 單卡                  |   _基線_ |   — | 來自上一個實戰專案        |
| 8   | EP=8（1 節點）        |        — |   — | 節點內 all-to-all（快）    |
| 16  | EP=8 × DP=2           |        — |   — | + 跨節點 DP                |
| 16  | PP=2 × TP=2 × EP=4     |        — |   — | 完整 3D + EP               |

（_填上你的量測結果；註明硬體、shape 與方法。_）擴展會隨通訊成長而變得次線性；曲線與理想線性
之間的差距，就是你還沒藏起來的通訊。用 profiler 去診斷，別用猜的。

## 步驟 6 — 大規模驗證正確性

分散式的 bug 很隱晦（[訓練穩定性](../moe/training-stability.md)）：

- **偏差控制器的計數必須在 DP rank／副本之間同步**，否則各自朝不同目標平衡。
- **損失/梯度範數應該對得上**單 GPU 跑幾步、固定種子的結果（用梯度累積當作 sanity check）。
- **router 數學一律用 fp32**，避免 bf16 打平造成跨 rank 的 routing 分歧。

## 要點

- 依**哪一項先溢位記憶體**來選並行策略；對 MoE，**EP** 是主軸，外面包 DP/ZeRO/TP/PP。
- **把 collective 對映到連結**：TP 節點內、EP/PP 跨節點、DP 最外層；把 device mesh 明確寫出來。
- EP 的 MoE 層就是那個[七步](../moe/systems-ep.md)：permute → all-to-all → grouped GEMM →
  all-to-all → unpermute；**把 all-to-all 重疊掉**就是 MFU 的勝負所在。
- **好好量強/弱擴展**，並**驗證正確性**（同步偏差計數、fp32 routing、損失對得上單 GPU）。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/capstones.md) 上。請先嘗試每個練習，再展開解答。

1. 為你的模型估算每張 GPU 的記憶體，分別為 8 卡與 64 卡選一組並行配置，並說明每個維度的理由。
2. 為 MoE 層實作 EP（或接上 DeepSpeed-MoE/Megatron），驗證損失在 50 步內對得上單 GPU。
3. 對 all-to-all 重疊做 profiling 並量化它；回報啟用分塊管線前後的 MFU。
4. 做一張強擴展表，並解釋偏離線性的原因。

## 參考文獻

- [分散式訓練](../performance/distributed-training.md)與[系統與 EP](../moe/systems-ep.md)
  （本實戰專案的基礎）。
- Rajbhandari et al. _DeepSpeed-MoE._ 2022；Shoeybi et al. _Megatron-LM._ 2019。
- DeepSeek-AI. _DeepSeek-V3 / DeepEP / DualPipe._ 2024。
