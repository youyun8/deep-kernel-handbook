# 解決方案

<div class="page-meta">
  <span class="chip"><strong>適用範圍：</strong>手冊中的每個練習</span>
  <span class="chip"><strong>格式：</strong> 可折疊工作答案</span>
</div>

解答了手冊中所有練習的答案，按部分組織。每個答案
是一個可折疊的塊 -**首先嘗試練習**，然後展開以檢查你的
推理和數字。完整給出了封閉形式的推導；建構和測量
練習給了預期的結果、正確的方法和陷阱
避免。

<div class="grid cards" markdown>

- :material-cube-outline:**[Part I · Foundations](foundations.md)**

  FLOP/位元組計數、KV 快取、Flashattention、數字。

- :material-set-split:**[Part II · Mixture-of-experts](moe.md)**

  稀疏數學、平衡、routing、穩定性、EP、kernels、serving。

- :material-speedometer:**[Part III · Performance](performance.md)**

  GPU 模型、Triton/CUDA/HIP、分散式 training、Quant、inference、分析。

- :material-rocket-launch-outline:**[Part IV · Capstones](capstones.md)**

  小型 MoE LM 的端到端建置和測量。

</div>

!!! tip "如何充分利用這些"
    這些數字使用圓形硬體規格（例如 A100 ≈ 312 TFLOP/s bf16 / 2 TB/s；
    H100 ≈ 990 TFLOP/秒/3.35 TB/秒； MI300X ≈ 1.3 PFLOP/秒/5.3 TB/秒）。你的確切
    數字會隨著你假設的晶片而變化——應該匹配的是
    **制度**（記憶體限制與計算限制）和**數量級**。
