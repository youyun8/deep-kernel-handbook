# 解答

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 手冊中每一道練習</span>
  <span class="chip"><strong>格式：</strong> 可折疊的詳解</span>
</div>

手冊裡所有練習的解答，依各部組織。每個答案都是一個可折疊區塊——**請先自己試做**，再展開對照 你的推理與數字。封閉形式的推導會完整給出；建構與量測類的題目則給出預期結果、正確方法，以及該 避開的陷阱。

<div class="grid cards" markdown>

- :material-cube-outline:&nbsp;**[基礎](foundations.md)**

  FLOP/bytes 計數、KV cache、FlashAttention、數值。

- :material-set-split:&nbsp;**[Mixture-of-Experts](moe.md)**

  稀疏數學、平衡、routing、穩定性、EP、kernel、serving。

- :material-speedometer:&nbsp;**[效能工程](performance.md)**

  GPU 模型、Triton/CUDA/HIP、分散式訓練、量化、inference、profiling。

- :material-rocket-launch-outline:&nbsp;**[實戰專案](capstones.md)**

  小型 MoE LM 的端到端建立與量測。

</div>

!!! tip "怎麼用這些解答"
    這些數字用的是整數化的硬體規格（例如 A100 ≈ 312 TFLOP/s BF16 / 2 TB/s；H100 ≈ 990 TFLOP/s / 3.35 TB/s；MI300X ≈ 1.3 PFLOP/s / 5.3 TB/s）。你的確切數字會隨假設的晶片而變——該對得上的是 **機制**（memory-bound vs compute-bound）與**數量級**。
