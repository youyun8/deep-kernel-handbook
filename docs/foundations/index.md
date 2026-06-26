# 第一部 · 現代機器學習系統的基礎

要讓模型跑得更快，你得先能**預測**它*應該*多快，並判斷某個操作是被 GPU 的算術單元卡住、
還是被它的記憶體匯流排卡住。第一部建立這種流暢度。

讀完本部，你將能夠：

- 從第一原理講清楚 Transformer **是什麼**——token、attention、多頭、FFN、完整 block——並
  端到端追蹤一個 token。
- 數出 Transformer 前向／反向傳播的 FLOP 與 bytes，把它的**算術強度**放上 **roofline**。
- 精確說明為什麼 LLM 的 **decode 是 memory-bound**、而 training/prefill（大多）是 compute-bound，
  以及 KV cache 的角色。
- 從第一原理推導 **FlashAttention**——tiling + online softmax——並解釋它如何把 $O(N^2)$ 的記憶體
  問題變成 $O(N)$。
- 在 **fp32 / bf16 / fp16 / fp8** 之間做選擇，並解釋溢位、下溢與 loss scaling。

## 頁面

1. **[從零實作 Transformer](transformer-from-scratch.md)**——Transformer 到底是什麼，一次一張圖：
   token、attention、多頭、FFN、完整 block。第一次接觸 Transformer 就從這裡開始。
2. **[作為系統的 Transformer](transformer-systems.md)**——roofline 模型、FLOP/bytes 計數，以及
   時間實際花在哪。
3. **[Attention 效率](attention-efficiency.md)**——KV cache、decode 的記憶體頻寬牆，以及
   PagedAttention。
4. **[從零實作 FlashAttention](flashattention.md)**——online softmax 與 tiling，附 numpy 參考實作。
5. **[數值與精度](numerics-precision.md)**——浮點格式、mixed precision、數值穩定性。

!!! tip "整本手冊的先備觀念"
    第一部最有用的兩個觀念是**算術強度**和 **roofline**。後面幾乎每一個優化——FlashAttention、
    grouped GEMM、融合 MoE router、量化——最終都是對 roofline 的出招。如果你只讀一頁，請讀
    [作為系統的 Transformer](transformer-systems.md)。
