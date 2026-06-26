# 第一部份·現代機器學習系統的基礎

在讓模型變得更快之前，你需要能夠**預測**它的速度
*應該*是 — 並了解給定的操作是否受到 GPU 的限制
算術單元或其記憶體匯流排。第一部分建立流暢性。

在本部分結束時，你將能夠：

- 從第一原理解釋 Transformer**是什麼**- tokens、attention、
  多頭、FFN 和完整區塊 — 並端到端追蹤 token。
- 計算 Transformer 前向/後向傳遞的 FLOP 和位元組數，並計算
  將其**算術強度**放置在**roofline**上。
- 準確解釋為什麼 LLM**decoding 受記憶體限制**，而 training/prefill
  （大部分）是計算密集型的，使用 KV 快取。
- 從第一原理推導出**Flashattention**— 平鋪 + 線上 softmax，
  並解釋為什麼它將 $O(N^2)$ 記憶體問題轉變為 $O(N)$ 問題。
- 在**fp32 / bf16 / fp16 / fp8**之間進行選擇，並解釋溢出的原因，
  下溢和損失縮放。

## 頁面

1. **[The transformer from scratch](transformer-from-scratch.md)**— Transformer*是*，一次建造一張圖：tokens、attention、多頭、
   FFN，完整區塊。如果 Transformer 是新的，請從這裡開始。
2. **[The transformer as a system](transformer-systems.md)**— roofline
   模型、FLOP/位元組計數以及時間實際去向。
3. **[attention efficiency](attention-efficiency.md)**— KV 緩存，
   decoding 中的記憶體頻寬牆和分頁 attention。
4. **[Flashattention from scratch](flashattention.md)**— 線上 softmax 和
   平鋪，帶有 numpy 參考實作。
5. **[Numerics & precision](numerics-precision.md)**— 浮點格式，
   mixed precision，數值穩定性。

!!! tip "整本手冊的先決條件"
    第一部分中最有用的想法是**算術強度**和
    **roofline**。幾乎後來的每一次優化－Flashattention，分組
    GEMM，融合了 MoE router，量化－最終是對
    roofline。如果你只閱讀一頁，請閱讀
    [the transformer as a system](transformer-systems.md)。
