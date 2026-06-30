# 數學基礎

要真正讀懂現代 ML 系統 —— 不只是會套公式，而是能**推導**它為什麼長這樣、**預測**它在哪裡會出問題 —— 你需要幾塊數學基石。本章把這些基石整理成最精簡的形式：夠用、夠深、直接對應手冊裡的技術。

讀完本章，你將能夠：

- 把矩陣乘法看成「線性映射的複合」，而不只是「把數字相乘再相加」。
- 解釋**低秩**是什麼意思，並理解 MLA 為什麼能用低秩 latent 壓縮 KV cache。
- 從零推導 softmax 的形狀，並說明 temperature 怎麼控制分佈的銳利度。
- 用鏈式法則展開任何前向計算圖的梯度，並解釋 autograd 的實作原理。

## 頁面

1. **[向量、矩陣與線性映射](linear-algebra.md)** —— 維度、點積、矩陣乘法、秩的幾何意義。不管學哪一門深度學習技術，這都是入口。
2. **[低秩矩陣與矩陣分解](low-rank.md)** —— SVD、低秩近似、LoRA 為什麼奏效、MLA 的 KV latent 是什麼。
3. **[機率、Softmax 與 Entropy](probability.md)** —— 離散機率、softmax、temperature、KL 散度與交叉熵損失。
4. **[梯度、反向傳播與自動微分](calculus.md)** —— 偏微分、鏈式法則、計算圖、autograd 如何實作反向傳播。

!!! Tip "和手冊其餘部分的關係"
    - **低秩**：[Decode 算子數學對照](../aiter/decode-math.md) 裡的 MLA 大量依賴低秩投影。先讀 [低秩矩陣與矩陣分解](low-rank.md)，再去看 $W^{DQ}$、$W^{DKV}$、$W^{UK}$ 的角色就會豁然開朗。
    - **Softmax**：[從零實作 Transformer](../foundations/transformer-from-scratch.md) 的 attention 與 MoE router 都用到 softmax。[機率、Softmax 與 Entropy](probability.md) 幫你補上數值穩定性的直覺。
    - **梯度**：[訓練穩定性](../moe/training-stability.md) 談到 gradient explosion、router z-loss 與 loss scaling。[梯度、反向傳播與自動微分](calculus.md) 提供背景。
