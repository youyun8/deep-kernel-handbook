# 訓練穩定性

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階→高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../load-balancing/">負載平衡</a>、<a href="../../foundations/numerics-precision/">數值與精度</a></span>
  <span class="chip"><strong>硬體：</strong> 無</span>
</div>

MoE 比密集模型更難訓練，因為 **routing 既離散又會自我強化**。微小的數值擾動就能翻轉 routing 決策，改變哪些參數拿到梯度，而這又會再次改變 routing。本頁談這些特定的病態與標準解法： **router z-loss**、**初始化**、router 的精度紀律，以及一些實務護欄。

## 為什麼 MoE training 很敏感

三個彼此耦合的問題：

1. **離散性。** top-$k$ 是一個硬性、不可微的選擇。gate 權重可微，但*哪些* expert 被啟用這件事 不可微。一個極小的 logit 變動就能把 token 移到不同 expert——在損失面上造成不連續的跳動。
2. **自我強化。** 如同 [負載平衡](load-balancing.md)，routing 是一個容易崩塌的正回饋迴圈。
3. **logit 膨脹。** router logits 本身沒有任何約束。一旦變大，softmax 飽和（routing 幾乎變成 one-hot 並「凍結」——沒有梯度可以逃離錯誤分配），而大的 logit 又和 [低精度](../foundations/numerics-precision.md)互動不良。

不處理的後果：損失尖峰、NaN、「死」expert，以及很早就鎖死、再也回不來的 routing。

## Router z 損失

router z-loss（出自 ST-MoE）直接懲罰過大的 router logits，讓 softmax 保持清醒。對每個 token 的 logits $x \in \mathbb{R}^{E}$：

$$ \mathcal{L}_{z} = \frac{\beta}{T}\sum_{t=1}^{T}\Big(\log\sum*{e=1}^{E} e^{x*{t,e}}\Big)^{2}. $$

其中 $\log\sum_e e^{x_e}$ 是 log-partition（softmax 的歸一化器）；對它平方並懲罰，會把 logits 往較小的值拉。效果：

- 讓 `exp` 的引數維持小 → 在 BF16/FP16 中**不溢位**，softmax 更穩定。
- 防止 gate 飽和到凍結狀態 → routing 保持*可塑*，能修正早期的錯誤。
- 係數很小（$\beta \approx 10^{-3}$）——它是正規化項，不是主要目標。

```python
def router_z_loss(logits, beta=1e-3):
    # logits: [T, E] pre-softmax router outputs (compute in fp32!)
    logsumexp = torch.logsumexp(logits.float(), dim=-1)      # [T]
    return beta * (logsumexp**2).mean()
```

MoE training 的總損失：

$$ \mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{aux}} + \beta\,\mathcal{L}\_{z}, $$

（$\mathcal{L}_{\text{aux}}$ 可選擇換成 [aux-loss-free 偏差](load-balancing.md)）。即使在 aux-loss-free 的配方裡，z-loss 仍會保留——它處理的是 logit 大小，這跟平衡是兩個不同的問題。

## Router 的精確紀律

這是 [數值與精度](../foundations/numerics-precision.md)和 MoE 正面相撞的地方。routing 是一個由 logit 之間「微小差距」驅動的「離散」決策，所以捨入雜訊會翻轉分配、破壞回饋迴路的穩定。

- **router logits、softmax/sigmoid 與 aux/z-loss 一律用 FP32 計算**，即使是 BF16 模型。router 矩陣很小，FP32 成本可忽略，但穩定性收益很大。
- **偏差控制器的計數必須以 FP32 reduce 並跨 data-parallel rank 同步**，否則不同 rank 會朝不同 目標平衡。
- 在任何 softmax 之前都先減掉 max（用 `logsumexp`／`log_softmax` 即免費取得）。

!!! warning "經典的無聲 bug"
    在 BF16 裡做 routing，可能讓兩個 expert 的 logit 打平（BF16 只有約 7 個 mantissa bit），於是 決勝（argmax/topk）變得任意、而且在 data parallel 下依 rank 而異——不同副本把同一個 token 路由到不同地方，破壞平衡統計。用 FP32 做 router 數學可以避免。

## 初始化

routing 在**早期**最脆弱，也就是 expert 還沒分化的時候。良好實務：

- **小的 router 初始化。** 用小尺度初始化 router 權重（例如 $\text{std}\sim 0.01$–$d^{-1/2}$ 再額外縮小），讓初始 logit 接近零 → routing 接近均勻 → 每個 expert 都拿到梯度、在迴圈崩塌前 先分化開來。（Switch 用截斷常態並刻意縮小初始化尺度，正是為此。）
- **標準的 expert 初始化。** expert 就是普通 FFN，照密集 FFN 的方式初始化即可。
- **router/容量 warmup。** 早期用較大的容量係數（routing 還隨機時少丟一點）加上 LR warmup，可 降低早期不穩定。
- **用共享 expert 當穩定器。** 一個 [共享 expert](routing-variants.md) 從第 0 步就保證有一條密集 梯度路徑，能平滑冷啟動。

## 其他實用護欄

- **梯度裁剪（global norm）**——MoE 的損失尖峰很常見；裁剪能防止單一尖峰毀掉整個訓練。
- **per-microbatch／per-sequence 的 auxiliary loss**，而不只是全域，以避免全域統計掩蓋掉 batch 內部的熱點。
- **監控死 expert**（連續多步零負載）與 routing 熵；熵突然下降是崩塌的早期警訊。
- **在 logit 上加 jitter/雜訊**（較舊的手法，例如 Switch 的乘法式輸入 jitter）增加探索，避免 routing 鎖死——有了 z-loss + 良好初始化後較少用，但仍是一項工具。
- **optimizer 狀態保持 FP32**（Adam 的動量）——標準做法，在損失面崎嶇時格外重要。

## 診斷有問題的 MoE 運行

| 症狀 | 可能原因 | 解法 |
| --- | --- | --- |
| 早期損失尖峰／NaN | router logit 膨脹；FP16 溢位 | 加入／提高 z-loss；router 走 FP32；裁剪梯度 |
| 少數 expert 拿走所有 token | 平衡太弱 | 提高 $\alpha$ 或啟用偏差控制器 |
| 死 expert 再也回不來 | 早期崩塌、gate 飽和 | 更小的 router 初始化、更大的早期容量、z-loss |
| 不同副本對 routing 意見不一 | BF16 router 打平 | router 走 FP32 數學；同步偏差計數 |
| drop rate 過高 | 容量太低／不平衡 | 提高容量係數；修好平衡 |

## 要點

- MoE 的不穩定源自**離散、自我強化的 routing**與**無界的 router logits**。
- **router z-loss** $\beta(\log\sum e^{x})^2$ 讓 logit 維持小 → softmax 穩定、不溢位、routing 可塑。即使在 aux-loss-free 配方裡也要保留它。
- **所有 router 數學都用 FP32**——基於微小 logit 差距的離散決策對精度敏感，BF16 的打平會造成 跨副本不一致。
- **小 router 初始化 +（可選）共享 expert + warmup** 讓脆弱的冷啟動得以存活；裁剪梯度，並監控 熵與死 expert。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 證明最小化 $\mathcal{L}_z$ 會縮小 $\|x\|$、把 softmax 從 one-hot 拉開。routing 分佈的熵會怎麼 變？
2. 構造一組 router logits，使 BF16 捨入會翻轉 argmax、但 FP32 不會。
3. 在玩具 MoE 上，用一個刻意偏大的 router 初始化，分別訓練有 z-loss 與沒有 z-loss 的版本；比較 損失尖峰頻率與死 expert 數。
4. 為什麼共享 expert 能緩解冷啟動？追蹤第 0 步時，一個 token 的梯度路徑（此時其路由 expert 幾乎 無差別）。

## 參考文獻

- Zoph et al. _ST-MoE: Designing Stable and Transferable Sparse Expert Models_（router z-loss）。2022。
- Fedus, Zoph, Shazeer. _Switch Transformer_（初始化、jitter、選擇性 FP32）。2021。
- Lepikhin et al. _GShard._ 2020。
- DeepSeek-AI. _DeepSeek-V3_（偏差控制器、穩定性配方）。2024。
- Micikevicius et al. _Mixed Precision Training._ 2017。
