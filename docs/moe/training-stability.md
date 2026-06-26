# training MoE 穩定性

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階→高階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../load-balancing/">負載平衡</a>、<a href="../../foundations/numerics-precision/">數位</a></span>
  <span class="chip"><strong>硬體：</strong> 無</span>
</div>

MoE 比密集模型更難訓練，因為**routing 是離散的並且
自我強化**。小數值擾動會翻轉 routing 決策，其中
更改哪些參數獲得梯度，這會再次更改 routing。本頁
涵蓋特定的病理學和標準修復：**router z-loss**，
**初始化**，router 的精確規則，以及一些實用的
護欄。

## 為什麼 MoE training 很敏感

三個耦合問題：

1. **離散性。**Top-$k$ 是一個硬性的、不可微分的選擇。大門
   權重是可微的，但*其中*experts 運作不可微。一個微小的改變
   logit 可以將 token 移到不同的 expert — 不連續跳轉
   損失面。
2. **自我強化。**與 [負載平衡](load-balancing.md) 一樣，routing 是
   容易崩潰的正向回饋環路。
3. **Logit 膨脹。**router logits 沒有任何內在限制。如果它們長大了
   大時，softmax 飽和（routing 幾乎變成熱狀態並且「凍結」——不
   梯度來逃避錯誤的分配），而大的 logits 與
   [low precision](../foundations/numerics-precision.md)。

未經處理的結果：損失尖峰、NaN、「死」experts 和 routing
很早就鎖定並且永遠不會恢復。

## router z 損失

router z 損失（來自 ST-MoE）直接懲罰大的 router logits 以保持
Softmax 處於理智狀態。對於每個 token 的 logits $x \in \mathbb{R}^{E}$：

$$ \mathcal{L}_{z} = \frac{\beta}{T}\sum_{t=1}^{T}\Big(\log\sum*{e=1}^{E} e^{x*{t,e}}\Big)^{2}. $$

術語 $\log\sum_e e^{x_e}$ 是對數分割區（softmax 歸一化器）；
對它進行平方和懲罰會將 logits 拉向較小的值。效果：

- 保持 `exp` 參數較小 → bf16/fp16 中**無溢出**，softmax 更穩定。
- 防止澆口飽和至結凍狀態 → routing 保留
  *塑膠*並且可以糾正早期的錯誤。
- 微小係數 ($\beta \approx 10^{-3}$) — 它是正規化器，而不是主係數
  客觀。

```python
def router_z_loss(logits, beta=1e-3):
    # logits: [T, E] pre-softmax router outputs (compute in fp32!)
    logsumexp = torch.logsumexp(logits.float(), dim=-1)      # [T]
    return beta * (logsumexp**2).mean()
```

MoE training 總損失：

$$ \mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{aux}} + \beta\,\mathcal{L}\_{z}, $$

（$\mathcal{L}_{\text{aux}}$ 可選地替換為
[aux-loss-free bias](load-balancing.md)）。即使在 aux-loss-free 中，z-loss 也被保留
食譜——它解決了 logit 大小，這是一個與平衡不同的問題。

## router 的精確紀律

這就是[numerics](../foundations/numerics-precision.md)和 MoE 的碰撞點。
routing 是一個由 logits 之間的「微小差異」驅動的「離散」決策，因此
舍入雜訊會翻轉分配並破壞回授迴路的穩定性。

-**計算 router logits、softmax/sigmoid 和 fp32 中的 aux/z 損失**，甚至
在 BF16 型號。 router 矩陣很小 - fp32 成本可以忽略不計
穩定性增益大。 -**偏移控制器的計數必須在 fp32**中減少並同步
跨數據並行排名，或不同排名針對不同目標進行平衡。

- 在任何 softmax 之前減去 max（透過 `logsumexp`/`log_softmax` 免費）。

!!! warning "經典的無聲蟲子"
    bf16 中的 routing 可以使兩個 experts' logits 平手（bf16 有 ~7 尾數
    位），並且決勝局（argmax/topk）變得任意且依賴於排名
    在資料並行性下 - 不同的副本以不同的方式路由相同的 token，
    破壞平衡統計。 fp32 router 數學避免了它。

## 初始化

routing 是最脆弱的**早期**，在 experts 分化之前。好
練習：

-**小 router 初始化。**使用小秤初始化 router 重量（例如
$\text{std}\sim 0.01$–$d^{-1/2}$ 具有額外收縮），因此初始 logits 為
接近零 → 接近均勻的 routing→ 每個 expert 都獲得梯度並微分
在循環崩潰之前。 （開關使用了截斷法線，並減少了
初始化規模正是為了這個。 ） -**標準 expert 初始化。**experts 是普通 FFN；像你一樣初始化它們
密集 FFN。 -**預熱 router/容量。**儘早獲得更大的容量係數（更少的跌落
而 routing 是隨機的）和 LR 預熱減少了早期的不穩定。 -**共用 expert 作為穩 ​​ 定器。**A [shared expert](routing-variants.md)
保證從步驟 0 開始的密集梯度路徑，平滑冷啟動。

## 其他實用護欄

-**梯度削波**（全球標準）－MoE 損失峰值很常見；剪輯
防止一顆尖峰破壞運轉。 -**平衡每個微批次/每個序列的輔助損耗**，而不僅僅是全局性的，以
避免全域統計資料隱藏的批次內熱點。 -**監控死 experts**（許多步驟零負載）和 routing 熵；一個
熵的突然下降是崩潰的預警訊號。 -**logits 上的抖動/雜訊**（較舊的方法，例如 Switch 的乘法
輸入抖動）增加了探索，因此 routing 不會鎖定 - 較少使用
z-loss + 很好的 init，但仍然是一個工具。 -**保持優化器狀態 fp32**（亞當矩），標準但雙重重要
當損失表面粗糙時。

## 診斷有問題的 MoE 運行

| 症狀                                | 可能的原因                    | 修復                                         |
| ----------------------------------- | ----------------------------- | -------------------------------------------- |
| 早期損失峰值/NaN                    | router logit 放大； fp16 溢出 | 添加/提高 z 損失； fp32 中的路線；剪輯畢業生 |
| 幾個 experts 得到全部 tokens        | 弱平衡                        | 提高$\alpha$或啟用偏移控制器                 |
| 死了的 experts 再也無法恢復         | 早期倒塌，飽和閘門            | 更小的 router init，更大的早期容量，z-loss   |
| 不同的複製品在 routing 上意見不一致 | BF16 router 領帶              | fp32 router 數學；同步偏差計數               |
| 高掉率                              | 容量過低/不平衡               | 提高容量係數；修復平衡                       |

## 要點

- MoE 的不穩定性源自於**離散、自我強化的 routing**和
  **無限 router logits**。 -**router z-loss**$\beta(\log\sum e^{x})^2$ 保持 logits 小 → 穩定的 softmax，
  無溢出，塑膠 routing。即使在無輔助損耗設定中也能保持它。 -**在 fp32 中執行所有 router 數學**- 對小 logit 差異的離散決策
  對精確度敏感，且 bf16 關係會導致跨副本不一致。 -**小型 router 初始化+（可選）共用 expert +預熱**使脆弱
  冷啟動可生存；剪輯梯度並監控熵/死 experts。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 證明最小化 $\mathcal{L}_z$ 會縮小 $\|x\|$ 並限制 softmax
   遠離一熱。 routing 分佈的熵會發生什麼變化？
2. 構造 router logits，其中 bf16 捨去會翻轉 argmax，但 fp32 不會。
3. 在玩具 MoE 上，使用故意較大的 router 進行有或沒有 z 損失的訓練
   初始化；比較損耗尖峰頻率和死亡 expert 計數。
4. 為什麼共用 expert 可以緩解冷啟動？追蹤步驟 0 上的梯度路徑
   對於 token，其佈線 experts 幾乎相同。

## 參考文獻

-佐夫等。 _ST-MoE：設計穩定且可轉移的稀疏 expert 模型_（router z 損失）。 2022 年。

- 費杜斯、佐夫、沙吉爾。 _開關 Transformer_（初始化、抖動、選擇性 fp32）。 2021 年。
- 萊皮欣等人。 _GShard。 _ 2020。
- DeepSeek-AI。 _DeepSeek-V3_（偏置控制器，穩定性配方）。 2024 年。
- Micikevicius 等人。 _mixed precision training。 _ 2017 年。
