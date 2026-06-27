# 負載平衡

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../moe-from-scratch/">從零實作 MoE layer</a></span>
  <span class="chip"><strong>程式碼：</strong> <code>code/moe/load_balancing.py</code>（已測試）</span>
</div>

放著不管的 router 會**崩塌**：少數 expert 拿走大部分 token，其餘逐漸萎縮，於是你為用不到的 容量付了錢——而那幾個熱門 expert 反而成了整層的瓶頸。負載平衡就是讓 token 在 expert 之間 攤開的機制。本頁涵蓋經典的 **auxiliary loss**、**expert capacity** 與 token drop，以及現代的 **aux-loss-free**（以偏差為基礎、DeepSeek 風格）做法。

## Router 為何崩潰

routing 是個贏者全拿的回饋迴圈。早期稍微好一點的 expert 拿到更多 token → 更多梯度 → 進步更快 → 又拿到更多 token。沒有反向壓力，分佈就會越來越集中。這帶來兩種不同的傷害：

- **品質**：用不到的 expert 等於浪費參數，模型表現得像一個更小的模型。
- **系統**：在 [expert parallelism](systems-ep.md) 下，每個 expert 落在某張 GPU 上、配著 **固定容量**的緩衝區。過載的 expert 會**丟棄** token（它們直接跳過該層），而負載不足的 GPU 則閒著。不平衡直接變成浪費的硬體與落後的 latency。

所以平衡同時是建模目標，也是系統需求。

## 輔助負載平衡損耗

標準解法（GShard、Switch）加入一個可微分的懲罰項，它在 token 均勻攤開時取得最小值。對一個 含 $T$ 個 token、$E$ 個 expert 的 batch，為每個 expert 定義：

- $f_e$ = _選到_ expert $e$ 的 token 比例（硬計數，落在 top-$k$ 內的比例），
- $P_e$ = 整個 batch 上分配給 $e$ 的平均 router _機率_（軟值）。

Switch 的 auxiliary loss 為

$$ \mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \, P_e. $$

直覺：$f_e$ 是不可微的計數，但它*乘上*可微的平均機率 $P_e$。梯度流過 $P_e$，把機率從已經 熱門（高 $f_e$）的 expert 推開。在固定的 $\sum f_e = k$、$\sum P_e = 1$ 之下，$\sum f_e P_e$ 在兩者都均勻時取得最小，亦即 $f_e = k/E$、$P_e = 1/E$。因子 $E$ 讓它與規模無關； $\alpha$（通常 $10^{-2}$）設定強度。

```python
def switch_aux_loss(router_probs, topk_idx, n_experts, alpha=1e-2):
    # router_probs: [T, E] softmax probs ; topk_idx: [T, k] selected experts
    T = router_probs.shape[0]
    P = router_probs.mean(dim=0)                              # [E] mean prob
    one_hot = torch.zeros(T, n_experts, device=router_probs.device)
    one_hot.scatter_(1, topk_idx, 1.0)
    f = one_hot.sum(dim=0) / T                                # [E] selection frac
    return alpha * n_experts * torch.sum(f * P)
```

你把 $\mathcal{L}_{\text{aux}}$ 加進語言建模損失。這裡有個張力：**$\alpha$ 太小 → 崩塌； $\alpha$ 太大 → router 被逼向均勻、忽略內容，損害品質。** $\alpha$ 很難調，這也催生了下面的 aux-loss-free 做法。

!!! note "每個 device 的平衡 vs 全域平衡"
    在 EP 下，你通常希望*每個 device 群組*都平衡，而不只是全域平衡——一個全域平衡但局部傾斜 的 batch，仍會在熱 GPU 上丟 token。DeepSeek 額外加了 device 級與通訊平衡的項；大型模型也 會套用 per-microbatch／per-sequence 的損失，以避免 batch 內部的熱點。

## Expert 容量、下降和溢出

為了讓批次計算高效、通訊緩衝區固定，每個 expert 每個 batch 最多只收固定數量的 token——這就是 它的**容量（capacity）**：

$$ C = \Big\lceil \text{capacity_factor} \cdot \frac{k \cdot T}{E} \Big\rceil. $$

$kT/E$ 是 tokens-per-expert 的平均；**容量係數（capacity factor）**（例如 1.0–2.0）加入一些 餘裕。然後：

- 若超過 $C$ 個 token 選到某 expert，**溢位**的 token 會被**丟棄**——它們略過 MoE（殘差仍把 它們帶過這一層）。
- 若到達的 token 較少，緩衝區會以零**填充（pad）**（浪費計算）。

這是一個直接的**品質 vs throughput** 旋鈕。容量係數 1.0 不浪費記憶體，但只要有不平衡就會丟 token；2.0 很少丟，但緩衝區（與 GEMM padding）翻倍。drop rate 是 training 的關鍵指標——健康的 訓練之所以維持低 drop，是*因為* auxiliary loss 在發揮作用，而不是因為容量開得很大。

```python
def apply_capacity(topk_idx, n_experts, capacity):
    # Returns a boolean keep-mask; drops tokens beyond capacity per expert (FIFO).
    keep = torch.ones_like(topk_idx, dtype=torch.bool)
    for e in range(n_experts):
        pos = (topk_idx == e).nonzero(as_tuple=False)        # assignments to e
        if pos.shape[0] > capacity:
            drop = pos[capacity:]                             # overflow
            keep[drop[:, 0], drop[:, 1]] = False
    return keep
```

**Expert-choice routing**（見 [Routing 變體](routing-variants.md)）讓每個 expert _挑自己的 top-$C$ token_，從而完全避免丟棄——天生完美平衡，代價是有些 token 比別人分到更多 expert。

## Aux 無損耗平衡（現代方式）

DeepSeek-V3 把這個做法發揚開來，幾乎完全捨棄 auxiliary loss，改成把**每個 expert 的偏差**加到 routing 分數上、_只影響 top-$k$ 的選擇_——不影響 gate 權重。想法如下：

$$ \text{select TopK of } \big(s_e + b_e\big), \qquad \text{but weight by the original } s_e. $$

- 每個 expert 有一個純量偏差 $b_e$（不是用梯度下降學的）。
- 每一步後，依最近負載**微調** $b_e$：把負載不足 expert 的 $b_e$ 調高（讓它更容易被選），把 過載 expert 的調低。一個簡單的控制器：

$$ b_e \leftarrow b_e + \gamma \cdot \text{sign}\big(\bar{c} - c_e\big), $$

其中 $c_e$ 是 expert $e$ 最近的 token 計數、$\bar c$ 是平均、$\gamma$ 是一個很小的更新率。

```python
@torch.no_grad()
def update_router_bias(bias, counts, gamma=1e-3):
    # counts: [E] tokens routed to each expert this step
    target = counts.float().mean()
    bias += gamma * torch.sign(target - counts.float())      # raise under-loaded
    return bias
```

為什麼這招好用：

- **不干擾梯度。** 偏差只影響*選擇*，不影響縮放 expert 輸出的 gate 權重，所以它能平衡負載 **而不扭曲損失地形**——你不必像大 $\alpha$ 那樣拿品質換平衡。
- **與 sigmoid gating（獨立 per-expert 分數）天生相配**，把偏差加上去很乾淨。
- **直接控制。** 這是針對你真正在意的量（負載）做的回饋控制器，而不是一個代理懲罰項。

DeepSeek-V3 回報，這比調 auxiliary loss 得到更好的平衡*又*更好的品質，只保留一個很小的輔助項 來防止病態情況。如今這已是大型 MoE 的常見預設。

!!! warning "偏差不是參數"
    $b_e$ 是由控制器更新、而非由 optimizer 更新，通常排除在權重衰減與梯度流之外。把它當成一個 running 統計量來看待（而且它必須跨 data-parallel rank 同步）。

## 測量平衡

每隔 N 步追蹤這些指標，它們會告訴你 routing 是否健康：

- **drop rate**——因容量而被丟棄的 token-expert 分配比例。
- **max/mean 負載比**——$\max_e c_e / \bar c$；1.0 為完美，>2 要警覺。
- **routing 熵**——$-\sum_e P_e \log P_e$；崩塌會表現為熵下降。
- $c_e$ 的**變異係數（CV）**。

[`code/moe/load_balancing.py`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/moe/load_balancing.py) 用測試實作了 auxiliary loss、capacity/drop 與偏差控制器，並展示偏差控制器能在幾百步內把一個 刻意偏斜的 router 拉回均勻負載。

## 要點

- router 缺乏反向壓力就會崩塌；不平衡同時浪費參數*與*硬體（被丟的 token、落後的 GPU）。
- **auxiliary loss** $\alpha E \sum_e f_e P_e$ 推動均勻 routing，但要靠難調的 $\alpha$ 拿品質換 平衡。
- **expert capacity** 為每個 expert 的 token 設上限；**容量係數**是品質（drop）vs throughput （padding）的旋鈕。
- **aux-loss-free** 平衡加入由控制器更新的**選擇偏差**（不是 gate 權重），平衡負載而不扭曲 梯度——現代預設，與 sigmoid gating 相配。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 在 $\sum f_e = k$、$\sum P_e = 1$ 的約束下，推導 $\sum_e f_e P_e$ 在均勻分佈時取得最小。
2. 對 $T{=}4096$、$E{=}64$、$k{=}2$、容量係數 1.25，計算 $C$；若某 expert 收到全部分配的 5%， drop rate 是多少？
3. 實作並調整偏差控制器：從一個傾斜的初始化開始，$\gamma$ 與 gate 類型（softmax vs sigmoid） 如何影響收斂後的平衡？
4. 在 `train_tiny_moe.py` 的玩具 MoE 上比較 aux-loss 與 aux-loss-free：分別回報最終損失*與* 負載 CV。

## 參考文獻

- Shazeer et al. _Sparsely-Gated MoE._ 2017（負載平衡損失的源頭）。
- Lepikhin et al. _GShard._ 2020（capacity、drop）。
- Fedus, Zoph, Shazeer. _Switch Transformer._ 2021（本頁用的 auxiliary loss 形式）。
- Zhou et al. _Mixture-of-Experts with Expert Choice Routing._ 2022。
- Wang et al. / DeepSeek-AI. _Auxiliary-Loss-Free Load Balancing_ 與 _DeepSeek-V3._ 2024。
