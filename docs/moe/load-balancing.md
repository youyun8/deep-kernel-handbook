#負載平衡

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../moe-from-scratch/">MoE 從頭開始</a></span>
  <span class="chip"><strong>代碼：</strong> <code>code/moe/load_balancing.py</code>（已測試）</span>
</div>

一台 router 留給自己的裝置**崩潰**：少數 experts 獲得了大部分
tokens，其餘的萎縮，你已經為不使用的容量付費了——而
流行的 experts 成為整個層的瓶頸。負載
平衡是使 tokens 分佈在 experts 上的機制。本頁
涵蓋了經典的**輔助損耗**、**expert 容量**和 token 下降，
以及現代的**aux-loss-free**（基於偏差、DeepSeek 風格）方法。

## router 為何崩潰

routing 是贏者全拿的回饋循環。稍微好一點的 expert
早期獲得更多 tokens → 更多梯度 → 提高更快 → 獲得更多 tokens。
如果沒有反壓力，分佈就會集中。兩種不同的危害：

-**品質**：未使用的 experts 廢棄物參數；該模型的行為就像一個
較小的一個。 -**系統**：對於 [expert parallelism](systems-ep.md)，每個 expert 都依賴
具有**固定容量**緩衝區的 GPU。過載的 expert**掉落**tokens
（他們跳過該層）而負載不足的 GPU 空閒。不平衡直接變成
浪費硬體和 latency 落後者。

因此，平衡同時是建模目標和系統需求。

## 輔助負載平衡損耗

標準修復（GShard、Switch）添加了最小化的可微分懲罰
當 tokens 均勻鋪開時。對於一批$T$ tokens 和$E$ experts，
定義每個 expert：

- $f_e$ = _選擇_ expert $e$ 的 tokens 的分數（硬計數，分數
  在頂部-$k$），
- $P_e$ = 批次上分配給 $e$ 的平均 router _機率_（軟）。

開關輔助損耗為

$$ \mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \, P_e. $$

直覺：$f_e$ 是一個不可微的計數，但它*乘以*
可微平均機率 $P_e$。梯度流過$P_e$並推動
遠離已經流行的 experts（高 $f_e$）的機率。總和
當兩者都時，$\sum f_e P_e$ 最小化（對於固定 $\sum f_e = k$、$\sum P_e = 1$）
是統一的，即$f_e = k/E$、$P_e = 1/E$。因子 $E$ 使其成為無標度；
$\alpha$（通常為 $10^{-2}$）設定強度。

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

你將 $\mathcal{L}_{\text{aux}}$ 加入語言建模損失。張力：
**$\alpha$太少 → 崩潰；太多的 $\alpha$ → router 被迫朝向
統一並忽略內容，損害品質。**$\alpha$ 的調優很挑剔，這
激發了以下的無輔助損耗方法。

!!! note "每個裝置與全域平衡"
    對於 EP，你通常希望*每個設備組*實現平衡，而不僅僅是全局 -
    全域平衡但局部傾斜的批次仍會在熱 GPU 上掉落 tokens。
    DeepSeek 新增了設備級和通訊平衡術語；還有大型號
    應用每個微批次/序列的損失以避免批次內熱點。

## expert 容量、下降和溢出

為了高效的批量計算和固定的通訊緩衝區，每個 expert 最多接受
每批次固定數量的 tokens — 其**容量**：

$$ C = \Big\lceil \text{capacity_factor} \cdot \frac{k \cdot T}{E} \Big\rceil. $$

$kT/E$ 是 tokens-per-expert 的平均值；**容量係數**（例如 1.0–2.0）
增加鬆弛。然後：

- 如果超過 $C$ tokens 選擇 expert，則**溢出**tokens 為
  **掉落**——它們繞過了 MoE（殘餘物仍然帶著它們通過）。
- 如果到達的數量較少，則緩衝區將用零**填充**（浪費計算）。

這是直接**品質與 throughput**旋鈕的比較。容量係數 1.0 浪費 無
內存，但在任何不平衡情況下都會下降 tokens； 2.0 很少下降但翻倍
緩衝區（和 GEMM 填充）。掉率是 training 的關鍵指標—健康運行
將其保持在低水平*因為*輔助損耗正在發揮作用，而不是因為容量巨大。

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

**expert-選擇 routing**（參見[routing variants](routing-variants.md)）迴避
透過讓每個 expert _選擇其頂部 - $C$ tokens_ 完全下降 - 完美平衡
透過構建，代價是一些 tokens 比其他人獲得更多的 experts。

## Aux 無損耗平衡（現代方式）

DeepSeek-V3 普及，幾乎完全放棄了輔助損失，取而代之
將**每個 expert 偏差**加到 routing 分數*僅適用於頂級 $k$
選擇* — 不適用於門權重。想法：

$$ \text{select TopK of } \big(s_e + b_e\big), \qquad \text{but weight by the original } s_e. $$

- 每個 expert 都有一個標量偏差 $b_e$（不是透過梯度下降學習的）。
- 每一步之後，根據最近的負載**微移**$b_e$：增加 $b_e$
  負載不足的 experts（使它們更有可能被選中），減少
  超載的。一個簡單的控制器：

$$ b_e \leftarrow b_e + \gamma \cdot \text{sign}\big(\bar{c} - c_e\big), $$

其中 $c_e$ 是 expert $e$ 最近的 token 計數，$\bar c$ 是平均值，$\gamma$ 是
更新率小。

```python
@torch.no_grad()
def update_router_bias(bias, counts, gamma=1e-3):
    # counts: [E] tokens routed to each expert this step
    target = counts.float().mean()
    bias += gamma * torch.sign(target - counts.float())      # raise under-loaded
    return bias
```

為什麼這很好：

-**無梯度幹擾。**偏差影響*選擇*，而非門
縮放 expert 輸出的權重，因此它可以平衡負載**，而不會扭曲
損失景觀**— 你不必像重型 $\alpha$ 那樣以品質換取平衡
確實如此。 -**它與 sigmoid 門控**（獨立的每個 expert 分數）配對，其中添加
偏見是乾淨的。 -**直接控制。**這是你關心的實際數量的回饋控制器
關於（負載），而不是代理懲罰。

DeepSeek-V3 報告比輔助損耗調諧更好的平衡*和*更好的質量，
只保留一個很小的輔助項來防止病理情況。現在這是一個
大型 MoE 的常見預設設定。

!!! warning "偏差不是參數"
    $b_e$ 由控制器更新，而不是由優化器更新，通常是
    排除權重衰減和梯度流。把它當作跑步一樣
    統計資料（它必須跨資料並行等級同步）。

## 測量平衡

每 N 步驟追蹤一次；他們會告訴你 routing 是否健康：

-**丟棄率**— 以容量丟棄的 token-expert 分配的分數。 -**最大/平均負載比**— $\max_e c_e / \bar c$； 1.0 是完美的，請注意 >2。 -**routing 熵**— $-\sum_e P_e \log P_e$；崩潰表現為熵下降。

- $c_e$ 的**變異係數**。

[`code/moe/load_balancing.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/load_balancing.py)
透過測試實現輔助損耗、容量/壓降和偏壓控制器
顯示偏壓控制器驅動故意偏斜的 router 回到
幾百步內的均勻負載。

## 要點

- router 無反壓塌陷；不平衡浪費參數*和*
  硬體（放棄了 tokens，落後的 GPU）。 -**輔助損耗**$\alpha E \sum_e f_e P_e$ 推動統一 routing
  但透過挑剔的 $\alpha$ 以品質換取平衡。 -**expert 容量**每個 expert 的 tokens 上限；**容量係數**是
  質量（下降）-vs-throughput（填充）旋鈕。 -**輔助無損耗**平衡添加了控制器更新的**選擇偏差**
  （不是門的重量），平衡負載而不扭曲梯度－現代
  默認，與 sigmoid 門控配對。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 推導$\sum_e f_e P_e$在均勻分佈主體處最小
   至 $\sum f_e = k$、$\sum P_e = 1$。
2. 對於 $T{=}4096$、$E{=}64$、$k{=}2$，容量係數 1.25，計算 $C$ 和
   如果一個 expert 收到所有作業的 5%，則掉落率。
3. 實現並調整偏置控制器：從傾斜的 init 開始，如何
   $\gamma$ 和閘類型（softmax vs sigmoid）影響收斂平衡？
4. 比較 `train_tiny_moe.py` 中玩具 MoE 上的 aux-loss 與 aux-loss-free：
   報告每個的最終損失*和*負載 CV。

## 參考文獻

- 沙澤爾等人。 _稀疏門控 MoE。 _ 2017（負載平衡損耗起源）。
- 萊皮欣等人。 _GShard._ 2020（容量，下降）。
- 費杜斯、佐夫、沙吉爾。 _開關 Transformer。 _ 2021（此處使用輔助損耗形式）。
- 周等人。 _experts 與 expert 選擇 routing 的混合物。 _ 2022 年。
- 王等人。 / DeepSeek-AI。 _輔助無損 負載平衡_ 和 _DeepSeek-V3._ 2024。
