# 從零實作 MoE layer

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../why-sparsity/">為什麼需要稀疏化</a>、PyTorch</span>
  <span class="chip"><strong>程式碼：</strong> <code>code/moe/</code>（CPU、已測試）</span>
</div>

現在我們用 PyTorch 把一個完整的 MoE 層做出來 —— Expert、router/gate、top-$k$ 選擇與加權合併。 先寫一個最乾淨、保證正確的版本，再重構成 [系統章節](systems-ep.md)與 [kernels](kernels.md) 要 優化的那種 dispatch 形式。這裡的所有程式碼都在 CPU 上可跑，並由 [`code/moe/`](https://github.com/youyun8/deep-kernel-handbook/tree/main/code/moe) 的測試把關。

## MoE 層的剖析

MoE FFN 把單一 FFN 換成：

1. **$E$ 個 expert** —— 各自獨立的 FFN（通常是 SwiGLU）：$\text{expert}_e(h) = W^{down}_e\,\big(\text{SiLU}(W^{gate}_e h)\odot (W^{up}_e h)\big)$。
2. **router/gate** —— 一個線性映射 $h \mapsto W_r h \in \mathbb{R}^{E}$，為每個 expert 產生一個 logit。
3. **top-$k$ 選擇** —— 為每個 token 挑出得分最高的 $k$ 個 expert。
4. **合併** —— 把 token 送過這 $k$ 個 expert，再用（歸一化後的）gate 分數加權求和它們的輸出。

對一個 token 表示 $h\in\mathbb{R}^d$、gate 權重 $g_e$：

$$ y = \sum_{e \in \text{TopK}(h)} g_e \cdot \text{expert}_e(h), \qquad g = \text{normalize}\big(\text{score}(W_r h)\big). $$

## 閘控：softmax 與 sigmoid

Gate 的計分函數，比它表面上看起來更關鍵。

**Softmax gating**（GShard、Switch、Mixtral）：對全部 $E$ 個 logit 做 softmax，取 top-$k$， 再把這 $k$ 個重新歸一化成總和為 1。

$$ p = \text{softmax}(W_r h), \quad g_e = \frac{p_e}{\sum_{j\in\text{TopK}} p_j}\ \text{for } e\in\text{TopK}. $$

權重彼此*競爭*：expert 共用一份固定預算，抬高一個就會壓低其他。乾淨，但這把各 expert 的 gate 綁在一起（也是 [訓練穩定性](training-stability.md)裡不穩定的來源之一）。

**Sigmoid gating**（DeepSeek-V3 與一些較新模型）：用 sigmoid *獨立*為每個 expert 計分，取 top-$k$，再把選中的歸一化。

$$ s_e = \sigma(W_r h), \quad g_e = \frac{s_e}{\sum_{j\in\text{TopK}} s_j}. $$

獨立計分把各 expert 解耦（沒有固定預算的零和競爭），這跟**細粒度 expert**與 **aux-loss-free** 的平衡偏差天生契合（見 [負載平衡](load-balancing.md)） —— 偏差可以直接加到 $s_e$ 上，而不會扭曲 softmax 的歸一化。現代大型 MoE 越來越愛用 sigmoid gating，正是這個原因。

!!! Note "top-k *之後*再歸一化"
    兩種變體都會把**選中的** $k$ 個 gate 重新歸一化成總和為 1，這樣層的輸出尺度就不取決於 哪些／多少個 expert 被啟用。要不要重新歸一化是一個真實的設計選擇：Switch（$k=1$）略過它， 大多數 $k\ge2$ 的模型則會做。

## 參考實作#1：可讀循環

最清楚、最正確的 MoE —— 容易驗證，而且刻意*不*追求快：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):
    """SwiGLU FFN, the standard expert."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MoELayerNaive(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2, gate="softmax"):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.top_k, self.gate = top_k, gate

    def forward(self, x):                       # x: [T, d_model] (tokens flattened)
        logits = self.router(x)                 # [T, E]
        if self.gate == "softmax":
            scores = logits.softmax(dim=-1)
        else:                                   # sigmoid
            scores = logits.sigmoid()
        topv, topi = scores.topk(self.top_k, dim=-1)   # [T, k]
        topv = topv / topv.sum(dim=-1, keepdim=True)   # renormalize selected
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):      # loop over experts
            mask = (topi == e)                  # [T, k] where this expert chosen
            tok, slot = mask.nonzero(as_tuple=True)    # tokens routed to e
            if tok.numel() == 0:
                continue
            out = expert(x[tok])                # run expert on its tokens only
            y.index_add_(0, tok, out * topv[tok, slot, None])
        return y
```

帶遮罩的 `for e in experts` 迴圈就是概念核心：**每個 expert 只在被 routing 到它的 token 上 執行。** 這是正確的、在 CPU 上拿來學也很好；但在 GPU 上，Python 迴圈加上不規則的 per-expert 批次很慢 —— 這正是後面 grouped GEMM 與 dispatch kernel 的全部動機。

## 參考實作#2：調度/排列形式

生產版本會按 expert 把 token 排序，於是每個 expert 看到的是一段*連續*的區塊 —— 正是 grouped GEMM 想要的排佈。這個「permute → grouped matmul → unpermute」的模式，就是 [MoE kernels](kernels.md) 要加速、[expert parallelism](systems-ep.md) 要透過網路傳送的東西。

```python
def moe_dispatch(x, topi, topv, experts, n_experts):
    """Permute tokens into per-expert contiguous groups, run, scatter back."""
    T, k = topi.shape
    # Flatten (token, slot) pairs; each pair is one token->expert assignment.
    flat_expert = topi.reshape(-1)                 # [T*k]
    flat_weight = topv.reshape(-1, 1)              # [T*k, 1]
    flat_token  = torch.arange(T, device=x.device).repeat_interleave(k)  # [T*k]

    order = torch.argsort(flat_expert)             # group by expert
    sorted_expert = flat_expert[order]
    sorted_token  = flat_token[order]
    counts = torch.bincount(sorted_expert, minlength=n_experts)  # tokens/expert

    x_sorted = x[sorted_token]                     # gather inputs, contiguous
    out_sorted = torch.empty_like(x_sorted)
    start = 0
    for e in range(n_experts):                     # each block is contiguous
        n = int(counts[e])
        if n:
            out_sorted[start:start+n] = experts[e](x_sorted[start:start+n])
        start += n

    out_sorted = out_sorted * flat_weight[order]   # apply gate weight
    y = torch.zeros_like(x)
    y.index_add_(0, sorted_token, out_sorted)      # scatter-add back (combine)
    return y
```

`argsort`／`bincount`／`index_add_` 這三件套，在 GPU 上對應的就是融合的 **permute** kernel； 而每個 expert 的連續區塊則餵進單一的 **grouped GEMM** 呼叫。我們已經把「參差不齊的遮罩迴圈」 變成了「排序 + 密集區塊」。

!!! Tip "兩種實作經測試一致"
    [`code/moe/test_moe.py`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/moe/test_moe.py) 斷言 `MoELayerNaive` 與 dispatch 形式在隨機輸入、兩種 gate、數種 $k$ 下產生相同輸出 （`torch.allclose`），並且單一 expert 在 $E{=}1$ 時退化成普通 FFN。執行 `pytest code/moe`。

## 將其放入 Transformer 塊中

直接替換：把密集 FFN 子層換成 MoE 層，attention 與 norm 保持不動。許多模型還會在路由 expert 之外加上一個共享 expert —— 見 [Routing 變體](routing-variants.md)：

```python
class MoEBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_experts, top_k):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)   # 來自基礎篇
        self.moe  = MoELayer(d_model, d_ff, n_experts, top_k)
        self.n1, self.n2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x
```

完整可訓練的版本 —— 含負載平衡損失與一個跑玩具任務的小型 training 迴圈 —— 放在 [`code/moe/train_tiny_moe.py`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/moe/train_tiny_moe.py)， 它也是[實戰專案](../capstones/build-moe.md)的起點。

## 要點

- MoE 層 = **expert + router + top-$k$ + 加權合併**。容量住在 expert 裡；router 只是一個極小的 線性映射。
- **Softmax gating** 讓 expert 爭搶固定預算；**Sigmoid gating** 獨立計分，和細粒度 expert 與 aux-loss-free 平衡更搭。
- 可讀的「帶遮罩 expert 迴圈」與生產的「permute → grouped GEMM → unpermute」算的是**同一件事**； 後者顯露出 kernel 與 EP 所利用的連續區塊。
- 把選中的 $k$ 個 gate 重新歸一化，讓輸出尺度與 routing 無關。

## 練習

!!! Tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 在 `MoELayerNaive` 裡實作 sigmoid gating，並用測試驗證。
2. 設 $k{=}1$（Switch 風格）並拿掉重新歸一化；輸出尺度會怎麼變、為什麼？
3. 在 CPU 上對 $T{=}8192$、$E{=}64$ 分析樸素迴圈與 dispatch 形式。時間都花在哪？預測兩者在 GPU 上的表現。
4. 加上一個共享 expert（總是套用），並確認不論 routing 如何，每步都有梯度流向它。

## 參考文獻

[1] N. Shazeer *et al.*, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *Proc. ICLR*, 2017.

[2] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity," *J. Mach. Learn. Res.*, vol. 23, no. 120, pp. 1-39, 2022.

[3] A. Q. Jiang *et al.*, "Mixtral of experts," *arXiv:2401.04088*, 2024.

[4] D. Dai *et al.*, "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," *arXiv:2401.06066*, 2024.

[5] T. Gale, D. Narayanan, C. Young, and M. Zaharia, "MegaBlocks: Efficient sparse training with mixture-of-experts," *arXiv:2211.15841*, 2022.
