# 從頭開始的 MoE 層

<div class="page-meta">
  <span class="chip"><strong>等級：</strong>中階</span>
  <span class="chip"><strong>先備知識：</strong> <a href="../why-sparsity/">為什麼稀疏</a>，PyTorch</span>
  <span class="chip"><strong>代碼：</strong> <code>code/moe/</code>（CPU，已測試）</span>
</div>

現在我們在 PyTorch 中建立一個完整的 MoE 層 - experts、router/gate、top-$k$
選擇和加權組合—從最乾淨的正確版本開始
並重構為調度表單
[systems pages](systems-ep.md) 和 [kernels](kernels.md) 最佳化。一切
這裡在 CPU 上運行並透過測試進行檢查
[`code/moe/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code/moe)。

## MoE 層的剖析

MoE FFN 將單一前饋塊替換為：

1. **$E$ experts**— 獨立 FFN（通常為 SwiGLU）：$\text{expert}_e(h) = W^{down}_e\,\big(\text{SiLU}(W^{gate}_e h)\odot (W^{up}_e h)\big)$。
2. **router/gate**— 線性映射 $h \mapsto W_r h \in \mathbb{R}^{E}$，每個 expert 產生一個 logit。
3. **頂級 $k$ 選擇**— 選擇每個 token 得分最高的 $k$ experts。
4. **組合**— 透過 $k$ experts 運作 token 並對它們的輸出求和，
   由（標準化）門分數加權。

對於 token 表示 $h\in\mathbb{R}^d$，閘權重為 $g_e$：

$$ y = \sum\_{e \in \text{TopK}(h)} g_e \cdot \text{expert}\_e(h), \qquad g = \text{normalize}\big(\text{score}(W_r h)\big). $$

## 閘控：softmax 與 sigmoid

門得分函數比它看起來更重要。

**Softmax 閘控**（GShard、Switch、Mixtral）：對所有 $E$ logits 進行 softmax，然後
取頂部的 $k$ 並將這些 $k$ 重新歸一化，使其總和為 1。

$$ p = \text{softmax}(W*r h), \quad g_e = \frac{p_e}{\sum*{j\in\text{TopK}} p_j}\ \text{for } e\in\text{TopK}. $$

重量具有*競爭力*：experts 共享固定預算，因此增加一個
壓制別人。乾淨，但將 experts 的門連接在一起（
[training stability](training-stability.md) 中的不穩定性）。

**Sigmoid 門控**（DeepSeek-V3、一些最新型號）：對每個 expert 進行評分
*獨立地*使用 sigmoid，取 top-$k$，然後將所選的進行標準化。

$$ s*e = \sigma(W_r h), \quad g_e = \frac{s_e}{\sum*{j\in\text{TopK}} s_j}. $$

獨立評分解耦 experts（無固定預算競賽），這對
自然地具有**細粒度 experts**和**aux-loss-free**平衡偏差
（請參閱 [負載平衡](load-balancing.md)）— 偏移可以加入到 $s_e$
不扭曲 softmax 歸一化。現代大型 MoE 越來越多使用
sigmoid 門控正是因為這個原因。

!!! note "*後*top-k 標準化"
    兩種變體都將**選定的**$k$ 閘重新歸一化，使其總和為 1，因此
    層的輸出比例不取決於發射哪一個/多少個 experts。是否
    重新規範化是一個真正的設計選擇；開關（$k=1$）跳過它，大多數
    $k\ge2$ 型號可以做到。

## 參考實作#1：可讀循環

最清晰正確的 MoE — 易於驗證，故意「不」快：

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

帶有屏蔽的 `for e in experts` 循環是概念核心：**每個 expert
僅在路由到它的 tokens 上運行。**這是正確的，並且在 CPU 上運行良好
學習，但在 GPU 上，Python 循環和不規則的 per-expert 批次速度很慢 —
這就是後來分組 GEMM 並派遣 kernels 的全部動機。

## 參考實作#2：調度/排列形式

生產形狀按 expert 對 tokens 進行排序，因此每個 expert 都會看到一個*連續的*
block — 正是分組 GEMM 想要的佈局。這個「排列 → 分組 matmul →
unpermute」模式是什麼 [MoE kernels](kernels.md) 加速什麼
[expert parallelism](systems-ep.md) 透過網路發送。

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

`argsort`/`bincount`/`index_add_` 三元組是 GPU 的**排列**
kernels 熔斷器和每個 expert 區塊是單一**分組的輸入
GEMM**致電。我們已將「參差不齊的屏蔽循環」轉變為「排序+密集區塊」。

!!! tip "這兩個實現經過測試一致"
    [`code/moe/test_moe.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/test_moe.py)
    斷言 `MoELayerNaive` 和調度形式產生相同的輸出
    (`torch.allclose`) 用於隨機輸入、兩個閘和幾個 $k$ — 並且
    單一 expert 與 $E{=}1$ 簡化為普通 FFN。運行 `pytest code/moe`。

## 將其放入 Transformer 塊中

Drop-in：以 MoE 層取代密集的 FFN 子層，保留 attention 和
規範。許多型號在
路由 experts — 請參閱 [routing variants](routing-variants.md)：

```python
class MoEBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_experts, top_k):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)   # from Part I
        self.moe  = MoELayer(d_model, d_ff, n_experts, top_k)
        self.n1, self.n2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x
```

完整的可訓練版本 - 具有負載平衡損耗和微小的 training 循環
執行一項玩具任務 — 處於
[`code/moe/train_tiny_moe.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/train_tiny_moe.py)
是 [capstone](../capstones/build-moe.md) 的起點。

## 要點

- MoE 層 =**experts + router + 頂部 $k$ + 加權聯合收割機**。容量為
  在 experts 中；router 是一個微小的線性地圖。 -**Softmax 門控**使 experts 爭奪固定預算；**乙狀結腸門控**
  對它們進行獨立評分，並與細粒度 experts 更好地配對，
  輔助無損耗平衡。
- 可讀的「帶遮罩的 experts 循環」表格和製作
  「排列 → 分組 GEMM→ 取消排列」形式計算**相同的東西**；的
  後者揭露了 kernels 和 EP 利用的連續區塊。
- 重新規範化選定的 $k$ 閘，以便輸出比例是 routing 不變的。

## 練習

!!! tip "解決方案"
    參考解答位於 [解答頁](../solutions/moe.md) 上。請先嘗試每個練習，再展開解答。

1. 在`MoELayerNaive`中實作 sigmoid 門控並透過測試進行驗證。
2. 設定$k{=}1$（Switch-style）並去掉重整化；輸出有什麼變化
   規模以及為什麼？
3. 分析 CPU 上 $T{=}8192$、$E{=}64$ 的樸素循環與調度形式。
   時間都去哪了？預測每個在 GPU 上的行為。
4. 新增共享的 expert（始終應用）並確認梯度流向它
   步驟與 routing 無關。

## 參考文獻

- 沙澤爾等人。 _稀疏門控 MoE。 _ 2017 年。
- 費杜斯、佐夫、沙吉爾。 _開關 Transformer。 _ 2021 年。
- 江等人。 _experts 的混合。 _ 2024 年。
- DeepSeek-AI。 _DeepSeekMoE_ 和 _DeepSeek-V3。 _ 2024 年。
- 大風等人。 _MegaBlocks：高效率稀疏 training 與 experts 的混合。 _ 2022 年。
