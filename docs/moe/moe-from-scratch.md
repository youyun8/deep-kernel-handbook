# MoE layer from scratch

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../why-sparsity/">why sparsity</a>, PyTorch</span>
  <span class="chip"><strong>Code:</strong> <code>code/moe/</code> (CPU, tested)</span>
</div>

Now we build a complete MoE layer in PyTorch — experts, router/gate, top-$k$
selection, and the weighted combine — starting from the cleanest correct version
and refactoring toward the dispatch form that the
[systems pages](systems-ep.md) and [kernels](kernels.md) optimize. Everything
here runs on CPU and is checked by tests in
[`code/moe/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code/moe).

## Anatomy of an MoE layer

An MoE FFN replaces the single feed-forward block with:

1. **$E$ experts** — independent FFNs (usually SwiGLU): $\text{expert}_e(h) = W^{down}_e\,\big(\text{SiLU}(W^{gate}_e h)\odot (W^{up}_e h)\big)$.
2. **A router/gate** — a linear map $h \mapsto W_r h \in \mathbb{R}^{E}$ producing one logit per expert.
3. **Top-$k$ selection** — pick the $k$ highest-scoring experts per token.
4. **Combine** — run the token through its $k$ experts and sum their outputs,
   weighted by the (normalized) gate scores.

For token representation $h\in\mathbb{R}^d$, with gate weights $g_e$:

$$ y = \sum_{e \in \text{TopK}(h)} g_e \cdot \text{expert}_e(h), \qquad g = \text{normalize}\big(\text{score}(W_r h)\big). $$

## Gating: softmax vs sigmoid

The gate score function matters more than it looks.

**Softmax gating** (GShard, Switch, Mixtral): softmax over all $E$ logits, then
take the top-$k$ and renormalize those $k$ to sum to 1.

$$ p = \text{softmax}(W_r h), \quad g_e = \frac{p_e}{\sum_{j\in\text{TopK}} p_j}\ \text{for } e\in\text{TopK}. $$

The weights are *competitive*: experts share a fixed budget, so boosting one
suppresses others. Clean, but ties the experts' gates together (a source of the
instabilities in [training stability](training-stability.md)).

**Sigmoid gating** (DeepSeek-V3, some recent models): score each expert
*independently* with a sigmoid, take top-$k$, then normalize the selected ones.

$$ s_e = \sigma(W_r h), \quad g_e = \frac{s_e}{\sum_{j\in\text{TopK}} s_j}. $$

Independent scoring decouples experts (no fixed-budget competition), which pairs
naturally with **fine-grained experts** and the **aux-loss-free** balancing bias
(see [load balancing](load-balancing.md)) — the bias can be added to $s_e$
without distorting a softmax normalization. Modern large MoEs increasingly use
sigmoid gating for exactly this reason.

!!! note "Normalize *after* top-k"
    Both variants renormalize the **selected** $k$ gates to sum to 1 so the
    layer's output scale doesn't depend on which/how-many experts fired. Whether
    to renormalize is a real design choice; Switch ($k=1$) skips it, most
    $k\ge2$ models do it.

## Reference implementation #1: the readable loop

The clearest correct MoE — easy to verify, deliberately *not* fast:

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

The `for e in experts` loop with masking is the conceptual core: **each expert
runs only on the tokens routed to it.** This is correct and runs fine on CPU for
learning, but on GPU the Python loop and ragged per-expert batches are slow —
which is the entire motivation for grouped GEMM and dispatch kernels later.

## Reference implementation #2: the dispatch/permute form

The production shape sorts tokens by expert so each expert sees a *contiguous*
block — exactly the layout a grouped GEMM wants. This "permute → grouped matmul →
unpermute" pattern is what [MoE kernels](kernels.md) accelerate and what
[expert parallelism](systems-ep.md) sends over the network.

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

The `argsort`/`bincount`/`index_add_` triple is the **permutation** the GPU
kernels fuse, and the per-expert blocks are the inputs to a single **grouped
GEMM** call. We've turned "ragged masked loops" into "sort + dense blocks."

!!! tip "These two implementations are tested to agree"
    [`code/moe/test_moe.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/test_moe.py)
    asserts `MoELayerNaive` and the dispatch form produce identical outputs
    (`torch.allclose`) for random inputs, both gates, and several $k$ — and that
    a single expert with $E{=}1$ reduces to a plain FFN. Run `pytest code/moe`.

## Putting it in a transformer block

Drop-in: replace the dense FFN sublayer with the MoE layer, keep attention and
norms. Many models add a **shared expert** (always-on dense FFN) alongside the
routed experts — see [routing variants](routing-variants.md):

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

The full, trainable version — with load-balancing loss and a tiny training loop
on a toy task — is in
[`code/moe/train_tiny_moe.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/train_tiny_moe.py)
and is the starting point for the [capstone](../capstones/build-moe.md).

## Key takeaways

- An MoE layer = **experts + router + top-$k$ + weighted combine**. Capacity is
  in the experts; the router is a tiny linear map.
- **Softmax gating** makes experts compete for a fixed budget; **sigmoid gating**
  scores them independently and pairs better with fine-grained experts and
  aux-loss-free balancing.
- The readable "loop over experts with masks" form and the production
  "permute → grouped GEMM → unpermute" form compute the **same thing**; the
  latter exposes the contiguous blocks that kernels and EP exploit.
- Renormalize the selected $k$ gates so output scale is routing-invariant.

## Exercises

1. Implement sigmoid gating in `MoELayerNaive` and verify against the test.
2. Set $k{=}1$ (Switch-style) and remove renormalization; what changes in output
   scale and why?
3. Profile the naive loop vs the dispatch form for $T{=}8192$, $E{=}64$ on CPU.
   Where does time go? Predict how each behaves on GPU.
4. Add a shared expert (always applied) and confirm gradients flow to it every
   step regardless of routing.

## References

- Shazeer et al. *Sparsely-Gated MoE.* 2017.
- Fedus, Zoph, Shazeer. *Switch Transformer.* 2021.
- Jiang et al. *Mixtral of Experts.* 2024.
- DeepSeek-AI. *DeepSeekMoE* and *DeepSeek-V3.* 2024.
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* 2022.
