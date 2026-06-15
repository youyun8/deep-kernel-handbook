# Capstone: build a small MoE LM end to end

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate → advanced</span>
  <span class="chip"><strong>Prereqs:</strong> all of Part II</span>
  <span class="chip"><strong>Code:</strong> <code>code/moe/train_tiny_moe.py</code> (CPU/GPU)</span>
</div>

This capstone assembles everything from [Part II](../moe/index.md) into a small
but complete MoE language model, trains it on a toy task, then optimizes it and
**reports measured speedups** with the
[profiling methodology](../performance/profiling.md). The reference is
[`code/moe/train_tiny_moe.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/train_tiny_moe.py),
which runs on CPU (small) and on a single GPU.

## Goal and design

Build a character-level (or tiny BPE) decoder-only LM where the FFN is replaced by
an MoE layer, small enough to train in minutes on a laptop/GPU. Config:

- $d_{model}=256$, $L=4$ layers, $n_{heads}=4$, $d_{ff}=512$ per expert.
- $E=8$ experts, top-$k=2$, **sigmoid gate** + optional **shared expert**.
- Balancing: [aux-loss-free bias controller](../moe/load-balancing.md) (+ small
  [z-loss](../moe/training-stability.md)).
- bf16 autocast on GPU; fp32 router math (the
  [precision discipline](../foundations/numerics-precision.md)).

## Step 1 — assemble the model

Reuse the components: causal attention (Part I), the
[MoE layer from scratch](../moe/moe-from-scratch.md), and the block wiring:

```python
class TinyMoELM(nn.Module):
    def __init__(self, vocab, d=256, L=4, n_heads=4, d_ff=512,
                 n_experts=8, top_k=2, shared=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(4096, d)
        self.blocks = nn.ModuleList(
            MoEBlock(d, d_ff, n_heads, n_experts, top_k, shared) for _ in range(L))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        aux = 0.0
        for blk in self.blocks:
            x, a = blk(x)          # block returns hidden + aux/z loss
            aux = aux + a
        return self.head(self.norm(x)), aux
```

## Step 2 — train with the balancing machinery

The training loop adds the MoE losses and updates the bias controller each step:

```python
for step, (xb, yb) in enumerate(loader):
    with torch.autocast(device, dtype=torch.bfloat16, enabled=cuda):
        logits, aux = model(xb)
        lm = F.cross_entropy(logits.flatten(0,1), yb.flatten())
    loss = lm + aux                       # aux = z-loss (+ tiny aux if used)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # MoE spikes!
    scaler.step(opt); scaler.update(); opt.zero_grad()
    update_all_router_biases(model)       # aux-loss-free controller step
    if step % 100 == 0:
        log_metrics(lm, load_cv(model), drop_rate(model), entropy(model))
```

Watch the [health metrics](../moe/load-balancing.md): loss down, **load CV** down
toward 0, **drop rate** low, **routing entropy** stable (not collapsing). If you
see spikes/NaNs, revisit [training stability](../moe/training-stability.md)
(z-loss, fp32 router, init, clipping).

??? success "What you should see"
    On the toy task, training loss should fall smoothly, the load coefficient of
    variation should drop from its initial value toward ~0.1–0.2 within a few
    hundred steps (the bias controller working), and drop rate should stay low.
    Disable balancing to *watch it collapse* — a few experts take everything,
    entropy crashes — which makes the balancing machinery's value concrete.

## Step 3 — optimize and measure

Now apply Part III. **Measure correctly** (warmup, CUDA events, sync, sweep,
locked clocks — see [profiling](../performance/profiling.md)) and report
before/after. Optimizations, in rough order of payoff for this model:

1. **Replace the Python expert loop with the dispatch form** (sort → grouped
   compute → scatter), per [MoE-from-scratch](../moe/moe-from-scratch.md).
2. **Use a grouped-GEMM kernel** for the experts ([Triton](../moe/kernels.md)) on
   GPU; fuse the gather/scatter into the kernel.
3. **bf16 autocast** + fused attention (FlashAttention) for the attention sublayer.
4. **CUDA graph** the decode step for generation (kills launch overhead).
5. **Quantize experts** (int8/fp8) for inference
   ([quantization](../performance/quantization.md)).

Report a table like this (fill with *your* measured numbers and state the
hardware/shapes — the values below are illustrative):

| Variant | Train step (ms) | Tokens/s | MFU | Notes |
|---|---:|---:|---:|---|
| Naive expert loop | *baseline* | — | — | Python loop, ragged batches |
| Dispatch (sorted) | — | — | — | contiguous per-expert blocks |
| + grouped-GEMM kernel | — | — | — | one launch, no padding |
| + fused gather/scatter | — | — | — | one HBM round-trip saved |
| + bf16 + flash-attn | — | — | — | |

The point of the capstone is the **discipline**: each row is a hypothesis ("the
expert loop is launch-bound") tested with a correct measurement against a roofline
target, not a vibe.

## Step 4 — sample from it

Confirm it learned something: generate text with the trained model (greedy or
temperature sampling). For a char-LM on a small corpus you should get locally
coherent text. The generation loop is a good place to add the
[inference optimizations](../performance/inference-optimization.md) (KV cache —
which you'll add — and CUDA graphs).

## Extensions

- Swap softmax↔sigmoid gating and aux-loss↔aux-loss-free; compare loss and load CV.
- Add [expert-choice routing](../moe/routing-variants.md) and observe the
  no-drop behavior.
- Scale $E$ up (fine-grained) and watch routing/comm overhead grow — motivating
  [Scaling it up](scaling.md).

## Key takeaways

- A complete MoE LM is just the Part II components wired into a transformer plus a
  training loop that carries the **balancing + stability** machinery.
- The optimization phase is an exercise in **measured, roofline-anchored**
  engineering: hypothesize the bottleneck, fix it, re-measure correctly, repeat.
- Toggling balancing on/off makes collapse — and the value of the load-balancing
  toolkit — viscerally clear.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/capstones.md). Try each exercise before expanding.

1. Run the reference, then remove balancing and quantify the collapse (entropy,
   load CV, final loss).
2. Implement the dispatch form and grouped GEMM; produce the before/after table
   with correct methodology.
3. Add a KV cache to the generation loop and measure decode latency vs the
   recompute-everything baseline.
4. Quantize the experts to int8 and report quality (val loss) vs speed.

## References

- All of [Part II](../moe/index.md) and [Part III](../performance/index.md).
- Karpathy. *nanoGPT* (the dense skeleton this extends).
- Gale et al. *MegaBlocks*; Jiang et al. *Mixtral*; DeepSeek-AI *DeepSeek-V3*.
