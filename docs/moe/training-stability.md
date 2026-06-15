# Training stability for MoE

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate → advanced</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../load-balancing/">load balancing</a>, <a href="../../foundations/numerics-precision/">numerics</a></span>
  <span class="chip"><strong>Hardware:</strong> none</span>
</div>

MoEs are harder to train than dense models because **routing is discrete and
self-reinforcing**. Small numerical perturbations flip routing decisions, which
change which parameters get gradients, which changes routing again. This page
covers the specific pathologies and the standard fixes: **router z-loss**,
**initialization**, precision discipline for the router, and a few practical
guardrails.

## Why MoE training is touchy

Three coupled issues:

1. **Discreteness.** Top-$k$ is a hard, non-differentiable selection. The gate
   weights are differentiable, but *which* experts run is not. A tiny change in a
   logit can move a token to a different expert — a discontinuous jump in the
   loss surface.
2. **Self-reinforcement.** As in [load balancing](load-balancing.md), routing is
   a positive-feedback loop prone to collapse.
3. **Logit blow-up.** Nothing inherently bounds the router logits. If they grow
   large, the softmax saturates (routing becomes nearly one-hot and *frozen* — no
   gradient to escape a bad assignment), and large logits interact badly with
   [low precision](../foundations/numerics-precision.md).

The result, untreated: loss spikes, NaNs, "dead" experts, and routing that
locks in early and never recovers.

## Router z-loss

The router z-loss (from ST-MoE) directly penalizes large router logits to keep
the softmax in a sane regime. For logits $x \in \mathbb{R}^{E}$ per token:

$$ \mathcal{L}_{z} = \frac{\beta}{T}\sum_{t=1}^{T}\Big(\log\sum_{e=1}^{E} e^{x_{t,e}}\Big)^{2}. $$

The term $\log\sum_e e^{x_e}$ is the log-partition (the softmax normalizer);
squaring and penalizing it pulls the logits toward small magnitudes. Effects:

- Keeps `exp` arguments small → **no overflow** in bf16/fp16, more stable softmax.
- Prevents the gate from saturating to a frozen one-hot → routing stays
  *plastic* and can correct early mistakes.
- Tiny coefficient ($\beta \approx 10^{-3}$) — it's a regularizer, not a primary
  objective.

```python
def router_z_loss(logits, beta=1e-3):
    # logits: [T, E] pre-softmax router outputs (compute in fp32!)
    logsumexp = torch.logsumexp(logits.float(), dim=-1)      # [T]
    return beta * (logsumexp ** 2).mean()
```

Total MoE training loss:

$$ \mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{aux}} + \beta\,\mathcal{L}_{z}, $$

(with $\mathcal{L}_{\text{aux}}$ optionally replaced by the
[aux-loss-free bias](load-balancing.md)). z-loss is kept even in aux-loss-free
recipes — it addresses logit magnitude, a different problem than balance.

## Precision discipline for the router

This is where [numerics](../foundations/numerics-precision.md) and MoE collide.
Routing is a *discrete* decision driven by *small differences* between logits, so
rounding noise can flip assignments and destabilize the feedback loop.

- **Compute router logits, softmax/sigmoid, and the aux/z losses in fp32**, even
  in a bf16 model. The router matrix is tiny — the fp32 cost is negligible and
  the stability gain is large.
- **The bias controller's counts must be reduced in fp32** and synchronized
  across data-parallel ranks, or different ranks balance toward different targets.
- Subtract-the-max before any softmax (it's free via `logsumexp`/`log_softmax`).

!!! warning "A classic silent bug"
    Routing in bf16 can make two experts' logits tie (bf16 has ~7 mantissa
    bits), and the tie-break (argmax/topk) becomes arbitrary and rank-dependent
    under data parallelism — different replicas route the same token differently,
    corrupting the balancing statistics. fp32 router math avoids it.

## Initialization

Routing is most fragile **early**, before experts have differentiated. Good
practice:

- **Small router init.** Initialize the router weights with a small scale (e.g.
  $\text{std}\sim 0.01$–$d^{-1/2}$ with extra shrinkage) so initial logits are
  near zero → near-uniform routing → every expert gets gradient and differentiates
  before the loop can collapse. (Switch used a truncated-normal with a reduced
  init scale for exactly this.)
- **Standard expert init.** Experts are normal FFNs; init them as you would a
  dense FFN.
- **Warm-up the router / capacity.** A larger capacity factor early (fewer drops
  while routing is random) and LR warm-up reduce early instability.
- **Shared expert as a stabilizer.** A [shared expert](routing-variants.md)
  guarantees a dense gradient path from step 0, smoothing the cold-start.

## Other practical guardrails

- **Gradient clipping** (global norm) — MoE loss spikes are common; clipping
  prevents one spike from wrecking the run.
- **Balance the aux loss per micro-batch / per sequence**, not just globally, to
  avoid within-batch hotspots that global stats hide.
- **Monitor dead experts** (zero load for many steps) and routing entropy; a
  sudden entropy drop is the early-warning sign of collapse.
- **Jitter / noise on logits** (older recipes, e.g. Switch's multiplicative
  input jitter) adds exploration so routing doesn't lock in — used less with
  z-loss + good init, but still a tool.
- **Keep the optimizer state fp32** (Adam moments), standard but doubly important
  when the loss surface is rough.

## Diagnosing a sick MoE run

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss spikes / NaN early | router logit blow-up; fp16 overflow | add/raise z-loss; route in fp32; clip grads |
| A few experts get all tokens | weak balancing | raise $\alpha$ or enable bias controller |
| Dead experts that never recover | early collapse, saturated gate | smaller router init, larger early capacity, z-loss |
| Different replicas disagree on routing | bf16 router ties | fp32 router math; sync bias counts |
| High drop rate | capacity too low / imbalance | raise capacity factor; fix balancing |

## Key takeaways

- MoE instability stems from **discrete, self-reinforcing routing** and
  **unbounded router logits**.
- **Router z-loss** $\beta(\log\sum e^{x})^2$ keeps logits small → stable softmax,
  no overflow, plastic routing. Keep it even in aux-loss-free setups.
- **Do all router math in fp32** — discrete decisions on small logit differences
  are precision-sensitive, and bf16 ties cause cross-replica disagreement.
- **Small router init + (optional) shared expert + warm-up** make the fragile
  cold-start survivable; clip gradients and monitor entropy/dead experts.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/moe.md). Try each exercise before expanding.

1. Show that minimizing $\mathcal{L}_z$ shrinks $\|x\|$ and bounds the softmax
   away from one-hot. What happens to the entropy of the routing distribution?
2. Construct router logits where bf16 rounding flips the argmax but fp32 does not.
3. On the toy MoE, train with and without z-loss using deliberately large router
   init; compare loss-spike frequency and dead-expert counts.
4. Why does a shared expert ease cold-start? Trace the gradient path on step 0
   for a token whose routed experts are all near-identical.

## References

- Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (router z-loss). 2022.
- Fedus, Zoph, Shazeer. *Switch Transformer* (init, jitter, selective fp32). 2021.
- Lepikhin et al. *GShard.* 2020.
- DeepSeek-AI. *DeepSeek-V3* (bias controller, stability recipe). 2024.
- Micikevicius et al. *Mixed Precision Training.* 2017.
