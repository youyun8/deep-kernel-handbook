# Load balancing

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../moe-from-scratch/">MoE from scratch</a></span>
  <span class="chip"><strong>Code:</strong> <code>code/moe/load_balancing.py</code> (tested)</span>
</div>

A router left to its own devices **collapses**: a few experts get most of the
tokens, the rest atrophy, and you've paid for capacity you don't use — while the
popular experts become stragglers that bottleneck the whole layer. Load
balancing is the machinery that keeps tokens spread across experts. This page
covers the classic **auxiliary loss**, **expert capacity** and token dropping,
and the modern **aux-loss-free** (bias-based, DeepSeek-style) approach.

## Why routers collapse

Routing is a winner-take-all feedback loop. An expert that's slightly better
early gets more tokens → more gradient → improves faster → gets even more tokens.
Without a counter-pressure, the distribution concentrates. Two distinct harms:

- **Quality**: unused experts waste parameters; the model behaves like a much
  smaller one.
- **Systems**: with [expert parallelism](systems-ep.md), each expert lives on a
  GPU with a **fixed capacity** buffer. An overloaded expert **drops** tokens
  (they skip the layer) while underloaded GPUs idle. Imbalance directly becomes
  wasted hardware and a latency straggler.

So balancing is simultaneously a modeling objective and a systems requirement.

## The auxiliary load-balancing loss

The standard fix (GShard, Switch) adds a differentiable penalty that's minimized
when tokens are spread uniformly. For a batch of $T$ tokens and $E$ experts,
define per-expert:

- $f_e$ = fraction of tokens that *selected* expert $e$ (a hard count, fraction
  in top-$k$),
- $P_e$ = mean router *probability* assigned to $e$ over the batch (soft).

The Switch auxiliary loss is

$$ \mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \, P_e. $$

Intuition: $f_e$ is a non-differentiable count, but it's *multiplied* by the
differentiable mean-probability $P_e$. Gradients flow through $P_e$ and push
probability *away* from already-popular experts (high $f_e$). The sum
$\sum f_e P_e$ is minimized (for fixed $\sum f_e = k$, $\sum P_e = 1$) when both
are uniform, i.e. $f_e = k/E$, $P_e = 1/E$. The factor $E$ makes it scale-free;
$\alpha$ (typically $10^{-2}$) sets the strength.

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

You add $\mathcal{L}_{\text{aux}}$ to the language-modeling loss. The tension:
**too little $\alpha$ → collapse; too much $\alpha$ → the router is forced toward
uniform and ignores content, hurting quality.** Tuning $\alpha$ is finicky, which
motivates the aux-loss-free method below.

!!! note "Per-device vs global balancing"
    With EP, you often want balance *per device group*, not just globally — a
    globally-balanced but locally-skewed batch still drops tokens on a hot GPU.
    DeepSeek adds device-level and communication-balance terms; large models also
    apply the loss per micro-batch/sequence to avoid within-batch hotspots.

## Expert capacity, drop, and overflow

For efficient batched compute and fixed comm buffers, each expert accepts at most
a fixed number of tokens per batch — its **capacity**:

$$ C = \Big\lceil \text{capacity\_factor} \cdot \frac{k \cdot T}{E} \Big\rceil. $$

$kT/E$ is the average tokens-per-expert; the **capacity factor** (e.g. 1.0–2.0)
adds slack. Then:

- If more than $C$ tokens pick an expert, the **overflow** tokens are
  **dropped** — they bypass the MoE (the residual still carries them through).
- If fewer arrive, the buffer is **padded** with zeros (wasted compute).

This is a direct **quality vs throughput** knob. Capacity factor 1.0 wastes no
memory but drops tokens under any imbalance; 2.0 rarely drops but doubles the
buffer (and the GEMM padding). Drop rate is a key training metric — a healthy run
keeps it low *because* the aux loss is working, not because capacity is huge.

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

**Expert-choice routing** (see [routing variants](routing-variants.md)) sidesteps
drops entirely by having each expert *pick its top-$C$ tokens* — perfect balance
by construction, at the cost of some tokens getting more experts than others.

## Aux-loss-free balancing (the modern way)

DeepSeek-V3 popularized dropping the auxiliary loss almost entirely, replacing it
with a **per-expert bias** added to the routing scores *only for the top-$k$
selection* — not for the gate weights. The idea:

$$ \text{select TopK of } \big(s_e + b_e\big), \qquad \text{but weight by the original } s_e. $$

- Each expert has a scalar bias $b_e$ (not learned by gradient descent).
- After each step, **nudge** $b_e$ based on recent load: increase $b_e$ for
  under-loaded experts (make them more likely to be picked), decrease for
  over-loaded ones. A simple controller:

$$ b_e \leftarrow b_e + \gamma \cdot \text{sign}\big(\bar{c} - c_e\big), $$

where $c_e$ is expert $e$'s recent token count, $\bar c$ the mean, and $\gamma$ a
small update rate.

```python
@torch.no_grad()
def update_router_bias(bias, counts, gamma=1e-3):
    # counts: [E] tokens routed to each expert this step
    target = counts.float().mean()
    bias += gamma * torch.sign(target - counts.float())      # raise under-loaded
    return bias
```

Why this is nice:

- **No gradient interference.** The bias affects *selection*, not the gate
  weights that scale expert outputs, so it balances load **without distorting the
  loss landscape** — you don't trade quality for balance the way a heavy $\alpha$
  does.
- **It pairs with sigmoid gating** (independent per-expert scores), where adding
  a bias is clean.
- **Direct control.** It's a feedback controller on the actual quantity you care
  about (load), not a proxy penalty.

DeepSeek-V3 reports better balance *and* better quality than aux-loss tuning,
keeping only a tiny aux term to prevent pathological cases. This is now a
common default for large MoEs.

!!! warning "The bias is not a parameter"
    $b_e$ is updated by the controller, not by the optimizer, and is typically
    excluded from weight decay and gradient flow. Treat it like a running
    statistic (it must be synchronized across data-parallel ranks).

## Measuring balance

Track these every N steps; they tell you if routing is healthy:

- **Drop rate** — fraction of token-expert assignments dropped by capacity.
- **Max/mean load ratio** — $\max_e c_e / \bar c$; 1.0 is perfect, watch for >2.
- **Routing entropy** — $-\sum_e P_e \log P_e$; collapse shows as falling entropy.
- **Coefficient of variation** of $c_e$.

[`code/moe/load_balancing.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/load_balancing.py)
implements the aux loss, capacity/drop, and the bias controller, with tests
showing the bias controller drives a deliberately-skewed router back toward
uniform load over a few hundred steps.

## Key takeaways

- Routers collapse without counter-pressure; imbalance wastes parameters *and*
  hardware (dropped tokens, straggler GPUs).
- The **auxiliary loss** $\alpha E \sum_e f_e P_e$ pushes toward uniform routing
  but trades quality for balance via the finicky $\alpha$.
- **Expert capacity** caps tokens per expert; the **capacity factor** is a
  quality(drop)-vs-throughput(padding) knob.
- **Aux-loss-free** balancing adds a controller-updated **bias to the selection**
  (not the gate weight), balancing load without distorting gradients — the modern
  default, paired with sigmoid gating.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/moe.md). Try each exercise before expanding.

1. Derive that $\sum_e f_e P_e$ is minimized at the uniform distribution subject
   to $\sum f_e = k$, $\sum P_e = 1$.
2. For $T{=}4096$, $E{=}64$, $k{=}2$, capacity factor 1.25, compute $C$ and the
   drop rate if one expert receives 5% of all assignments.
3. Implement and tune the bias controller: starting from a skewed init, how do
   $\gamma$ and the gate type (softmax vs sigmoid) affect convergence to balance?
4. Compare aux-loss vs aux-loss-free on the toy MoE in `train_tiny_moe.py`:
   report final loss *and* load CV for each.

## References

- Shazeer et al. *Sparsely-Gated MoE.* 2017 (load-balancing loss origin).
- Lepikhin et al. *GShard.* 2020 (capacity, drop).
- Fedus, Zoph, Shazeer. *Switch Transformer.* 2021 (aux loss form used here).
- Zhou et al. *Mixture-of-Experts with Expert Choice Routing.* 2022.
- Wang et al. / DeepSeek-AI. *Auxiliary-Loss-Free Load Balancing* & *DeepSeek-V3.* 2024.
