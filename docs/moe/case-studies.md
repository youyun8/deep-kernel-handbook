# Case studies: real MoE architectures

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate → advanced</span>
  <span class="chip"><strong>Prereqs:</strong> all prior Part II pages</span>
  <span class="chip"><strong>Hardware:</strong> none</span>
</div>

Having built every component, we can now read real frontier MoEs and see *which*
design choices they made and *why*. We dissect four: **Mixtral** (the clean
classic), **DeepSeek-V3** (the systems-co-designed flagship), **Qwen-MoE** (the
pragmatic productionizer), and **Kimi K2/K2.5** (extreme sparsity at trillion
scale).

!!! warning "Verify the exact numbers against primary sources"
    Configurations below reflect the published technical reports as of early 2026
    and are meant to illustrate *design patterns*, not as a spec sheet. Newer
    point releases (e.g. a K2.5 refresh) may tweak counts. Always confirm exact
    figures in the linked papers/model cards before quoting them.

## Cross-model comparison

| Model | Total / active params | Experts (routed + shared) | Top-$k$ | Gate | Balancing | Attention | Notable systems |
|---|---|---|---|---|---|---|---|
| **Mixtral 8×7B** | ~47B / ~13B | 8, no shared | 2 | softmax | aux loss | GQA | clean SMoE, dense-ish |
| **DeepSeek-V3** | 671B / 37B | 256 + 1 shared | 8 | sigmoid | aux-loss-free bias | **MLA** | fp8 train, node-limited routing, DualPipe/DeepEP, MTP |
| **Qwen3-MoE (235B)** | ~235B / ~22B | 128 + 0–shared* | 8 | softmax | aux loss | GQA | productionized, broad tooling |
| **Kimi K2** | ~1T / ~32B | 384 + 1 shared | 8 | sigmoid | aux-loss-free style | MLA | extreme sparsity, MuonClip optimizer |

<small>*Qwen variants differ across versions; check the specific model card.</small>

The trend is unmistakable: from a few big experts (Mixtral) to **many
fine-grained experts + a shared expert + sigmoid gating + aux-loss-free
balancing** (DeepSeek, Kimi), with attention compressed via
[MLA](../foundations/attention-efficiency.md) to fight the KV-cache cost.

## Mixtral 8×7B — the clean classic

The model that made open SMoE mainstream. It's the
[from-scratch design](moe-from-scratch.md) almost verbatim:

- **8 experts per layer, top-2**, softmax gating with renormalization of the two
  selected gates. No shared expert.
- **GQA** attention to bound the KV cache.
- Trained with the standard **auxiliary load-balancing loss**.

Why it matters pedagogically: it's the minimal SMoE that works at scale, so it
isolates the core idea (sparse FFN routing) without the later complexity.
~47B params, but each token uses ~13B — the [why-sparsity](why-sparsity.md)
decoupling in its simplest form. Its limitation — only 8 coarse experts → only 28
expert combinations — is exactly what fine-grained designs improve on.

## DeepSeek-V3 — systems co-design end to end

The flagship example of *every* technique in this part working together. It is
worth studying because the modeling and the systems were designed jointly.

**Architecture**

- **DeepSeekMoE**: 256 fine-grained routed experts + **1 shared expert**, top-8
  routing — billions of expert combinations vs Mixtral's 28
  ([routing variants](routing-variants.md)).
- **Sigmoid gating** with **aux-loss-free** balancing: a per-expert
  [bias controller](load-balancing.md) adjusts selection without a heavy aux loss
  distorting the objective — better balance *and* quality.
- **Multi-head Latent Attention (MLA)**: compresses K/V into a low-rank latent,
  shrinking the [KV cache](../foundations/attention-efficiency.md) dramatically —
  critical for long context and cheap decode.
- **Multi-Token Prediction (MTP)**: extra heads predict multiple future tokens,
  improving data efficiency and enabling speculative-decoding-like inference.

**Systems** (the part most relevant here)

- **fp8 training** of the GEMMs with high-precision accumulation and bf16/fp32 for
  sensitive parts — the [numerics](../foundations/numerics-precision.md) recipe at
  the frontier.
- **Node-limited routing**: a token's experts span ≤4 nodes, bounding cross-node
  [all-to-all](systems-ep.md) traffic.
- **DualPipe + DeepEP**: a pipeline schedule and a communication library built to
  **overlap the all-to-all with compute** almost completely — the single biggest
  EP optimization, productionized.

671B total / **37B active**: you pay ~37B-model inference compute for far-larger-
model quality, *because* the systems work keeps the EP overhead small.

## Qwen-MoE — the pragmatic productionizer

The Qwen MoE line (Qwen1.5-MoE-A2.7B → Qwen2-57B-A14B → Qwen3-235B-A22B) shows
the design choices a team makes when **broad deployment and tooling** matter:

- **Fine-grained experts** (e.g. 128 routed, top-8) — adopting the
  many-small-experts lesson.
- **GQA** attention (well-supported everywhere) rather than the more exotic MLA.
- Standard **aux-loss** balancing — robust and easy to reason about, prioritizing
  reliability over squeezing the last bit of balance.
- Strong **ecosystem support** (quantized variants, serving integrations), which
  is itself a systems decision: an architecture is only as useful as the kernels
  and servers that run it.

The takeaway: there's a spectrum from "research-frontier exotic" (DeepSeek/Kimi)
to "battle-tested and portable" (Qwen), and the right point depends on whether you
control the whole stack.

## Kimi K2 / K2.5 — extreme sparsity at trillion scale

Moonshot's Kimi K2 pushes sparsity hard: on the order of **~1T total parameters
with only ~32B active**, via a very large fine-grained expert pool (~384 routed +
a shared expert, top-8) and MLA attention. Two things stand out:

- **A very high sparsity ratio** (active/total ≈ 3%) — further along the
  [why-sparsity](why-sparsity.md) trade than even DeepSeek-V3, betting heavily
  that cheap memory capacity buys quality.
- **Training-stability engineering** at this scale, including the **MuonClip**
  optimizer/clipping approach reported to tame the loss spikes and attention-logit
  growth that plague trillion-parameter MoE training — a direct response to the
  [stability pathologies](training-stability.md) we covered.

K2.5 is a subsequent refinement in the same family; treat specific counts as
version-dependent and confirm against Moonshot's current model card. The
*architectural lesson* is stable: extreme sparsity is viable if (and only if) the
balancing, stability, and serving-memory problems are solved together.

!!! tip "See it run, kernel by kernel"
    The [Anatomy of an MoE decode](decode-anatomy.md) page profiles a decode step
    of a model in exactly this class (MLA + fine-grained MoE + shared expert) and
    shows where the time actually goes — routing, grouped expert GEMMs, the shared
    expert, and the per-layer all-reduce — plus the fusion and concurrency levers
    that move it.

## What to take away

Reading these side by side, the modern MoE consensus is:

1. **Many fine-grained experts + a shared expert** beat few coarse experts.
2. **Sigmoid gating + aux-loss-free bias** balances better than heavy aux loss.
3. **Compress attention** (MLA/GQA) so the KV cache doesn't eat the savings.
4. **The systems work (all-to-all overlap, node-limited routing, fp8) is not
   optional** — it's what makes the FLOP-decoupling survive contact with real
   hardware.
5. **Stability engineering scales with size** — z-loss, fp32 routing, careful
   optimizers (MuonClip) matter more as $E$ and total params grow.

## Key takeaways

- Mixtral = the minimal clean SMoE (8 experts, top-2, softmax, aux loss).
- DeepSeek-V3 = the co-designed flagship (fine-grained + shared, sigmoid +
  aux-loss-free, MLA, fp8, node-limited routing, overlapped all-to-all).
- Qwen-MoE = pragmatic, portable, ecosystem-first.
- Kimi K2/K2.5 = extreme sparsity at ~1T params with heavy stability engineering.
- The field has converged on fine-grained + shared experts, sigmoid + bias
  balancing, compressed attention, and serious comm/overlap systems work.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/moe.md). Try each exercise before expanding.

1. For each model, compute the number of expert combinations $\binom{E}{k}$ and
   relate it to the fine-grained quality argument.
2. Estimate each model's KV-cache size per 1k tokens; quantify how much MLA saves
   over GQA over plain MHA.
3. DeepSeek-V3 vs Mixtral: compare active/total ratios and argue how that changes
   the serving memory vs latency trade.
4. Pick one model and map every component back to a Part II page (gate, balancing,
   routing variant, EP strategy, attention, precision).

## References

- Jiang et al. *Mixtral of Experts.* 2024.
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024 (MLA, aux-loss-free, fp8, DualPipe, MTP).
- Dai et al. *DeepSeekMoE.* 2024.
- Qwen Team. *Qwen2 / Qwen3 Technical Reports.* 2024–2025.
- Moonshot AI. *Kimi K2 Technical Report* (MuonClip, large-scale MoE). 2025.
