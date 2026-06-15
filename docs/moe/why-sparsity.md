# Why sparsity

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../../foundations/transformer-systems/">transformer as a system</a></span>
  <span class="chip"><strong>Hardware:</strong> none</span>
</div>

Before building an MoE, it's worth being precise about *what problem sparsity
solves* and *what it costs*. The headline: MoE **decouples parameter count from
FLOPs per token**. This page makes that statement quantitative and honest about
the trade-offs, so the rest of Part II has a clear target.

## The dense bottleneck

In a dense transformer, the FFN dominates both parameters and FLOPs. From
[Part I](../foundations/transformer-systems.md): forward cost is $\approx 2P$
FLOPs per token, where $P$ is the parameter count — *every parameter touches
every token*. To make the model "know more" you grow $P$, and your compute bill
grows lockstep. Scaling laws say loss falls smoothly with both parameters and
data, but compute $\approx 6 P D$ (params × tokens) is the budget you actually
pay.

The question MoE asks: **can we add parameters without adding proportional
FLOPs?**

## Conditional computation

Yes — if each token only uses a *subset* of the parameters. Replace one FFN with
$E$ expert FFNs and a router that activates $k$ of them per token ($k \ll E$,
often $k=1$ or $2$). Then:

- **Total parameters** scale with $E$ (all experts exist, store knowledge).
- **Active parameters per token** scale with $k$ (only $k$ experts run).
- **FLOPs per token** track *active* params, not total.

Define the **sparsity ratio** $k/E$. A model with $E=64$ experts, $k=2$ has
$\sim$32× more FFN parameters than its active-compute-equivalent dense model. Real
examples: Mixtral 8×7B has 47B total but ~13B active; DeepSeek-V3 has 671B total
but only **37B active** per token. You pay ~37B-model FLOPs and get closer to a
671B-model's quality.

$$ \underbrace{P_{\text{total}}}_{\text{capacity / memory}} \;\propto\; E, \qquad \underbrace{P_{\text{active}}}_{\text{FLOPs, speed}} \;\propto\; k. $$

## The scaling argument

Empirically (Switch Transformer, GShard, and follow-ups), at a **fixed training
FLOP budget**, sparse models reach a given loss faster than dense ones, and at a
**fixed active-parameter budget**, adding experts keeps improving quality with
sub-linear extra compute. Intuitions for *why*:

- **Specialization.** Different experts can specialize (loosely — by token type,
  topic, or syntactic role), so the effective function class is richer than a
  single FFN of the same active size.
- **More parameters = more memorized knowledge** without more per-token math; the
  router acts as a learned sparse lookup.
- **Capacity where it's cheap.** Parameters are cheap to *store* (HBM/offload);
  FLOPs are expensive to *run*. MoE buys capacity in the cheap currency.

!!! note "It's not free quality"
    Sparse models are less *parameter-efficient* than dense ones — a 671B sparse
    model is not as good as a hypothetical 671B dense model would be. The win is
    **quality per FLOP** and **quality per dollar at inference**, not quality per
    parameter. You're trading abundant memory for scarce compute.

## What sparsity costs (the rest of Part II)

Conditional computation is not a free lunch; it imports a stack of systems
problems that dense models never face:

| Cost | Where it bites | Covered in |
|---|---|---|
| **Load imbalance** — routers collapse to a few popular experts | wasted experts, stragglers | [load balancing](load-balancing.md) |
| **Discrete routing** — top-k is non-differentiable, unstable | training divergence | [training stability](training-stability.md) |
| **All-to-all communication** — tokens must travel to their expert's GPU | network-bound layers | [systems & EP](systems-ep.md) |
| **Memory footprint** — all experts must be stored/loaded | huge HBM / offload | [inference & serving](inference-serving.md) |
| **Irregular compute** — variable tokens-per-expert breaks dense GEMM | kernel inefficiency | [kernels](kernels.md) |
| **Capacity & padding** — fixed buffers waste or drop tokens | quality/throughput trade | [load balancing](load-balancing.md), [systems](systems-ep.md) |

The art of MoE is paying these costs efficiently enough that the
FLOPs-decoupling win survives. Everything else in this part is about that.

## A back-of-envelope comparison

Compare a dense FFN vs an MoE FFN at the same *active* compute, hidden $d$,
$d_{ff}=4d$, $E$ experts, top-$k$:

- Dense FFN params: $\approx 8 d^2$ (up+down). FLOPs/token: $\approx 16 d^2$.
- MoE: params $\approx 8 d^2 E$; FLOPs/token $\approx 16 d^2 k$ (plus a tiny
  router $d\times E$). Same FLOPs as dense when $k=1$, $E$× the parameters.

So at $k=1$ you get $E$× the FFN capacity for the *same* FLOPs, plus a negligible
router cost — minus the systems overheads above. The whole engineering question
is how small you can make those overheads.

## Key takeaways

- MoE **decouples total parameters (capacity) from active parameters (FLOPs)**.
  Capacity scales with $E$; compute scales with $k$.
- The win is **quality per FLOP / per inference dollar**, achieved by buying
  capacity in cheap memory rather than expensive compute. It is *not* better
  quality-per-parameter.
- Sparsity imports load-balancing, communication, memory, and kernel-irregularity
  costs — the subject of the rest of Part II.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/moe.md). Try each exercise before expanding.

1. For $E=128$, $k=2$, $d=4096$, compute total vs active FFN parameters and the
   FLOPs-per-token ratio against the dense $E=1$ baseline.
2. DeepSeek-V3: 671B total, 37B active. What effective sparsity ratio is that?
   How does it compare to Mixtral 8×7B?
3. Argue both sides: when would you prefer a dense 37B model over a 671B/37B-active
   sparse one? Consider memory, latency at batch 1, and fine-tuning.
4. If experts are offloaded to CPU/NVMe and streamed in, which roofline axis
   (compute or bandwidth) becomes the new limiter? (Foreshadows
   [inference & serving](inference-serving.md).)

## References

- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* 2017.
- Lepikhin et al. *GShard.* 2020.
- Fedus, Zoph, Shazeer. *Switch Transformer.* 2021.
- Clark et al. *Unified Scaling Laws for Routed Language Models.* 2022.
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024.
