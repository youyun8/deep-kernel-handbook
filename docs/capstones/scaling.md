# Capstone: scaling it up

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> advanced</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../build-moe/">build a small MoE LM</a>, <a href="../../performance/distributed-training/">distributed training</a></span>
  <span class="chip"><strong>Hardware:</strong> multi-GPU to fully execute</span>
</div>

The [previous capstone](build-moe.md) built and optimized a single-GPU MoE LM.
This one is a **planning and implementation guide** for taking it multi-GPU,
applying the [parallelism techniques](../performance/distributed-training.md) and
the [expert-parallel all-to-all](../moe/systems-ep.md). It's structured as a
decision process you can follow for any model/cluster.

!!! warning "Partially hands-on"
    <span class="status-badge wip">SCAFFOLDED</span> Fully executing this needs a
    multi-GPU (ideally multi-node) cluster. The *reasoning, planning, and code
    structure* below are complete and runnable in skeleton form; the measured
    multi-node numbers are left for you to fill in on your hardware. Nothing here
    is hidden behind "TODO" — but the cluster-scale benchmark table is yours to
    populate.

## Step 1 — decide what to shard, and why

Walk the memory budget. For each GPU, estimate (bf16 + Adam): parameters (2
bytes), gradients (2), optimizer state (fp32 moments + master ≈ 12), and peak
activations. Then pick dimensions by what overflows first:

```text
fits on 1 GPU?                      → just DP/DDP for throughput
optimizer/grad state too big?       → ZeRO-1/2 (or FSDP)
parameters too big?                 → ZeRO-3/FSDP, or TP for the big matmuls
a single layer too big?             → TP (intra-node, NVLink/IF)
too many layers for memory?         → PP (cross-node)
experts dominate parameters? (MoE)  → EP (shard experts, all-to-all)
context too long?                   → SP/CP (shard the sequence)
```

For our MoE, experts are most of the parameters, so **EP is the headline
dimension**, composed with DP/ZeRO on the outside and (at larger scale) TP for
attention and PP across stages.

## Step 2 — map onto the topology

Place the chattiest collectives on the fastest links
([distributed training](../performance/distributed-training.md)):

- **TP** within a node (per-layer all-reduce needs NVLink/Infinity Fabric).
- **EP** across nodes is acceptable *if* you overlap the all-to-all and bound it
  with [node-limited routing](../moe/routing-variants.md); keep EP groups as
  local as the expert count allows.
- **PP** across nodes (only activations cross stage boundaries).
- **DP/ZeRO** outermost (gradient all-reduce/reduce-scatter overlaps with backward).

Write down the device mesh explicitly, e.g. for 16 GPUs (2 nodes × 8):
`DP=2 × PP=2 × TP=2 × EP=4` (degrees multiply to the device count along the right
axes; EP typically shares the data-parallel axis).

## Step 3 — implement EP for the MoE layer

The single-process [dispatch form](../moe/moe-from-scratch.md) becomes a real
all-to-all (from [Systems & EP](../moe/systems-ep.md)). Skeleton:

```python
# Per MoE layer, with an expert-parallel process group `ep_group`:
# 1. router -> top-k -> per-token destination expert (and thus dest rank)
# 2. sort local tokens by destination rank; compute send_counts
# 3. all_to_all_single(send_counts) -> recv_counts
# 4. all_to_all_single(recv_buf, send_buf, recv_counts, send_counts)  # dispatch
# 5. local grouped GEMM over resident experts on recv_buf
# 6. reverse all_to_all                                               # combine
# 7. unpermute + weighted sum into the residual
```

Use existing libraries where possible (Megatron-LM, DeepSpeed-MoE, or DeepEP for
the optimized all-to-all overlap) rather than hand-rolling the comm — but
understanding the seven steps is what lets you debug imbalance and stalls.

## Step 4 — overlap communication with compute

This is where MFU is won or lost ([Systems & EP](../moe/systems-ep.md)):

- Chunk the token batch and pipeline dispatch with the previous chunk's expert
  GEMM.
- Overlap the **shared-expert** FFN (dense, no comm) with the routed-expert
  all-to-all.
- Overlap DP gradient reduce-scatter with the backward pass.
- Profile the [timeline](../performance/profiling.md): serialized all-to-all
  appears as a gap with comm on the critical path — the #1 thing to fix.

## Step 5 — measure scaling

Report **strong** and **weak** scaling and watch for the usual cliffs:

| GPUs | Parallel config | Tokens/s | MFU | Notes |
|---|---|---:|---:|---|
| 1 | single | *baseline* | — | from previous capstone |
| 8 | EP=8 (1 node) | — | — | intra-node all-to-all (fast) |
| 16 | EP=8 × DP=2 | — | — | + cross-node DP |
| 16 | PP=2×TP=2×EP=4 | — | — | full 3D+EP |

(*Fill with your measurements; state hardware, shapes, and methodology.*) Expect
sub-linear scaling as communication grows; the gap between your curve and linear
is the comm you haven't hidden. Diagnose with the profiler, not guesswork.

## Step 6 — validate correctness at scale

Distributed bugs are subtle ([training stability](../moe/training-stability.md)):

- **Bias-controller counts must be synchronized** across DP ranks, or replicas
  balance toward different targets.
- **Loss/grad norms should match** a single-GPU run for a few steps with fixed
  seeds (gradient accumulation as a sanity check).
- **fp32 router math** everywhere, to avoid cross-rank routing disagreement from
  bf16 ties.

## Key takeaways

- Choose parallelism by **what overflows memory first**; for MoE, **EP** is the
  headline dimension, composed with DP/ZeRO/TP/PP.
- **Map collectives to links**: TP intra-node, EP/PP cross-node, DP outermost;
  write the device mesh explicitly.
- The EP MoE layer is the [seven-step](../moe/systems-ep.md) permute→all-to-all→
  grouped-GEMM→all-to-all→unpermute; **overlapping the all-to-all** is where MFU
  is won.
- **Measure strong/weak scaling correctly** and **validate correctness** (synced
  bias counts, fp32 routing, matched loss vs single-GPU).

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/capstones.md). Try each exercise before expanding.

1. For your model, compute per-GPU memory and choose a parallel config for 8 and
   64 GPUs; justify each dimension.
2. Implement EP for the MoE layer (or wire up DeepSpeed-MoE/Megatron) and verify
   loss matches the single-GPU run for 50 steps.
3. Profile and quantify the all-to-all overlap; report MFU before/after enabling
   chunked pipelining.
4. Produce the strong-scaling table and explain the deviation from linear.

## References

- [Distributed training](../performance/distributed-training.md) and
  [Systems & EP](../moe/systems-ep.md) (this capstone's foundations).
- Rajbhandari et al. *DeepSpeed-MoE.* 2022; Shoeybi et al. *Megatron-LM.* 2019.
- DeepSeek-AI. *DeepSeek-V3 / DeepEP / DualPipe.* 2024.
