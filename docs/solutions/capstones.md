# Solutions — Part IV · Capstones

<div class="page-meta">
  <span class="chip"><strong>Covers:</strong> Build a small MoE LM, Scaling it up</span>
  <span class="chip"><strong>Use:</strong> attempt first, then check</span>
</div>

The capstone exercises are **build-and-measure** tasks against the toy model in
[`code/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code). There is no
single numeric answer; below is the expected result, the correct methodology, and
the traps to avoid for each.

## Build a small MoE LM

??? success "1 — Remove balancing and quantify the collapse"
    Train the reference, then disable the aux loss / bias controller. Expected
    **collapse signature**: routing **entropy crashes** (a few experts win),
    **load CV rises** sharply (from ~0.1–0.2 toward ≫1), several experts go
    **dead** (zero load), and **final val loss is worse**. Log all three
    (entropy, CV, loss) per step so the divergence from the balanced run is
    visible — this is the concrete demonstration that balancing is load-bearing,
    not cosmetic.

??? success "2 — Dispatch form + grouped GEMM, before/after table"
    Implement the permute→grouped-GEMM→unpermute path and compare to the naive
    masked loop. **Methodology that makes the table trustworthy:** identical
    weights/seed, warmup iters, `synchronize()` around timed loops, report
    median over many runs, and **verify outputs match** (max abs diff ~1e-3 bf16)
    before trusting the speed. Expect the dispatch form to win on GPU (contiguous
    grouped-GEMM vs many tiny masked matmuls), with the gap growing in $E$.

??? success "3 — KV cache vs recompute-everything"
    Add a KV cache to the generation loop: store K,V per layer, append one token's
    K,V each step instead of recomputing the whole prefix. Measure decode latency
    vs the recompute baseline. Expected: recompute is $O(N^2)$ over the generation
    (each new token re-attends to all prior tokens **and recomputes their K,V**),
    while cached decode is $O(N)$ — so the speedup **grows with sequence length**,
    from modest at short lengths to large (10×+) at long ones.

??? success "4 — int8 experts: quality vs speed"
    Quantize expert weights to int8, keep router + attention in bf16. Report **val
    loss** (quality) and **decode latency / weight bytes** (speed). Expected: val
    loss nearly unchanged (experts are quantization-tolerant — see
    [quantization ex. 4](performance.md#quantization-compression)), weight memory
    ~2× smaller, decode faster in the memory-bound regime. This reproduces, in
    miniature, the real MoE serving recipe.

## Scaling it up

??? success "1 — Per-GPU memory and a parallel config for 8 and 64 GPUs"
    Compute per-GPU state (16 B/param for bf16+Adam; see
    [distributed ex. 2](performance.md#distributed-training)) plus activations and
    KV. **8 GPUs (single node):** TP=8 over NVLink if a layer doesn't fit, else
    ZeRO-3/FSDP for a pure-DP memory cut; EP across the 8 for the MoE layer.
    **64 GPUs (multi-node):** compose — TP=8 **intra-node**, then PP and/or EP
    **across nodes**, DP/ZeRO on the outside. Justify each: TP where bandwidth is
    highest, PP/EP where it's lower, DP outermost (most tolerant of slow links).

??? success "2 — Implement EP and verify loss matches single-GPU"
    Wire up expert parallelism (hand-rolled all-to-all dispatch/combine, or
    DeepSpeed-MoE / Megatron). **Correctness check:** with the same seed/data, the
    EP run's loss must track the single-GPU run for ~50 steps to within
    floating-point noise. If it drifts, the usual culprits are **router math not
    in fp32** (cross-rank routing disagreement) or **un-synced balancing counts**
    — exactly the bugs from [training stability](../moe/training-stability.md).

??? success "3 — Quantify all-to-all overlap; MFU before/after"
    Profile with chunked pipelining **off**, then **on**. Off: the all-to-all
    appears as an exposed gap on the timeline (GPU idle during comm) → lower MFU.
    On: comm overlaps independent compute (shared expert / next-chunk attention) →
    the gap shrinks and **MFU rises**. Report MFU =
    $6P\cdot\text{tok/s}/\text{peak}$ for both; the delta is the value of overlap,
    the single most important EP optimization.

??? success "4 — Strong-scaling table and deviation from linear"
    Hold the problem size fixed, increase GPUs, plot speedup vs GPU count. It will
    **fall below the linear ideal** because: (a) **communication grows** with
    scale (all-reduce/all-to-all), (b) the **pipeline bubble** $\frac{P-1}{m+P-1}$
    eats compute, (c) **per-GPU work shrinks** until kernels are launch-/latency-
    bound (low occupancy), and (d) load imbalance. Attribute the gap to these
    terms — naming *why* scaling deviates is the real deliverable, not the table.
