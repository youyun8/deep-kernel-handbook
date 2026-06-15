# Solutions — Part II · Mixture-of-Experts

<div class="page-meta">
  <span class="chip"><strong>Covers:</strong> all nine MoE pages</span>
  <span class="chip"><strong>Use:</strong> attempt first, then check</span>
</div>

Worked answers to the exercises across [Part II](../moe/index.md). Several refer
to the toy model in [`code/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code);
where a derivation has a clean closed form we give it, otherwise we give the
method and the expected qualitative result.

## Why sparsity

??? success "1 — Total vs active params, FLOPs ratio ($E{=}128, k{=}2, d{=}4096$)"
    One FFN ≈ $8d^2$ params (up $4d^2$ + down $4d^2$, with $d_{ff}=4d$).

    - **Total experts:** $128 \times 8d^2 = 1024d^2 \approx 1.72\times10^{10}$.
    - **Active per token:** $2 \times 8d^2 = 16d^2 \approx 2.68\times10^{8}$.

    FLOPs scale with *active* params, so the FLOPs-per-token ratio vs dense
    ($E{=}1$, i.e. $8d^2$ active) is $16d^2/8d^2 = \mathbf{2\times}$ — you pay for
    $k=2$ experts. But you **store** $128\times$ the FFN params: huge capacity,
    constant-ish compute. That gap is the entire value proposition of sparsity.

??? success "2 — DeepSeek-V3 sparsity vs Mixtral"
    DeepSeek-V3: $37/671 \approx 5.5\%$ active (≈ **18× sparsity**). Mixtral
    8×7B: top-2 of 8 experts, $\approx 13/47 \approx 28\%$ active (≈ 3.6×). V3 is
    far sparser — many fine-grained experts with low activation — which is the
    modern trend: more, smaller experts give more combinations (next exercise)
    at the same active cost.

??? success "3 — When to prefer dense 37B over 671B/37B-active sparse"
    Same active FLOPs, but the sparse model needs **~18× the memory** to hold all
    experts resident. Prefer **dense** when: (a) memory is tight (single GPU /
    edge); (b) **batch-1 latency** matters and you can't amortize expert loading
    — sparse decode may touch many experts for few tokens, hurting locality;
    (c) **fine-tuning** on a small dataset, where most experts get little
    gradient and risk staleness. Prefer **sparse** for high-throughput serving and
    pretraining quality-per-FLOP.

??? success "4 — Offloaded experts: which roofline axis binds?"
    Stream experts from CPU/NVMe over PCIe (~tens of GB/s) vs HBM (TB/s) — a
    1–2 order-of-magnitude bandwidth cliff. The limiter becomes **bandwidth**
    (PCIe/NVMe transfer of expert weights), not compute. Streaming is only hidden
    if tokens-per-expert is large enough that the GEMM time exceeds the transfer
    time — the exact condition derived in [inference & serving](#inference-serving)
    exercise 2.

## MoE layer from scratch

??? success "1 — Sigmoid gating in `MoELayerNaive`"
    Replace `softmax(logits)` with `sigmoid(logits)` for the top-$k$ gate values;
    unlike softmax, sigmoid gates are **independent** (they don't sum to 1), so
    each expert's weight is an absolute "should this fire" score. After selecting
    top-$k$, renormalize the chosen gates if you want a convex combination. Verify
    output shape/finiteness against the test; expect comparable loss but different
    balancing dynamics (sigmoid + bias controller is the DeepSeek-V3 recipe).

??? success "2 — $k{=}1$ (Switch) without renormalization"
    With $k=1$ and no renorm, the output is $g\cdot \text{expert}(h)$ where
    $g=\text{softmax}(\cdot)_{\text{top1}} \in (0,1)$. So the expert output is
    **scaled down by $g<1$**, shrinking activation magnitude (and coupling the
    residual scale to gate confidence). Switch keeps the gate as a multiplier on
    purpose — it provides the differentiable signal to the router — but you must
    account for the reduced scale (init/LR), or renormalize so the kept gate is 1.

??? success "3 — Naive loop vs dispatch form ($T{=}8192, E{=}64$, CPU)"
    The **naive loop** iterates experts, masking $T$ tokens each time → it touches
    all $T$ tokens $E$ times (lots of wasted masked work, but trivially
    vectorized). The **dispatch form** permutes tokens into per-expert contiguous
    groups and runs each expert on only its tokens → far less wasted compute, but
    pays a gather/scatter. On CPU the dispatch form usually wins once $E$ is large
    (the naive $O(TE)$ masking dominates). On **GPU** the gap widens: the dispatch
    form's contiguous grouped-GEMM is what the hardware wants, while the naive
    loop launches many tiny masked matmuls (launch + low-occupancy bound).

??? success "4 — Add a shared expert; confirm gradients every step"
    Add $y = \text{shared}(h) + \sum_{e\in\text{TopK}} g_e\,\text{expert}_e(h)$.
    Because `shared(h)` is on the path for **every** token regardless of routing,
    `shared.weight.grad` is non-`None` and non-zero on every step (check after
    `backward`). Routed experts only get gradient when selected; the shared expert
    is the always-on dense path that stabilizes cold-start (see training
    stability ex. 4).

## Load balancing

??? success "1 — Uniform distribution minimizes $\sum_e f_e P_e$"
    Minimize $\sum_e f_e P_e$ subject to $\sum f_e = k$, $\sum P_e = 1$. The
    Switch aux loss multiplies the fraction of **tokens** routed to $e$ ($f_e$) by
    the mean **gate prob** to $e$ ($P_e$); it is minimized when load is spread.
    Formally, by rearrangement/Cauchy–Schwarz the coupled sum is smallest when
    both vectors are flat: $f_e = k/E$ and $P_e = 1/E$ for all $e$, giving
    $\sum_e f_e P_e = E\cdot\frac{k}{E}\frac{1}{E} = k/E$. Any concentration
    (some $f_e,P_e$ large together) raises the product — so descending this loss
    pushes routing toward uniform. *(Aux loss uses $f$ = hard counts, $P$ =
    differentiable probs, so the gradient flows through $P$.)*

??? success "2 — Capacity and drop rate ($T{=}4096, E{=}64, k{=}2$, cf 1.25)"
    Expected tokens/expert $= Tk/E = 4096\cdot2/64 = 128$. Capacity
    $C = \lceil \text{cf}\cdot Tk/E\rceil = \lceil 1.25\times128\rceil = 160$.
    If one expert gets 5% of all $Tk = 8192$ assignments $= 410$ tokens, it
    overflows by $410-160 = 250$; those are **dropped**. Drop rate from that one
    expert $\approx 250/8192 \approx 3.0\%$ of all assignments (others assumed
    within capacity). Raising cf to 2.0 ($C=256$) still drops $410-256=154$ —
    showing capacity alone can't fix a badly skewed router; you need balancing.

??? success "3 — Tuning the bias controller"
    The aux-loss-free controller nudges a per-expert bias: $b_e \leftarrow b_e +
    \gamma\,\text{sign}(\text{target} - \text{load}_e)$. **Larger $\gamma$** →
    faster convergence but oscillation/overshoot around balance; **smaller
    $\gamma$** → smooth but slow. **Sigmoid** gates respond more cleanly than
    **softmax** because sigmoid scores are independent — shifting one bias doesn't
    rescale the others through a shared normalizer, so the controller's per-expert
    adjustments don't fight each other. Expect load CV to fall toward ~0.1–0.2
    within a few hundred steps at a well-tuned $\gamma$.

??? success "4 — Aux-loss vs aux-loss-free on the toy MoE"
    Run `train_tiny_moe.py` both ways. Expected: **aux-loss-free** reaches similar
    or slightly **lower final LM loss** at the same load CV, because it balances
    via a bias on routing *counts* without adding a gradient term that competes
    with the LM objective (the aux loss slightly distorts the loss surface). Both
    should drive load CV down to ~0.1–0.2; the aux-loss-free run avoids the
    tug-of-war between balance and quality. Report both final loss and CV side by
    side — that pairing is the whole point.

## Routing variants

??? success "1 — Expert combinations and fine-grained gains"
    Combinations $= \binom{E}{k}$:

    - $(8,2): \binom{8}{2} = 28$
    - $(64,8): \binom{64}{8} \approx 4.4\times10^{9}$
    - $(256,8): \binom{256}{8} \approx 4.1\times10^{14}$

    Each token selects one combination, so more, finer experts give exponentially
    more specialized "mixtures" at the **same active $k$** — the fine-grained
    expert argument (DeepSeekMoE). Combinatorial capacity, constant compute.

??? success "2 — Why expert-choice breaks autoregressive decode"
    Expert-choice has each expert pick its top-$C$ tokens **from the whole batch**
    — it assumes all tokens are present at once (true in training/prefill). In
    autoregressive **decode** you generate one token at a time; an expert can't
    "choose" among future tokens that don't exist yet, and the top-$C$ over a
    batch-of-one is meaningless. Token-choice routes each new token independently,
    so it's the natural inference-time scheme.

??? success "3 — Shared expert effect on stability"
    Add a shared expert to the toy MoE and compare runs. Expected: **lower loss
    variance** (fewer spikes) and equal-or-better final loss, because the shared
    path delivers a dense gradient every step, so early routing mistakes don't
    starve the model of signal. The effect is largest early in training (cold
    start) and shrinks as routed experts differentiate.

??? success "4 — Growing $E$ shrinks all-to-all messages"
    At fixed total params, more experts ⇒ each expert smaller ⇒ each token's
    payload to a given expert-GPU is unchanged, but tokens **spread across more
    destinations**, so each all-to-all message gets *smaller and more numerous*.
    Many tiny messages hurt network efficiency (latency- and overhead-bound, poor
    link utilization). **Node-limited routing** caps how many nodes a token's
    experts span, keeping messages large enough to stay bandwidth-bound rather
    than latency-bound.

## Training stability

??? success "1 — $\mathcal{L}_z$ shrinks $\|x\|$ and bounds softmax off one-hot"
    $\mathcal{L}_z = \beta(\log\sum_e e^{x_e})^2$. The log-sum-exp grows with the
    logits' magnitude, so penalizing its square pulls logits toward small values
    → $\|x\|$ shrinks. Smaller logits ⇒ softmax closer to uniform ⇒ **higher
    routing entropy** and probabilities bounded away from one-hot (0/1). Concretely
    it keeps the gate **plastic**: a near-saturated softmax has vanishing gradient
    and can't escape a bad assignment; z-loss prevents that frozen state.

??? success "2 — bf16 flips argmax, fp32 doesn't"
    bf16 has 8 mantissa bits → ULP near 1.0 is $2^{-7}\approx0.0078$. Take logits
    $x_1 = 1.0000,\ x_2 = 1.0039$ (true argmax = 2). Both round to the **same**
    bf16 value $1.0$, so the bf16 argmax/topk tie-break is arbitrary (and
    rank-dependent under data parallelism), while fp32 (23 mantissa bits) keeps
    them distinct and picks expert 2. This is the "silent bug": replicas disagree
    on routing and corrupt the shared balancing counts.

??? success "3 — Large router init, with/without z-loss"
    On the toy MoE, initialize the router with a deliberately large scale and
    train both ways. Expected: **without z-loss**, logits blow up early → frequent
    loss spikes/NaNs and several **dead experts** (saturated gate locks routing).
    **With z-loss**, logits stay bounded → spikes rare, dead-expert count near
    zero, smoother loss. This makes z-loss's role concrete: it's cheap insurance
    against logit blow-up.

??? success "4 — Why a shared expert eases cold-start"
    On step 0 the routed experts are near-identical (undifferentiated), so the
    routed gradient is noisy and nearly symmetric — little signal to specialize.
    The shared expert sits **outside** the top-$k$ selection, so $\partial
    \mathcal{L}/\partial\,\text{shared}$ is a full dense gradient on **every**
    token from step 0, giving the model a reliable learning path while routing
    sorts itself out. Trace: $y = \text{shared}(h)+\sum g_e e(h)$ →
    $\nabla_{\text{shared}}$ never depends on the (random) routing decision.

## Systems & expert parallelism

??? success "1 — All-to-all bytes vs expert GEMM time ($T{=}4096/\text{GPU}, d{=}4096$)"
    Per all-to-all, each GPU moves ≈ its tokens × $d$ × 2 B $= 4096\times4096\times2
    \approx 3.4\times10^{7}$ B = 34 MB; **two** per layer (dispatch+combine) → ~67
    MB/GPU/layer. Over 60 layers ≈ **4 GB/GPU** of traffic. On NVLink (~300 GB/s
    intra-node) that's ~13 ms; cross-node IB (~50 GB/s) ~80 ms.
    Expert GEMM time: per token $2\cdot k\cdot 8d^2$ FLOPs; at $k=2$,
    $T=4096$, that's $\approx 4096\cdot2\cdot8\cdot4096^2 \approx 1.1\times10^{12}$
    FLOPs/layer → on an H100 (~990 TFLOP/s) ~1.1 ms/layer, ~66 ms over 60 layers.
    So **cross-node EP is comm-bound** (80 ms comm vs 66 ms compute) unless the
    all-to-all is overlapped/node-limited; intra-node it's roughly balanced —
    which is exactly why overlap and node-limited routing matter.

??? success "2 — Node-limited routing bounds cross-node traffic"
    If each token's $k$ experts may land on $k$ different nodes you pay cross-node
    bandwidth up to $k$ times per token. Capping experts to $\le M$ nodes bounds
    worst-case cross-node messages per token at $M$ (DeepSeek-V3 uses $M=4$ with
    $k=8$). **Cost:** the router can no longer pick the globally best $k$ experts
    if they're scattered across $>M$ nodes — a small quality hit traded for a hard
    bandwidth ceiling.

??? success "3 — Padding waste (batched, cf 2.0) vs grouped GEMM (CV 0.5)"
    Batched GEMM at capacity factor 2.0 pads **every** expert to $2\times$ mean
    load, so even a perfectly balanced expert wastes ~50% of its slots; with
    real load the padded tensor is sized for the worst case → large wasted FLOPs.
    Grouped GEMM runs each expert on exactly its token count (no padding), so
    waste $\approx 0$ regardless of CV. For CV = 0.5 the batched form wastes
    roughly the capacity-vs-actual gap (tens of percent); grouped wins clearly —
    the reason modern MoE kernels use grouped/variable-length GEMM.

??? success "4 — Chunked schedule overlapping dispatch with shared-expert compute"
    Split the token batch into chunks; while chunk $i$'s dispatch all-to-all is in
    flight, compute the **shared expert** (and/or attention) for chunk $i-1$,
    which needs no routing. See the converted pipeline diagram on
    [Systems & EP](../moe/systems-ep.md). **Overlap is limited by** the smaller of
    (comm time, independent-compute time): if the shared-expert/attention work is
    shorter than the all-to-all, comm is exposed; also by chunking overhead and
    available SMs/queues for concurrent kernels.

## MoE kernels

??? success "1 — Triton gather + inverse permutation, fused into the epilogue"
    Alongside the forward permutation `perm[i]` (sorted-position → original), emit
    `inv_perm[perm[i]] = i` in the same kernel (one scatter write). Then the
    grouped-GEMM **epilogue** can scatter each expert's output rows directly to
    their original token positions using `inv_perm`, avoiding a separate
    scatter-read pass over HBM — one fused write instead of compute-then-permute.

??? success "2 — Wavefront-agnostic CUDA `gather_rows`; block-size sweep"
    Replace any hard-coded 32 with `warpSize` (32 on NVIDIA, 64 on CDNA) for
    intra-warp logic, and parametrize the block size. Benchmarking 128/256/512:
    throughput rises then plateaus/falls — small blocks underuse the SM (low
    occupancy, more launch overhead), very large blocks hit register/SMEM limits
    and reduce resident blocks. The peak is where occupancy saturates memory
    bandwidth without spilling — typically 256 for a memory-bound gather.

??? success "3 — Padded batched vs grouped GEMM: time vs capacity factor"
    Plot wall-time against cf. **Batched** time grows roughly **linearly with cf**
    (you literally compute padded zeros). **Grouped** is ~flat (independent of cf
    — it processes real tokens only). They cross near cf ≈ 1 + a small constant;
    below the crossover the simpler batched kernel can win on launch simplicity,
    above it grouped dominates. Real MoE runs at cf ≥ 1.25, so grouped is the
    standard choice.

??? success "4 — Profile fused vs unfused dispatch (HBM-bytes)"
    The fused gather→GEMM avoids materializing the permuted token tensor in HBM.
    In the profiler's memory counters (e.g. `dram__bytes` / equivalent) the fused
    path should show **lower HBM read+write bytes** by roughly the size of that
    intermediate ($T\times d\times 2$ B), confirming the win is *traffic*, not
    FLOPs — exactly what you'd predict from the memory-bound nature of permute.

## Inference & serving

??? success "1 — HBM for DeepSeek-V3 weights: bf16 / fp8 / int4"
    671B params: **bf16** $= 671\times2 = 1342$ GB; **fp8** $= 671$ GB; **int4**
    $\approx 336$ GB. On 80 GB GPUs (weights only, before KV): bf16 → $\lceil
    1342/80\rceil = 17$; fp8 → 9; int4 → 5. Quantization directly buys you fewer
    GPUs — the headline reason MoE serving leans hard on low precision.

??? success "2 — Condition for hiding expert streaming"
    Streaming an expert is hidden when its **GEMM time ≥ transfer time**:

    $$ \frac{2\,n_e\,(8d^2)}{\text{FLOP/s}} \;\ge\; \frac{8d^2 \cdot \text{bytes}}{\text{PCIe BW}}, $$

    where $n_e$ = tokens routed to the expert. The $8d^2$ cancels, giving a
    threshold on **tokens-per-expert**: $n_e \ge \tfrac{1}{2}\cdot
    \tfrac{\text{FLOP/s}}{\text{PCIe BW/byte}}$. Big batches (many tokens/expert)
    hide the transfer; batch-1 decode never does — so offloading helps throughput
    serving, not low-latency single-stream.

??? success "3 — Distinct experts touched (batch 256, $E{=}256, k{=}8$)"
    Assignments $= 256\times8 = 2048$ over $E=256$ experts. Expected distinct
    experts (balls-in-bins): $E(1-(1-1/E)^{2048}) = 256(1-(255/256)^{2048})
    \approx 256(1-e^{-8}) \approx 256\times0.99966 \approx \mathbf{256}$ — i.e.
    **essentially all experts** are touched. Mean tokens/expert $= 2048/256 = 8$,
    Poisson-ish spread (std ≈ √8 ≈ 2.8). Implication: at serving batch sizes you
    can't avoid loading most experts, so resident-weight strategies beat caching.

??? success "4 — Expert-cache eviction by popularity; failure mode"
    Use LFU/LRU keyed on observed routing frequency: keep hot experts in HBM,
    stream cold ones. **Failure mode:** if the routing distribution **shifts at
    runtime** (e.g. a domain change in the request stream), the cache is now
    populated with formerly-hot, now-cold experts → high miss rate and a latency
    cliff while it re-warms. Mitigate with an adaptive window / decay so popularity
    tracks recent traffic, and a floor of always-resident experts.

## Case studies

??? success "1 — $\binom{E}{k}$ per model and the fine-grained argument"
    Compute $\binom{E}{k}$ for each studied model (e.g. Mixtral $\binom{8}{2}=28$;
    DeepSeek-V3 $\binom{256}{8}\approx4\times10^{14}$ for routed experts).
    More/finer experts ⇒ astronomically more token-specific mixtures at the same
    active $k$ — the quantitative core of the "fine-grained experts" claim and the
    reason $E$ has grown across model generations.

??? success "2 — KV-cache per 1k tokens: MLA vs GQA vs MHA"
    Per token-layer: **MHA** caches $2\,n_h d_h$; **GQA** $2\,n_{kv}d_h$ (with
    $n_{kv}\ll n_h$); **MLA** a latent $d_c$. For a model with $n_h=128, d_h=128,
    n_{kv}=8, d_c\approx512$: MHA $=2\cdot128\cdot128=32768$ B; GQA
    $=2\cdot8\cdot128=2048$ B; MLA $\approx512\cdot2=1024$ B per token-layer.
    Multiply by $L$ and 1000 tokens. MLA saves ~2× over GQA and ~30× over MHA —
    the lever that makes long-context DeepSeek serving affordable.

??? success "3 — DeepSeek-V3 vs Mixtral: active/total and the serving trade"
    V3 ≈ 5.5% active (671B/37B), Mixtral ≈ 28% (47B/13B). V3's low activation
    ratio means **more memory but less compute per token** — favoring
    high-throughput, memory-rich serving (many GPUs, big batches). Mixtral's
    higher ratio is lighter on memory, friendlier to smaller deployments and
    batch-1 latency. The active/total ratio is the dial between
    memory-cost and compute-cost.

??? success "4 — Map one model back to Part II pages"
    E.g. **DeepSeek-V3**: sigmoid gate + bias controller →
    [load balancing](../moe/load-balancing.md); fine-grained + shared experts →
    [routing variants](../moe/routing-variants.md); z-loss/fp32 router →
    [training stability](../moe/training-stability.md); node-limited routing +
    DualPipe →  [systems & EP](../moe/systems-ep.md); MLA →
    [attention efficiency](../foundations/attention-efficiency.md); fp8 →
    [numerics](../foundations/numerics-precision.md). Every production trick traces
    to a page — that's the handbook's thesis.
