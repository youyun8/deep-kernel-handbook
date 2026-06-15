# Solutions — Part III · Performance

<div class="page-meta">
  <span class="chip"><strong>Covers:</strong> all seven performance pages</span>
  <span class="chip"><strong>Use:</strong> attempt first, then check</span>
</div>

Worked answers to the exercises in [Part III](../performance/index.md). Kernel
exercises are open-ended ("run it, benchmark it"); we give the expected result
and the reasoning so you can check your numbers.

## GPU programming model

??? success "1 — Why a 32-lane reduction is wrong on CDNA"
    A warp reduction hard-coded with `offset = 16,8,4,2,1` and a mask of 32 lanes
    assumes a 32-wide warp (NVIDIA). AMD CDNA wavefronts are **64 lanes**, so a
    32-lane shuffle only reduces *half* the wavefront — the upper 32 lanes are
    ignored, giving a wrong (partial) sum. Fix: start the shuffle loop at
    `warpSize/2` and use `warpSize` everywhere instead of the literal 32, so the
    same code reduces 32 or 64 lanes correctly.

??? success "2 — Occupancy limiter (64 regs/thread, 48 KB SMEM/block)"
    Per SM: 64K registers, 100 KB SMEM. Take a 256-thread block.

    - **Registers:** $64\times256 = 16384$ regs/block → $65536/16384 = 4$ blocks.
    - **SMEM:** $100/48 = 2.08 →$ **2 blocks**.

    SMEM is the tighter constraint → 2 resident blocks (512 threads). **Shared
    memory is the occupancy limiter**; cutting SMEM per block (or block size)
    would raise occupancy.

??? success "3 — Coalesced for row-major, not for its transpose"
    For a row-major $[M,N]$ tensor, threads in a warp indexed by the **column**
    read `A[row, col0+lane]` — consecutive addresses → one coalesced transaction.
    Reading the **transpose** (threads walk down a column) gives stride-$N$
    addresses → $N$ separate transactions, ~32× the memory traffic. This is
    exactly the MoE **gather**: scattered token indices break coalescing, which is
    why the gather kernel is memory-bound and worth fusing.

??? success "4 — When lowering occupancy raises throughput"
    A register-heavy matmul tile keeps more of the working set in **registers**
    (the fastest memory), cutting SMEM/HBM traffic and instruction count per
    output. That raises register pressure → fewer resident warps → lower
    occupancy, yet **higher throughput** because each warp does far more useful
    work and the kernel is compute- not latency-bound. Max occupancy only helps
    when you need many warps to hide memory latency; a well-tiled GEMM doesn't.

## Triton track

??? success "1 — Vector-add and softmax vs PyTorch"
    Both Triton kernels should match `torch` outputs to bf16/fp32 tolerance.
    Vector-add is memory-bound → expect throughput near HBM bandwidth, on par with
    the native op. Fused softmax should **beat** a naive three-pass torch softmax
    on bandwidth (one read + one write vs three passes) and roughly tie
    `torch.softmax` (which is itself fused).

??? success "2 — AMD-oriented autotune configs"
    Add configs varying `num_warps` (4/8) and `BLOCK` with **wavefront-64** in
    mind — on CDNA, `num_warps=4` already means 256 lanes/block, so the
    sweet-spot block sizes differ from NVIDIA's 32-lane warps. Best config is
    GPU-specific; the lesson is that a config autotuned on one vendor is rarely
    optimal on the other — always re-autotune per target.

??? success "3 — Softmax for rows wider than one `BLOCK`"
    Loop over the row in `BLOCK`-sized tiles maintaining a running max $m$ and sum
    $\ell$ via the **online-softmax combiner** (same recurrence as FlashAttention
    ex. 1): for each tile, $m' = \max(m, \max_{\text{tile}})$, rescale $\ell$ by
    $e^{m-m'}$, add the tile's contribution. A second pass (or cached numerators)
    normalizes. This removes the "row must fit in one block" limit.

??? success "4 — Fused vs three-pass softmax bytes"
    Three-pass reads the row **3×** (max, exp-sum, normalize) and writes once.
    Fused reads once, writes once. For an $[R,C]$ tensor the byte ratio is
    ≈ $(3+1)/(1+1) = 2\times$ less traffic for the fused kernel — and since
    softmax is memory-bound, that's ≈ a 2× speedup, which your benchmark should
    confirm.

## CUDA / HIP track

??? success "1 — Port tiled matmul to HIP"
    `hipify` or hand-port: `__shared__` stays, `cudaMalloc→hipMalloc`,
    `<<<>>>` launch is identical syntax under `hipcc`. Build with `hipcc` (or via
    PyTorch on ROCm) and verify against cuBLAS/hipBLAS to fp tolerance. The point:
    HIP is source-compatible — the same kernel runs on both vendors, the only real
    portability bug is wavefront width (next exercise).

??? success "2 — 32-lane reduction fails on 64-wide wavefront"
    Take a reduction that shuffles only `offset = 16…1`. On CDNA the 64-lane
    wavefront means lanes 32–63 never contribute → result is the sum of the lower
    half only. Demonstrate by reducing an all-ones vector of length 64: you get
    32, not 64. Fix with `for (offset = warpSize/2; offset>0; offset>>=1)`.

??? success "3 — `TILE` ∈ {8,16,32} sweep"
    Larger tiles → more data reuse from SMEM per global load (higher arithmetic
    intensity) but more SMEM/registers per block → lower occupancy. Typically
    `TILE=16` or `32` wins: 8 is too small (poor reuse, memory-bound), 32 may spill
    or cut occupancy on smaller GPUs. Relate the best value to warps/wavefronts per
    block and SMEM budget on your specific card.

??? success "4 — Replace inner product with matrix-core (`wmma`/rocWMMA)"
    Swapping the scalar inner loop for tensor-core/matrix-core MMA fragments gives
    a large speedup (often 4–10×) because matrix cores do a whole tile-MMA per
    instruction at much higher FLOP/s than the scalar FMA path. The catch:
    fragments require specific tile shapes/dtypes (e.g. 16×16×16 bf16) and careful
    SMEM layout — measure both correctness and speedup vs the scalar baseline.

## Distributed training

??? success "1 — all-reduce = reduce-scatter + all-gather; ZeRO-2 volume"
    **Identity:** reduce-scatter sums and leaves each rank one shard of the result;
    all-gather then distributes all shards → every rank has the full reduced
    tensor = an all-reduce. Ring cost of each step ≈ $S(G{-}1)/G$ bytes/rank, so
    the two together ≈ $2S(G{-}1)/G$ = the all-reduce cost.
    **ZeRO-2** shards gradients, so instead of an all-reduce of the full gradient
    it does a **reduce-scatter** (each rank keeps its grad shard, updates its
    optimizer shard) and an **all-gather of parameters** — same total volume as
    DDP's all-reduce ($\approx 2S$), but it never materializes the full gradient
    or optimizer state, saving memory at equal communication.

??? success "2 — Per-GPU memory, 70B bf16 + Adam, 8 GPUs"
    Mixed-precision Adam state per param: 2 B (bf16 weight) + 2 B (grad) + 4 B
    (fp32 master) + 4 + 4 B (fp32 m, v) = **16 B/param**. For 70B that's
    $70\text{B}\times16 = 1120$ GB total.

    - **DDP:** every GPU stores all 16 B/param → **~1120 GB/GPU** (impossible on 80 GB — needs sharding).
    - **ZeRO-1** (shard optimizer, 12 of 16 B): 2+2 + 12/8 = 4 + 1.5 = **~5.5 B/param × 70B ≈ 385 GB**… still split? Per-GPU: unsharded 4 B/param (weight+grad, 280 GB) + sharded 12/8 = 1.5 B (105 GB) ≈ **385 GB/GPU**.
    - **ZeRO-2** (also shard grads): weight 2 B (140 GB) + (2+12)/8 = 1.75 B (122 GB) ≈ **262 GB/GPU**.
    - **ZeRO-3** (shard everything): 16/8 = 2 B/param × 70B ≈ **140 GB/GPU**.

    The trend is the point: ZeRO-3 cuts per-GPU state ~$G\times$ vs DDP, turning an
    impossible model into a fittable one (with more all-gather traffic).

??? success "3 — Pipeline bubble fraction"
    With $P$ stages and $m$ micro-batches the bubble fraction is

    $$ \text{bubble} = \frac{P-1}{m + P - 1}. $$

    To keep it $<10\%$: $\frac{P-1}{m+P-1} < 0.1 \Rightarrow m > 9(P-1)$. So for
    $P=8$ you need $m > 63$ micro-batches; for $P=4$, $m>27$. More stages ⇒ many
    more micro-batches required to amortize the fill/drain — the core PP tension.

??? success "4 — Why TP intra-node, EP can cross nodes"
    **TP** does an all-reduce **inside every layer** (twice: fwd + bwd) on the full
    activation — enormous, latency-sensitive volume per step → it must ride the
    fastest links (intra-node NVLink/Infinity Fabric). **EP** does two all-to-alls
    per MoE layer but the per-token payload is smaller and, crucially, **overlaps
    with compute** and can be **node-limited**; it tolerates slower cross-node
    bandwidth. Map the chattiest collective (TP) to the fastest link, the
    overlappable one (EP) to the slower fabric.

## Quantization & compression

??? success "1 — int8 affine quantize/dequantize and max error"
    Affine: $q = \text{round}(x/s) + z$, $\hat x = s(q - z)$, with
    $s = (\max-\min)/255$ for int8. Max error per element is **half a step**,
    $s/2$. **Per-tensor** uses one $s$ for the whole tensor, so an outlier channel
    inflates $\max$ → large $s$ → big error on all the *small* channels.
    **Per-channel** gives each channel its own $s$, so the outlier's large $s$
    doesn't pollute the others → much lower error. This is why per-channel (and
    AWQ) exist.

??? success "2 — AWQ salient-channel scaling"
    AWQ scales up the **salient** (high-activation-magnitude) weight channels
    before quantizing and compensates by scaling the corresponding activations
    down, so the important channels get effectively more bits. Implement on one
    linear layer: identify salient channels by activation stats, apply the
    per-channel scale, quantize to int4, dequantize, measure perplexity. Expect
    **noticeably lower perplexity than naive per-channel int4** at the same
    bit-width.

??? success "3 — Decode-latency gain from W4 on a 13B model"
    Decode is **memory-bound**: latency ≈ weight-bytes / HBM-BW. bf16 weights =
    $13\text{B}\times2 = 26$ GB; int4 ≈ $13\text{B}\times0.5 = 6.5$ GB. Bytes drop
    ~4×, so per-token decode latency drops toward **~4×** (minus overhead for
    dequant and unquantized activations/KV). The win comes purely from moving
    fewer weight bytes per token — the [memory-bound](../foundations/attention-efficiency.md)
    argument made quantitative.

??? success "4 — Why routed experts tolerate int4 better than router/attention"
    Routed experts are **redundant and averaged** — each token sees only $k$ of
    many experts, and quantization noise across many experts washes out in the
    weighted sum. The **router** makes discrete decisions on **small logit
    differences** (precision-critical — see MoE stability), and **attention**
    feeds the KV cache where errors **compound over the sequence**. So aggressive
    int4 goes on the experts (most of the params, most tolerant), while router and
    attention stay higher precision — the standard MoE serving recipe.

## Inference optimization

??? success "1 — Speculative decoding speedup"
    With draft acceptance rate $\alpha$ and proposal length $\gamma$, the expected
    number of tokens accepted per verify step is

    $$ \mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}. $$

    Speedup ≈ that, divided by the per-step cost ratio (one big-model verify +
    $\gamma$ cheap drafts). High $\alpha$ and modest $\gamma$ give the best
    return; as $\alpha\to1$ you approach $\gamma+1$ tokens per verify. Acceptance
    rate, not draft speed, dominates.

??? success "2 — Continuous vs static batching, lengths uniform [64,1024]"
    Static batching pads every request to the **longest in the batch** and waits
    for the slowest to finish, so short requests waste compute/slots; with lengths
    uniform in [64,1024], mean length ≈ 544 but the batch runs at ≈ 1024 → ~40–50%
    wasted. **Continuous batching** evicts finished sequences and admits new ones
    each step, keeping the batch full → throughput gain on the order of the
    padding-waste fraction (roughly **1.5–2×** for this spread, more with higher
    variance).

??? success "3 — Prefix-cache KV saved, 100 requests, 2k shared prompt"
    Without sharing, each request stores the 2k-token system prompt's KV
    separately → $100\times$ copies. With prefix caching the shared prefix is
    stored **once** and reused, saving $99\times$ the prefix KV. If one token-layer
    of KV is $b$ bytes and there are $L$ layers, saved $= 99\times2000\times L\times
    b$ — typically several GB. Pure win for any workload with a common system
    prompt.

??? success "4 — Prefill/decode disaggregation: help vs hurt"
    Disaggregation puts compute-bound **prefill** and memory-bound **decode** on
    separate pools, each tuned to its bottleneck (prefill: big batches, high MFU;
    decode: high memory bandwidth). It **helps** when the two phases would
    otherwise contend (bursty long prompts starving decode). It **hurts** when the
    **KV-cache transfer** between pools (prefill→decode handoff) costs more than
    the contention it avoids — i.e. short prompts / small KV, or slow
    interconnect. Reason about KV bytes vs link bandwidth vs contention saved.

## Profiling & methodology

??? success "1 — Benchmark a Triton softmax wrong, then right"
    **Wrong:** time the first call with no warmup and no `torch.cuda.synchronize()`
    — you measure kernel-launch + compile (JIT) latency and CPU-side async return,
    not GPU time, often off by 10–100×. **Right:** warm up several iterations
    (trigger autotune/compile), then time many iterations with a `synchronize()`
    bracketing the loop. The corrected number is the real per-call GPU time;
    quantify the gap.

??? success "2 — Profile a decode step: who dominates?"
    Profile one decode step of a small transformer. At **batch 1** decode, expect
    **launch overhead and memory-bound attention/FFN** to dominate — many tiny
    kernels, each reading weights/KV from HBM, with the GPU underutilized.
    Common fix: **CUDA graphs** (kill launch overhead) + batching (raise
    arithmetic intensity). If attention dominates, KV layout/Flash-decoding helps;
    if the FFN, weight quantization helps.

??? success "3 — Compute MFU; diagnose 15%"
    $\text{MFU} = \frac{6P \cdot \text{tokens/s}}{\text{GPU peak FLOP/s}}$ (training,
    $6P$ FLOPs/token). A 15% result means you're using a sixth of the math units.
    Diagnose in order: **input pipeline stalls** (GPU starved), **small batch /
    low occupancy**, **comm not overlapped** (DP/TP/EP exposed), **memory-bound
    ops** (unfused norms/activations), **recompute** overhead. Profile to find
    which, then fix the top contributor — MFU is the single best training-health
    number.

??? success "4 — Dead-code elimination hides the kernel"
    If a benchmark computes a result the program never reads, the compiler may
    **eliminate** the kernel entirely → you "measure" ~0 time. Construct it by
    discarding the output, observe an absurd speed, then **fix by consuming the
    output** (e.g. accumulate into a value you print/return, or add a data
    dependency). Always make the result observable so the work can't be optimized
    away — a classic microbenchmark trap.
