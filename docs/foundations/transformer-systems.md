# The transformer as a system

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> beginner</span>
  <span class="chip"><strong>Prereqs:</strong> matmul, basic transformer</span>
  <span class="chip"><strong>Hardware:</strong> none (pen & paper)</span>
</div>

You will leave this page able to look at a model config and a GPU spec sheet and
say, *before running anything*, roughly how fast a layer should run and **what
is limiting it**. This is the most important skill in the handbook: every later
optimization is a deliberate move on the **roofline**.

## The two numbers that govern everything

Every GPU has two headline throughputs:

- **Compute**: peak floating-point ops per second, $\pi$ (FLOP/s).
- **Memory bandwidth**: peak bytes per second moved between HBM and the chip, $\beta$ (B/s).

| Accelerator | BF16 dense (TFLOP/s) | HBM BW (TB/s) | Ridge point $\pi/\beta$ (FLOP/byte) |
|---|---|---|---|
| NVIDIA A100 80GB (SXM) | ~312 | ~2.0 | ~156 |
| NVIDIA H100 (SXM) | ~990 | ~3.35 | ~296 |
| AMD Instinct MI300X | ~1300 | ~5.3 | ~245 |

!!! note "These are peak, marketing-sheet numbers"
    Real kernels reach maybe 50–80% of peak compute and 70–90% of peak
    bandwidth. Use peak for *ratios and intuition*; use the
    [profiler](../performance/profiling.md) for truth. Sparse/structured FLOP
    numbers on spec sheets are usually 2× the dense number — ignore them unless
    you're actually using sparsity.

A kernel that does $W$ FLOPs and moves $Q$ bytes has **arithmetic intensity**

$$ I = \frac{W}{Q} \quad \text{[FLOP/byte]}. $$

The achievable performance is the **roofline**:

$$ P = \min(\pi,\; \beta \cdot I). $$

If $I$ is below the **ridge point** $\pi/\beta$, you are **memory-bound** —
performance is $\beta \cdot I$ and adding FLOPs is free until you hit the ridge.
Above it, you are **compute-bound** — only fewer FLOPs or faster math help.

<figure class="roofline-figure">
<svg viewBox="0 0 760 430" role="img" aria-labelledby="roofline-title roofline-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="roofline-title">Roofline performance model</title>
  <desc id="roofline-desc">A log-log plot of achievable performance versus arithmetic intensity, rising with memory bandwidth until the ridge point and then flattening at peak compute.</desc>
  <defs>
    <linearGradient id="roofline-line" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="#00bcd4" />
      <stop offset="55%" stop-color="#5e35b1" />
      <stop offset="100%" stop-color="#7c4dff" />
    </linearGradient>
    <marker id="axis-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
  </defs>
  <rect class="roofline-panel" x="24" y="20" width="712" height="360" rx="24" />
  <g class="roofline-grid">
    <path d="M130 320 H650" />
    <path d="M130 260 H650" />
    <path d="M130 200 H650" />
    <path d="M130 140 H650" />
    <path d="M210 80 V320" />
    <path d="M330 80 V320" />
    <path d="M450 80 V320" />
    <path d="M570 80 V320" />
  </g>
  <g class="roofline-axis">
    <path d="M110 320 H675" marker-end="url(#axis-arrow)" />
    <path d="M110 320 V64" marker-end="url(#axis-arrow)" />
  </g>
  <path class="roofline-slope" d="M135 305 L365 115" />
  <path class="roofline-cap" d="M365 115 H650" />
  <line class="roofline-ridge" x1="365" y1="115" x2="365" y2="320" />
  <circle class="roofline-ridge-dot" cx="365" cy="115" r="5" />
  <text class="roofline-label roofline-y" x="44" y="62">performance (FLOP/s, log)</text>
  <text class="roofline-label roofline-x" x="462" y="365">arithmetic intensity I (log)</text>
  <text class="roofline-tick" x="86" y="121">π</text>
  <text class="roofline-label" x="424" y="98">compute-bound: P = π</text>
  <text class="roofline-label" x="198" y="194">memory-bound: P = β · I</text>
  <text class="roofline-label roofline-slope-label" x="232" y="153">slope = β</text>
  <text class="roofline-label" x="323" y="350">ridge = π/β</text>
</svg>
<figcaption>The roofline. Left of the ridge you are limited by bandwidth; right of it, by the math units.</figcaption>
</figure>

The whole game of ML performance engineering is: **(1) figure out which regime
you're in, (2) if memory-bound, raise $I$ (fuse ops, reuse data in SRAM,
quantize), (3) if compute-bound, cut FLOPs or use faster precision.**

## Counting FLOPs in a transformer

Take a decoder-only transformer: $L$ layers, hidden size $d$, FFN hidden
$d_{ff}$ (often $4d$), sequence length $N$, batch $B$, vocabulary $V$.

A matmul of $(m\times k)\cdot(k\times n)$ costs $2mkn$ FLOPs (the 2 = one
multiply + one add per inner-product term).

**Per layer, per token** (ignoring the $O(N^2)$ attention score term for now):

| Sub-block | Matmul shapes | FLOPs / token |
|---|---|---|
| QKV projection | $d \to 3d$ | $2 \cdot d \cdot 3d = 6d^2$ |
| Attention output proj | $d \to d$ | $2d^2$ |
| FFN up | $d \to d_{ff}$ | $2 d\, d_{ff}$ |
| FFN down | $d_{ff} \to d$ | $2 d\, d_{ff}$ |

With $d_{ff}=4d$ the FFN is $16d^2$ and attention projections are $8d^2$, so a
layer is about $24d^2$ FLOPs per token for the **linear** parts. Over $L$ layers
and $BN$ tokens the forward pass is roughly

$$ W_{\text{fwd}} \approx 24\, L\, d^2 \cdot BN. $$

There's a classic shortcut: with $P \approx 12 L d^2$ non-embedding parameters,
this is $W_{\text{fwd}} \approx 2 P \cdot BN$ — **2 FLOPs per parameter per
token**. The backward pass costs about twice the forward, giving the famous

$$ \boxed{\;W_{\text{train}} \approx 6\, P \cdot (\text{tokens})\;} $$

used in compute budgeting (e.g. Chinchilla). Keep this in your head — it's how
you sanity-check a training run's MFU (model FLOPs utilization) in seconds.

### The $N^2$ attention term

The score matrix $QK^\top$ is $(N\times d)\cdot(d\times N)$ and the
$\text{softmax}\cdot V$ step is $(N\times N)\cdot(N\times d)$, each
$2N^2 d$ FLOPs per head-group, so attention scores cost $\approx 4 L N^2 d \cdot B$
total. Compare to the linear $24 L d^2 BN$:

$$ \frac{\text{attention}}{\text{linear}} \approx \frac{4 N^2 d}{24 d^2 N} = \frac{N}{6d}. $$

So attention's FLOP share grows with $N/d$. At $N=2048, d=4096$ it's ~8% of
FLOPs; at $N=128\text{k}$ it dominates. This is *why* long-context work obsesses
over attention, and why FlashAttention matters.

## Counting bytes: the part people forget

FLOPs are only half the roofline. Consider a single FFN-up matmul during
**training** at batch×seq = $BN$ tokens, weights $W \in \mathbb{R}^{d\times d_{ff}}$
in bf16 (2 bytes):

- Bytes moved: read activations $BN\cdot d \cdot 2$, read weights $d\, d_{ff}\cdot 2$, write output $BN\, d_{ff}\cdot 2$.
- FLOPs: $2\, BN\, d\, d_{ff}$.

When $BN \gg d$ (big batch), the weight read amortizes and intensity approaches
$I \approx \tfrac{2 BN d\, d_{ff}}{2(BN d + BN d_{ff})} \approx \tfrac{BN d\, d_{ff}}{BN(d+d_{ff})}$,
which for $d_{ff}=4d$ is $\approx 0.8\,d$ — hundreds of FLOP/byte, comfortably
**compute-bound**. Large matmuls with large batch are the GPU's happy place.

Now do the same matmul during **decoding** with batch $B=1$, one new token
($N=1$): you read the entire weight matrix to produce a single token's
activations. Intensity collapses to $\approx 1$ FLOP/byte — deeply
**memory-bound**. Same math, opposite regime, purely because of batch size.

!!! important "The central tension of LLM serving"
    **Training / prefill** processes many tokens at once → compute-bound → you
    want kernels that hit peak FLOPs. **Decoding** generates one token at a time
    → memory-bound → you want to move fewer bytes (quantize weights, batch many
    requests together, cache KV). Almost every serving trick
    ([continuous batching](../performance/inference-optimization.md),
    [weight quantization](../performance/quantization.md),
    [speculative decoding](../performance/inference-optimization.md)) is an
    attack on this one fact.

## A worked roofline example

H100: $\pi = 990$ TFLOP/s, $\beta = 3.35$ TB/s, ridge $= 296$ FLOP/byte.

A bf16 GEMM with $m=n=k=8192$: $W = 2\cdot8192^3 \approx 1.1\times10^{12}$ FLOP;
bytes $= 3\cdot 8192^2 \cdot 2 \approx 4.0\times10^8$ B; intensity
$I \approx 2730$ FLOP/byte $\gg 296$ → compute-bound. Best-case time
$\approx 1.1\times10^{12} / 9.9\times10^{14} \approx 1.1$ ms. If your profiler
says 2.2 ms you're at ~50% MFU — now you have a *target*, not a vibe.

A bf16 element-wise GELU on the same tensor: ~$10\cdot8192^2$ FLOP but
$2\cdot 8192^2\cdot 2$ bytes → $I\approx 2.5$ → memory-bound, time set by
bandwidth. This is exactly why we **fuse** GELU into the matmul epilogue:
standalone, it's pure memory traffic.

## Key takeaways

- Two GPU numbers — peak FLOP/s $\pi$ and bandwidth $\beta$ — and one kernel
  number — arithmetic intensity $I = W/Q$ — predict performance via
  $P=\min(\pi, \beta I)$.
- Transformer training costs $\approx 6 P$ FLOPs per token; forward is
  $\approx 2P$. Attention's FLOP share scales as $N/6d$.
- **Large-batch matmuls are compute-bound; single-token decoding is
  memory-bound.** Most of this handbook is about moving operations rightward
  on the roofline.
- Always compute a *target* time from the roofline before trusting a measured
  one.

## Exercises

1. For a 7B-parameter model ($P\approx7\times10^9$), estimate the forward FLOPs
   for one sequence of 4096 tokens. How long should it take on an MI300X at 60%
   MFU?
2. At what sequence length $N$ does attention-score FLOPs equal the linear-layer
   FLOPs for $d=5120$? What does that imply for context-length scaling?
3. A bf16 LayerNorm over a $[B{=}32, N{=}2048, d{=}4096]$ tensor: estimate FLOPs
   and bytes, compute $I$, and decide the regime on an A100. Should it be fused?
4. Re-derive the $6P$ rule and identify every place the factor of 2 (fwd vs bwd)
   and the factor of 3 (fwd + 2× bwd) enter.

## References

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 2009.
- Kaplan et al. *Scaling Laws for Neural Language Models.* 2020.
- Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla). 2022.
- Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models.* 2022.
- NVIDIA H100 and AMD CDNA3 (MI300) architecture whitepapers.
