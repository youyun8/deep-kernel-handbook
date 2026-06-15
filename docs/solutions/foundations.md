# Solutions — Part I · Foundations

<div class="page-meta">
  <span class="chip"><strong>Covers:</strong> Transformer-as-system, Attention efficiency, FlashAttention, Numerics</span>
  <span class="chip"><strong>Use:</strong> attempt first, then check</span>
</div>

Worked answers to the exercises in [Part I](../foundations/index.md). Numbers
use round hardware specs (A100 ≈ 312 TFLOP/s bf16 / 2.0 TB/s; H100 ≈ 990 TFLOP/s
bf16 / 3.35 TB/s; MI300X ≈ 1.3 PFLOP/s bf16 / 5.3 TB/s); your exact deltas will
shift with the chip you assume, but the *regime* (memory- vs compute-bound) is
what matters.

## Transformer as a system

??? success "1 — Forward FLOPs for a 7B model, 4096 tokens"
    Forward is $\approx 2P$ FLOPs per token (one multiply-add per parameter):

    $$ 2 \times 7\times10^9 \times 4096 \approx 5.7\times10^{13}\ \text{FLOPs}. $$

    On an MI300X at 60% MFU the sustained rate is $0.6 \times 1.3\times10^{15}
    \approx 7.8\times10^{14}$ FLOP/s, so

    $$ t \approx \frac{5.7\times10^{13}}{7.8\times10^{14}} \approx 73\ \text{ms}. $$

    The attention-score term ($\propto N^2$) is small here — at $N=4096$ it adds
    only a few percent — which is exactly why the $2P$ rule is a good first
    estimate until context gets long.

??? success "2 — Sequence length where attention = linear FLOPs"
    Per layer, parameters $\approx 12d^2$ (attention $4d^2$ for $W_{q,k,v,o}$ +
    FFN $8d^2$), so **linear** forward FLOPs $\approx 2\cdot 12d^2 \cdot N = 24Nd^2$.
    **Attention-score** FLOPs (QKᵀ + AV) $\approx 4N^2 d$. Setting equal:

    $$ 4N^2 d = 24 N d^2 \;\Rightarrow\; N = 6d. $$

    For $d=5120$, $N \approx 3.1\times10^{4}$ tokens. Below that, the matmuls
    dominate the FLOP bill; past it attention's quadratic term takes over — which
    is why long-context work lives or dies on attention efficiency, not the FFN.

??? success "3 — bf16 LayerNorm: FLOPs, bytes, regime"
    Elements $= 32 \times 2048 \times 4096 \approx 2.68\times10^{8}$.

    - **FLOPs:** mean, variance, normalize, scale+shift ≈ ~10 FLOPs/elem →
      $\approx 2.7\times10^{9}$.
    - **Bytes:** read input + write output in bf16 = $2+2 = 4$ bytes/elem →
      $\approx 1.07\times10^{9}$ bytes (the $\gamma,\beta$ vectors are negligible).
    - **Intensity:** $I \approx 2.7\times10^9 / 1.07\times10^9 \approx 2.5$ FLOP/byte.

    A100 ridge $= 312\text{T}/2.0\text{T} \approx 156$ FLOP/byte. Since
    $2.5 \ll 156$, LayerNorm is **deeply memory-bound** — almost pure data
    movement. **Yes, fuse it** into the adjacent matmul/residual so the tensor is
    never round-tripped to HBM just to be normalized.

??? success "4 — Re-deriving the 6P rule"
    Per token:

    - **Forward:** every parameter is used in one multiply-add = **2 FLOPs** →
      $2P$. *(The factor of 2 is the MAC: one multiply + one add.)*
    - **Backward:** you compute the gradient w.r.t. the layer **input** ($2P$)
      *and* w.r.t. the **weights** ($2P$) → $4P$.

    Total $2P + 4P = 6P$. *(The factor of 3 = one forward + two backward passes;
    $2P \times 3 = 6P$.)* Every place a matmul appears, its transpose appears
    twice in the backward — that is where the 3 comes from.

## Attention efficiency

??? success "1 — KV-cache size for a GQA model"
    Per token per layer: $2\ (\text{K,V}) \times n_{kv} \times d_h \times 2\
    \text{bytes} = 2 \times 8 \times 128 \times 2 = 4096$ B $= 4$ KB.

    $$ 4\,\text{KB} \times L(32) \times N(8192) \times B(16) \approx 1.7\times10^{10}\ \text{B} \approx 17\ \text{GB}. $$

    The 7B weights in bf16 are $\approx 14$ GB. So at this batch/length the **KV
    cache already exceeds the weights** — concrete motivation for GQA/MLA and for
    why decode is memory-bound.

??? success "2 — Arithmetic intensity of one decode step"
    One query attends to $t$ cached keys. FLOPs $\approx \underbrace{2td_h}_{QK^T}
    + \underbrace{2td_h}_{AV} = 4td_h$. Bytes (read K,V) $\approx 2\cdot t d_h \cdot
    2 = 4td_h$. Therefore

    $$ I = \frac{4td_h}{4td_h} = O(1)\ \text{FLOP/byte}, $$

    independent of $t$. Decode reads the *entire* KV cache to emit **one** token
    — the canonical memory-bound op, and the reason batching many requests is the
    main throughput lever.

??? success "3 — Fragmentation waste with 16-token blocks"
    A length-$\ell$ sequence uses $\lceil \ell/16\rceil$ blocks; wasted slots
    $= 16\lceil\ell/16\rceil - \ell$. With $\ell$ uniform, $\ell \bmod 16$ is
    ~uniform on $\{0..15\}$, so mean waste $\approx 7.5$ slots/sequence. As a
    fraction of used memory: $7.5 / \overline{\ell} = 7.5/2048 \approx 0.37\%$.
    Negligible — that is precisely why paged KV uses small blocks instead of
    pre-reserving max length (which wastes ~50%).

??? success "4 — MLA cache ratio vs GQA"
    GQA caches $2\,n_{kv}d_h$ per token-layer; MLA caches a single latent of dim
    $d_c$ (K and V are reconstructed by an up-projection at compute time):

    $$ \frac{\text{MLA}}{\text{GQA}} \approx \frac{d_c}{2\,n_{kv}d_h}. $$

    With DeepSeek-style $d_c \approx 512$ this is several-fold to an
    order-of-magnitude smaller cache. The **trade**: less memory and bandwidth at
    decode (the binding constraint) in exchange for extra FLOPs to up-project the
    latent back to K/V each step — a good deal precisely because decode is
    memory-bound, so the added compute is nearly free.

## FlashAttention

??? success "1 — Online-softmax combiner is exact"
    For two chunks with local maxes $m_1,m_2$, denominators $\ell_1,\ell_2$, and
    partial outputs $O_1,O_2$, let $m=\max(m_1,m_2)$ and

    $$ \ell = \ell_1 e^{m_1-m} + \ell_2 e^{m_2-m},\qquad
       O = \frac{O_1\,\ell_1 e^{m_1-m} + O_2\,\ell_2 e^{m_2-m}}{\ell}. $$

    Because $e^{x_i-m_1}\cdot e^{m_1-m} = e^{x_i-m}$, every term is re-expressed
    against the *global* max, so $\ell$ becomes $\sum_i e^{x_i-m}$ and $O$ becomes
    $\sum_i \mathrm{softmax}(x)_i v_i$ over the union — **identical** to one-pass
    softmax. Folding is associative, so any number of chunks is exact.

??? success "2 — Why subtract the running max"
    With a score of $+100$, naive fp16 computes $e^{100}$, which overflows
    (fp16 max $= 65504 \ll e^{100}$) → `inf` → `nan` after normalization. The
    stable form subtracts the max first: $e^{100-100}=1$, all terms in $[0,1]$,
    no overflow. Subtracting the max changes nothing mathematically (it cancels
    in the ratio) but everything numerically.

??? success "3 — Skipping fully-masked tiles under causality"
    Causal masking means query-tile $i$ only needs key-tiles $j \le i$. Of an
    $n\times n$ tile grid you compute the lower triangle, $n(n+1)/2$ tiles. For
    $N=4096$ with tile $128$, $n=32$: computed $= 32\cdot33/2 = 528$ of $1024$ →
    **~48% of the score FLOPs eliminated**, approaching the $\tfrac{n-1}{2n}\to
    50\%$ asymptote.

??? success "4 — HBM bytes: naive vs flash at N=8192, d=128 (per head)"
    Naive materializes the $N\times N$ score matrix (write then re-read for
    softmax): $\approx 2 \cdot N^2 \cdot 2\,\text{B} = 4N^2 \approx 2.7\times10^8$
    B ≈ **270 MB**. Flash never writes $S$; it streams Q,K,V once and writes O:
    $\approx (3{\cdot}Nd + Nd)\cdot 2 = 8Nd \approx 8\times10^6$ B ≈ **8 MB** —
    roughly **30× less traffic**. On an H100 (ridge ≈ $990\text{T}/3.35\text{T}
    \approx 295$ FLOP/byte) the naive version sits left of the ridge
    (memory-bound on the $S$ round-trip); flash raises $I$ past the ridge into
    compute-bound territory.

## Numerics & precision

??? success "1 — Largest finite-`exp` logit: fp16 vs bf16"
    `exp(x)` is finite while $x < \ln(\text{max normal})$.

    - **fp16** (5 exponent bits, max $65504$): $x < \ln 65504 \approx 11.1$.
    - **bf16** (8 exponent bits, max $\approx 3.4\times10^{38}$): $x < \ln(3.4\times10^{38}) \approx 88.7$.

    The 8 vs 5 exponent bits give bf16 fp32-level *range*, so softmax logits that
    instantly overflow fp16 are comfortably safe in bf16 — a core reason bf16 is
    the default training dtype.

??? success "2 — bf16 loses a $10^6 \times$ `1e-3` sum; fp32 recovers it"
    True sum $= 1000$. bf16 has 8 mantissa bits (~2–3 decimal digits). Once the
    running total passes ~256, the ULP exceeds $10^{-3}$, so each new addend
    rounds away (**swamping**) and the sum stalls far below 1000. An fp32
    accumulator (23 mantissa bits, ~7 digits) keeps $10^{-3}$ significant well
    past 1000, recovering the right answer. Lesson: **reduce in fp32** even when
    inputs are bf16.

??? success "3 — Dynamic loss scaling; why bf16 rarely halves"
    Keep a scale $S$: multiply the loss by $S$ before `backward`, unscale grads
    before `step`. If any grad is `inf`/`nan`, **skip the step and halve $S$**;
    after $N$ clean steps, **double $S$**. The halve branch fires only on
    overflow, which is an fp16 problem (5 exponent bits, tiny range). bf16 shares
    fp32's exponent range, so gradients essentially never overflow/underflow —
    loss scaling is an fp16 fix and is usually unnecessary in bf16.

??? success "4 — fp8 E4M3 with per-tensor scale, max = 1000"
    E4M3 max representable $\approx 448$. Pick scale $s = 448/1000 = 0.448$ so the
    tensor max lands near the top of range. **Quantize** $q = \text{cast}_{E4M3}(x
    \cdot s)$; **dequantize** $\hat x = q / s$. With 3 mantissa bits the
    half-ULP relative error is $\le 2^{-(3+1)} = 6.25\%$ (a few percent typical).
    The error is dominated by the **coarse mantissa**, not the scale choice —
    which is why fp8 needs fine-grained (per-tensor/per-block) scaling and is
    applied to tolerant tensors (e.g. routed-expert weights), not the router.
