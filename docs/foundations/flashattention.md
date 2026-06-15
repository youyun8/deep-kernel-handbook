# FlashAttention from scratch

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../attention-efficiency/">attention efficiency</a>, softmax, roofline</span>
  <span class="chip"><strong>Code:</strong> <code>code/attention/</code> (runs on CPU)</span>
</div>

FlashAttention is the textbook example of the roofline playbook: it computes
*exactly the same* attention output but **never writes the $N\times N$ score
matrix to HBM**, turning a memory-bound op into a compute-bound one. The trick
is **online softmax** — computing a numerically-stable softmax in a single
streaming pass — combined with **tiling**. We derive both here and give a numpy
reference you can run and check against PyTorch.

## The problem with naive attention

Standard attention for one query block:

```python
S = Q @ K.T / sqrt(d)      # [N, N]  <- materialized in HBM
P = softmax(S, axis=-1)    # [N, N]  <- read + write again
O = P @ V                  # [N, d]
```

The score matrix $S$ is $N\times N$. At $N=8192$ that's 67M entries — written to
HBM, read back for the softmax, written again, read again for $PV$. The matmuls
are cheap relative to this traffic; we're **memory-bound on a quadratic
tensor**. We want to produce $O$ while only ever holding small *tiles* of $S$ in
on-chip SRAM.

The obstacle: **softmax needs a normalizer over the whole row** ($\sum_j
e^{s_j}$), which seems to require all of $S$ at once. Online softmax removes that
obstacle.

## Online softmax: the running-max trick

We want $\text{softmax}(x)_i = e^{x_i - m} / \sum_j e^{x_j - m}$ where
$m=\max_j x_j$ (subtracting the max is what keeps it numerically stable — see
[numerics](numerics-precision.md)). Suppose we see $x$ in two chunks
$x^{(1)}, x^{(2)}$ and want to combine partial results.

Maintain a running max $m$ and running denominator $\ell$. After chunk 1:

$$ m_1 = \max(x^{(1)}), \qquad \ell_1 = \sum_{j} e^{x^{(1)}_j - m_1}. $$

When chunk 2 arrives with local max $m_2' = \max(x^{(2)})$, the **new global
max** is $m_2 = \max(m_1, m_2')$. The old denominator was computed against the
*old* max, so we **rescale** it by $e^{m_1 - m_2}$ before adding the new chunk's
contribution:

$$ \ell_2 = e^{m_1 - m_2}\,\ell_1 + \sum_j e^{x^{(2)}_j - m_2}. $$

That correction factor $e^{m_{\text{old}} - m_{\text{new}}}$ is the entire idea.
It lets us fold in chunks one at a time and get the *exact* softmax denominator
at the end — never needing all of $x$ simultaneously.

### Extending to the weighted sum $O = PV$

Attention doesn't just need the denominator; it needs $O = \sum_j p_j v_j$ with
$p_j = e^{s_j - m}/\ell$. Keep an **unnormalized** running output
$\tilde{O} = \sum_j e^{s_j - m} v_j$ and rescale it by the *same* factor whenever
the max updates:

$$ \tilde{O} \leftarrow e^{m_{\text{old}} - m_{\text{new}}}\,\tilde{O} + \sum_{j \in \text{block}} e^{s_j - m_{\text{new}}}\, v_j. $$

At the very end, $O = \tilde{O} / \ell$. We now have a streaming algorithm that
touches each key/value block exactly once.

## The tiled algorithm

Split $K, V$ into blocks of $B_c$ rows and $Q$ into blocks of $B_r$ rows. For
each query block, loop over key/value blocks, maintaining $(m, \ell, \tilde O)$
per query row:

```text
for each query block Qi:                      # outer (rows of output)
    m = -inf;  l = 0;  O_acc = 0              # per-row running state
    for each key/value block (Kj, Vj):        # inner (streaming)
        S = Qi @ Kj.T / sqrt(d)               # [Br, Bc]  small tile, stays in SRAM
        apply causal mask to S if needed
        m_new = max(m, rowmax(S))             # update running max
        P = exp(S - m_new)                     # [Br, Bc]
        alpha = exp(m - m_new)                 # correction for old state
        l = alpha * l + rowsum(P)
        O_acc = alpha * O_acc + P @ Vj
        m = m_new
    Oi = O_acc / l                            # normalize once at the end
    write Oi to HBM                            # the ONLY N×d write
```

Memory traffic to HBM is now $O(N d)$ (read $Q,K,V$ once, write $O$ once) instead
of $O(N^2)$. The score tiles live and die in SRAM. The FLOPs are unchanged — so
on the roofline we've moved sharply **right** (higher intensity) and the kernel
becomes compute-bound. That's the whole win.

!!! note "The backward pass"
    The backward pass recomputes the tiles of $S$ on the fly (cheap) rather than
    storing them — trading a little extra compute for a massive memory saving.
    FlashAttention-2 improves work partitioning (fewer rescalings, parallelize
    over the sequence dim); FlashAttention-3 exploits Hopper's async copy
    (TMA) and fp8. The *math above is identical* across all versions.

## Reference implementation (runnable)

A faithful, readable numpy implementation lives in
[`code/attention/flash_attention_numpy.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/attention/flash_attention_numpy.py).
The core loop:

```python
import numpy as np

def flash_attention(Q, K, V, block=64, causal=True):
    """Tiled, online-softmax attention. Matches softmax(QK^T/sqrt(d))V exactly."""
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((N, d), dtype=np.float32)
    for i in range(0, N, block):                      # query tile
        qi = Q[i:i+block] * scale
        m = np.full((qi.shape[0], 1), -np.inf)        # running max
        l = np.zeros((qi.shape[0], 1))                # running denom
        acc = np.zeros((qi.shape[0], d))              # unnormalized output
        for j in range(0, N, block):                  # key/value tile
            if causal and j > i + block - 1:
                break                                 # skip fully-masked tiles
            kj, vj = K[j:j+block], V[j:j+block]
            s = qi @ kj.T                             # [Br, Bc] in "SRAM"
            if causal:                                # mask within the diagonal tile
                qpos = (i + np.arange(qi.shape[0]))[:, None]
                kpos = (j + np.arange(kj.shape[0]))[None, :]
                s = np.where(kpos <= qpos, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1, keepdims=True))
            p = np.exp(s - m_new)                      # [Br, Bc]
            alpha = np.exp(m - m_new)                  # rescale old state
            l = alpha * l + p.sum(axis=1, keepdims=True)
            acc = alpha * acc + p @ vj
            m = m_new
        O[i:i+block] = acc / l                         # normalize once
    return O
```

The test [`code/attention/test_attention.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/attention/test_attention.py)
checks it against a dense PyTorch reference with `torch.allclose` (atol 1e-5) for
random inputs, with and without the causal mask. Run it:

```bash
pip install -r code/requirements.txt
pytest code/attention -q
```

A standalone [`online_softmax.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/attention/online_softmax.py)
shows the running-max combiner in isolation and proves it equals a one-shot
softmax — read that first if the rescaling feels like magic.

The GPU version (a real tiled Triton kernel) is in the
[Triton track](../performance/triton-track.md); the same online-softmax math
reappears, just with `tl.load`/`tl.dot` over SRAM tiles.

## Key takeaways

- Softmax's row-normalizer seems to block streaming, but **online softmax**
  computes the exact result in one pass via a running max $m$, running
  denominator $\ell$, and the correction factor $e^{m_{\text{old}}-m_{\text{new}}}$.
- FlashAttention tiles $Q,K,V$ and keeps score tiles in SRAM, cutting HBM traffic
  from $O(N^2)$ to $O(Nd)$ — a roofline move from memory-bound to compute-bound.
- The output is **numerically identical** to naive attention (up to fp rounding);
  it's a systems optimization, not an approximation.
- The same algorithm underlies FA-2/FA-3 and every fused attention kernel,
  including the MoE-friendly ones.

## Exercises

1. Prove the online-softmax combiner is exact: show that folding in chunks gives
   the same $\ell$ and $\tilde O$ as computing softmax over the full row.
2. Why subtract the running max at all? Construct inputs (e.g. scores of +100)
   where skipping it overflows fp16, and confirm the stable version survives.
3. Modify the reference to skip key tiles that are *fully* masked under causality
   (already partially done) and measure the FLOP reduction for $N=4096$.
4. Estimate HBM bytes moved by naive vs flash attention at $N=8192, d=128$ and
   place both on an H100 roofline.

## References

- Milakov & Gimelshein. *Online normalizer calculation for softmax.* 2018.
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* 2022.
- Dao. *FlashAttention-2.* 2023.
- Shah et al. *FlashAttention-3.* 2024.
- Rabe & Staats. *Self-attention Does Not Need $O(n^2)$ Memory.* 2021.
