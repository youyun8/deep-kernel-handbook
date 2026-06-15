---
title: ML Perf Handbook
hide:
  - navigation
  - toc
---

<section class="home-hero" markdown>
<div class="home-hero__copy" markdown>

# Learn SOTA ML systems from scratch

Modern machine learning is a *systems* discipline. Knowing the math of a
transformer is table stakes; the leverage is in understanding **how it runs**
on real hardware: the FLOPs, bytes, kernels, collectives, and trade-offs that
decide whether a model is practical.

This handbook teaches state-of-the-art ML techniques together with the systems
work that makes them efficient. Each core component moves from **intuition** to
**math**, then to a **clean reference implementation**, then to the optimized
version that scales across GPUs.

<div class="home-actions" markdown>
[Start the reading path :material-arrow-right:](reading-path.md){ .md-button .md-button--primary }
[Jump to the MoE flagship](moe/index.md){ .md-button }
</div>

</div>

<div class="home-hero__panel" aria-label="Course scope" markdown="0">
<div class="home-kicker">Flagship track</div>
<div class="home-panel-title">Mixture-of-Experts</div>
<div class="home-panel-copy">From sparse scaling and router math to all-to-all dispatch, grouped GEMM, inference serving, and case studies.</div>
<div class="home-metrics">
<div><strong>4</strong><span>parts</span></div>
<div><strong>9</strong><span>MoE chapters</span></div>
<div><strong>3</strong><span>kernel paths</span></div>
</div>
</div>
</section>

---

## What makes this different

<div class="grid cards" markdown>

-   :material-function-variant: **Intuition → math → code → systems**

    No hand-waving on the hard parts. When a step is subtle — online-softmax
    rescaling, aux-loss-free routing, all-to-all bucketing — you see the
    algebra *and* the array shapes.

-   :material-flash: **Measured, not asserted**

    Every performance topic shows the naive version, profiles it, then the
    optimized version with **before/after numbers** and the methodology to
    reproduce them.

-   :material-expansion-card: **CUDA *and* ROCm/HIP, side by side**

    AMD Instinct (MI300-class) is a first-class target, not an afterthought.
    We flag where warp vs wavefront (32 vs 64), occupancy, and APIs differ.

-   :material-play-circle: **Runnable from scratch**

    Reference implementations live in a tested `code/` tree — checked against
    PyTorch with `torch.allclose`, not just pasted into prose.

</div>

---

## The curriculum

<div class="curriculum-grid" markdown>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">Part I</span>

### :material-cube-outline: Foundations

Get fluent in FLOPs, bytes, arithmetic intensity, attention memory traffic, and
precision before optimizing anything.

[Open foundations](foundations/index.md){ .md-button }
</section>

<section class="curriculum-card curriculum-card--feature" markdown>
<span class="curriculum-card__eyebrow">Part II · flagship</span>

### :material-star-circle: Mixture-of-Experts

Build a production-shaped MoE stack: sparsity, routing, load balancing, expert
parallelism, kernels, serving, and case studies.

[Open MoE track](moe/index.md){ .md-button .md-button--primary }
</section>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">Part III</span>

### :material-speedometer: Performance

Work through GPU programming, Triton, CUDA/HIP, distributed training,
quantization, inference optimization, and profiling.

[Open performance](performance/index.md){ .md-button }
</section>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">Part IV</span>

### :material-trophy: Capstones

Train a small MoE LM, optimize it, report measured speedups, then apply the
parallelism techniques that make it scale.

[Open capstones](capstones/index.md){ .md-button }
</section>

</div>

---

## Who this is for

You know Python and basic deep learning (you can read a training loop and you
know what a softmax is) and you want to understand modern ML *systems* deeply —
from first principles up to production-grade implementations. You do **not**
need prior GPU-programming experience; Part III builds it from the memory
hierarchy up.

!!! tip "How to read it"
    If you're new, follow the [reading path](reading-path.md) top to bottom.
    If you came for MoE, skim [Part I](foundations/index.md) for the systems
    vocabulary (roofline, arithmetic intensity, all-to-all) then dive into the
    [MoE flagship](moe/index.md).

---

*Content is licensed CC BY 4.0; code is MIT. Contributions welcome — see the
[contributor guide](https://github.com/youyun8/ml-perf-handbook/blob/main/CONTRIBUTING.md).*
