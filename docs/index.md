---
title: ML Perf Handbook
hide:
  - navigation
  - toc
---

# Learn SOTA ML systems — from scratch

> Modern machine learning is a *systems* discipline. Knowing the math of a
> transformer is table stakes; the leverage is in understanding **how it runs**
> on real hardware — the FLOPs, the bytes, the kernels, the collectives — and
> rebuilding every piece yourself until there is no magic left.

This handbook teaches state-of-the-art ML techniques **together with the
performance-engineering and systems work** that makes them efficient. Every core
component is built three times: first the **math and intuition**, then a
**clean reference implementation** you can read in one sitting, then the
**performance-optimized version** and how it **scales across many GPUs**.

The flagship, deepest track is **Mixture-of-Experts (MoE)** — the architecture
behind today's frontier sparse models (Kimi K2.5, DeepSeek-V3, Mixtral,
Qwen-MoE). We go all the way from "why does sparsity help at all?" to writing
the all-to-all dispatch and the grouped-GEMM kernels that make it fast.

[Start the reading path :material-arrow-right:](reading-path.md){ .md-button .md-button--primary }
[Jump to the MoE flagship :material-arrow-right:](moe/index.md){ .md-button }

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

### :material-cube-outline: Part I — Foundations of modern ML systems

Get fluent in the language of performance before optimizing anything.

- [The transformer as a system](foundations/transformer-systems.md) — FLOPs, memory traffic, arithmetic intensity, and the roofline model.
- [Attention efficiency](foundations/attention-efficiency.md) — KV cache, memory-bound decoding, paged attention.
- [FlashAttention from scratch](foundations/flashattention.md) — tiling and online softmax, derived step by step.
- [Numerics & precision](foundations/numerics-precision.md) — fp32/bf16/fp16/fp8, mixed precision, and stability.

### :material-star-circle: Part II — Mixture-of-Experts *(flagship)*

The deepest series. Build a production-shaped MoE stack from the ground up.

- [Why sparsity](moe/why-sparsity.md) · [MoE layer from scratch](moe/moe-from-scratch.md) · [Load balancing](moe/load-balancing.md)
- [Routing variants](moe/routing-variants.md) · [Training stability](moe/training-stability.md)
- [Systems & expert parallelism](moe/systems-ep.md) · [MoE kernels (Triton/CUDA/HIP)](moe/kernels.md)
- [Inference & serving](moe/inference-serving.md) · [Case studies](moe/case-studies.md)

### :material-speedometer: Part III — Performance & systems engineering

The general toolkit that powers everything above.

- [GPU programming model](performance/gpu-programming.md) · [Triton track](performance/triton-track.md) · [CUDA / HIP track](performance/cuda-hip-track.md)
- [Distributed training](performance/distributed-training.md) · [Quantization & compression](performance/quantization.md)
- [Inference optimization](performance/inference-optimization.md) · [Profiling & methodology](performance/profiling.md)

### :material-trophy: Part IV — Capstones

- [Build a small MoE LM end to end](capstones/build-moe.md), then optimize it and report the speedups.
- [Scaling it up](capstones/scaling.md) with the parallelism techniques.

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
