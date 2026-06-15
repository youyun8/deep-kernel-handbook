# Part III · Performance & systems engineering

The general toolkit that powers everything in the MoE flagship and beyond: how
GPUs actually execute work, how to write custom kernels in Triton and CUDA/HIP,
how to spread training across many devices, how to compress models, how to serve
them fast, and — underpinning all of it — how to **measure** so you optimize the
right thing.

Read this part **alongside** [Part II](../moe/index.md); the MoE systems and
kernels pages link directly into it.

## Pages

**Kernels (build up from the hardware)**

1. [GPU programming model](gpu-programming.md) — the execution and memory
   hierarchy, CUDA and ROCm/HIP side by side.
2. [Triton track](triton-track.md) — productive kernel writing; vector add →
   fused softmax → matmul → attention.
3. [CUDA / HIP track](cuda-hip-track.md) — the low level, with portability across
   NVIDIA and AMD as a first-class concern.

**Scale**

4. [Distributed training](distributed-training.md) — data / tensor / pipeline /
   sequence / expert parallelism, ZeRO, and the collectives underneath.

**Deploy**

5. [Quantization & compression](quantization.md) — PTQ/QAT, GPTQ/AWQ, pruning,
   distillation.
6. [Inference optimization](inference-optimization.md) — continuous batching,
   speculative decoding, KV-cache management, serving systems.

**Always**

7. [Profiling & methodology](profiling.md) — how to measure, what to trust, and
   the benchmarking pitfalls that produce fake speedups. **Read this early.**

!!! tip "The throughline"
    Every page here is an application of the [roofline](../foundations/transformer-systems.md):
    kernels raise arithmetic intensity, parallelism trades compute for
    communication, quantization cuts bytes, and profiling tells you which wall
    you're actually hitting.
