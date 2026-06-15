# Glossary

Concise definitions of the terms used across the handbook. Links point to the
page where each is developed.

## Systems & performance

**Arithmetic intensity ($I$)**
: FLOPs performed per byte moved from memory, $I = W/Q$. Determines roofline
regime. See [transformer as a system](foundations/transformer-systems.md).

**Roofline model**
: Performance bound $P=\min(\pi, \beta I)$ from peak compute $\pi$ and bandwidth
$\beta$. The organizing idea of the whole handbook.

**Compute-bound / memory-bound**
: Limited by the math units (right of the roofline ridge) vs by memory bandwidth
(left of it). Decoding is memory-bound; large-batch matmuls are compute-bound.

**MFU (model FLOPs utilization)**
: Achieved model FLOPs ÷ peak FLOPs; $\approx 6P\cdot\text{tok/s}/\pi$ for
training. The headline efficiency metric. See [profiling](performance/profiling.md).

**HBM**
: High-bandwidth memory — the GPU's main DRAM; the bandwidth $\beta$ in the
roofline.

**SRAM / shared memory / LDS**
: Fast on-chip scratchpad. "Shared memory" (NVIDIA) = "LDS" (AMD). Staging tiles
here is how kernels raise intensity.

**Warp / wavefront**
: The lock-step SIMT execution group — **32 threads** on NVIDIA (warp), **64**
on AMD CDNA (wavefront). A frequent portability trap. See
[GPU programming](performance/gpu-programming.md).

**Occupancy**
: Resident warps/wavefronts per SM/CU; a latency-hiding means, not the goal.

**Coalescing**
: Consecutive lanes accessing consecutive addresses so memory transactions merge.

**Operator fusion**
: Combining ops to avoid HBM round-trips (e.g. FlashAttention, fused MLP) — raises
intensity.

**FLOPs**
: Floating-point operations; a matmul $(m{\times}k)(k{\times}n)$ costs $2mkn$.

## Precision

**bf16 / fp16 / fp8**
: 16/16/8-bit floats. bf16 keeps fp32's exponent range (8 bits) and won training;
fp16 has more mantissa but narrow range; fp8 (E4M3/E5M2) is the frontier. See
[numerics](foundations/numerics-precision.md).

**Mixed precision**
: Low-precision storage/matmul + fp32 accumulation + fp32 master weights.

**Loss scaling**
: Multiplying the loss to keep fp16 gradients in range; largely unneeded with
bf16.

## Attention

**KV cache**
: Stored keys/values of past tokens so decoding is $O(N)$ not $O(N^2)$; often the
dominant inference memory. See [attention efficiency](foundations/attention-efficiency.md).

**MQA / GQA / MLA**
: Multi-Query / Grouped-Query / Multi-head Latent Attention — architectural ways
to shrink the KV cache (fewer or compressed KV heads).

**FlashAttention**
: IO-aware attention that tiles $Q,K,V$ and uses online softmax to avoid
materializing the $N{\times}N$ score matrix. See
[FlashAttention](foundations/flashattention.md).

**Online softmax**
: Single-pass, numerically-stable softmax via a running max and the correction
factor $e^{m_{old}-m_{new}}$.

**PagedAttention**
: Block-based KV-cache allocation (like virtual memory paging) that removes
fragmentation and enables sharing.

## Mixture-of-Experts

**MoE (Mixture-of-Experts)**
: A layer with many expert FFNs and a router that activates a few per token,
decoupling total parameters from per-token FLOPs. See [Part II](moe/index.md).

**Expert**
: One of the parallel FFNs in an MoE layer (usually SwiGLU).

**Router / gate**
: The small network producing per-expert scores; top-$k$ selects experts.
**Softmax gating** makes experts compete; **sigmoid gating** scores them
independently.

**Top-$k$ routing**
: Each token uses its $k$ highest-scoring experts.

**Token-choice vs expert-choice**
: Tokens pick experts (coverage guaranteed, balance not) vs experts pick their
top-$C$ tokens (balance guaranteed, coverage not). See
[routing variants](moe/routing-variants.md).

**Shared expert**
: An always-on FFN added to the routed experts to absorb common knowledge and
stabilize training.

**Fine-grained experts**
: Many small experts instead of few large ones, enlarging the combinatorial mix
space at fixed active compute.

**Auxiliary (load-balancing) loss**
: Penalty $\alpha E\sum_e f_e P_e$ encouraging uniform routing. See
[load balancing](moe/load-balancing.md).

**Aux-loss-free balancing**
: Balancing via a controller-updated per-expert **bias on selection** (not the
gate weight), avoiding gradient distortion (DeepSeek-style).

**Expert capacity / capacity factor**
: Max tokens per expert per batch; the factor trades dropped tokens (quality) vs
padding/buffer size (throughput/memory).

**Token dropping / overflow**
: Tokens beyond capacity skip the MoE layer (carried by the residual).

**Router z-loss**
: $\beta(\log\sum_e e^{x_e})^2$ penalizing large router logits for stability. See
[training stability](moe/training-stability.md).

**Expert parallelism (EP)**
: Sharding experts across GPUs; tokens reach their expert via all-to-all. See
[systems & EP](moe/systems-ep.md).

**Grouped GEMM**
: One kernel doing many different-sized matmuls (one per expert) without padding.

**MegaBlocks / block-sparse MoE**
: Reformulating the MoE FFN as a block-sparse matmul to avoid token dropping and
padding.

## Distributed training

**Collectives**
: All-reduce, all-gather, reduce-scatter, all-to-all, broadcast/P2P (NCCL/RCCL).
See [distributed training](performance/distributed-training.md).

**Data / tensor / pipeline / sequence / expert parallelism (DP/TP/PP/SP/EP)**
: The dimensions along which training is split; composed into N-D parallelism.

**ZeRO / FSDP**
: Sharding optimizer state (1), gradients (2), and parameters (3) across the DP
group to cut memory.

**All-to-all**
: The collective where each rank sends a distinct chunk to every other rank — the
MoE dispatch/combine primitive.

**Node-limited routing**
: Capping how many nodes a token's experts span, to bound cross-node all-to-all.

## Inference & compression

**Prefill / decode**
: Processing the prompt (many tokens, compute-bound) vs generating one token at a
time (memory-bound).

**Continuous (in-flight) batching**
: Iteration-level scheduling that swaps finished sequences for waiting ones,
keeping the GPU full. See [inference optimization](performance/inference-optimization.md).

**Speculative decoding**
: A cheap draft proposes tokens, the target verifies them in one pass; lossless,
exploits decode's spare compute.

**PTQ / QAT**
: Post-Training Quantization (calibration only) vs Quantization-Aware Training.
See [quantization](performance/quantization.md).

**GPTQ / AWQ / SmoothQuant**
: PTQ methods: error-corrected weight quantization / activation-aware weight
scaling / migrating activation outliers into weights.

**Pruning**
: Removing weights or structures (unstructured, structured, or 2:4
semi-structured) to compress.

**Distillation**
: Training a small student to mimic a large teacher.

**TTFT / TPOT (ITL)**
: Time To First Token (prefill latency) / Time Per Output Token (decode latency).

## Hardware quick reference

Round numbers for back-of-envelope estimates (the exercises and
[solutions](solutions/index.md) use these). Peak FLOP/s is dense bf16; real
kernels hit a fraction (MFU). Bandwidth is HBM. The **ridge** $\pi/\beta$ is the
arithmetic intensity where a chip flips from memory- to compute-bound.

| GPU | bf16 peak ($\pi$) | HBM BW ($\beta$) | HBM size | Ridge $\pi/\beta$ |
|---|---|---|---|---|
| A100 (80 GB) | ~312 TFLOP/s | ~2.0 TB/s | 80 GB | ~156 FLOP/byte |
| H100 (SXM) | ~990 TFLOP/s | ~3.35 TB/s | 80 GB | ~295 FLOP/byte |
| H200 | ~990 TFLOP/s | ~4.8 TB/s | 141 GB | ~206 FLOP/byte |
| MI300X | ~1.3 PFLOP/s | ~5.3 TB/s | 192 GB | ~245 FLOP/byte |

Interconnect (for [distributed training](performance/distributed-training.md)):
intra-node **NVLink** ~0.9 TB/s/GPU (NVLink 4) or **Infinity Fabric** on MI300;
cross-node **InfiniBand/RoCE** ~25–50 GB/s/GPU; host **PCIe Gen5** ~64 GB/s. The
1–2 order-of-magnitude drop from HBM → NVLink → IB → PCIe is why parallelism is
mapped so the chattiest collectives ride the fastest links.

!!! note "Using these"
    Numbers vary by SKU, clocks, and sparsity claims (vendors often quote
    2× with structured sparsity — halve for dense). What should be robust in your
    estimates is the **regime** and the **order of magnitude**, not the third
    significant figure.
