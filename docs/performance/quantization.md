# Quantization & compression

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> intermediate</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../../foundations/numerics-precision/">numerics & precision</a>, <a href="../../foundations/attention-efficiency/">memory-bound decoding</a></span>
  <span class="chip"><strong>Hardware:</strong> none for theory; GPU to benchmark</span>
</div>

Compression makes models cheaper to store and faster to run — especially at
[memory-bound decode](../foundations/attention-efficiency.md), where halving the
weight bytes roughly halves latency. This page covers **quantization** (the big
lever: PTQ vs QAT, and the GPTQ/AWQ family), then **pruning** and
**distillation**. The [MoE serving page](../moe/inference-serving.md) applies all
of this to experts specifically.

## Quantization basics

Quantization maps high-precision values to a small integer (or low-bit float)
grid. The standard affine scheme for a tensor $x$:

$$ q = \text{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s\,(q - z), $$

where $s$ (scale) and $z$ (zero-point) are chosen so the integer grid covers the
tensor's range. **Granularity** matters enormously:

- **Per-tensor**: one $(s,z)$ for the whole tensor — cheapest, least accurate.
- **Per-channel / per-row**: one scale per output channel — standard for weights.
- **Per-group** (e.g. 128 elements): one scale per small block — the sweet spot
  for 4-bit weights; pairs with [microscaling/MX formats](../foundations/numerics-precision.md).

Two axes of *what* you quantize:

- **Weight-only** (W8/W4, activations stay in bf16): great for **decode**, which
  is weight-bandwidth-bound — you cut the bytes read per token. Compute uses
  dequantized values, so it doesn't speed up compute-bound prefill much.
- **Weight + activation** (W8A8 / fp8): also speeds up the **matmul** (integer/fp8
  tensor cores), helping compute-bound prefill/training — but activations are
  harder to quantize (outliers).

## PTQ vs QAT

- **Post-Training Quantization (PTQ)**: quantize a trained model using a small
  **calibration** set to pick scales (and, for GPTQ/AWQ, to correct error). Cheap,
  fast, no retraining — the default for LLM inference.
- **Quantization-Aware Training (QAT)**: simulate quantization during training
  (straight-through estimator for the round) so the model learns to be robust.
  Recovers more accuracy at very low bits, but costs a training run.

The hard part of PTQ for LLMs is **activation outliers**: a few channels have
huge magnitudes that, if quantized naively, blow up the scale and crush
everything else. The GPTQ/AWQ family is about handling this.

### GPTQ — error-corrected weight quantization

GPTQ quantizes weights one column at a time and, after rounding each, **updates
the remaining unquantized weights to compensate** for the error introduced (a
second-order/Hessian-based correction using the calibration activations). The
result: accurate 3–4 bit *weight-only* quantization with small accuracy loss.
Weight-only → ideal for decode.

### AWQ — activation-aware weight quantization

AWQ observes that not all weights matter equally: weights multiplying
high-magnitude (salient) activation channels are the most important. It **scales
those salient channels up before quantizing** (and compensates), protecting them
from rounding error — no backprop needed. Often matches or beats GPTQ at 4-bit
and is simple/fast.

### SmoothQuant — make activations quantizable

For W8A8, SmoothQuant **migrates** the activation outlier magnitude into the
weights (per-channel) via a mathematically-equivalent rescaling, so both
activations and weights become easy to quantize to int8 — enabling fast int8
matmuls for prefill/serving.

```python
# The shared idea: choose per-channel scales so the *product* is unchanged
# but each operand is easier to quantize. (Conceptual sketch.)
# y = (x / s) @ (s * W)   # s absorbs outliers from x into W or vice-versa
```

!!! tip "Which to use"
    Memory-bound **decode**, want simple + accurate → **AWQ or GPTQ** (W4,
    weight-only). Want faster **prefill/serving compute** too → **SmoothQuant**
    or **fp8** (W8A8). Pushing below 4-bit with accuracy → consider **QAT**.

## fp8 as quantization

fp8 (E4M3/E5M2) is quantization with a floating grid — better for the wide
dynamic range of activations than int8, and natively accelerated on H100/MI300.
With per-tensor/per-block scales it's used for both inference (W8A8-style) and,
increasingly, **training** (DeepSeek-V3). See
[numerics & precision](../foundations/numerics-precision.md) for the format
details and scaling discipline.

## Pruning

Remove weights/structures instead of shrinking their precision:

- **Unstructured** (zero out individual small weights): high compression in
  theory, but irregular sparsity is hard to accelerate on GPUs.
- **Structured** (remove whole heads/channels/layers): less compression, but
  yields a smaller *dense* model that runs fast on stock kernels.
- **2:4 semi-structured** (2 of every 4 weights zero): a hardware-supported
  middle ground — NVIDIA Sparse Tensor Cores give ~2× on these. A practical
  sweet spot when supported.

Pruning usually needs fine-tuning to recover quality; it's most attractive when
you can exploit hardware sparsity or want a smaller dense model.

## Distillation

Train a small **student** to mimic a large **teacher** (matching output
distributions / soft logits, and sometimes intermediate features). Unlike
quantization/pruning, it produces a genuinely smaller architecture and can
transfer capabilities, at the cost of a training run and access to the teacher.
Often combined with the above (distill, then quantize the student).

## Key takeaways

- Quantization maps to a low-bit grid via scale/zero-point; **granularity**
  (per-tensor → per-group) trades accuracy for cost.
- **Weight-only (GPTQ/AWQ, W4)** speeds up memory-bound **decode**;
  **weight+activation (SmoothQuant/fp8, W8A8)** also speeds up compute-bound
  **prefill/training** but must tame **activation outliers**.
- **PTQ** (calibration only) is the LLM default; **QAT** buys accuracy at very low
  bits for the price of training.
- **Pruning** (esp. 2:4 structured) and **distillation** are complementary
  compression tools; combine them.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/performance.md). Try each exercise before expanding.

1. Derive the int8 affine quantize/dequantize and the max quantization error for
   per-tensor vs per-channel scales on a tensor with one outlier channel.
2. Implement AWQ's salient-channel scaling on a single linear layer and measure
   perplexity vs naive per-channel int4.
3. Estimate decode-latency improvement from W4 weight-only on a 13B model (use the
   [memory-bound](../foundations/attention-efficiency.md) argument).
4. For MoE, argue why routed experts tolerate aggressive (int4) quantization
   better than the router or attention. (Connect to [MoE serving](../moe/inference-serving.md).)

## References

- Frantar et al. *GPTQ.* 2022.
- Lin et al. *AWQ.* 2023.
- Xiao et al. *SmoothQuant.* 2022.
- Dettmers et al. *LLM.int8() / GPTQ-era outlier analysis.* 2022.
- Mishra et al. *2:4 structured sparsity.* 2021; Hinton et al. *Distilling the Knowledge in a Neural Network.* 2015.
