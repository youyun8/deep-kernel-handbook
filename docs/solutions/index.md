# Solutions

<div class="page-meta">
  <span class="chip"><strong>Scope:</strong> every exercise in the handbook</span>
  <span class="chip"><strong>Format:</strong> collapsible worked answers</span>
</div>

Worked answers to all of the handbook's exercises, organized by Part. Each answer
is a collapsible block — **attempt the exercise first**, then expand to check your
reasoning and numbers. Closed-form derivations are given in full; build-and-measure
exercises give the expected result, the correct methodology, and the traps to
avoid.

<div class="grid cards" markdown>

- :material-cube-outline: **[Part I · Foundations](foundations.md)**

    FLOP/byte counting, KV cache, FlashAttention, numerics.

- :material-set-split: **[Part II · Mixture-of-Experts](moe.md)**

    Sparsity math, balancing, routing, stability, EP, kernels, serving.

- :material-speedometer: **[Part III · Performance](performance.md)**

    GPU model, Triton/CUDA/HIP, distributed training, quant, inference, profiling.

- :material-rocket-launch-outline: **[Part IV · Capstones](capstones.md)**

    End-to-end build-and-measure for the small MoE LM.

</div>

!!! tip "How to get the most from these"
    The numbers use round hardware specs (e.g. A100 ≈ 312 TFLOP/s bf16 / 2 TB/s;
    H100 ≈ 990 TFLOP/s / 3.35 TB/s; MI300X ≈ 1.3 PFLOP/s / 5.3 TB/s). Your exact
    figures will shift with the chip you assume — what should match is the
    **regime** (memory- vs compute-bound) and the **order of magnitude**.
