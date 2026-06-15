# Part IV · Capstones

Two end-to-end projects that put the whole handbook together. The first is fully
worked with runnable code; the second is a structured guide for taking that model
to a multi-GPU setting.

## Pages

1. **[Build a small MoE LM](build-moe.md)** — assemble a tiny MoE language model
   from the Part II components, train it on a toy corpus, then optimize it and
   **report the measured speedups** using the
   [profiling methodology](../performance/profiling.md).
2. **[Scaling it up](scaling.md)** — apply the
   [parallelism techniques](../performance/distributed-training.md) (DP/ZeRO, TP,
   PP, and [EP](../moe/systems-ep.md)) to the model you built, with a planning
   guide for mapping it onto real hardware.

!!! tip "These are checkpoints, not just reading"
    The capstones reference real, tested code under
    [`code/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code). The
    goal is that you can run, modify, and measure — not just read.
