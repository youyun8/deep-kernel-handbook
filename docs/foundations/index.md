# Part I · Foundations of modern ML systems

Before you can make a model fast, you need to be able to **predict** how fast it
*should* be — and to know whether a given operation is limited by the GPU's
arithmetic units or by its memory bus. Part I builds that fluency.

By the end of this part you will be able to:

- Explain what a transformer **is** from first principles — tokens, attention,
  multi-head, the FFN, and the full block — and trace a token end to end.
- Count the FLOPs and bytes of a transformer forward/backward pass, and compute
  its **arithmetic intensity** to place it on a **roofline**.
- Explain precisely why LLM **decoding is memory-bound** while training/prefill
  is (mostly) compute-bound, using the KV cache.
- Derive **FlashAttention** — tiling + online softmax — from first principles,
  and explain why it turns an $O(N^2)$ memory problem into an $O(N)$ one.
- Choose between **fp32 / bf16 / fp16 / fp8**, and reason about overflow,
  underflow, and loss scaling.

## Pages

1. **[The transformer from scratch](transformer-from-scratch.md)** — what a
   transformer *is*, built one diagram at a time: tokens, attention, multi-head,
   FFN, the full block. Start here if transformers are new.
2. **[The transformer as a system](transformer-systems.md)** — the roofline
   model, FLOP/byte accounting, and where the time actually goes.
3. **[Attention efficiency](attention-efficiency.md)** — KV cache, the
   memory-bandwidth wall in decoding, and paged attention.
4. **[FlashAttention from scratch](flashattention.md)** — online softmax and
   tiling, with a numpy reference implementation.
5. **[Numerics & precision](numerics-precision.md)** — floating-point formats,
   mixed precision, and numerical stability.

!!! tip "Prerequisite for the whole handbook"
    The single most useful idea in Part I is **arithmetic intensity** and the
    **roofline**. Almost every optimization later — FlashAttention, grouped
    GEMM, fusing the MoE router, quantization — is ultimately a move on the
    roofline. If you read only one page first, read
    [the transformer as a system](transformer-systems.md).
