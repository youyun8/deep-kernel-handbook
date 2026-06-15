# Bibliography & annotated references

A curated, annotated reading list grouped by topic. These are the primary sources
behind the handbook — start here when you want the original detail. (Look up the
latest version/venue; ariv IDs change with revisions.)

!!! note "How to use this list"
    Each entry says *why it matters* so you can prioritize. For MoE, read in the
    order: Shazeer 2017 → Switch/GShard → DeepSeekMoE → DeepSeek-V3 → MegaBlocks.
    For kernels: the Triton paper + tutorials, then FlashAttention.

## Foundations: systems, scaling, precision

- **Williams, Waterman, Patterson — "Roofline: An Insightful Visual Performance
  Model" (CACM 2009).** The model the whole handbook is organized around. Read
  for the compute-vs-bandwidth framing.
- **Kaplan et al. — "Scaling Laws for Neural Language Models" (2020).** Where the
  $\sim$power-law loss scaling and the $6P$-FLOP intuition come from.
- **Hoffmann et al. — "Training Compute-Optimal LLMs" (Chinchilla, 2022).**
  Compute-optimal token/parameter trade; essential for budgeting.
- **Micikevicius et al. — "Mixed Precision Training" (2017).** The fp16 + fp32
  master weights + loss scaling recipe.
- **Kalamkar et al. — "A Study of BFLOAT16 for Deep Learning Training" (2019).**
  Why bf16's range beats fp16 for training.
- **Micikevicius et al. — "FP8 Formats for Deep Learning" (2022).** E4M3/E5M2 and
  why each is used where.

## Attention efficiency

- **Milakov & Gimelshein — "Online normalizer calculation for softmax" (2018).**
  The streaming-softmax trick FlashAttention builds on.
- **Dao, Fu, Ermon, Rudra, Ré — "FlashAttention" (2022).** IO-aware exact
  attention; the canonical fuse-to-save-bandwidth result. Follow-ups:
  **FlashAttention-2 (2023)**, **FlashAttention-3 (2024)**.
- **Shazeer — "Fast Transformer Decoding" (MQA, 2019)** and **Ainslie et al. —
  "GQA" (2023).** Architectural KV-cache reduction.
- **Kwon et al. — "Efficient Memory Management for LLM Serving with
  PagedAttention" (vLLM, 2023).** Paging for the KV cache.

## Mixture-of-Experts: algorithm

- **Shazeer et al. — "Outrageously Large Neural Networks: The Sparsely-Gated MoE
  Layer" (2017).** The origin: gating, top-$k$, the load-balancing loss.
- **Lepikhin et al. — "GShard" (2020).** Capacity, token dropping, all-to-all
  dispatch at scale.
- **Fedus, Zoph, Shazeer — "Switch Transformer" (2021).** Top-1 routing, the
  simplified aux loss used in the handbook, init/stability tricks.
- **Zoph et al. — "ST-MoE" (2022).** Router z-loss and transfer/stability lessons.
- **Zhou et al. — "Mixture-of-Experts with Expert Choice Routing" (2022).** The
  expert-choice dual to token-choice.
- **Dai et al. — "DeepSeekMoE" (2024).** Fine-grained + shared experts; the modern
  recipe's foundation.
- **Clark et al. — "Unified Scaling Laws for Routed Language Models" (2022).**
  How sparse models scale.

## Mixture-of-Experts: systems & kernels

- **Gale et al. — "MegaBlocks: Efficient Sparse Training with MoE" (2022).**
  Block-sparse, dropless MoE; grouped-GEMM thinking.
- **Rajbhandari et al. — "DeepSpeed-MoE" (2022).** Training and inference systems
  for MoE at scale.
- **Tillet, Kung, Cox — "Triton" (2019).** The tile-based kernel language; pair
  with the official tutorials.
- **DeepSeek-AI — "DeepSeek-V3 Technical Report" (2024).** The flagship case
  study: aux-loss-free balancing, MLA, fp8 training, node-limited routing,
  DualPipe/DeepEP overlap, MTP. The single most useful modern MoE systems paper.

## Distributed training

- **Shoeybi et al. — "Megatron-LM" (2019)** and **Narayanan et al. — "Efficient
  Large-Scale LM Training on GPU Clusters" (2021).** Tensor + pipeline parallelism
  and N-D composition.
- **Rajbhandari et al. — "ZeRO" (2020)** and **Zhao et al. — "PyTorch FSDP"
  (2023).** Sharding optimizer/grad/param state.
- **Huang et al. — "GPipe" (2019).** Pipeline parallelism and micro-batching.
- **Liu et al. — "Ring Attention" (2023)** and **Korthikanti et al. —
  "Reducing Activation Recomputation / Sequence Parallelism" (2022).** Long-context
  parallelism and activation memory.

## Compression & inference

- **Frantar et al. — "GPTQ" (2022).** Error-corrected low-bit weight quantization.
- **Lin et al. — "AWQ" (2023).** Activation-aware weight quantization.
- **Xiao et al. — "SmoothQuant" (2022).** Migrating activation outliers for W8A8.
- **Yu et al. — "Orca" (2022).** Iteration-level (continuous) batching.
- **Leviathan et al. / Chen et al. — "Speculative decoding" (2023).** Lossless
  draft-and-verify; see also **Medusa (2024)** and **EAGLE (2024)**.
- **Zhong et al. — "DistServe" (2024)** and **Patel et al. — "Splitwise" (2024).**
  Prefill/decode disaggregation.
- **Hinton et al. — "Distilling the Knowledge in a Neural Network" (2015).**

## Models referenced in the case studies

- **Jiang et al. — "Mixtral of Experts" (2024).** The clean open SMoE.
- **DeepSeek-AI — "DeepSeek-V2 / V3 Technical Reports" (2024).** MLA, DeepSeekMoE,
  fp8, the full modern stack.
- **Qwen Team — "Qwen2 / Qwen3 Technical Reports" (2024–2025).** Productionized
  fine-grained MoE.
- **Moonshot AI — "Kimi K2 Technical Report" (2025).** Trillion-scale extreme
  sparsity; MuonClip stability. (Confirm K2.5 specifics in the current model card.)

## Hardware & tooling docs

- **NVIDIA CUDA C++ Programming Guide; CUTLASS; Nsight Systems/Compute.**
- **AMD ROCm / HIP Programming Guide; CDNA3 (MI300) ISA; Composable Kernel;
  rocWMMA; rocprof / Omniperf.** The first-class ROCm references for the
  CUDA/HIP track.
- **PyTorch docs:** `torch.autocast`, FSDP, `torch.profiler`,
  `torch.utils.cpp_extension`.
- **Triton:** official language reference and tutorials; ROCm backend notes.

## Books

- **Hwu, Kirk, El Hajj — *Programming Massively Parallel Processors*.** The
  standard GPU-architecture-and-CUDA text.
