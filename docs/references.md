# 書目與帶註解的參考資料

依主題分組、附簡短註解的精選閱讀清單。這些是手冊背後的一手來源 —— 想看原始細節時從這裡開始。引用採 **IEEE 格式**並全域連續編號 [1]–[41]；每條後附簡短中文註解，說明*為什麼重要*。（請自行找最新的版本／出處；arXiv 編號與會議卷期會隨修訂而變。）

!!! Note "怎麼用這份清單"
    MoE 的閱讀順序建議：[11] → [13]／[12] → [16] → [21] → [18]。kernel 則先讀 [20] 與其官方教學，再讀 [8]。

## 基礎：系統、scaling、精度

- **[1]** S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Commun. ACM*, vol. 52, no. 4, pp. 65–76, 2009. —— 整本手冊都圍繞這個模型組織；用來建立算力 vs 頻寬的框架。
- **[2]** J. Kaplan *et al.*, "Scaling laws for neural language models," *arXiv:2001.08361*, 2020. —— 冪律損失縮放與 $6P$-FLOP 直覺的出處。
- **[3]** J. Hoffmann *et al.*, "Training compute-optimal large language models," *arXiv:2203.15556* (Chinchilla), 2022. —— 計算最優的 token/參數取捨；做預算時必讀。
- **[4]** P. Micikevicius *et al.*, "Mixed precision training," in *Proc. ICLR*, 2018. —— FP16 + FP32 master 權重 + loss scaling 的配方。
- **[5]** D. Kalamkar *et al.*, "A study of BFLOAT16 for deep learning training," *arXiv:1905.12322*, 2019. —— 為什麼 training 上 BF16 的範圍勝過 FP16。
- **[6]** P. Micikevicius *et al.*, "FP8 formats for deep learning," *arXiv:2209.05433*, 2022. —— E4M3/E5M2，以及各自用在哪。

## Attention 效率

- **[7]** M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax," *arXiv:1805.02867*, 2018. —— FlashAttention 所依賴的串流 softmax 技巧。
- **[8]** T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," in *Proc. NeurIPS*, 2022. 後續：T. Dao, "FlashAttention-2," 2023；J. Shah *et al.*, "FlashAttention-3," 2024. —— IO-aware 的精確 attention；靠融合省頻寬的典範。
- **[9]** N. Shazeer, "Fast transformer decoding: One write-head is all you need," *arXiv:1911.02150* (MQA), 2019；J. Ainslie *et al.*, "GQA: Training generalized multi-query transformer models from multi-head checkpoints," in *Proc. EMNLP*, 2023. —— 架構層級的 KV cache 縮減。
- **[10]** W. Kwon *et al.*, "Efficient memory management for large language model serving with PagedAttention," in *Proc. SOSP*, 2023 (vLLM). —— KV cache 的分頁。

## MoE (Mixture of Experts)：演算法

- **[11]** N. Shazeer *et al.*, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *Proc. ICLR*, 2017. —— 源頭：gating、top-$k$、負載平衡損失。
- **[12]** D. Lepikhin *et al.*, "GShard: Scaling giant models with conditional computation and automatic sharding," *arXiv:2006.16668*, 2020. —— Capacity、token drop、規模化的 all-to-all dispatch。
- **[13]** W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformer: Scaling to trillion parameter models with simple and efficient sparsity," *J. Mach. Learn. Res.*, 2022. —— Top-1 routing、本手冊採用的簡化 auxiliary loss、初始化/穩定性技巧。
- **[14]** B. Zoph *et al.*, "ST-MoE: Designing stable and transferable sparse expert models," *arXiv:2202.08906*, 2022. —— Router z-loss 與穩定/可遷移性的經驗。
- **[15]** Y. Zhou *et al.*, "Mixture-of-experts with expert choice routing," in *Proc. NeurIPS*, 2022. —— Expert-choice，token-choice 的對偶。
- **[16]** D. Dai *et al.*, "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," in *Proc. ACL*, 2024. —— 細粒度 + 共享 expert；現代配方的基礎。
- **[17]** A. Clark *et al.*, "Unified scaling laws for routed language models," in *Proc. ICML*, 2022. —— 稀疏模型如何 scaling。

## MoE (Mixture of Experts)：系統與 kernel

- **[18]** T. Gale, D. Narayanan, C. Young, and M. Zaharia, "MegaBlocks: Efficient sparse training with mixture-of-experts," in *Proc. MLSys*, 2023. —— 區塊稀疏、dropless MoE；grouped GEMM 的思路。
- **[19]** S. Rajbhandari *et al.*, "DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale," in *Proc. ICML*, 2022. —— 大規模 serve MoE 的 training 與 inference 系統。
- **[20]** P. Tillet, H. T. Kung, and D. Cox, "Triton: An intermediate language and compiler for tiled neural network computations," in *Proc. MAPL*, 2019. —— 以 tile 為基礎的 kernel 語言；搭配官方教學一起看。
- **[21]** DeepSeek-AI, "DeepSeek-V3 technical report," *arXiv:2412.19437*, 2024. —— 旗艦案例：aux-loss-free 平衡、MLA、FP8 training、node-limited routing、DualPipe/DeepEP 重疊、MTP。最有用的現代 MoE 系統論文。

## 分散式訓練

- **[22]** M. Shoeybi *et al.*, "Megatron-LM: Training multi-billion parameter language models using model parallelism," *arXiv:1909.08053*, 2019；D. Narayanan *et al.*, "Efficient large-scale language model training on GPU clusters using Megatron-LM," in *Proc. SC*, 2021. —— Tensor + pipeline 並行與 N 維組合。
- **[23]** S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, "ZeRO: Memory optimizations toward training trillion parameter models," in *Proc. SC*, 2020；Y. Zhao *et al.*, "PyTorch FSDP: Experiences on scaling fully sharded data parallel," *Proc. VLDB Endow.*, 2023. —— 切分 optimizer/梯度/參數狀態。
- **[24]** Y. Huang *et al.*, "GPipe: Efficient training of giant neural networks using pipeline parallelism," in *Proc. NeurIPS*, 2019. —— Pipeline 並行與 micro-batching。
- **[25]** H. Liu, M. Zaharia, and P. Abbeel, "Ring attention with blockwise transformers for near-infinite context," *arXiv:2310.01889*, 2023；V. A. Korthikanti *et al.*, "Reducing activation recomputation in large transformer models," in *Proc. MLSys*, 2023. —— 長上下文並行與 activation 記憶體。

## 壓縮與 inference

- **[26]** E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers," in *Proc. ICLR*, 2023. —— 誤差修正的低位元權重量化。
- **[27]** J. Lin *et al.*, "AWQ: Activation-aware weight quantization for LLM compression and acceleration," in *Proc. MLSys*, 2024. —— Activation-aware 權重量化。
- **[28]** G. Xiao *et al.*, "SmoothQuant: Accurate and efficient post-training quantization for large language models," in *Proc. ICML*, 2023. —— 為 W8A8 把 activation 離群值遷移出去。
- **[29]** G.-I. Yu *et al.*, "Orca: A distributed serving system for transformer-based generative models," in *Proc. OSDI*, 2022. —— 迭代級（continuous）batching。
- **[30]** Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *Proc. ICML*, 2023；C. Chen *et al.*, "Accelerating large language model decoding with speculative sampling," *arXiv:2302.01318*, 2023. 另見 Medusa (T. Cai *et al.*, 2024) 與 EAGLE (Y. Li *et al.*, 2024). —— 無損的 draft-then-verify。
- **[31]** Y. Zhong *et al.*, "DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving," in *Proc. OSDI*, 2024；P. Patel *et al.*, "Splitwise: Efficient generative LLM inference using phase splitting," in *Proc. ISCA*, 2024. —— Prefill/decode 拆分。
- **[32]** G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv:1503.02531*, 2015. —— 知識蒸餾的源頭。

## 案例研究引用的模型

- **[33]** A. Q. Jiang *et al.*, "Mixtral of experts," *arXiv:2401.04088*, 2024. —— 乾淨的開放 SMoE。
- **[34]** DeepSeek-AI, "DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model," *arXiv:2405.04434*, 2024（V3 見 [21]）. —— MLA、DeepSeekMoE、FP8，完整的現代堆疊。
- **[35]** Qwen Team, "Qwen2 technical report," *arXiv:2407.10671*, 2024；Qwen Team, "Qwen3 technical report," 2025. —— 量產化的細粒度 MoE。
- **[36]** Moonshot AI, "Kimi K2 technical report," 2025. —— 兆級的極端稀疏；MuonClip 穩定性。（K2.5 細節請以當前模型卡為準。）

## 硬體與工具文件

- **[37]** NVIDIA, *CUDA C++ Programming Guide*; *CUTLASS*; *Nsight Systems / Nsight Compute*. [Online]. —— CUDA 路線的一手參考。
- **[38]** AMD, *ROCm / HIP Programming Guide*; *CDNA3 (MI300) ISA*; *Composable Kernel*; *rocWMMA*; *rocprof / Omniperf*. [Online]. —— CUDA/HIP 路線的一手 ROCm 參考。
- **[39]** PyTorch, *Documentation*: `torch.autocast`、FSDP、`torch.profiler`、`torch.utils.cpp_extension`. [Online]. —— 框架層 API 參考。
- **[40]** Triton, *Language Reference and Tutorials*; ROCm backend notes. [Online]. —— Triton 官方語言參考與教學。

## 書籍

- **[41]** W. W. Hwu, D. B. Kirk, and I. El Hajj, *Programming Massively Parallel Processors*, 4th ed. Morgan Kaufmann. —— 標準的 GPU 架構與 CUDA 教科書。
