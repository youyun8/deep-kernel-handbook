# 書目與帶註解的參考資料

依主題分組、附簡短註解的精選閱讀清單。這些是手冊背後的一手來源——想看原始細節時從這裡開始。
（請自行找最新的版本／出處；arXiv 編號會隨修訂而變。）

!!! note "怎麼用這份清單"
    每一條都說明*為什麼重要*，方便你排優先序。MoE 的閱讀順序建議：Shazeer 2017 → Switch/GShard
    → DeepSeekMoE → DeepSeek-V3 → MegaBlocks。kernel 則先讀 Triton 論文 + 教學，再讀 FlashAttention。

## 基礎：系統、scaling、精度

- **Williams, Waterman, Patterson — "Roofline: An Insightful Visual Performance Model"（CACM 2009）。** 整本手冊都圍繞這個模型組織；用來建立算力 vs 頻寬的框架。
- **Kaplan et al. — "Scaling Laws for Neural Language Models"（2020）。** 冪律損失縮放與 $6P$-FLOP 直覺的出處。
- **Hoffmann et al. — "Training Compute-Optimal LLMs"（Chinchilla，2022）。** 計算最優的 token/參數取捨；做預算時必讀。
- **Micikevicius et al. — "Mixed Precision Training"（2017）。** fp16 + fp32 master 權重 + loss scaling 的配方。
- **Kalamkar et al. — "A Study of BFLOAT16 for Deep Learning Training"（2019）。** 為什麼 training 上 bf16 的範圍勝過 fp16。
- **Micikevicius et al. — "FP8 Formats for Deep Learning"（2022）。** E4M3/E5M2，以及各自用在哪。

## Attention 效率

- **Milakov & Gimelshein — "Online Normalizer Calculation for Softmax"（2018）。** FlashAttention 所依賴的串流 softmax 技巧。
- **Dao, Fu, Ermon, Rudra, Ré — "FlashAttention"（2022）。** IO-aware 的精確 attention；靠融合省頻寬的典範。後續：**FlashAttention-2（2023）**、**FlashAttention-3（2024）**。
- **Shazeer — "Fast Transformer Decoding"（MQA，2019）** 與 **Ainslie et al. — "GQA"（2023）。** 架構層級的 KV cache 縮減。
- **Kwon et al. — "Efficient Memory Management for LLM Serving with PagedAttention"（vLLM，2023）。** KV cache 的分頁。

## Mixture-of-Experts：演算法

- **Shazeer et al. — "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"（2017）。** 源頭：gating、top-$k$、負載平衡損失。
- **Lepikhin et al. — "GShard"（2020）。** capacity、token drop、規模化的 all-to-all dispatch。
- **Fedus, Zoph, Shazeer — "Switch Transformer"（2021）。** top-1 routing、本手冊採用的簡化 auxiliary loss、初始化/穩定性技巧。
- **Zoph et al. — "ST-MoE"（2022）。** router z-loss 與穩定/可遷移性的經驗。
- **Zhou et al. — "Mixture-of-Experts with Expert Choice Routing"（2022）。** expert-choice，token-choice 的對偶。
- **Dai et al. — "DeepSeekMoE"（2024）。** 細粒度 + 共享 expert；現代配方的基礎。
- **Clark et al. — "Unified Scaling Laws for Routed Language Models"（2022）。** 稀疏模型如何 scaling。

## Mixture-of-Experts：系統與 kernel

- **Gale et al. — "MegaBlocks: Efficient Sparse Training with MoE"（2022）。** 區塊稀疏、dropless MoE；grouped GEMM 的思路。
- **Rajbhandari et al. — "DeepSpeed-MoE"（2022）。** 大規模 serve MoE 的 training 與 inference 系統。
- **Tillet, Kung, Cox — "Triton"（2019）。** 以 tile 為基礎的 kernel 語言；搭配官方教學一起看。
- **DeepSeek-AI — "DeepSeek-V3 Technical Report"（2024）。** 旗艦案例：aux-loss-free 平衡、MLA、fp8 training、node-limited routing、DualPipe/DeepEP 重疊、MTP。最有用的現代 MoE 系統論文。

## 分散式訓練

- **Shoeybi et al. — "Megatron-LM"（2019）** 與 **Narayanan et al. — "Efficient Large-Scale LM Training on GPU Clusters"（2021）。** tensor + pipeline 並行與 N 維組合。
- **Rajbhandari et al. — "ZeRO"（2020）** 與 **Zhao et al. — "PyTorch FSDP"（2023）。** 切分 optimizer/梯度/參數狀態。
- **Huang et al. — "GPipe"（2019）。** pipeline 並行與 micro-batching。
- **Liu et al. — "Ring Attention"（2023）** 與 **Korthikanti et al. — "Reducing Activation Recomputation / Sequence Parallelism"（2022）。** 長上下文並行與 activation 記憶體。

## 壓縮與 inference

- **Frantar et al. — "GPTQ"（2022）。** 誤差修正的低位元權重量化。
- **Lin et al. — "AWQ"（2023）。** activation-aware 權重量化。
- **Xiao et al. — "SmoothQuant"（2022）。** 為 W8A8 把 activation 離群值遷移出去。
- **Yu et al. — "Orca"（2022）。** 迭代級（continuous）batching。
- **Leviathan et al. / Chen et al. — "Speculative Decoding"（2023）。** 無損的 draft-then-verify；另見 **Medusa（2024）** 與 **EAGLE（2024）**。
- **Zhong et al. — "DistServe"（2024）** 與 **Patel et al. — "Splitwise"（2024）。** prefill/decode 拆分。
- **Hinton et al. — "Distilling the Knowledge in a Neural Network"（2015）。**

## 案例研究引用的模型

- **Jiang et al. — "Mixtral of Experts"（2024）。** 乾淨的開放 SMoE。
- **DeepSeek-AI — "DeepSeek-V2 / V3 Technical Report"（2024）。** MLA、DeepSeekMoE、fp8，完整的現代堆疊。
- **Qwen Team — "Qwen2 / Qwen3 Technical Report"（2024–2025）。** 量產化的細粒度 MoE。
- **Moonshot AI — "Kimi K2 Technical Report"（2025）。** 兆級的極端稀疏；MuonClip 穩定性。（K2.5 細節請以當前模型卡為準。）

## 硬體與工具文件

- **NVIDIA CUDA C++ Programming Guide；CUTLASS；Nsight Systems/Compute。**
- **AMD ROCm / HIP Programming Guide；CDNA3（MI300）ISA；Composable Kernel；rocWMMA；rocprof / Omniperf。** CUDA/HIP 路線的一手 ROCm 參考。
- **PyTorch 文件：** `torch.autocast`、FSDP、`torch.profiler`、`torch.utils.cpp_extension`。
- **Triton：** 官方語言參考與教學；ROCm 後端說明。

## 書籍

- **Hwu, Kirk, El Hajj — _Programming Massively Parallel Processors_。** 標準的 GPU 架構與 CUDA 教科書。
