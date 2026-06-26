# 參考書目和帶註釋的參考文獻

按主題分組的精選、附註釋的閱讀清單。這些是主要來源
手冊背後——當你想要原始細節時從這裡開始。 （查找
最新版本/地點； aiv ID 隨修訂而變更。 ）

!!! note "如何使用此清單"
    每個條目都會說明*為什麼重要*，以便你可以確定優先順序。對於 MoE，請閱讀
    順序：Shazeer 2017 → Switch/GShard → DeepSeekMoE → DeepSeek-V3 → MegaBlocks。
    對於 kernels：Triton 論文 + 教程，然後是 Flashattention。

## 基礎：系統、縮放、精確度

-**Williams、Waterman、Patterson —「roofline：富有洞察力的視覺表演
模型」（CACM 2009）。**整本手冊都是圍繞著該模型組織的。閱讀
用於計算與頻寬框架。 -**卡普蘭等。 —「神經語言模型的縮放定律」（2020）。**其中
$\sim$冪律損失縮放和$6P$-FLOP 直覺來自。 -**霍夫曼等人。 —「training 計算最佳法學碩士」（Chinchilla，2022 年）。**
計算最優 token/參數交易；對於預算至關重要。 -**Micikevicius 等人。 —“mixed precision training”（2017）。**fp16 + fp32
掌握重量+損失縮放配方。 -**Kalamkar 等人。 —「用於深度學習 training 的 BFLOAT16 研究」（2019）。**
為什麼 training 的 bf16 範圍優於 fp16。 -**Micikevicius 等人。 —「深度學習的 FP8 格式」(2022)。**E4M3/E5M2 和
為什麼每個都用在哪裡。

## attention 效率

-**Milakov 和 Gimelshein —「softmax 的線上標準化器計算」(2018)。**
Flashattention 建立在串流 softmax 技巧之上。 -**Dao、Fu、Ermon、Rudra、Ré —「Flashattention」(2022)。**IO 感知精確
attention；典型的熔斷以節省頻寬結果。後續行動：
**Flashattention-2 (2023)**、**Flashattention-3 (2024)**。 -**Shazeer —「Fast Transformer decoding」（MQA，2019 年）**和**Ainslie 等人。 —
「GQA」（2023）。**架構 KV 快取減少。 -**Kwon 等人。 —「LLM serving 的高效能記憶體管理
Pagedattention" (vLLM, 2023).**KV 快取的分頁。

## experts 的混合：演算法

-**Shazeer 等人。 —「極為龐大的神經網路：稀疏門控的 MoE
Layer」（2017）。**起源：閘控，top-$k$，負載平衡損失。 -**Lepikhin 等人。 —“GShard”(2020).**容量、token 下降、all-to-all
規模調度。 -**Fedus、Zoph、Shazeer —「開關 Transformer」(2021)。**Top-1 routing，
手冊中使用的簡化輔助損失，初始化/穩定性技巧。 -**佐夫等。 —“ST-MoE”（2022）。**router z 損失和轉移/穩定性課程。 -**周等人。 —「experts 與 expert 選擇 routing 的混合」(2022)。**
expert-選擇雙至 token-選擇。 -**戴等。 —“DeepSeekMoE”（2024）。**細粒度+共享 experts；現代的
食譜的基礎。 -**克拉克等人。 —「路由語言模型的統一縮放法則」（2022）。**
稀疏模型如何擴充。

## experts 的混合：系統和 kernels

-**蓋爾等。 —「MegaBlocks：使用 MoE 的高效稀疏 training」（2022 年）。**
區塊稀疏、無滴 MoE；分組 GEMM 思維。 -**Rajbhandari 等人。 —“DeepSpeed-MoE”(2022)。**training 和 inference 系統
為 MoE 大規模服務。 -**Tillet, Kung, Cox —「Triton」(2019)。**基於圖塊的 kernel 語言；一對
與官方教程。 -**DeepSeek-AI — 《DeepSeek-V3 技術報告》（2024）。**旗艦案例
研究：輔助無損平衡、MLA、fp8 training、節點限制 routing、
DualPipe/DeepEP 重疊，MTP。最有用的現代 MoE 系統論文。

## 分散式 training

-**Shoeybi 等人。 —“Megatron-LM”(2019)**和**Narayanan 等人。 —《高效
GPU 叢集上的大規模 LM training」(2021)。**張量 + 管道並行
和 N 維組成。 -**Rajbhandari 等人。 —“ZeRO”（2020）**和**Zhao 等人。 —“PyTorch FSDP”
(2023).**分片優化器/梯度/參數狀態。 -**黃等人。 —“GPipe”(2019)。**管道並行性和微批處理。 -**劉等人。 —“Ring attention”(2023)**和**Korthikanti 等人。 —
「減少活化重新計算/序列並行性」（2022）。**長上下文
並行性和激活記憶。

## 壓縮 & inference

-**Frantar 等人。 —「GPTQ」（2022）。**糾錯低位權重量化。 -**林等。 —“AWQ”(2023)。**激活感知權重量化。 -**肖等人。 —“SmoothQuant”(2022)。**遷移 W8A8 的活化異常值。 -**於等。 —“Orca”(2022)。**迭代級（連續）批次處理。 -**利維坦等。 /陳等人。 —「推測 decoding」（2023）。**無損
起草並驗證；另請參閱**Medusa (2024)**和**EAGLE (2024)**。 -**鐘等人。 —“DistServe”(2024)**和**Patel 等人。 —「分裂」（2024）。**
prefill/decode 分解。 -**Hinton 等人。 —「在神經網路中提取知識」（2015）。**

## 案例研究中引用的模型

-**江等人。 —「experts 的混合」(2024)。**乾淨開放的 SMoE。 -**DeepSeek-AI —“DeepSeek-V2 / V3 技術報告”(2024)。**MLA、DeepSeekMoE、
fp8，完整的現代堆疊。 -**Qwen 團隊 — “Qwen2 / Qwen3 技術報告”（2024–2025）。**生產化
細粒度的 MoE。 -**Moonshot AI — 《Kimi K2 技術報告》（2025）。**兆級極限
稀疏性； MuonClip 穩定性。 （確認目前型號卡中的 K2.5 詳細資訊。）

## 硬體和工具文檔

-**NVIDIA CUDA C++ 程式指南；彎刀； Nsight 系統/運算。** -**AMD ROCm / HIP 程式指南； CDNA3 (MI300) ISA；可組合 kernel；
rocWMMA； rocprof / Omniperf。**一流的 ROCm 參考資料
CUDA/HIP 軌道。 -**PyTorch 文件：**`torch.autocast`、FSDP、`torch.profiler`、
`torch.utils.cpp_extension`。 -**Triton：**官方語言參考和教程；ROCm 後端註釋。

## 書籍

-**Hwu, Kirk, El Hajj — _大規模平行處理器程式設計_.**
標準 GPU 架構和 CUDA 文字。
