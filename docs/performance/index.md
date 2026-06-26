# 第三部分·效能與系統工程

為 MoE 旗艦及其他產品提供支援的通用工具包：如何
GPU 實際執行工作，如何在 Triton 和 CUDA/HIP 中編寫自訂 kernels，
如何在多種設備上傳播 training，如何壓縮模型，如何服務
它們速度很快，而且——支撐這一切——如何**測量**以便優化
正確的事。

**與**[Part II](../moe/index.md) 一起閱讀本部分；MoE 系統和
kernels 頁面直接連結到其中。

## 頁面

**kernels（由硬體建構）**

1. [GPU programming model](gpu-programming.md) — 執行與內存
   層級結構，CUDA 和 ROCm/HIP 並列。
2. [Triton track](triton-track.md) — 高效率的 kernel 寫作；向量相加 →
   融合 softmax → matmul → attention。
3. [CUDA / HIP track](cuda-hip-track.md) — 低級別，具有跨平台移植性
   NVIDIA 和 AMD 也同樣值得關注。

**規模**

4. [Distributed training](distributed-training.md) — 數據/張量/管道/
   序列/expert 並行性、ZeRO 以及下面的集合。

**部署**

5. [Quantization & compression](quantization.md) — PTQ/QAT、GPTQ/AWQ、修剪、
   蒸餾。
6. [inference optimization](inference-optimization.md)－連續配料，
   推測性 decoding、KV 快取管理、serving 系統。

**永遠**

7. [Profiling & methodology](profiling.md) — 如何衡量、信任什麼以及
   產生虛假加速的基準測試陷阱。**儘早閱讀本文。**

!!! tip "直通線"
    這裡的每一頁都是 [roofline](../foundations/transformer-systems.md) 的應用：
    kernels 提高 算術強度，並行性換算為
    通訊、量化減少位元組、分析告訴你哪堵牆
    你確實在打。
