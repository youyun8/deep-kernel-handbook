# 效能與系統工程

支撐 MoE 旗艦與其餘一切的通用工具箱：GPU 實際如何執行工作、如何用 Triton 與 CUDA/HIP 寫自訂
kernel、如何把 training 攤到多裝置、如何壓縮模型、如何把它們 serve 得快——以及撐起這一切的根本
技能：如何**量測**，才能優化到對的東西上。

**請和[MoE 篇](../moe/index.md)一起讀**；MoE 的系統與 kernel 章節會直接連到這裡。

## 頁面

**kernel（由硬體往上建）**

1. [GPU 程式設計模型](gpu-programming.md)——執行與記憶體階層，CUDA 與 ROCm/HIP 對照。
2. [Triton 路線](triton-track.md)——高生產力的 kernel 撰寫：向量加法 → 融合 softmax → matmul →
   attention。
3. [CUDA / HIP 路線](cuda-hip-track.md)——更底層，把 NVIDIA 與 AMD 的跨平台可移植性放在同等地位。

**規模**

4. [分散式訓練](distributed-training.md)——data/tensor/pipeline/sequence/expert 並行、ZeRO，以及
   底層的 collective。

**部署**

5. [量化與壓縮](quantization.md)——PTQ/QAT、GPTQ/AWQ、剪枝、蒸餾。
6. [推論最佳化](inference-optimization.md)——Continuous Batching、speculative decoding、KV cache
   管理、serving 系統。

**永遠**

7. [Profiling 與方法論](profiling.md)——如何量測、該相信什麼，以及會造出假加速的 benchmark 陷阱。
   **這篇要趁早讀。**

!!! tip "貫穿全部的那條線"
    這裡每一頁都是 [roofline](../foundations/transformer-systems.md) 的應用：kernel 提高算術強度、
    並行把計算換成通訊、量化減少 bytes、profiling 告訴你自己到底在撞哪一面牆。
