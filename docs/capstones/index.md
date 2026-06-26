# 第四部分·Capstone

將整本手冊整合在一起的兩個端到端項目。第一個是完全
使用可運行的程式碼；第二個是採用該模型的結構化指南
到多 GPU 設定。

## 頁面

1. **[Build a small MoE LM](build-moe.md)**— 組裝一個微型 MoE 語言模型
   從第二部分的組件中，在玩具語料庫上進行訓練，然後對其進行最佳化並
   **使用以下方法報告測量的加速**
   [profiling methodology](../performance/profiling.md)。
2. **[Scaling it up](scaling.md)**— 應用
   [parallelism techniques](../performance/distributed-training.md)（DP/ZeRO、TP、
   PP 和 [EP](../moe/systems-ep.md)）到你建立的模型，並進行規劃
   將其映射到真實硬體的指南。

!!! tip "這些是檢查點，而不僅僅是閱讀"
Capstone 參考了真實的、經過測試的程式碼
[`code/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code)。這
目標是你可以運行、修改和測量——而不僅僅是閱讀。
