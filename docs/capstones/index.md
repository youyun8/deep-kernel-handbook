# 實戰專案

兩個把整本手冊串起來的端到端專案。第一個有完整可執行的程式碼；第二個是把該模型搬到多 GPU 設定的結構化指南。

## 頁面

1. **[建立小型 MoE LM](build-moe.md)**——用MoE 篇的元件拼出一個微型 MoE 語言模型，在玩具語料庫 上訓練它，再優化、並用 [profiling 方法論](../performance/profiling.md) **回報量測到的加速**。
2. **[擴展到更大規模](scaling.md)**——把 [並行技術](../performance/distributed-training.md)（DP/ZeRO、TP、PP 與 [EP](../moe/systems-ep.md)） 套用到你建好的模型，並給出一份把它對映到真實硬體的規劃指南。

!!! tip "這些是檢查點，不只是讀物"
    實戰專案參照 [`code/`](https://github.com/youyun8/deep-kernel-handbook/tree/main/code) 裡真實、 已測試的程式碼。目標是讓你能跑、能改、能量測——而不只是讀過。
