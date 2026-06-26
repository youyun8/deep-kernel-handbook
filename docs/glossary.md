# 詞彙表

手冊中用到的術語的簡明定義。連結指向各自詳述的頁面。

## 系統和效能

**算術強度 ($I$)**
：每從記憶體搬一個 byte 所執行的 FLOP 數，$I = W/Q$。決定你落在 roofline 的哪個機制。見
[作為系統的 Transformer](foundations/transformer-systems.md)。

**roofline 模型**
：由峰值算力 $\pi$ 與頻寬 $\beta$ 給出的效能上限 $P=\min(\pi, \beta I)$。整本手冊的組織主軸。

**compute-bound / memory-bound**
：被數學單元卡住（脊點右側）vs 被記憶體頻寬卡住（脊點左側）。decode 受記憶體限制；大 batch
的 matmul 受計算限制。

**MFU（模型 FLOP 利用率）**
：達到的模型 FLOP ÷ 峰值 FLOP；training 時 $\approx 6P\cdot\text{tok/s}/\pi$。招牌效率指標。見
[profiling](performance/profiling.md)。

**HBM**
：高頻寬記憶體——GPU 的主 DRAM；就是 roofline 裡的頻寬 $\beta$。

**SRAM / 共享記憶體 / LDS**
：快速的晶片內暫存。「共享記憶體」（NVIDIA）=「LDS」（AMD）。把 tile 暫存在這裡，正是 kernel
提高算術強度的手段。

**warp / wavefront**
：鎖步的 SIMT 執行群組——NVIDIA 上是 **32 個 thread**（warp），AMD CDNA 上是 **64**（wavefront）。
常見的可移植性陷阱。見 [GPU 程式設計](performance/gpu-programming.md)。

**占用率（occupancy）**
：每個 SM/CU 上常駐的 warp/wavefront 數；它是藏 latency 的手段，不是目標。

**coalescing（合併存取）**
：相鄰 lane 存取相鄰位址，使記憶體交易能被合併。

**算子融合**
：把多個操作合在一起以避免 HBM 往返（例如 FlashAttention、融合 MLP）——提高算術強度。

**FLOP**
：浮點運算次數；matmul $(m{\times}k)(k{\times}n)$ 成本為 $2mkn$。

## 精度

**bf16 / fp16 / fp8**
：16/16/8 位元浮點。bf16 保留 fp32 的指數範圍（8 位）因而在 training 上勝出；fp16 mantissa 較多
但範圍較窄；fp8（E4M3/E5M2）是前沿。見 [數值與精度](foundations/numerics-precision.md)。

**mixed precision**
：低精度儲存/matmul+fp32 累加+fp32 主權重。

**loss scaling（損失縮放）**
：把損失乘上一個因子，讓 fp16 梯度落在可表示範圍內；bf16 基本上不需要。

## attention

**KV cache**
：儲存過去 token 的 key/value，使 decode 變成 $O(N)$ 而非 $O(N^2)$；常常是 inference 記憶體的
主角。見 [Attention 效率](foundations/attention-efficiency.md)。

**MQA / GQA / MLA**
：Multi-Query / Grouped-Query / Multi-head Latent Attention——從架構層面縮小 KV cache（減少或壓縮
KV 頭）。

**FlashAttention**
：IO-aware 的 attention，把 $Q,K,V$ 分塊、用 online softmax 避免具現化 $N{\times}N$ 分數矩陣。見
[FlashAttention](foundations/flashattention.md)。

**online softmax**
：靠 running max 與校正因子 $e^{m_{old}-m_{new}}$，單趟傳遞就算出數值穩定的 softmax。

**PagedAttention**
：以區塊為單位配置 KV cache（如虛擬記憶體分頁），消除碎片並支援共享。

## experts 的混合物

**MoE（Mixture-of-Experts）**
：含許多 expert FFN 與一個 router 的層，每個 token 只啟動其中幾個，把總參數和每 token FLOP 解耦。
見[第二部](moe/index.md)。

**expert**
：MoE 層裡並列的其中一個 FFN（通常是 SwiGLU）。

**router / gate**
：產生 per-expert 分數的小網路；由 top-$k$ 選出 expert。**softmax gating** 讓 expert 互相競爭；
**sigmoid gating** 則獨立為它們計分。

**top-$k$ routing**
：每個 token 使用自己分數最高的 $k$ 個 expert。

**token-choice vs expert-choice**
：token 挑 expert（保證覆蓋、不保證平衡）vs expert 挑自己的 top-$C$ token（保證平衡、不保證覆蓋）。
見 [Routing 變體](moe/routing-variants.md)。

**共享 expert**
：在路由 expert 之外再加一個總是啟用的 FFN，用來吸收常識並穩定 training。

**細粒度 expert**
：用許多小 expert 取代少數大 expert，在固定活躍計算下放大 expert 組合空間。

**auxiliary（負載平衡）損失**
：懲罰項 $\alpha E\sum_e f_e P_e$，鼓勵均勻 routing。見 [負載平衡](moe/load-balancing.md)。

**aux-loss-free 平衡**
：靠控制器更新的 per-expert **選擇偏差**（不是 gate 權重）來平衡，避免扭曲梯度（DeepSeek 風格）。

**expert capacity / 容量係數**
：每個 batch 每個 expert 的 token 上限；此係數在丟 token（品質）與 padding/緩衝區大小
（throughput/記憶體）之間做取捨。

**token drop / overflow**
：超出 capacity 的 token 略過 MoE 層（由殘差帶過去）。

**router z-loss**
：$\beta(\log\sum_e e^{x_e})^2$，懲罰過大的 router logit 以求穩定。見
[訓練穩定性](moe/training-stability.md)。

**expert parallelism（EP）**
：把 expert 跨 GPU 切分；token 經由 all-to-all 抵達它的 expert。見 [系統與 EP](moe/systems-ep.md)。

**grouped GEMM**
：單一 kernel 執行許多個不同大小的矩陣乘法（每個 expert 一個），無需 padding。

**MegaBlocks / 區塊稀疏 MoE**
：把 MoE FFN 改寫成區塊稀疏 matmul，藉此避免 token drop 與 padding。

## 分散式 training

**collective（集合通訊）**
：all-reduce、all-gather、reduce-scatter、all-to-all、broadcast/P2P（NCCL/RCCL）。見
[分散式訓練](performance/distributed-training.md)。

**data / tensor / pipeline / sequence / expert 並行（DP/TP/PP/SP/EP）**
：切分 training 的各個維度；可組合成 N 維並行。

**ZeRO / FSDP**
：在 DP 群組間切分 optimizer 狀態（1）、梯度（2）與參數（3）以省記憶體。

**all-to-all**
：每個 rank 向其他每個 rank 各送一塊不同資料的 collective——MoE dispatch/combine 的原語。

**node-limited routing**
：限制一個 token 的 expert 能跨多少節點，以約束跨節點 all-to-all。

## inference & 壓縮

**prefill / decode**
：處理 prompt（很多 token、compute-bound）vs 一次生成一個 token（memory-bound）。

**continuous（in-flight）batching**
：以迭代為單位排程，隨時把跑完的序列換成等待中的序列，讓 GPU 保持滿載。見
[推論最佳化](performance/inference-optimization.md)。

**speculative decoding**
：用一個便宜的 draft 模型提出 token，再由目標模型一次驗證；無損，利用 decode 閒置的算力。

**PTQ / QAT**
：訓練後量化（只做校準）vs 量化感知訓練。見 [量化](performance/quantization.md)。

**GPTQ / AWQ / SmoothQuant**
：PTQ 方法——誤差修正的權重量化／activation-aware 權重縮放／把 activation 離群值搬進權重。

**剪枝（pruning）**
：移除權重或結構（非結構化、結構化，或 2:4 半結構化）以壓縮模型。

**蒸餾（distillation）**
：訓練一個小的 student 模型去模仿大的 teacher。

**TTFT / TPOT（ITL）**
：產生第一個 token 的時間（prefill latency）／每個輸出 token 的時間（decode latency）。

## 硬體快速參考

用來做粗略估計的整數（練習與 [解答](solutions/index.md) 都用這些）。峰值 FLOP/s 為密集 bf16；真實
kernel 只達到其中一部分（MFU）。頻寬指 HBM。**脊點** $\pi/\beta$ 是晶片由 memory-bound 轉成
compute-bound 的那個算術強度。

| GPU           | bf16 峰值 ($\pi$) | HBM BW ($\beta$) | HBM 大小 | 脊點 $\pi/\beta$ |
| ------------- | ----------------- | ---------------- | -------- | ---------------- |
| A100（80 GB） | ~312 TFLOP/s      | ~2.0 TB/s        | 80 GB    | ~156 FLOP/byte   |
| H100 (SXM)    | ~990 TFLOP/s      | ~3.35 TB/s       | 80 GB    | ~295 FLOP/byte   |
| H200          | ~990 TFLOP/s      | ~4.8 TB/s        | 141 GB   | ~206 FLOP/byte   |
| MI300X        | ~1.3 PFLOP/s      | ~5.3 TB/s        | 192 GB   | ~245 FLOP/byte   |

互連（用於 [分散式訓練](performance/distributed-training.md)）：節點內 **NVLink** ~0.9 TB/s/GPU
（NVLink 4）或 MI300 上的 **Infinity Fabric**；跨節點 **InfiniBand/RoCE** ~25–50 GB/s/GPU；主機
**PCIe Gen5** ~64 GB/s。從 HBM → NVLink → IB → PCIe 每一級掉 1–2 個數量級——這正是為什麼要把並行
小心對映，好讓最多話的 collective 走最快的連結。

!!! note "怎麼用這些數字"
    數字會隨 SKU、時脈與稀疏性宣稱而變（廠商常引用 2× 結構稀疏——對密集要減半）。你的估計裡該
    穩健的是**機制**與**數量級**，而不是第三位有效數字。
