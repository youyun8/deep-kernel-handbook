# 術語表

手冊中使用的術語的簡明定義。連結指向
每個開發的頁面。

## 系統和效能

**算術強度 ($I$)**
：從記憶體中移動的每個位元組執行的 FLOP 數，$I = W/Q$。確定 roofline
政權。參見 [transformer as a system](foundations/transformer-systems.md)。

**roofline 型號**
：峰值計算 $\pi$ 和頻寬的效能限制 $P=\min(\pi, \beta I)$
$\beta$。整本手冊的組織思路。

**計算限制/記憶體限制**
：受數學單位限制（roofline 脊右側）與 記憶體頻寬 限制
（它的左邊）。 decoding 受記憶體限制；大批量 matmul 是受計算限制的。

**MFU（模型 FLOP 利用率）**
：達到的模型 FLOP 數 ÷ 峰值 FLOP 數； $\approx 6P\cdot\text{tok/s}/\pi$ 為
training。標題效率指標。參見 [profiling](performance/profiling.md)。

**HBM**
：高頻寬記憶體－GPU 的主 DRAM；頻寬$\beta$中
roofline。

**SRAM/共享記憶體/LDS**
：快速片上暫存器。 「共享記憶體」(NVIDIA) =「LDS」(AMD)。暫存磁磚
這是 kernels 如何提高強度的。

**扭曲/波前**
：鎖步 SIMT 執行群組 — NVIDIA 上的**32 個執行緒**（扭曲），**64**
AMD CDNA（波前）。常見的可移植性陷阱。參見
[GPU programming](performance/gpu-programming.md)。

**入住率**
：每個 SM/CU 的駐留扭曲/波前； latency-隱藏手段，而不是目標。

**合併**
：連續通道存取連續位址，以便記憶體事務合併。

**算子融合**
：組合作業以避免 HBM 往返（例如 Flashattention、融合 MLP）- 提高
強度。

**失敗次數**
：浮點運算； matmul $(m{\times}k)(k{\times}n)$ 的成本為 $2mkn$。

## 精度

**bf16 / fp16 / fp8**
：16/16/8 位元浮點數。 bf16 保留了 fp32 的指數範圍（8 位）並贏得了 training；
fp16 尾數較多，但範圍較窄； fp8（E4M3/E5M2）是前緣。參見
[numerics](foundations/numerics-precision.md)。

**mixed precision**
：低精度儲存/matmul+fp32 累加+fp32 主權重。

**損失縮放**
：乘以損失以將 fp16 梯度保持在範圍內；基本上不需要
BF16。

## attention

**KV 緩存**
：儲存過去 tokens 的鍵/值，因此 decoding 是$O(N)$而不是$O(N^2)$；常常是
主導的 inference 記憶體。參見 [attention efficiency](foundations/attention-efficiency.md)。

**MQA / GQA / MLA**
：多查詢/分組查詢/多頭潛在 attention — 架構方式
縮小 KV 快取（減少或壓縮 KV 頭）。

**Flashattention**
：IO 感知 attention，平鋪$Q,K,V$並使用線上 softmax 來避免
具體化 $N{\times}N$ 分數矩陣。參見
[Flashattention](foundations/flashattention.md)。

**線上 softmax**
：透過運行最大值和校正的單通道、數值穩定的 softmax
係數 $e^{m_{old}-m_{new}}$。

**已分頁 attention**
：基於區塊的 KV 快取分配（如虛擬記憶體分頁），消除了
碎片化並實現共享。

## experts 的混合物

**MoE（experts 的混合）**
：具有許多 expert FFN 和一個 router 的層，每個 token 激活一些，
將總參數與每個 token FLOP 解耦。參見 [Part II](moe/index.md)。

**expert**
：MoE 層中的平行 FFN 之一（通常是 SwiGLU）。

**router / 門**
：產生每 expert 分數的小型網路； top-$k$ 選擇 experts。
**Softmax 門控**使 experts 競爭；**sigmoid 門控**對它們進行評分
獨立。

**頂部-$k$ routing**
：每個 token 都使用其 $k$ 最高得分的 experts。

**token-選擇與 expert-選擇**
：tokens 選擇 experts（保證覆蓋，不平衡） vs experts 選擇他們的
頂部-$C$ tokens（保證餘額，不承保）。參見
[routing variants](moe/routing-variants.md)。

**共享 expert**
：路由 experts 中新增了始終在線的 FFN，以吸收常識和
穩定 training。

**細粒度 experts**
：許多小的 experts 而不是幾個大的，擴大了組合混合
固定活動計算的空間。

**輔助（負載平衡）損耗**
：處罰 $\alpha E\sum_e f_e P_e$ 鼓勵制服 routing。參見
[負載平衡](moe/load-balancing.md)。

**輔助無損耗平衡**
：透過控制器更新的 per-expert**選擇偏差**（不是
門權重），避免梯度失真（DeepSeek 式）。

**expert 容量/容量係數**
：每批次每個 expert 的最大 tokens；因子交易下降了 tokens（品質）與
填滿/緩衝區大小（throughput/記憶體）。

**token 掉落/溢出**
：tokens 超出容量跳過 MoE 層（由殘差攜帶）。

**router z 損耗**
：$\beta(\log\sum_e e^{x_e})^2$ 懲罰大 router 的穩定性。參見
[training stability](moe/training-stability.md)。

**expert 並行度 (EP)**
：跨 GPU 分片 experts； tokens 透過 all-to-all 到達 expert。參見
[systems & EP](moe/systems-ep.md)。

**分組 GEMM**
：一台 kernel 執行許多不同大小的矩陣乘法（每個 expert 一個），無需填滿。

**巨型塊/塊稀疏 MoE**
：將 MoE FFN 重新表述為區塊稀疏 matmul，以避免 token 丟棄和
填充。

## 分散式 training

**集體**
：all-reduce，全聚集，減少分散，all-to-all，廣播/P2P（NCCL/RCCL）。
參見 [distributed training](performance/distributed-training.md)。

**資料/張量/管道/序列/expert 並行性(DP/TP/PP/SP/EP)**
：training 分割的尺寸；組成 N 維並行。

**零/FSDP**
：整個 DP 的分片優化器狀態 (1)、梯度 (2) 和參數 (3)
組來削減記憶。

**all-to-all**
：每個等級向每個其他等級發送不同區塊的集體 -
MoE 調度/組合原語。

**節點限制 routing**
：限制 token 的 experts 跨度的節點數，以綁定跨節點 all-to-all。

## inference & 壓縮

**prefill / decode**
：處理提示（許多 tokens，受計算限制）與在某個時間產生一個 token
時間（受記憶體限制）。

**連續（飛行中）配料**
：迭代級調度，將已完成的序列交換為等待的序列，
保持 GPU 滿載。參見 [inference optimization](performance/inference-optimization.md)。

**投機 decoding**
：一個廉價的草案提出了 tokens，目標一次性驗證它們；無損,
利用 decode 的備用計算。

**PTQ / QAT**
：training 後量化（僅校準）與量化感知 training。
參見 [quantization](performance/quantization.md)。

**GPTQ / AWQ / SmoothQuant**
：PTQ 方法：糾錯權重量化/激活感知權重
將激活異常值縮放/遷移到權重中。

**修剪**
：去除重量或結構（非結構化、結構化或 2:4
半結構化）進行壓縮。

**蒸餾**
：training 小學生模仿大老師。

**TTFT / TPOT (ITL)**
：首次 token (prefill latency) 的時間/每次輸出 token (decode latency) 的時間。

## 硬體快速參考

用於粗略估計的整數（練習和
[solutions](solutions/index.md) 使用這些）。峰值 FLOP/s 密集 bf16；真實的
kernels 達到了分數 (MFU)。頻寬是 HBM。**山脊**$\pi/\beta$ 是
算術強度，其中晶片從記憶體限制轉變為計算限制。

| 圖形處理器    | bf16 峰值 ($\pi$) | HBM BW ($\beta$) | HBM 尺寸 | 山脊 $\pi/\beta$        |
| ------------- | ----------------- | ---------------- | -------- | ----------------------- |
| A100（80 GB） | ~312 TFLOP/秒     | ~2.0 TB/秒       | 80GB     | ~156 FLOP/位元組        |
| H100 (SXM)    | ~990 TFLOP/秒     | ~3.35 TB/秒      | 80GB     | ~295 FLOP/位元組        |
| H200          | ~990 TFLOP/秒     | ~4.8 TB/秒       | 141 GB   | 141 GB ~206 FLOP/位元組 |
| 小米 300X     | ~1.3 PFLOP/s      | ~5.3 TB/秒       | 192 GB   | 192 GB ~245 FLOP/位元組 |

互連（適用於 [distributed training](performance/distributed-training.md)）：
MI300 上的節點內**NVLink**~0.9 TB/s/GPU (NVLink 4) 或**Infinity Fabric**；
跨節點**InfiniBand/RoCE**~25–50 GB/s/GPU；主機**PCIe Gen5**~64 GB/s。的
從 HBM → NVLink → IB → PCIe 下降了 1-2 個數量級，這就是為什麼並行性變得如此重要的原因
進行映射，以便最健談的集體乘坐最快的鏈接。

!!! note "使用這些"
    數字因 SKU、時鐘和稀疏性聲明而異（供應商經常引用
    2× 結構稀疏－減半為稠密）。你的哪些內容應該是穩健的
    估計是**制度**和**數量級**，而不是第三個
    重要數字。
