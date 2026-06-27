# Mixture-of-Experts

這是手冊的核心。Mixture-of-Experts（MoE）把一個密集 FFN 換成許多個「expert」FFN，再讓
每個 token 只走其中幾個——藉此**把總參數量（容量）和每個 token 的計算量（FLOP）解耦**。
前沿的開放權重模型（DeepSeek-V3、Qwen3-MoE、Kimi-K2）幾乎全是 MoE，原因正在於此。

但稀疏化不是免費的午餐。它換來一整套密集模型從來不會遇到的系統問題：負載不平衡、離散
routing 的訓練不穩定、all-to-all 通訊、巨大的記憶體足跡，以及不規則的 GEMM。本篇把 MoE
當成一個**完整系統**來拆解——從建模到 kernel 到 serving。

讀完本篇，你將能夠：

- 量化說明 MoE 用什麼換什麼，以及稀疏率 $k/E$ 的意義。
- 從零寫出一個 MoE 層：expert、router/gate、top-$k$ 選擇與加權合併。
- 診斷並修復 router 崩潰：auxiliary loss、expert capacity，以及現代的 aux-loss-free 偏差控制。
- 在 token-choice / expert-choice、共享 expert 與細粒度 expert 之間做設計取捨。
- 把 expert 跨 GPU 切分（expert parallelism），並理解 all-to-all 為何是瓶頸、怎麼藏起來。
- 看懂 grouped GEMM 與 permute kernel，以及它們在真實 decode trace 裡長什麼樣。

## 頁面

<div class="grid cards" markdown>

- :material-scale-balance:&nbsp;**[為什麼需要稀疏化](why-sparsity.md)**

    容量與 FLOP 的解耦、稀疏率，以及代價的全貌。

- :material-layers-triple-outline:&nbsp;**[從零實作 MoE layer](moe-from-scratch.md)**

    expert + router + top-$k$ + 合併，可讀版與 dispatch 版兩種實作。

- :material-scale:&nbsp;**[負載平衡](load-balancing.md)**

    auxiliary loss、capacity、token drop，與 aux-loss-free 偏差控制器。

- :material-call-split:&nbsp;**[Routing 變體](routing-variants.md)**

    token-choice vs expert-choice、共享 expert、細粒度 expert。

- :material-shield-check-outline:&nbsp;**[訓練穩定性](training-stability.md)**

    router z-loss、精度紀律、初始化與實務護欄。

- :material-lan:&nbsp;**[系統與 expert parallelism](systems-ep.md)**

    all-to-all dispatch/combine、通訊重疊、grouped GEMM、capacity 取捨。

- :material-chip:&nbsp;**[MoE kernels](kernels.md)**

    permute 與 grouped GEMM 在 Triton / CUDA / HIP 上的實作。

- :material-server-network:&nbsp;**[推論與 serving](inference-serving.md)**

    expert 記憶體、offload、量化與 batch 動態。

- :material-book-open-page-variant-outline:&nbsp;**[案例研究](case-studies.md)**

    Mixtral、DeepSeek-V3、Qwen-MoE、Kimi-K2 的設計選擇。

- :material-pulse:&nbsp;**[MoE decode 剖析](decode-anatomy.md)**

    用真實逐 token decode profile 把所有元件放上時鐘來量。

</div>

!!! tip "和效能工程篇一起讀"
    本篇後半（系統、kernel、serving）會直接連回[效能工程](../performance/index.md)的
    collective、Triton/CUDA 與 profiling 章節，兩篇並行閱讀效果最好。讀完之後，
    [AITER](../aiter/index.md) 會把這些觀念對到一條真實的 Kimi-K2.5 decode 執行路徑上。
