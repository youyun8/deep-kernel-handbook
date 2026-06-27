# 實戰專案：端到端建立小型 MoE LM

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 中階 → 高階</span>
  <span class="chip"><strong>先備知識：</strong> MoE 篇全部</span>
  <span class="chip"><strong>程式碼：</strong> <code>code/moe/train_tiny_moe.py</code>（CPU/GPU）</span>
</div>

這個實戰專案把[MoE 篇](../moe/index.md)的所有東西拼成一個小但完整的 MoE 語言模型，在玩具
任務上訓練它，再優化、並用 [profiling 方法論](../performance/profiling.md) **回報量測到的加速**。
參考實作是
[`code/moe/train_tiny_moe.py`](https://github.com/youyun8/deep-kernel-handbook/blob/main/code/moe/train_tiny_moe.py)，
它在 CPU（小規模）與單張 GPU 上都能跑。

## 目標與設計

做一個 char 級（或微型 BPE）的 decoder-only LM，把 FFN 換成 MoE 層，小到可以在筆電/GPU 上幾分鐘
內訓完。組態：

- $d_{model}=256$、$L=4$ 層、$n_{heads}=4$、每個 expert $d_{ff}=512$。
- $E=8$ 個 expert、$k=2$、**sigmoid gate** + 選用的**共享 expert**。
- 平衡：[aux-loss-free 偏差控制器](../moe/load-balancing.md)（外加一點點
  [z-loss](../moe/training-stability.md)）。
- GPU 上用 bf16 autocast；router 數學走 fp32（[精度紀律](../foundations/numerics-precision.md)）。

## 第 1 步 — 組裝模型

重用既有元件：因果 attention（基礎篇）、[從零實作 MoE layer](../moe/moe-from-scratch.md)，以及
block 接線：

```python
class TinyMoELM(nn.Module):
    def __init__(self, vocab, d=256, L=4, n_heads=4, d_ff=512,
                 n_experts=8, top_k=2, shared=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(4096, d)
        self.blocks = nn.ModuleList(
            MoEBlock(d, d_ff, n_heads, n_experts, top_k, shared) for _ in range(L))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        aux = 0.0
        for blk in self.blocks:
            x, a = blk(x)          # block returns hidden + aux/z loss
            aux = aux + a
        return self.head(self.norm(x)), aux
```

## 第 2 步 — 帶平衡機制訓練

訓練迴圈把 MoE 損失加進去，並在每一步更新偏差控制器：

```python
for step, (xb, yb) in enumerate(loader):
    with torch.autocast(device, dtype=torch.bfloat16, enabled=cuda):
        logits, aux = model(xb)
        lm = F.cross_entropy(logits.flatten(0,1), yb.flatten())
    loss = lm + aux                       # aux = z-loss (+ tiny aux if used)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # MoE spikes!
    scaler.step(opt); scaler.update(); opt.zero_grad()
    update_all_router_biases(model)       # aux-loss-free controller step
    if step % 100 == 0:
        log_metrics(lm, load_cv(model), drop_rate(model), entropy(model))
```

盯著 [健康指標](../moe/load-balancing.md)：損失下降、**負載 CV** 趨近 0、**drop rate** 維持低、
**routing 熵**穩定（不崩塌）。若看到尖峰/NaN，回去看 [訓練穩定性](../moe/training-stability.md)
（z-loss、fp32 router、初始化、梯度裁剪）。

??? success "你應該看到什麼"
    在玩具任務上，訓練損失應平穩下降，負載 CV 應在幾百步內從初始值降到約 0.1–0.2（偏差控制器
    在發揮作用），drop rate 維持低。把平衡關掉「看它崩塌」——少數 expert 吃掉一切、熵崩潰——
    平衡機制的價值就具體了。

## 第 3 步 — 優化並量測

現在套用效能工程篇。**好好量測**（warmup、CUDA event、同步、掃描、鎖頻——見
[profiling](../performance/profiling.md)），並回報前後對比。依這個模型大致的報酬順序來優化：

1. **用 dispatch 形式取代 Python expert 迴圈**（排序 → grouped 計算 → scatter），依
   [從零實作 MoE layer](../moe/moe-from-scratch.md)。
2. **在 GPU 上為 expert 用 grouped GEMM kernel**（[Triton](../moe/kernels.md)）；把 gather/scatter
   融進 kernel。
3. **bf16 autocast** + attention 子層用融合 attention（FlashAttention）。
4. **CUDA graph** 包住 decode 步驟（消除啟動開銷）。
5. **量化 expert**（int8/fp8）做 inference（[量化](../performance/quantization.md)）。

回報一張像這樣的表（填上*你的*量測數字，並註明硬體/shape——下面的值僅供示意）：

| 變體                  | 訓練步長 (ms) | tok/s | MFU | 備註                         |
| --------------------- | ------------: | ----: | --: | ---------------------------- |
| 樸素 expert 迴圈      |        _基線_ |     — |   — | Python 迴圈、參差不齊的批次  |
| dispatch（已排序）    |             — |     — |   — | 每個 expert 連續區塊         |
| + grouped GEMM kernel |             — |     — |   — | 一次啟動、無 padding         |
| + 融合 gather/scatter |             — |     — |   — | 省下一次 HBM 往返            |
| + bf16 + 融合 attention |           — |     — |   — |                              |

這個實戰專案的重點是**紀律**：每一行都是一個假設（「expert 迴圈是 launch-bound」），用對照
roofline 目標的正確量測去檢驗，而不是憑感覺。

## 第 4 步 — 從模型取樣

確認它真的學到了東西：用訓練好的模型生成文字（greedy 或 temperature 取樣）。對小語料上的
char-LM，你應該在本地就能拿到通順的文字。這個生成迴圈也是加上
[Inference 優化](../performance/inference-optimization.md)（KV cache——你會連同 CUDA graph 一起加）
的下手處。

## 擴充

- 把 softmax↔sigmoid gating、aux-loss↔aux-loss-free 互換；比較損失與負載 CV。
- 加入 [expert-choice routing](../moe/routing-variants.md)，觀察 dropless 行為。
- 把 $E$ 往上擴（細粒度），觀察 routing/通訊開銷如何成長——這是 [擴展到更大規模](scaling.md) 的
  動機。

## 重點

- 一個完整的 MoE LM，不過就是接進 Transformer 的MoE 篇元件，加上一個承載**平衡 + 穩定性**機制
  的訓練迴圈。
- 優化階段是一場**以量測、以 roofline 為錨**的工程練習：假設瓶頸、修掉它、重新正確量測、再重複。
- 把平衡開關切來切去（看它崩塌），會讓負載平衡工具箱的價值變得一目了然。

## 練習

!!! tip "解答"
    參考解答在 [解答頁](../solutions/capstones.md)。請先試做每一題，再展開對照。

1. 跑參考實作，再拿掉平衡、量化它的崩塌（熵、負載 CV、最終損失）。
2. 實作 dispatch 與 grouped GEMM；用正確的方法論產生前/後對比表。
3. 在生成迴圈加上 KV cache，量測 decode latency，並與「每步全部重算」的基線比較。
4. 把 expert 量化成 int8，回報品質（val 損失）與速度。

## 參考

- 整個[MoE 篇](../moe/index.md)與[效能工程篇](../performance/index.md)。
- Karpathy. _nanoGPT_（可擴充的密集骨架）。
- Gale et al. _MegaBlocks_；Jiang et al. _Mixtral_；DeepSeek-AI _DeepSeek-V3_。
