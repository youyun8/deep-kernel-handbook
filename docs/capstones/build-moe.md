# Capstone：端到端建造小型 MoE LM

<div class="page-meta">
  <span class="chip"><strong>等級：</strong> 中階 → 高階</span>
  <span class="chip"><strong>先修改條件：</strong> 第 II 部分的全部 </span>
  <span class="chip"><strong>代碼：</strong> <code>code/moe/train_tiny_moe.py</code> (CPU/GPU)</span>PH
</div>

這個 Capstone 將[Part II](../moe/index.md)的所有內容拼成一個小
但完整的 MoE 語言模型，在玩具任務上訓練它，然後優化它並
**報告測量的加速度**
[profiling methodology](../performance/profiling.md)。參考是
[`code/moe/train_tiny_moe.py`](https://github.com/youyun8/ml-perf-handbook/blob/main/code/moe/train_tiny_moe.py),
它在 CPU（小型）和單一 GPU 上運行。

## 目標與設計

現在一個字元級（或微型 BPE）解碼器專用 LM，其中 FFN 被替換為
MoE 層，夠小，可以在筆記型電腦/GPU 上在四分之一內進行訓練。配置：

- $d_{model}=256$、$L=4$ 層、$n_{heads}=4$、$d_{ff}=512$ 抽屜專家。
- $E=8$ 專家，嚴格$k=2$，**sigmoid 門**+ 選擇性**共享專家**。
- 平衡度：[aux-loss-free bias controller](../moe/load-balancing.md)（+小）
  [z-loss](../moe/training-stability.md)）。
- GPU 上的 bf16 自動投射；fp32 路由器數學（
  [precision discipline](../foundations/numerics-precision.md)）。

## 第 1 步 — 組裝模型

重複使用組件：因果焦點（第一部分）、
[MoE layer from scratch](../moe/moe-from-scratch.md)，及塊接線：

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

## 步驟 2 — 使用平衡機進行訓練

訓練循環添加 MoE 損失並每一步更新偏差控制器：

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

觀看 [health metrics](../moe/load-balancing.md)：損失下降，**負載 CV**下降
接近 0，**丟棄率**低，**路由熵**穩定（不崩潰）。如果你
查看尖峰/NaN，重新造訪 [training stability](../moe/training-stability.md)
（z-loss、fp32 路由器、初始化、剪輯）。

??? success "你應該看到什麼"
在玩具任務上，訓練損失應該會平穩下降，負載係數為
變化應在幾內從其初始值下降到~0.1–0.2
一百步（偏置控制器工作），且下降率應保持較低。
禁用平衡以“看著它崩潰”——少數專家拿走了一切，
熵崩潰－這使得平衡機械的價值變得具體。

## 第 3 步 — 優化與衡量

現在應用第三部分。**正確測量**（預熱、CUDA 事件、同步、掃描、
鎖定時鐘 — 請參閱 [profiling](../performance/profiling.md)）與報告
之前/之後。優化，按照該模型的大致回報順序：

1. **用調度形式替換 Python 專家循環**（排序 → 分組
   計算 → 分散），依據 [MoE-from-scratch](../moe/moe-from-scratch.md)。
2. **為專家 ([Triton](../moe/kernels.md)) 使用分組 GEMM 核心**
   圖形處理器；將聚集/分散融合到核心。
3. **bf16 autocast**+ 注意力子層的融合注意力（FlashAttention）。
4. **CUDA 圖**產生的解碼步驟（消除啟動開銷）。
5. **量化專家**(int8/fp8) 用於推理
   （[quantization](../performance/quantization.md)）。

報告這樣的表格（填寫*你的*測量數字並說明
硬體/形狀 - 下面的值是說明性的）：

| 變體                | 訓練步長（毫秒） | 代幣 | MFU | 筆記                        |
| ------------------- | ---------------: | ---: | --: | --------------------------- |
| 天真的專家循環      |           _基線_ |    — |   — | Python 循環，參差不齊的批次 |
| 派遣（已排序）      |                — |    — |   — | 連續的每個專家區塊          |
| + 分組 GEMM 核心    |                — |    — |   — | 一次發射，無填充            |
| + 融合聚集/分散     |                — |    — |   — | 節省了一次 HBM 往返行程     |
| + bf16 + 閃光燈附加 |                — |    — |   — |                             |

Capstone 的要點是**規則**：每一行都是一個假設（“
專家循環是發射限制的」）透過對屋頂線的正確測量進行測試
目標，而不是氛圍。

## 步驟 4 — 從中取樣

確認它學到了一些東西：使用經過訓練的模型生成文字（貪婪或
溫度採樣）。對於小型語料庫上的 char-LM，你應該在本地獲取
連貫的文字。生成循環是添加
[inference optimizations](../performance/inference-optimization.md)（KV 快取—
你將添加它以及 CUDA 圖表）。

## 擴充

- 交換 softmax↔sigmoid 門控和 aux-loss↔aux-loss-free；比較損耗和負載 CV。
- 加入 [expert-choice routing](../moe/routing-variants.md) 並觀察
  無掉落行為。
- 向上擴展 $E$（細微）並觀察路由/通訊開銷的成長 — 激勵
  [Scaling it up](scaling.md)。

## 重點

- 完整的 MoE LM 只是連接到 Transformer 的第二部分組件加上一個
  承載**平衡+穩定性**機械的訓練循環。
- 優化階段是**測量、屋頂線錨定**的練習
  工程：假設瓶頸，修復它，重新正確測量，重複。
- 切換平衡開/關會導致崩潰——以及負載平衡的價值
  工具包－內心清晰。

## 練習

!!! tip "解決方案"
參考解答位於 [解答頁](../solutions/capstones.md) 上。請先嘗試每個練習，再展開解答。

1. 運行參考，然後刪除平衡並量化崩潰（熵，
   負載 CV，最終損失）。
2. 實施調度表和分組 GEMM；產生前/後表
   用正確的方法論。
3. 將 KV 快取添加到生成循環並測量解碼延遲與
   重新計算一切基線。
4. 將專家量化為 int8 並報告品質（val 損失）與速度。

## 參考

- 所有 [Part II](../moe/index.md) 和 [Part III](../performance/index.md)。
- 卡帕蒂。 _nanoGPT_（延伸的密集骨架）。
- 大風等人。 _巨型塊_；江等人。 _混合_； DeepSeek-AI _DeepSeek-V3_。
