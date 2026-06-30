# 解答 — 實戰專案

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 建立小型 MoE LM、擴展到更大規模</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

實戰專案的練習是針對 [`code/`](https://github.com/youyun8/deep-kernel-handbook/tree/main/code) 裡玩具模型的**動手建構與量測**任務。沒有單一數字答案；以下是預期結果、正確的做法，以及每個人都要避免的陷阱。

## 建立一個小型 MoE LM

??? Success "1 — 移除負載平衡，量化 routing collapse"
    先訓練參照版本，再關掉 auxiliary loss/bias controller。預期出現**collapse 的徵兆**：routing **熵崩潰**（少數幾個 expert 贏家全拿）、**負載 CV 急速上升**（從 ~0.1–0.2 飆到 ≫1）、幾個 expert 變成**dead**（負載歸零），而且**最終 val loss 變差**。把這三項（熵、CV、loss）每一步都記下來，這樣跟有做負載平衡的那次跑就能明顯比較出差異——這正具體證明了負載平衡是真正承重的機制，不是裝飾用的。

??? Success "2 — permute + grouped GEMM：有無對照表"
    實作 permute → grouped GEMM → unpermute 的路徑，跟 naive 的 masked loop 路徑比較。**讓對照表可信的做法：**用相同的權重/種子，先 warmup，計時迴圈外面包 `synchronize()`，報告多次跑的中位數，並在相信速度數字之前**先驗證輸出一致**（BF16 下最大絕對誤差 ~1e-3）。預期 permute 形式在 GPU 上會贏（連續的 grouped GEMM 對上許多微小的、被 mask 蓋住大半的 matmul），而且差距會隨著 $E$ 變大而擴大。

??? Success "3 — KV cache 與每步重算所有東西"
    在生成迴圈裡加上 KV cache：每層存 K、V，每一步只附加新 token 的 K、V，而不是重算整段前綴。把 decode latency 跟「每步重算全部」的 baseline 比較。預期：不快取的話整個生成過程是 $O(N^2)$（每個新 token 都要重新關注所有先前 token、**還要重新算它們的 K、V**），而有快取的 decode 是 $O(N)$——所以加速比**會隨序列長度增加而變大**，短序列時只是小幅提升，長序列時可以到 10 倍以上。

??? Success "4 — int8 expert：品質與速度"
    把 expert 權重量化成 int8，router + attention 維持 BF16。報告**val loss**（品質）與**decode latency / 權重位元組數**（速度）。預期：val loss 幾乎不變（expert 對量化容忍度高——見 [量化練習 4](performance.md#quantization-compression)），權重的記憶體佔用縮小約 2 倍，decode 在 memory-bound 的範圍內變快。這在玩具規模上重現了真正的 MoE serving 配方。

## 擴展到更大規模

??? Success "1 — 每 GPU 記憶體，以及 8 與 64 GPU 的並行配置"
    算出每 GPU 的狀態量（BF16+Adam 是 16 B/param；見 [分散式 training 練習 2](performance.md#distributed-training)），再加上 activation 和 KV。**8 GPU（單一節點）：**如果單層裝不下，就用 NVLink 上的 TP=8；否則用 ZeRO-3/FSDP 做純 DP 的記憶體切分；MoE 層則用跨 8 張卡的 EP。**64 GPU（多節點）：**組合方式是——TP=8 留在**節點內**，PP 和/或 EP 跨**節點**，DP/ZeRO 放在最外層。理由：TP 對頻寬要求最高，放在最快的連結上；PP/EP 次之；DP 放最外層（最能忍受慢速連結）。

??? Success "2 — 實作 EP，驗證 loss 與單 GPU 對得上"
    接上 expert parallelism（自己刻 all-to-all 的 dispatch/combine，或用 DeepSpeed-MoE/Megatron-LM）。**正確性檢查：**用相同的種子/資料，EP 跑出來的 loss 在約 50 步內應該要跟單 GPU 跑的結果吻合到浮點誤差範圍內。如果開始漂移，常見元兇是**router 的數學沒有用 FP32**（導致不同 rank 的 routing 結果不一致）或**負載平衡計數沒有同步**——正好對應 [訓練穩定性](../moe/training-stability.md) 裡提到的錯誤。

??? Success "3 — 量化 all-to-all 的重疊效果；前後 MFU"
    先 profile 關掉分塊 pipeline 的版本，再 profile 打開的版本。**關閉時：**all-to-all 在時間軸上是一段暴露的空檔（通訊期間 GPU 閒著）→ MFU 較低。**打開時：**通訊跟可獨立計算的工作（shared expert / 下一個 chunk 的 attention）重疊 → 空檔縮小，**MFU 上升**。兩種情況都報告 MFU = $6P\cdot\text{tok/s}/\text{peak}$；兩者的差值就是重疊帶來的收益，也是 EP 裡最重要的一項優化。

??? Success "4 — strong scaling 表與偏離線性的原因"
    固定問題規模、增加 GPU 數量，畫出加速比對 GPU 數的關係。結果會**低於理想線性**，原因有：(a) **通訊量隨規模成長**（all-reduce/all-to-all）；(b) **pipeline bubble** $\frac{P-1}{m+P-1}$ 吃掉算力；(c) **每 GPU 的工作量變小**到 kernel 變成 launch-/latency-bound（occupancy 低）；以及 (d) 負載不平衡。把偏離的部分拆給這幾項——說清楚「為什麼」會偏離線性才是真正的交付成果，不是表格本身。
