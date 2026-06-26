# 解答 — 第五部 · 實戰專案

<div class="page-meta">
  <span class="chip"><strong>涵蓋：</strong> 建立小型 MoE LM、擴展到更大規模</span>
  <span class="chip"><strong>用法：</strong> 先自己試，再對照</span>
</div>

實戰專案練習是針對玩具模型的**構建和測量**任務
[`code/`](https://github.com/youyun8/ml-perf-handbook/tree/main/code)。沒有
單一數字答案；以下是預期結果、正確方法，以及
每個人要避免的陷阱。

## 建立一個小型 MoE LM

??? success "1 — 刪除平衡並量化崩潰"
    訓練參考，然後停用輔助損耗/偏壓控制器。預計
    **崩潰簽名**：routing**熵崩潰**（一些 experts 獲勝），
    **負載 CV 急遽上升**（從〜0.1–0.2 到 ≫1），幾個 experts 去
    **死**（零負載），並且**最終的 val 損失更糟**。記錄所有三個
    每一步（熵、CV、損失），因此與平衡運行的差異為
    可見——這是平衡是承重的具體體現，
    不是化妝品。

??? success "2 — 調度表 + 分組 GEMM，前/後表"
    實現排列 → 分組 GEMM→ 取消排列路徑並與樸素的路徑進行比較
    屏蔽循環。**使表格值得信賴的方法：**相同
    權重/種子、預熱迭代、`synchronize()` 圍繞定時循環、報告
    多次運行的中位數，並**驗證輸出匹配**（最大 abs 差異~1e-3 bf16）
    在相信速度之前。預計調度形式會在 GPU 上獲勝（連續
    分組 GEMM 與許多微小的掩模矩陣相乘），$E$ 中的差距越來越大。

??? success "3 — KV 快取與重新計算一切"
    在生成循環中新增 KV 快取：每層儲存 K,V，附加一個 token
    K,V 每一步而不是重新計算整個前綴。測量 decode latency
    與重新計算基線。預期：一代中重新計算為 $O(N^2)$
    （每個新的 token 都會重新關注所有先前的 tokens**並重新計算它們的 K,V**），
    而快取的 decode 是 $O(N)$ — 因此加速**隨著序列長度的成長而成長**，
    從短長度的適度到長長度的大（10×+）。

??? success "4 — int8 experts：質量與速度"
    將 expert 權重化為 int8，將 router + attention 保留在 bf16 中。報告**值
    損失**（質量）和**decode latency /重量位元組**（速度）。預期： 值
    損失幾乎不變（experts 具有量化容忍 — 請參閱
    [quantization ex. 4](performance.md#quantization-compression))、重量記憶
    體積縮小約 2 倍，decode 在記憶體限制範圍內速度更快。這再現了，在
    微型，真正的 MoE serving 配方。

## 擴大規模

??? success "1 — 每 GPU 記憶體以及 8 和 64 個 GPU 的平行配置"
    計算每個 GPU 狀態（bf16+Adam 為 16 B/param；請參閱
    [distributed ex. 2](performance.md#distributed-training)）加上啟動和
    KV。**8 個 GPU（單一節點）：** 如果某個層不適合，則透過 NVLink TP=8，否則
    ZeRO-3/FSDP 用於純 DP 記憶體切割； EP 跨 8 為 MoE 層。
    **64 個 GPU（多節點）：**組成 — TP=8**節點內**，然後是 PP 和/或 EP
    **跨節點**，DP/ZeRO 在外部。證明每個： TP 頻寬所在
    最高，PP/EP 較低，DP 最外層（最能容忍慢速連結）。

??? success "2 — 實作 EP 並驗證損失是否與單 GPU 相符"
    連接 expert 並行性（手捲 all-to-all 調度/組合，或
    DeepSpeed-MoE/威震天）。**正確性檢查：**使用相同的種子/數據，
    EP 運行的損失必須追蹤單 GPU 運行約 50 步以內
    浮點噪音。如果它發生漂移，通常的罪魁禍首是**router 數學不是
    在 fp32**（跨等級 routing 分歧）或**未同步的平衡計數**
    — 正是 [training stability](../moe/training-stability.md) 中的錯誤。

??? success "3 — 量化 all-to-all 重疊；MFU 之前/之後"
    分析分塊管線**關閉**，然後**開啟**。關閉：all-to-all
    在時間軸上顯示為暴露的間隙（通訊期間 GPU 空閒）→ 較低的 MFU。
    開啟：通訊與獨立計算重疊（共用 expert / 下一個區塊 attention）→
    差距縮小，**MFU 上升**。報告 MFU =
    $6P\cdot\text{tok/s}/\text{peak}$ 兩者皆適用； delta 為重疊值，
    最重要的 EP 優化。

??? success "4 — 強縮放表和線性偏差"
    固定問題大小，增加 GPU，繪製加速與 GPU 數量的關係。它將
    **低於線性理想**，因為： (a)**通訊成長**
    規模 (all-reduce/all-to-all)，(b)**管道氣泡**$\frac{P-1}{m+P-1}$
    消耗運算能力，(c)**每 GPU 工作量縮減**，直到 kernels 發布-/latency-
    界（低佔用率），以及（d）負載不平衡。將差距歸咎於這些
    術語 - 命名「為什麼」縮放偏差是真正的可交付成果，而不是表格。
