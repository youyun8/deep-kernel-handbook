# Kimi K2.5 decode 概觀與模型組態

<div class="page-meta" markdown>
<span class="chip"><strong>Model:</strong> Kimi K2.5</span>
<span class="chip"><strong>Backend:</strong> SGLang + AITER MoE</span>
<span class="chip"><strong>Target:</strong> gfx950 / MI355X ×4 / TP4</span>
<span class="chip"><strong>Trace:</strong> PyTorch profiler + CUDA Graph</span>
</div>

本章把 Kimi K2.5 在 decode 階段的單層執行路徑拆開，逐一把 profiler 裡的 GPU kernel 對回 模型結構與 AITER operator。kernel 名稱與順序都直接來自實測的 Chrome/Kineto trace。內容依主題 拆成數頁：本頁先固定模型組態，後續頁面分別處理 decode 算子數學、shared-expert fusion 開 / 關 對照、MoE GEMM 與吸收式 BMM，以及 TP 通訊與重現方式。

核心對照是 **shared-expert fusion 開啟 / 關閉** 兩組 trace。兩者的 attention 路徑相同， 差異集中在 MoE：fusion 開啟時，shared expert 被併進 routed grouped GEMM；fusion 關閉時， 它是一條獨立的 shared MLP pipeline。

```text
record_function("Decode") window
  → one DeepSeek-V2 decode layer, anchored by input RMSNorm
  → ordered GPU kernels
  → SGLang AITER operators
  → tuned_fmoe.csv entries that select MoE kernels
```

---

## 模型組態與 `moe_tp_size`

後面的 shape、FLOPs 與 bandwidth 推導都依賴這些維度。數值來自 runtime log 與模型 config，而不是從 trace 反推。

| 參數                             | 值                              | 來源 / 備註                                         |
| -------------------------------- | ------------------------------- | --------------------------------------------------- |
| Transformer 層數                 | 61（layer 0 dense + 1–60 MoE）  | `num_hidden_layers=61`、`first_k_dense_replace=1`   |
| hidden size $H$                  | 7168                            | `hidden_size=7168`                                  |
| MoE intermediate（全域）         | 2048                            | `moe_intermediate_size=2048`                        |
| MoE intermediate（每 partition） | $I=256$                         | `intermediate_size_per_partition=256`（= 2048 / 8） |
| routed experts                   | 384                             | `n_routed_experts=384`                              |
| fused shared expert              | 1                               | `n_shared_experts=1`、`num_fused_shared_experts=1`  |
| top-k                            | 9（8 routed + 1 shared）        | `num_experts_per_tok=8`、runtime `top_k=9`          |
| 權重格式                         | MXFP4（`per_1x32` block scale） | `w13/w2 = float4_e2m1fn_x2`，scale `uint8`          |
| 每專家 W13（gate+up）            | `[512, 7168]` FP4               | `w13_up_dim=512`（= 2×256）                         |
| 每專家 W2（down）                | `[7168, 256]` FP4               | `w2_down_dim=128`（fp4x2 packed）                   |

!!! Note "`moe_tp_size=8`"

    1. **runtime log 直接印出 `moe_tp_size=8`**： `FusedMoE.__init__: ... Num_experts=385, num_fused_shared_experts=1, moe_ep_size=1, moe_tp_size=8`
    2. **與 server_args 一致**：profiling 用 `--tensor-parallel-size 4`，server_args 為 `tp_size=4, moe_ep_size=1, moe_dp_size=1` （`shared_expert_fusion_on/.../server.log` 第 8 行）。
    3. **與 SGLang 推導式一致**： `moe_tp_size = tp_size // moe_ep_size // moe_dp_size` （`/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py:1118`）。

    關鍵是 Kimi K2.5 對 MoE 採用 **attention DP + MoE TP** 的切法：attention 走 TP4， MoE 權重則沿 intermediate 維度切成 8 份（`2048 / 8 = 256`）。因此 `moe_tp_size=8` 與 `tp_size=4` 並不矛盾；它們描述的是不同並行軸。本章所有 `inter_dim=256` 的 kernel shape 都直接來自這個 MoE TP 切分。

注意 **gate+up = 512 = 2 × 256**。stage-1 的輸出寬度是 stage-2 輸入寬度的兩倍， 這也是後面 MoE GEMM 推導中 stage-1 約為 stage-2 兩倍成本的結構原因。

---

## 本章導覽

這條 decode 路徑的細節分散在以下頁面，建議依序閱讀：

1. [Decode 算子數學對照](decode-math.md) —— 把整個 decode step 的數學路徑（attention + MoE）逐一對齊。
2. [Shared-expert fusion 開 / 關](fusion.md) —— 怎麼從 trace 框出一層，以及 fusion 開啟 / 關閉兩組 kernel 順序對照。
3. [MoE GEMM 與吸收式 BMM](moe-gemm.md) —— 為什麼 stage-1 ≈ 2× stage-2、吸收式 BMM，以及 tuned_fmoe.csv 的完整 kernel 名稱。
4. [TP 通訊、查表與重現](comms-repro.md) —— All-reduce 成本、trace pattern 回原始碼的查表，以及如何重現這兩組 trace。
5. [AITER 原始碼解析](source-breakdown.md) —— AITER repo 架構、tuning 機制與「要改 MoE 該動哪裡」。
