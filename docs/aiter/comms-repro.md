# TP 通訊、查表與重現

這頁收尾：decode 一層的 TP all-reduce 成本、把 trace pattern 對回 AITER 原始碼的查表， 以及如何重現本章用到的兩組 trace。kernel 順序見 [Shared-expert fusion 開 / 關](fusion.md)。

## TP communication（all-reduce）

每層有 **2 次 all-reduce**：一次在 attention o_proj 後，一次在 MoE down + combine 後。 Trace 中對應兩個 `allreduce_fusion_kernel_1stage`。decode 的 hidden-state 訊息很小：

$$
\text{每次 all-reduce bytes} = \text{bs} \cdot 7168 \cdot 2
= \begin{cases} 448\,\text{KB} & (\text{bs}=32) \\ 896\,\text{KB} & (\text{bs}=64) \end{cases}
$$

訊息小代表 collective 主要受 latency 限制，不容易被 batch 攤平，因此它是 MoE GEMM 後面 固定存在的尾巴。concurrency 很高時，fused all-reduce 會從 1-stage 切到 2-stage reduce-scatter + load-RMSNorm （`reduce_scatter_cross_device_store` / `local_device_load_rmsnorm`，見 [Shared-expert fusion 開 / 關](fusion.md) 同 trace 的 其他 rank）。tuning 入口：`aiter/ops/custom_all_reduce.py`、`aiter/dist/communication_op.py`。

## 從 trace 回到原始碼的查表

| trace pattern                                       | 功能                                      | 優先看的檔案                                                             |
| --------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| `fused_qk_rmsnorm`                                  | input / QK RMSNorm + quant                | `aiter/ops/fused_qk_norm_rope_cache_quant.py`、`aiter/ops/rmsnorm.py`    |
| `hgemm_bf16_*`                                      | QKV / o_proj / router GEMM                | `aiter/tuned_gemm.py`、`aiter/ops/gemm_op_a16w16.py`                     |
| `_batched_gemm_a16wfp4_*`                           | K-absorb / V-absorb BMM                   | `aiter/ops/batched_gemm_op_bf16.py`、`aiter/ops/gemm_op_a4w4.py`         |
| `_fused_qk_rope_cat_and_cache_mla`                  | RoPE + KV cache write                     | `aiter/ops/rope.py`、`aiter/ops/cache.py`                                |
| `mla_a8w8_*`                                        | MLA core attention                        | `aiter/mla.py`、`aiter/aot/asm_mla_decode_fwd.py`、`csrc/cpp_itfs/mla/*` |
| `kn_mla_reduce_v1_ps`                               | split-KV reduce                           | `aiter/ops/attention.py`                                                 |
| `allreduce_fusion_kernel_1stage`                    | TP all-reduce fusion                      | `aiter/ops/custom_all_reduce.py`、`aiter/dist/communication_op.py`       |
| `grouped_topk_kernel`                               | biased grouped top-k                      | `aiter/ops/topk.py`、`aiter/ops/moe_op.py`                               |
| `opus_moe_sorting_entry`                            | MoE sort（token→expert 分桶）             | `aiter/ops/moe_sorting_opus.py`                                          |
| `fused_mx_quant_moe_sort` / `mxfp4_moe_sort`        | routed input MXFP4 quant + sort           | `aiter/ops/quant.py`、`aiter/utility/fp4_utils.py`                       |
| `mfma_moe1` / `flydsl_moe1`                         | MoE GEMM1 gate/up + SwiGLU                | `aiter/fused_moe.py`、`aiter/ops/flydsl/kernels/moe_gemm_2stage.py`      |
| `mfma_moe2` / `flydsl_moe2` / `moe_ck2stages_gemm2` | MoE GEMM2 down + combine                  | `aiter/fused_moe.py`、`csrc/ck_gemm_moe_2stages_codegen/*`               |
| `_dynamic_mxfp4_quant` / `_gemm_afp4wfp4*`          | standalone shared expert（fusion 關閉時） | `aiter/ops/quant.py`、`aiter/ops/gemm_op_a4w4.py`                        |
| `add_rmsnorm_quant`                                 | residual add + norm + quant               | `aiter/ops/rmsnorm.py`                                                   |

## 重現

兩組 trace 由 `run_multistream_profile_comparison.sh` 產生。A/B 列表分別設成 baseline （fusion 開）與 `--disable-shared-experts-fusion`（fusion 關）：

```bash
# 在 run_multistream_profile_comparison.sh 的 RUN_LIST 內保留：
#   "1k shared experts fusion|"                 -> shared_expert_fusion_on
#   "1k nofuse|--disable-shared-experts-fusion" -> shared_expert_fusion_off
./run_multistream_profile_comparison.sh --platform amd --moe-runner-backend aiter
```

輸出是 PyTorch profiler 的 CUDA Graph trace：

```text
shared_expert_fusion_on/amd_isl1024_osl1024_cuda_graph_profile_logs/traces/conc_4_isl_1024_osl_1024/*/<ts>-TP-0.trace.json.gz
shared_expert_fusion_off/amd_isl1024_osl1024_cuda_graph_profile_logs/traces/conc_4_isl_1024_osl_1024/*/<ts>-TP-0.trace.json.gz
```

解析單層 decode kernel 流程時使用自寫 parser；它只讀 Chrome trace，並以 `record_function("Decode")` 與 `fused_qk_rmsnorm` 作為 anchor：

```bash
python3 decode_analysis/parse_decode_layer.py --full shared_expert_fusion_on/.../conc_4_isl_1024_osl_1024/*/<ts>-TP-0.trace.json.gz
python3 decode_analysis/parse_decode_layer.py --full shared_expert_fusion_off/.../conc_4_isl_1024_osl_1024/*/<ts>-TP-0.trace.json.gz
```

!!! Note "判讀邊界"
    這裡的 kernel 名稱與順序對應 Kimi K2.5（MXFP4 權重）、gfx950、TP4（attention）/ moe_tp_size=8（MoE）、KV cache fp8_e4m3、conc4 / ISL1024。架構結論可遷移，但具體 tile、kernel 名稱與比例會隨 hidden / intermediate size、top-k、context length、batch 與 tuned config 改變。若要把這條 decode 路徑放回更一般的脈絡，回頭看 [MoE decode 剖析](../moe/decode-anatomy.md) 與 [Profiling 與方法論](../performance/profiling.md)。
