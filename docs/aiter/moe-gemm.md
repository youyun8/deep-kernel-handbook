# MoE GEMM 與吸收式 BMM

這頁解釋 decode 一層裡兩個最重的數學結構：MoE 的兩段 grouped GEMM，以及 MLA 的吸收式 BMM， 最後列出 tuned_fmoe.csv 實際選到的完整 kernel 名稱。kernel 順序對照見 [Shared-expert fusion 開 / 關](fusion.md)，模型維度見 [概觀與模型組態](index.md)。

## MoE GEMM 的數學：為什麼 stage-1 ≈ 2× stage-2

先看 **單一被選中的 token-expert row**。如果某個 expert 實際處理 $M$ 個 row， stage-1 與 stage-2 都乘上同一個 $M$，所以兩者的比例不變。

1. Stage-1

$$
\text{stage-1 (gate+up): } y = a\, W_{13}, \quad
a \in \mathbb{R}^{1 \times 7168}, \;
W_{13} \in \mathbb{R}^{7168 \times 512}, \;
y \in \mathbb{R}^{1 \times 512}.
$$

$$
\mathrm{FLOPs}_1 = 2 \cdot M \cdot N \cdot K = 2 \cdot 1 \cdot 512 \cdot 7168
= 7.34 \times 10^{6}\ \text{FLOP}.
$$

$$
\text{SwiGLU: } x = \operatorname{SiLU}(y_{:256}) \odot y_{256:}, \quad
x \in \mathbb{R}^{1 \times 256}, \quad 512 = 2 \times 256.
$$

2. Stage-2

$$
\text{stage-2 (down): } o = x\, W_{2}, \quad
x \in \mathbb{R}^{1 \times 256}, \;
W_{2} \in \mathbb{R}^{256 \times 7168}, \;
o \in \mathbb{R}^{1 \times 7168}.
$$

$$
\mathrm{FLOPs}_2 = 2 \cdot M \cdot N \cdot K = 2 \cdot 1 \cdot 7168 \cdot 256
= 3.67 \times 10^{6}\ \text{FLOP}.
$$

$$
\frac{\mathrm{FLOPs}_1}{\mathrm{FLOPs}_2} = \frac{512}{256} = 2.0.
$$

**Decode 時 MoE GEMM 主要受權重頻寬限制，而不是受 compute peak 限制。** FP4 權重 每個元素 0.5 byte；即使只算少量 token，每個被命中的 expert 仍要 stream 權重：

$$
\text{bytes}(W_{13}) = 512 \cdot 7168 \cdot 0.5 = 1.84\ \text{MB}, \quad
\text{bytes}(W_{2}) = 7168 \cdot 256 \cdot 0.5 = 0.92\ \text{MB}.
$$

stage-1 的 arithmetic intensity 可以用每個 expert 的 row 數 $m$ 表示：

$$
\mathrm{AI}_{\text{stage-1}}
= \frac{2 \cdot 7168 \cdot 512 \cdot m}{512 \cdot 7168 \cdot 0.5}
= 4m\ \ \text{FLOP/byte}.
$$

$m=1$ 時只有 **4 FLOP/byte**，$m=8$ 也只有 32。MI355X 的 FP4 compute / HBM bandwidth ridge point 遠高於此，因此 decode MoE GEMM 落在 roofline 的 bandwidth slope 上，離 compute roof 很遠。這也是 `mfma_moe1` / `mfma_moe2` 成為單層最重 kernel、且 stage-1 （gate/up）是 tuning 首要槓桿的原因。

## 吸收式（absorption）BMM

MLA（Multi-head Latent Attention）把 KV cache 存成低秩 latent；decode 時只讀 壓縮後的 cache。trace 中的 `_batched_gemm_a16wfp4`（kernel 3 與 7）就是這裡的 「吸收式 BMM」。

吸收（absorption）的意思是：原本 attention 會把 latent $c$ 透過 up-projection $W^{UK}, W^{UV}$ 還原成 per-head K/V 再計算；MLA 則把這些 up-projection 併進 query / output 側。結果是 decode 不需要展開完整 KV，只在 latent 維度上做 batched matmul。

令 $d_c$ 為 KV latent 維度，$d_n$ 為 non-RoPE key/query 維度，$W^{UK}\in \mathbb{R}^{d_c\times d_n}$ 為 K up-projection。未吸收的單 head logit 可寫成：

$$
s = q^{\top}(c W^{UK})
  = \big(q {W^{UK}}^{\top}\big)c^{\top}.
$$

先把 K up-projection 吸收到 query 側：

$$
\tilde q \equiv q {W^{UK}}^{\top} \in \mathbb{R}^{d_c},
\qquad
s = \tilde q\,c^{\top}.
$$

因此 K 側只需要在 $d_c$ 維度做 BMM（kernel 3），不必展開完整 K。輸出側同理把 $W^{UV}$ 吸收進 o-projection（kernel 7 的 V-absorb BMM）：

$$
o = \Big(\sum_u \alpha_u\, c_u\Big) W^{UV} W^{O},
\qquad
\widetilde{W}^{O} \equiv W^{UV} W^{O},
$$

其中 $\alpha_u$ 是 attention 權重。吸收的好處是 FLOPs 與 KV-cache 讀取量都從完整 per-head KV 的 $O(n_h d_h)$ 降到 latent cache 的 $O(d_c)$：

$$
\text{未吸收每步 KV 讀取} = 2\, n_h\, d_h\, L, \qquad
\text{吸收後} = d_c\, L \quad (d_c \ll n_h\, d_h),
$$

$L$ 是 context length。decode 時 MLA 的主要成本因此變成「讀低秩 KV cache 的頻寬」； `_batched_gemm_a16wfp4` 則用 FP4 權重與小 batch GEMM 來執行吸收後的投影。

## tuned_fmoe.csv 裡的完整 stage-1 / stage-2 kernel 名稱

trace 中的 `mfma_moe1/2` 是執行期實際選到的 kernel；選擇邏輯由 `get_2stage_cfgs()` 依 lookup key 從 `tuned_fmoe.csv` 查出。以下針對 `model_dim=7168`、`expert=385`、`topk=9`、FP4 這組 shape，列出 tuned csv 內各個 padded-M tier 的完整 `kernelName1` / `kernelName2`。decode 的小 M 會被 `get_padded_M` 補到 power-of-two，因此實際命中的是 `token` 欄對應的 tier。

來源：`kimik2_fp4_tp4_tuned_fmoe.csv`，篩 `inter_dim=256`、 `expert=385`、`topk=9`（即 `moe_tp_size=8` 的 routed+shared 9-way）。

<div class="aiter-stage-table" markdown>

| token（padded M） | kernelName1（stage-1 gate/up+SwiGLU）                      | kernelName2（stage-2 down+combine）                                                                                        |
| ----------------: | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
|                 1 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb14_fp4`       | `moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16` |
|                 2 | `flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_kb4_bnt0_go_fp4` | `moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16` |
|                 4 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_kb7_bnt0_fp4`   | `flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic`                                                                            |
|                 8 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic`                                                                            |
|                16 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic`                                                                            |
|                32 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w4_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t16x256x256_atomic_bnt2_sbm32`                                                                 |
|                64 | `flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4`             | `flydsl_moe2_afp4_wfp4_bf16_t16x256x256_atomic_bnt2_persist_sbm32`                                                         |
|               128 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic_bnt2_persist`                                                               |
|               256 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t16x256x256_atomic_bnt2_persist_sbm32`                                                         |
|               512 | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2_persist`                                                               |
|              1024 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_fp4`            | `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_persist_sbm64`                                                              |

</div>

這張表可以直接讀出幾個行為：

- **小 M（1–2）的 stage-2 走 CK**（`moe_ck2stages_gemm2_*`），M ≥ 4 才換成 FlyDSL `flydsl_moe2_*_atomic`。這跟 [Shared-expert fusion 開 / 關](fusion.md) trace 看到的 `mfma_moe2`（FlyDSL，conc4 對應 padded M=8）一致。
- **stage-2 大多是 `atomic` combine**（直接 atomic accumulate 到 `[M, 7168]`）；只有 更大的 prefill tier（M ≥ 2048）才改用 `reduce`。decode 全程在 atomic 範圍。
- runtime log（`kimi_k25_rocm_path.md` 第 267–292 行）顯示 conc 較高時 padded M=8 命中 `flydsl_moe1_..._t64x128x256_w4_fp4` + `flydsl_moe2_..._t64x128x256_atomic`，M=1 命中 `t32x128x256_w3_kb14_fp4` + CK stage2，與上表 token=8 / token=1 列吻合，表示 runtime 確實命中 tuned config，而不是 fallback。
