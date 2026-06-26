---
hide:
  - navigation
  - toc
---

<section class="home-hero" markdown>
<div class="home-hero__copy" markdown>

<p class="home-kicker">ML 系統效能路線</p>

# 從模型到 GPU kernel 的完整效能手冊

現代機器學習已經是系統工程。理解 Transformer 的數學只是起點；真正
決定模型能不能被有效部署的，是 FLOPs、記憶體流量、kernel、collective、
量化格式，以及這些選擇在 GPU 上互相牽制的方式。

這份手冊把每個核心主題拆成**直覺 → 數學 → 參考實作 → 效能化版本**。
你會看到乾淨的 PyTorch/Triton/CUDA/HIP 版本，也會看到實際 profiling
如何指出瓶頸，最後如何把瓶頸對應回可修改的原始碼。

<div class="home-actions" markdown>
[開始閱讀路線 :material-arrow-right:](reading-path.md){ .md-button .md-button--primary }
[查看 AITER 深入解析](aiter/index.md){ .md-button }
</div>

</div>

<div class="home-hero__panel" aria-label="Course scope" markdown="0">
<div class="home-kicker">核心深挖</div>
<div class="home-panel-title">Kimi-K2.5 · MoE · AITER</div>
<div class="home-panel-copy">從 routing、top-k、sort、MXFP4 quant 到 stage-1 / stage-2 MoE GEMM，使用真實 decode trace 串起整條執行路徑。</div>
<div class="home-metrics">
<div><strong>5</strong><span>主題部件</span></div>
<div><strong>25</strong><span>decode stages</span></div>
<div><strong>TP4</strong><span>MI355X 追蹤</span></div>
</div>
</div>
</section>

---

## 本站如何使用

<div class="grid cards" markdown>

- :material-book-open-variant:**讀手冊**

  先建立共同語彙：FLOPs、bytes、roofline、attention memory traffic、
  MoE routing、expert parallelism 與 all-reduce。

- :material-speedometer:**看 profiling**

  每個效能結論都要能回到 trace、kernel 名稱、shape 與測量方法。
  不只記住「哪裡慢」，也要知道為什麼慢。

- :material-code-braces:**對原始碼**

  章節會標出 AITER / SGLang / kernel wrapper 的實際檔案位置，讓你能
  從 profiling bucket 直接跳到需要修改或 tune 的路徑。

</div>

---

## 課程地圖

<div class="curriculum-grid" markdown>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">第一部</span>

### :material-cube-outline: 基礎

建立效能分析需要的數學與系統直覺：Transformer、attention、precision、
roofline 與資料搬移。

[開啟基礎篇](foundations/index.md){ .md-button }

</section>

<section class="curriculum-card curriculum-card--feature" markdown>
<span class="curriculum-card__eyebrow">第二部 · flagship</span>

### :material-expansion-card: Mixture-of-Experts

從稀疏化、routing、load balancing 到 expert parallelism、MoE kernels
與 serving，把 MoE 看成完整系統。

[開啟 MoE 路線](moe/index.md){ .md-button .md-button--primary }

</section>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">第三部</span>

### :material-flash: 效能工程

練習 GPU programming、Triton、CUDA/HIP、distributed training、量化與
profiling 方法論。

[開啟效能篇](performance/index.md){ .md-button }

</section>

<section class="curriculum-card curriculum-card--feature" markdown>
<span class="curriculum-card__eyebrow">第四部 · production trace</span>

### :material-chart-timeline-variant: AITER 深入解析

用 Kimi-K2.5 MXFP4 decode trace 解析 AITER MoE stack：moe gemm 1、
moe gemm 2、routing/sort/quant、shared expert 與 all-reduce fusion。

[閱讀 AITER 章節](aiter/index.md){ .md-button .md-button--primary }

</section>

</div>

---

## 近期重點

<div class="ml-stats-grid" markdown="0">
<div class="ml-stat"><strong>52%</strong><span>8k conc32/64 decode 中 MoE expert GEMM 約占比</span></div>
<div class="ml-stat"><strong>2×</strong><span>stage-1 gate/up + SwiGLU 約為 stage-2 down 的時間</span></div>
<div class="ml-stat"><strong>11-18%</strong><span>TP all-reduce 在 decode 中不易隨 batch 攤平</span></div>
</div>

!!! tip "閱讀建議"
    若你剛開始，先照 [閱讀路線](reading-path.md) 建立系統語彙；若你正在處理
    Kimi-K2.5 / SGLang / AITER profiling，直接看 [AITER decode 深入解析](aiter/index.md)，
    再回到 MoE 與效能章節補齊背景。

---

_內容採 CC BY 4.0 授權；程式碼採 MIT 授權。_
