---
hide:
  - toc
---

<section class="home-hero" markdown="0">
<div class="home-hero__copy">

<p class="home-kicker">深核手冊 · ML 系統效能路線</p>

<h1 class="home-title">從模型數學到 <span class="home-title__hl">GPU Kernel</span></h1>

<p class="home-lead">模型的部署效率取決於 FLOPs、記憶體流量、kernel、collective 與量化格式，以及它們在 GPU 上的交互。本手冊把這些因素逐一拆解、量測並對應到實作。</p>

<p class="home-sub">每個主題分成「直覺 → 數學 → 參考實作 → 效能化版本」，並把 profiling 結果對應回可修改的原始碼。</p>

<div class="home-actions">
<a class="md-button md-button--primary" href="reading-path/">開始閱讀路線&nbsp;→</a>
<a class="md-button" href="aiter/">AITER 深入解析</a>
</div>

<ul class="home-points">
<li><span aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polygon points="15.5 8.5 11 11 8.5 15.5 13 13" fill="currentColor" stroke="none"/></svg></span> 從共同語彙到 roofline，先建立分析直覺</li>
<li><span aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 14l4-4 3 3 5-6"/></svg></span> 每個結論都能回到 trace、kernel 與 shape</li>
<li><span aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 8 5 12 9 16"/><polyline points="15 8 19 12 15 16"/></svg></span> 從 profiling bucket 直接跳到可 tune 的原始碼</li>
</ul>

</div>

<aside class="home-hero__panel" aria-label="核心深挖">
<div class="home-panel__eyebrow">核心深挖</div>
<div class="home-panel__title">Kimi-K2.5 · MoE · AITER</div>
<p class="home-panel__copy">從 routing、top-k、sort、MXFP4 quant 到 stage-1 / stage-2 MoE GEMM，使用真實 decode trace 串起整條執行路徑。</p>
<div class="home-metrics">
<div class="home-metric"><strong>5</strong><span>主題章節</span></div>
<div class="home-metric"><strong>17</strong><span>decode kernels</span></div>
<div class="home-metric"><strong>TP4</strong><span>MI355X 追蹤</span></div>
</div>
</aside>
</section>

---

## 本站如何使用

<div class="grid cards" markdown>

- :material-book-open-variant: **讀手冊**

  先建立共同語彙：FLOPs、bytes、roofline、attention memory traffic、
  MoE routing、expert parallelism 與 all-reduce。

- :material-speedometer: **看 profiling**

  每個效能結論都要能回到 trace、kernel 名稱、shape 與測量方法。
  不只記住「哪裡慢」，也要知道為什麼慢。

- :material-code-braces: **對原始碼**

  章節會標出 AITER / SGLang / kernel wrapper 的實際檔案位置，讓你能
  從 profiling bucket 直接跳到需要修改或 tune 的路徑。

</div>

---

## 課程地圖

<div class="curriculum-grid" markdown>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">入門 · 分析直覺</span>

### :material-cube-outline: 基礎

建立效能分析需要的數學與系統直覺：Transformer、attention、precision、
roofline 與資料搬移。

[開啟基礎篇](foundations/index.md){ .md-button }

</section>

<section class="curriculum-card curriculum-card--feature" markdown>
<span class="curriculum-card__eyebrow">主線 · 稀疏模型系統</span>

### :material-expansion-card: Mixture-of-Experts

從稀疏化、routing、load balancing 到 expert parallelism、MoE kernels
與 serving，把 MoE 看成完整系統。

[開啟 MoE 路線](moe/index.md){ .md-button .md-button--primary }

</section>

<section class="curriculum-card" markdown>
<span class="curriculum-card__eyebrow">工具 · GPU 效能工程</span>

### :material-flash: 效能工程

練習 GPU programming、Triton、CUDA/HIP、distributed training、量化與
profiling 方法論。

[開啟效能篇](performance/index.md){ .md-button }

</section>

<section class="curriculum-card curriculum-card--feature" markdown>
<span class="curriculum-card__eyebrow">實戰 · production trace</span>

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
    Kimi-K2.5 / SGLang / AITER profiling，直接看 [AITER decode 一層的 kernel 流程](aiter/index.md)，
    再回到 MoE 與效能章節補齊背景。

---

_內容採 CC BY 4.0 授權；程式碼採 MIT 授權。_
