# ML Perf Handbook

**Learn state-of-the-art ML systems from scratch** — from the math and a clean
reference implementation, to the performance-engineered, multi-GPU version that
actually runs efficiently on modern accelerators. The flagship, deepest series
is **Mixture-of-Experts (MoE)**, motivated by the wave of modern sparse models
(Kimi K2.5, DeepSeek-V3, Mixtral, Qwen-MoE).

> Live site: <https://youyun8.github.io/ml-perf-handbook/>
> (published automatically by GitHub Actions on every push to `main`)

- **Suggested repo name:** `ml-perf-handbook`
- **One-line GitHub "About":** *From-scratch course on SOTA ML systems and GPU performance engineering, with a deep Mixture-of-Experts track (PyTorch + Triton + CUDA/ROCm).*

---

## Why this stack: MkDocs + Material

I evaluated four candidates for a **math- and code-heavy curriculum that must
deploy cleanly to GitHub Pages**: MkDocs Material, Quarto / Jupyter Book,
Astro + MDX, and Docusaurus. **I chose MkDocs Material.** It hits every hard
requirement out of the box with the least moving parts: LaTeX via
`pymdownx.arithmatex` + KaTeX, syntax highlighting with one-click copy buttons
(`content.code.copy`), fast client-side full-text search, a hierarchical
left-nav with section indexes, dark mode, and a responsive layout — all
configured in a single `mkdocs.yml`. Deployment to Pages is a ~30-line GitHub
Actions workflow with no Node toolchain to maintain, and authors write plain
Markdown so contributions have a near-zero learning curve. Jupyter Book is
superb for *executing* notebooks but heavier and slower to build; Astro/MDX and
Docusaurus are more flexible for bespoke UI but add a JavaScript build and
component layer that buys little for a docs site and costs maintenance. Because
runnable code is a first-class goal, the heavy, GPU-dependent implementations
live in a separately-tested `code/` tree (run under `pytest`) and are linked
from the prose — keeping the docs build fast and CPU-only while the code stays
genuinely runnable and verified.

---

## What's inside

| Part | Focus |
|------|-------|
| **I · Foundations** | Transformer as a *system* (FLOPs, memory, arithmetic intensity, roofline); attention efficiency (KV cache, FlashAttention from scratch, paged attention); numerics & precision (fp32/bf16/fp16/fp8). |
| **II · Mixture-of-Experts (flagship)** | Why sparsity; MoE layer from scratch; load balancing (aux loss, capacity, aux-loss-free bias); routing variants; training stability; expert parallelism & all-to-all; MoE kernels in Triton/CUDA/HIP; inference & serving; case studies (DeepSeek-V3, Mixtral, Qwen, Kimi). |
| **III · Performance** | GPU programming model (CUDA & ROCm/HIP side by side); Triton and CUDA/HIP kernel tracks; distributed training (DP/TP/PP/SP/EP, ZeRO, collectives); quantization, pruning, distillation; inference optimization; profiling methodology. |
| **IV · AITER** | A named, measurable decode execution path on AMD: mapping Kimi-K2.5 MXFP4 profiler traces back through the SGLang→AITER call path, Python dispatchers, and the underlying HIP/CK/FlyDSL kernels — with a roofline explanation of why MoE expert GEMM dominates decode. |
| **V · Capstones** | Build a small MoE LM end-to-end, then optimize it and report measured speedups; a guide to scaling it with the parallelism techniques. |

ROCm/HIP is treated as a **first-class target** alongside CUDA throughout —
including where warp/wavefront width (32 vs 64), occupancy, and API names differ.

## Project status

This repo is built to be **usable on first deploy**: the Foundations and the
Mixture-of-Experts flagship pages are written in depth with runnable code.
Some Part III/IV pages are intentionally shipped as clearly-labeled scaffolds
(they carry a "部分實作" banner). Nothing unfinished
is dressed up as finished.

---

## Local development

Requires Python 3.10+.

```bash
git clone https://github.com/youyun8/ml-perf-handbook.git
cd ml-perf-handbook

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdocs serve          # live preview at http://127.0.0.1:8000
```

Build commands:

```bash
mkdocs build            # render static site into ./site
mkdocs build --strict   # what CI runs — fails on broken links or nav refs
```

### Running the reference code

The runnable implementations have their own (heavier) dependency set:

```bash
pip install -r code/requirements.txt
pytest code/                       # CPU-only tests pass without a GPU
pytest code/ -m gpu                # GPU/Triton tests (need CUDA or ROCm)
python code/moe/train_tiny_moe.py  # trains a tiny MoE on a toy task
```

Hardware assumptions: PyTorch examples run on CPU. Triton kernels need an
NVIDIA (CUDA) or AMD (ROCm) GPU; CUDA/HIP `.cu`/`.cpp` examples need the
respective toolchain (`nvcc` or `hipcc`). Every page states its assumptions.

---

## One-time GitHub Pages setup

The deploy workflow (`.github/workflows/deploy.yml`) is ready to run. To turn
it on:

1. Push this repo to GitHub (`main` branch).
2. In the repo, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, select **GitHub Actions**
   (not "Deploy from a branch").
4. Push any commit to `main` (or run the workflow manually from the
   **Actions** tab → *Build & deploy docs* → *Run workflow*).
5. The site publishes to `https://<user>.github.io/ml-perf-handbook/`.
   Update `site_url` and `repo_url` in `mkdocs.yml` if your username/repo
   differ from `youyun8/ml-perf-handbook`.

No `gh-pages` branch is needed — the workflow uploads a Pages artifact and
deploys it directly.

---

## Repository layout

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full layout and a guide to
adding pages and runnable code.

## License

Dual-licensed: **code = MIT**, **prose = CC BY 4.0**. See [`LICENSE`](LICENSE).
