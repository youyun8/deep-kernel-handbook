# Deep Kernel Handbook

**Learn modern ML systems from scratch and make them fast on real GPUs.** Each topic starts from the math and a clean reference implementation, then works up to the performance-engineered, multi-GPU version that runs efficiently on modern accelerators. The deepest track is **Mixture-of-Experts (MoE)**, driven by today's wave of sparse models (Kimi K2.5, DeepSeek-V3, Mixtral, Qwen-MoE), and it culminates in a named, measurable **AITER** decode trace on AMD hardware.

> Live site: <https://youyun8.github.io/deep-kernel-handbook/> (published automatically by GitHub Actions on every push to `main`)

- **Repo name:** `deep-kernel-handbook`
- **One-line GitHub "About":** _From-scratch handbook on modern ML systems and GPU kernel performance, with a deep Mixture-of-Experts track and a real AITER decode trace (PyTorch + Triton + CUDA/ROCm)._

---

## What's inside

| Part | Focus |
| --- | --- |
| **I · Foundations** | Transformer as a _system_ (FLOPs, memory, arithmetic intensity, roofline); attention efficiency (KV cache, FlashAttention from scratch, paged attention); numerics & precision (FP32/BF16/FP16/FP8). |
| **II · Mixture-of-Experts** | Why sparsity; MoE layer from scratch; load balancing (aux loss, capacity, aux-loss-free bias); routing variants; training stability; expert parallelism & all-to-all; MoE kernels in Triton/CUDA/HIP; inference & serving; case studies (DeepSeek-V3, Mixtral, Qwen, Kimi). |
| **III · Performance** | GPU programming model (CUDA & ROCm/HIP side by side); Triton and CUDA/HIP kernel tracks; distributed training (DP/TP/PP/SP/EP, ZeRO, collectives); quantization, pruning, distillation; inference optimization; profiling methodology. |
| **IV · AITER** | A named, measurable decode execution path on AMD: mapping Kimi K2.5 (MXFP4) profiler traces back through the SGLang→AITER call path, Python dispatchers, and the underlying HIP/CK/FlyDSL kernels — with a roofline explanation of why MoE expert GEMM dominates decode. |
| **V · Capstones** | Build a small MoE LM end-to-end, then optimize it and report measured speedups; a guide to scaling it with the parallelism techniques. |

ROCm/HIP is treated as a **first-class target** alongside CUDA throughout — including where warp/wavefront width (32 vs 64), occupancy, and API names differ.

---

## Local development

Requires Python 3.10+.

```bash
git clone https://github.com/youyun8/deep-kernel-handbook.git
cd deep-kernel-handbook

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

Hardware assumptions: PyTorch examples run on CPU. Triton kernels need an NVIDIA (CUDA) or AMD (ROCm) GPU; CUDA/HIP `.cu`/`.cpp` examples need the respective toolchain (`nvcc` or `hipcc`). Every page states its assumptions.

---

## One-time GitHub Pages setup

The deploy workflow (`.github/workflows/deploy.yml`) is ready to run. To turn it on:

1. Push this repo to GitHub (`main` branch).
2. In the repo, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, select **GitHub Actions** (not "Deploy from a branch").
4. Push any commit to `main` (or run the workflow manually from the **Actions** tab → _Build & deploy docs_ → _Run workflow_).
5. The site publishes to `https://<user>.github.io/deep-kernel-handbook/`. Update `site_url` and `repo_url` in `mkdocs.yml` if your username/repo differ from `youyun8/deep-kernel-handbook`.

No `gh-pages` branch is needed — the workflow uploads a Pages artifact and deploys it directly.

---

## Repository layout

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full layout and a guide to adding pages and runnable code.

## License

Dual-licensed: **code = MIT**, **prose = CC BY 4.0**. See [`LICENSE`](LICENSE).
