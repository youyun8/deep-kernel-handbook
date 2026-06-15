# Profiling & methodology

<div class="page-meta">
  <span class="chip"><strong>Level:</strong> all</span>
  <span class="chip"><strong>Prereqs:</strong> <a href="../../foundations/transformer-systems/">roofline</a></span>
  <span class="chip"><strong>Hardware:</strong> GPU to run profilers</span>
</div>

**Read this early and reread it often.** Every optimization in this handbook is
justified by a measurement, and most measurement is wrong the first time. This
page covers how to benchmark a GPU correctly, the pitfalls that produce fake
speedups, and how to read a profile to find the real bottleneck.

## Measure against a target, not a vibe

Always start from the [roofline](../foundations/transformer-systems.md): compute
the *theoretical* time for your op (FLOPs/π if compute-bound, bytes/β if
memory-bound). That number is your target. "It takes 2 ms" is meaningless;
"it takes 2 ms against a 1.1 ms roofline → 50% of peak" is actionable. Without a
target you can't tell a good kernel from a bad one.

Compute **MFU** (model FLOPs utilization) for training: $\text{MFU} =
\frac{6 P \cdot \text{tokens/s}}{\pi}$. Healthy large-model training is roughly
40–55% MFU; if you're at 15%, communication or a stall — not the matmuls — is the
problem.

## Benchmarking a GPU correctly

GPU kernels launch **asynchronously**, so naive timing measures launch latency,
not execution. The rules:

```python
import torch
def benchmark(fn, iters=100, warmup=20):
    for _ in range(warmup):           # 1. WARM UP: JIT, autotune, caches, clocks
        fn()
    torch.cuda.synchronize()          # 2. SYNC before starting the timer
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()          # 3. SYNC before reading the timer
    return start.elapsed_time(end) / iters    # ms per iter (use CUDA events!)
```

The non-negotiables:

1. **Warm up.** The first call pays JIT/autotune, allocator, and cuDNN/cuBLAS
   algorithm-selection costs, and the GPU may be at a low clock. Discard it.
2. **Use CUDA events** (`torch.cuda.Event`), not `time.time()` — events measure
   on-device time and bracket async work correctly.
3. **`synchronize()`** before and after timing; otherwise you time the launch, not
   the kernel.
4. **Repeat and aggregate.** Report median (robust to outliers), and the spread.
5. **Lock clocks if you can** (`nvidia-smi -lgc` / `rocm-smi --setperflevel`) so
   thermal/boost variation doesn't masquerade as a regression.

On AMD the same applies with `torch.cuda` (HIP-backed) APIs;
`triton.testing.do_bench` handles warmup + events for you and is the easy default.

## The pitfalls that produce fake speedups

These bite everyone at least once:

- **No warmup** → you "optimized away" first-call overhead, not the kernel.
- **No sync** → you measured launch latency (~microseconds), reporting an
  absurd speedup.
- **Dead-code elimination** → the compiler dropped your kernel because the output
  is unused. *Consume the output* (sum it, return it).
- **Caching / constant folding** → identical inputs every iter let caches or the
  framework short-circuit. Vary inputs or clear caches if relevant.
- **Including H2D/D2H transfers** in a compute benchmark → you measured PCIe.
  Keep tensors on-device; time the copy separately if it matters.
- **Tiny problem sizes** → dominated by launch overhead; not representative of the
  real workload. Benchmark realistic shapes.
- **Clock drift / thermal throttling** → run long enough or lock clocks; a "2%
  regression" is often just boost variance.
- **Comparing apples to oranges** → different precision, batch, or sequence length
  between baseline and optimized. Change one thing at a time.
- **Cherry-picking one shape** → report a sweep; a kernel fast at one size can be
  slow at others (this is why we [autotune](triton-track.md)).

!!! warning "The cardinal rule: verify correctness first"
    A fast kernel that returns wrong numbers is infinitely slow at its job. Every
    `code/` kernel here checks `torch.allclose` against a reference *before* any
    timing. Benchmark only verified code.

## Reading a profile

Wall-clock tells you *how slow*; a profiler tells you *why*. Two views:

- **Timeline / system view** (Nsight Systems; rocprof + Perfetto): shows kernels,
  memcpys, and gaps on a timeline. Look for **gaps** (CPU-bound launch overhead,
  Python, sync points), **serialized comm** (an all-reduce/all-to-all not
  overlapping compute — the [MoE](../moe/systems-ep.md) failure mode), and which
  kernels dominate.
- **Kernel view** (Nsight Compute; Omniperf): per-kernel counters — achieved
  occupancy, memory throughput vs peak, compute throughput vs peak,
  warp-stall reasons. This tells you the **regime**: if memory throughput is near
  peak and compute is low, you're memory-bound (fuse / raise intensity); the
  reverse means compute-bound (lower precision / fewer FLOPs).

The PyTorch profiler is the easy on-ramp (no external tool):

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    model(x); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# prof.export_chrome_trace("trace.json")  # view in chrome://tracing / Perfetto
```

## A profiling workflow

1. **Roofline first** — compute the target time and the expected regime.
2. **Timeline view** — is time in kernels, gaps, or comm? Fix gaps/comm before
   micro-optimizing kernels (often the bigger win).
3. **Kernel view on the top kernel** — confirm the regime, find the limiter
   (memory throughput? occupancy? stalls?).
4. **Optimize the limiter**, not what's easy — raise intensity if memory-bound,
   cut FLOPs/precision if compute-bound, overlap if comm-bound.
5. **Re-measure correctly** (warmup, events, sync, sweep) and compare to target.
6. **Repeat** until you're near the roofline or out of headroom.

| Profiler | NVIDIA | AMD |
|---|---|---|
| Timeline | Nsight Systems | rocprof (+ Perfetto) |
| Kernel counters | Nsight Compute | Omniperf / rocprof |
| In-framework | PyTorch profiler | PyTorch profiler (ROCm) |
| Quick kernel timing | `triton.testing.do_bench` | same |

## Key takeaways

- **Compute a roofline target first** — a measured time means nothing without one;
  track **MFU** for training.
- Benchmark with **warmup + CUDA events + synchronize + repeat + locked clocks**,
  and **consume the output** so it isn't optimized away.
- Most "speedups" are artifacts: no sync, no warmup, dead-code elimination,
  included transfers, tiny shapes, clock drift. Change one variable at a time.
- Use the **timeline view** to find gaps/serialized comm and the **kernel view**
  to find the per-kernel limiter; **optimize the limiter**. Verify correctness
  before timing.

## Exercises

!!! tip "Solutions"
    Worked answers are on the [Part solutions page](../solutions/performance.md). Try each exercise before expanding.

1. Take the [Triton softmax](triton-track.md), benchmark it wrong (no
   warmup/sync) then right; quantify the difference.
2. Profile a small transformer's decode step; identify whether attention, the FFN,
   or launch overhead dominates, and propose a fix.
3. Compute MFU for a training run given tokens/s and GPU peak; diagnose a 15% MFU
   result.
4. Construct a benchmark where dead-code elimination hides the kernel, then fix it
   by consuming the output.

## References

- Williams et al. *Roofline.* 2009.
- NVIDIA Nsight Systems / Nsight Compute documentation.
- AMD rocprof / Omniperf documentation.
- PyTorch Profiler and `triton.testing` docs.
