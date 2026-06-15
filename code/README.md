# Runnable reference code

Tested, from-scratch implementations that back the prose in the
[handbook](https://youyun8.github.io/ml-perf-handbook/). The reference
implementations are checked against PyTorch with `torch.allclose` so you can
trust them as ground truth and as a base to optimize.

## Layout

```
code/
├── requirements.txt
├── pytest.ini              # registers the `gpu` marker
├── attention/
│   ├── online_softmax.py        # the running-max softmax combiner
│   ├── flash_attention_numpy.py # tiled, online-softmax attention (numpy)
│   └── test_attention.py        # checks both vs dense reference (CPU)
├── moe/
│   ├── moe_layer.py             # experts, router, top-k, naive + dispatch forms
│   ├── load_balancing.py        # aux loss, capacity/drop, aux-loss-free bias
│   ├── train_tiny_moe.py        # trains a tiny MoE LM on a toy task
│   └── test_moe.py              # naive == dispatch; balancing behaviour (CPU)
└── kernels/
    ├── softmax_triton.py        # fused softmax Triton kernel
    ├── triton_grouped_gemm.py   # grouped GEMM (per-expert matmuls, one launch)
    ├── moe_permute.cu           # CUDA gather/scatter for MoE dispatch
    └── moe_permute_hip.cpp      # the same in ROCm/HIP (wavefront-aware notes)
```

## Running

```bash
pip install -r code/requirements.txt

pytest code/                 # all CPU tests (no GPU needed)
pytest code/ -m "not gpu"    # explicitly CPU-only
pytest code/ -m gpu          # Triton/GPU tests (needs CUDA or ROCm + triton)

python code/moe/train_tiny_moe.py            # train the toy MoE
python code/moe/train_tiny_moe.py --no-balance  # watch routing collapse
```

GPU tests `skip` automatically when no GPU or Triton is available, so
`pytest code/` is always green on a CPU box.

## Hardware notes

- The numpy/PyTorch references run on **CPU**.
- Triton kernels run on **NVIDIA (CUDA)** or **AMD (ROCm)** GPUs. On AMD, recall
  wavefront = 64 (vs 32): re-autotune for best performance — see the
  [CUDA/HIP track](https://youyun8.github.io/ml-perf-handbook/performance/cuda-hip-track/).
- The `.cu` / `_hip.cpp` files are illustrative kernels for the
  [MoE kernels page](https://youyun8.github.io/ml-perf-handbook/moe/kernels/);
  build with `nvcc` / `hipcc` respectively.
