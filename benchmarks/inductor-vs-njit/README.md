# Inductor vs njit: CPU Latency by Number of Kernels

Measures wall-clock time of running an inductor-compiled graph through
`NjitInductorGraph` (the njit wrapper) vs the standard `torch.compile`
Python wrapper.  Tensors are small (32×64) so GPU compute is negligible —
only the CPU orchestration overhead is measured.  No `cudaDeviceSynchronize`
is called.

The independent variable is the number of Triton kernels in the graph.
Each graph is a chain of `torch.softmax` calls with alternating dims
(`dim=i%2`), which forces inductor to produce exactly one kernel per op.

## Results

![overhead_vs_kernels](overhead_vs_kernels.png)

### Linear fit (outliers removed, 2σ threshold)

|         | model     | k (µs/kernel) | b (µs)  |
|---------|-----------|---------------|---------|
| njit    | y = kx+b  | 1.9299        | 18.8674  |
| compile | y = kx+b  | 5.4319        | 46.5505  |

- **k** (slope) is the **per-kernel cost** — the marginal time (in µs)
  added by each additional Triton kernel launch.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs)
  for entering and leaving the wrapper, independent of how many kernels
  are launched.

### Raw data

| Kernels | njit (µs) | compile (µs) |
|---------|-----------|--------------|
| 1 | 14.70 | 37.43 |
| 2 | 20.00 | 49.45 |
| 3 | 25.67 | 58.75 |
| 4 | 24.03 | 64.88 |
| 5 | 28.43 | 72.13 |
| 6 | 28.63 | 76.72 |
| 7 | 31.96 | 84.77 |
| 8 | 31.65 | 88.06 |
| 9 | 36.65 | 94.97 |
| 10 | 37.53 | 99.32 |
| 11 | 40.78 | 104.78 |
| 12 | 41.31 | 109.84 |
| 13 | 44.96 | 120.04 |
| 14 | 45.32 | 123.16 |
| 15 | 48.35 | 128.83 |
| 16 | 51.08 | 133.36 |
| 17 | 53.66 | 139.32 |
| 18 | 54.92 | 144.81 |
| 19 | 56.38 | 150.84 |
| 20 | 58.35 | 155.27 |
| 21 | 59.77 | 163.87 |
| 22 | 61.46 | 166.63 |
| 23 | 64.86 | 173.12 |
| 24 | 65.19 | 177.97 |
| 25 | 68.06 | 184.72 |
| 26 | 68.68 | 190.19 |
| 27 | 71.12 | 197.67 |
| 28 | 72.69 | 201.82 |
| 29 | 77.52 | 204.45 |
| 30 | 77.38 | 209.91 |
| 31 | 79.03 | 213.88 |
| 32 | 80.66 | 220.18 |
| 33 | 83.52 | 226.97 |
| 34 | 85.18 | 231.79 |
| 35 | 83.94 | 240.00 |
| 36 | 86.79 | 248.33 |
| 37 | 91.29 | 249.95 |
| 38 | 93.07 | 251.90 |
| 39 | 94.81 | 260.62 |
| 40 | 96.33 | 261.68 |
| 41 | 101.72 | 268.80 |
| 42 | 100.30 | 277.10 |
| 43 | 101.14 | 277.23 |
| 44 | 104.00 | 284.55 |
| 45 | 106.14 | 286.18 |
| 46 | 107.46 | 297.63 |
| 47 | 108.80 | 297.84 |
| 48 | 111.92 | 311.41 |
| 49 | 113.80 | 312.43 |
| 50 | 116.33 | 317.50 |
| 51 | 115.66 | 323.45 |
| 52 | 119.56 | 331.58 |
| 53 | 122.03 | 335.34 |
| 54 | 123.19 | 341.53 |
| 55 | 123.61 | 343.37 |
| 56 | 131.92 | 347.80 |
| 57 | 127.05 | 359.20 |
| 58 | 130.45 | 359.33 |
| 59 | 131.52 | 363.50 |
| 60 | 133.75 | 369.48 |
| 61 | 136.57 | 372.53 |
| 62 | 137.58 | 383.22 |
| 63 | 137.38 | 391.12 |
| 64 | 143.71 | 396.78 |

> 1000 iterations per data point, 50 warmup iterations.

## Why njit is faster

There are two independent sources of savings, matching the two model
parameters *k* and *b*.

### Per-kernel cost: 1.93 vs 5.43 µs/kernel

When `torch.compile` runs the inductor-generated Python wrapper, each
Triton kernel launch crosses the Python/C boundary multiple times:

1. Python grid lambda is evaluated (`lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`)
2. `CachingAutotuner.__call__` is invoked — a Python method that looks up
   the best config and packages arguments
3. The Python-side launcher calls into the Triton C extension to fire
   `cuLaunchKernelEx`

Inside a compiled **njit** function, this entire chain is replaced by
LLVM-compiled machine code.  The grid is a compile-time integer constant
(computed once during `NjitInductorGraph.__init__`), and each kernel fires
through a lightweight C trampoline (`_generate_launch_trampoline_src`) that
calls `cuLaunchKernelEx` directly — no Python frames, no argument-parsing,
no autotuner lookup.  The ~3.5 µs/kernel savings (5.43 → 1.93 µs/kernel)
is the cost of Python's per-launch interpreter overhead.

### Fixed overhead: 18.87 vs 46.55 µs

Every `torch.compile` call pays a fixed Python cost before the first
kernel even launches: the dynamo/inductor graph wrapper must check guards
(shape guards, device guards, etc.) in Python, unpack the argument tuple,
and resolve the cached compiled artifact.  This accounts for the ~46.6 µs
baseline.

The njit wrapper's ~18.9 µs baseline comes from the Numba dispatcher
(one C-level function call) plus tensor unboxing — extracting the
`TensorImpl*` from each PyTorch tensor argument so it can be passed as
a raw pointer into compiled code.

### Break-even

The njit wrapper wins immediately: even at 1 kernel (14.7 µs vs 37.4 µs)
the lower fixed cost more than compensates for any overhead.  The gap
widens linearly with graph size at ~3.5 µs per additional kernel.

## Benchmark environment

| Component | Details |
|-----------|---------|
| CPU | aarch64 |
| GPU | NVIDIA GB200 |
| CUDA | 13.2 |
| Driver | 580.65.06 |
| Python | 3.12.3 |
| PyTorch | 2.11.0a0+a6c236b9fd.nvinternal.main.46298116 |
| Numba | 0.64.0 |
| Triton | 3.6.0 |
| OS | Linux-6.11.0-1011-nvidia-64k-aarch64-with-glibc2.39 |

## Running

```bash
PYTHONPATH=src python benchmarks/inductor-vs-njit/run.py
```
