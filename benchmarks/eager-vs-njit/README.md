# Eager vs njit: CPU Dispatch Overhead

Measures wall-clock time of launching GPU ops through `numba.njit` vs plain
eager PyTorch.  All tensors are tiny (4×4) so GPU compute is negligible — only
the CPU dispatch overhead is measured.  No `cudaDeviceSynchronize` is called.

Each graph is simply `for i in range(N): x = torch.relu(x)`.

## Results

![overhead_vs_ops](overhead_vs_ops.png)

### Linear fit: `y = k * x + b`

|       | k (µs/op) | b (µs)  |
|-------|-----------|---------|
| njit  | 5.1176    | 7.0313  |
| eager | 6.7711    | 0.2970  |

- **k** (slope) is the **per-op cost** — the marginal time (in µs) added by
  each additional `torch.relu` call.  A smaller *k* means each op dispatches
  faster.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs) for
  entering and leaving the function, independent of how many ops it contains.
  This captures things like the Python → njit transition cost or the eager
  Python function-call overhead.

### Raw data

| Ops | njit (µs) | eager (µs) |
|-----|-----------|------------|
| 1 | 14.21 | 7.59 |
| 2 | 18.94 | 14.47 |
| 3 | 24.58 | 20.57 |
| 4 | 30.18 | 27.81 |
| 5 | 31.90 | 32.37 |
| 6 | 38.56 | 39.98 |
| 7 | 44.10 | 48.21 |
| 8 | 50.07 | 54.70 |
| 9 | 52.73 | 57.19 |
| 10 | 57.87 | 70.47 |
| 11 | 61.84 | 74.96 |
| 12 | 64.83 | 79.00 |
| 13 | 69.49 | 87.95 |
| 14 | 77.16 | 93.07 |
| 15 | 82.26 | 106.80 |
| 16 | 87.04 | 106.06 |
| 17 | 92.31 | 111.12 |
| 18 | 93.20 | 117.82 |
| 19 | 97.81 | 126.29 |
| 20 | 98.65 | 128.38 |
| 21 | 109.41 | 145.16 |
| 22 | 117.56 | 154.13 |
| 23 | 120.27 | 149.35 |
| 24 | 129.97 | 158.81 |
| 25 | 135.02 | 172.62 |
| 26 | 133.42 | 186.97 |
| 27 | 139.02 | 183.81 |
| 28 | 140.00 | 176.95 |
| 29 | 153.19 | 199.79 |
| 30 | 168.32 | 205.62 |
| 31 | 158.86 | 222.18 |
| 32 | 168.02 | 221.93 |
| 33 | 175.39 | 216.80 |
| 34 | 173.03 | 235.95 |
| 35 | 185.16 | 246.00 |
| 36 | 181.21 | 243.78 |
| 37 | 191.25 | 257.40 |
| 38 | 208.40 | 239.90 |
| 39 | 205.55 | 279.05 |
| 40 | 190.76 | 268.64 |
| 41 | 211.45 | 284.45 |
| 42 | 399.09 | 268.18 |
| 43 | 296.32 | 274.86 |
| 44 | 224.32 | 300.85 |
| 45 | 227.62 | 314.80 |
| 46 | 236.74 | 304.25 |
| 47 | 234.00 | 316.02 |
| 48 | 274.55 | 343.33 |
| 49 | 247.45 | 343.80 |
| 50 | 250.23 | 323.01 |
| 51 | 267.31 | 357.18 |
| 52 | 262.64 | 368.74 |
| 53 | 257.06 | 359.67 |
| 54 | 275.85 | 347.06 |
| 55 | 274.28 | 379.86 |
| 56 | 277.71 | 391.14 |
| 57 | 312.99 | 400.44 |
| 58 | 296.02 | 418.31 |
| 59 | 277.75 | 399.31 |
| 60 | 299.06 | 383.24 |
| 61 | 310.24 | 390.44 |
| 62 | 325.84 | 424.50 |
| 63 | 318.94 | 428.93 |
| 64 | 365.73 | 410.94 |

> 1000 iterations per data point, 50 warmup iterations.

## Benchmark environment

| Component | Details |
|-----------|---------|
| CPU | aarch64 |
| GPU | NVIDIA GB200 |
| CUDA | 13.2 |
| Driver | 580.65.06 |
| Python | 3.12.3 |
| PyTorch | 2.11.0a0+a6c236b9fd.nvinternal.main.45821058 |
| Numba | 0.64.0 |
| OS | Linux-6.14.0-1007-nvidia-64k-aarch64-with-glibc2.39 |

## Running

```bash
PYTHONPATH=src python benchmarks/eager-vs-njit/run.py
```
