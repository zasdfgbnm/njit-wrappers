# Eager vs njit: CPU Dispatch Overhead

Measures wall-clock time of launching GPU ops through `numba.njit` vs plain
eager PyTorch.  All tensors are tiny (4×4) so GPU compute is negligible — only
the CPU dispatch overhead is measured.  No `cudaDeviceSynchronize` is called.

Each graph is simply `for i in range(N): x = torch.relu(x)`.

## Results

![overhead_vs_ops](overhead_vs_ops.png)

### Linear fit (outliers removed, 2σ threshold)

|       | model     | k (µs/op) | b (µs)  |
|-------|-----------|-----------|---------|
| njit  | y = kx+b  | 5.6901    | 6.3319  |
| eager | y = kx    | 8.7210    | 0 (forced) |

- **k** (slope) is the **per-op cost** — the marginal time (in µs) added by
  each additional `torch.relu` call.  A smaller *k* means each op dispatches
  faster.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs) for
  entering and leaving the function, independent of how many ops it contains.
  For njit, this captures the Numba dispatcher + tensor borrow/wrap cost.
  For eager, *b* is forced to 0 because plain Python function calls have no
  meaningful fixed overhead beyond the per-op cost itself.

### Raw data

| Ops | njit (µs) | eager (µs) |
|-----|-----------|------------|
| 1 | 14.67 | 9.79 |
| 2 | 22.86 | 18.45 |
| 3 | 24.33 | 23.99 |
| 4 | 31.45 | 33.53 |
| 5 | 39.06 | 42.14 |
| 6 | 41.78 | 49.54 |
| 7 | 50.25 | 61.80 |
| 8 | 52.96 | 71.35 |
| 9 | 59.50 | 78.99 |
| 10 | 65.80 | 83.08 |
| 11 | 69.86 | 94.41 |
| 12 | 76.96 | 101.46 |
| 13 | 79.63 | 109.89 |
| 14 | 89.91 | 121.76 |
| 15 | 91.33 | 133.17 |
| 16 | 101.65 | 142.89 |
| 17 | 106.24 | 142.96 |
| 18 | 102.52 | 148.77 |
| 19 | 110.18 | 162.82 |
| 20 | 121.89 | 176.81 |
| 21 | 118.16 | 173.75 |
| 22 | 133.29 | 196.20 |
| 23 | 141.65 | 196.27 |
| 24 | 138.39 | 199.93 |
| 25 | 141.76 | 214.21 |
| 26 | 145.34 | 221.74 |
| 27 | 175.08 | 244.73 |
| 28 | 160.66 | 243.43 |
| 29 | 171.02 | 250.14 |
| 30 | 167.53 | 268.93 |
| 31 | 180.97 | 276.52 |
| 32 | 190.69 | 287.50 |
| 33 | 201.30 | 272.56 |
| 34 | 189.26 | 307.43 |
| 35 | 200.62 | 287.98 |
| 36 | 217.58 | 331.21 |
| 37 | 210.85 | 324.46 |
| 38 | 220.47 | 324.19 |
| 39 | 229.76 | 378.40 |
| 40 | 255.63 | 370.14 |
| 41 | 228.31 | 347.07 |
| 42 | 252.05 | 361.98 |
| 43 | 254.19 | 388.77 |
| 44 | 263.94 | 404.76 |
| 45 | 252.01 | 380.08 |
| 46 | 277.63 | 384.91 |
| 47 | 270.54 | 398.51 |
| 48 | 272.10 | 443.89 |
| 49 | 282.17 | 433.79 |
| 50 | 276.27 | 454.23 |
| 51 | 287.94 | 444.03 |
| 52 | 295.59 | 426.96 |
| 53 | 304.83 | 447.94 |
| 54 | 310.15 | 468.82 |
| 55 | 321.13 | 482.63 |
| 56 | 326.63 | 506.19 |
| 57 | 330.76 | 502.83 |
| 58 | 340.65 | 511.76 |
| 59 | 352.97 | 530.31 |
| 60 | 354.73 | 515.22 |
| 61 | 360.08 | 526.47 |
| 62 | 357.95 | 530.05 |
| 63 | 376.81 | 532.56 |
| 64 | 391.75 | 524.21 |

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
