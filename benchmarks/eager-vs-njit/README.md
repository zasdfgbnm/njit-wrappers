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
| njit  | 5.0751    | 16.8765  |
| eager | 8.0992    | 1.3311  |

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
| 1 | 14.29 | 9.42 |
| 2 | 21.10 | 18.38 |
| 3 | 28.12 | 26.42 |
| 4 | 31.99 | 34.73 |
| 5 | 38.20 | 39.53 |
| 6 | 38.50 | 46.70 |
| 7 | 45.76 | 56.24 |
| 8 | 50.58 | 64.81 |
| 9 | 54.23 | 75.13 |
| 10 | 59.85 | 82.47 |
| 11 | 63.83 | 89.45 |
| 12 | 68.50 | 98.46 |
| 13 | 71.76 | 102.98 |
| 14 | 78.52 | 117.59 |
| 15 | 87.45 | 125.82 |
| 16 | 84.68 | 129.13 |
| 17 | 96.86 | 134.11 |
| 18 | 255.33 | 142.67 |
| 19 | 181.19 | 146.84 |
| 20 | 208.37 | 169.25 |
| 21 | 113.77 | 162.04 |
| 22 | 117.00 | 185.53 |
| 23 | 128.92 | 190.30 |
| 24 | 120.29 | 184.65 |
| 25 | 132.45 | 208.14 |
| 26 | 139.37 | 204.21 |
| 27 | 157.68 | 236.80 |
| 28 | 152.58 | 218.57 |
| 29 | 156.67 | 238.90 |
| 30 | 168.26 | 243.13 |
| 31 | 165.34 | 257.83 |
| 32 | 175.38 | 254.77 |
| 33 | 178.26 | 253.96 |
| 34 | 183.67 | 260.28 |
| 35 | 182.89 | 285.57 |
| 36 | 178.32 | 278.89 |
| 37 | 195.37 | 310.72 |
| 38 | 196.49 | 311.29 |
| 39 | 204.38 | 315.94 |
| 40 | 231.41 | 360.11 |
| 41 | 218.04 | 355.68 |
| 42 | 233.87 | 324.08 |
| 43 | 228.94 | 367.67 |
| 44 | 241.40 | 369.40 |
| 45 | 227.63 | 376.17 |
| 46 | 246.47 | 367.14 |
| 47 | 242.51 | 380.41 |
| 48 | 244.57 | 414.54 |
| 49 | 246.40 | 385.11 |
| 50 | 282.81 | 382.64 |
| 51 | 266.75 | 422.91 |
| 52 | 283.01 | 448.93 |
| 53 | 275.21 | 445.68 |
| 54 | 291.80 | 477.17 |
| 55 | 274.32 | 465.98 |
| 56 | 285.56 | 445.75 |
| 57 | 311.37 | 447.37 |
| 58 | 309.96 | 458.04 |
| 59 | 329.54 | 454.54 |
| 60 | 329.23 | 474.06 |
| 61 | 342.96 | 490.99 |
| 62 | 343.51 | 497.46 |
| 63 | 344.14 | 504.72 |
| 64 | 378.77 | 503.34 |

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
