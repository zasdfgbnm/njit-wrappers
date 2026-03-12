# Eager vs njit: CPU Dispatch Overhead

Measures wall-clock time of launching GPU ops through `numba.njit` vs plain
eager PyTorch.  All tensors are tiny (4×4) so GPU compute is negligible — only
the CPU dispatch overhead is measured.  No `cudaDeviceSynchronize` is called.

Each graph is simply `for i in range(N): x = torch.relu(x)`.

## Results

![overhead_vs_ops](overhead_vs_ops.png)

| Ops | njit (µs) | eager (µs) | ratio |
|-----|-----------|------------|-------|
| 1 | 14.44 | 9.00 | 0.62× |
| 2 | 19.94 | 16.28 | 0.82× |
| 4 | 30.79 | 29.28 | 0.95× |
| 8 | 48.34 | 62.76 | 1.30× |
| 16 | 86.69 | 128.25 | 1.48× |
| 32 | 154.59 | 256.27 | 1.66× |
| 64 | 292.30 | 492.45 | 1.68× |

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
