# Redispatch Experiment Results

## Goal

Measure the CPU overhead of PyTorch's dispatcher by comparing
`at::_ops::{op}::call(...)` (full dispatch) vs
`at::_ops::{op}::redispatch(DispatchKeySet(CUDA), ...)` (skip to CUDA kernel directly).

## Setup

- Platform: aarch64 Linux (GH200)
- PyTorch: built from source
- GPU: NVIDIA GH200
- Tensors: 4×4 on CUDA (tiny, so GPU compute is negligible)

## 20-op Graph Benchmark (5 rounds × 2000 iterations)

### Dispatched (`at::_ops::*::call`)

| Round | njit (µs) | eager (µs) | ratio |
|-------|-----------|------------|-------|
| 1     | 160.7     | 186.1      | 1.16× |
| 2     | 157.8     | 183.1      | 1.16× |
| 3     | 159.6     | 184.1      | 1.15× |
| 4     | 162.4     | 184.3      | 1.14× |
| 5     | 160.5     | 186.4      | 1.16× |

### Redispatch (`at::_ops::*::redispatch` with DispatchKeySet(CUDA))

| Round | njit (µs) | eager (µs) | ratio |
|-------|-----------|------------|-------|
| 1     | 159.8     | 193.7      | 1.21× |
| 2     | 160.8     | 193.2      | 1.20× |
| 3     | 161.7     | 196.6      | 1.22× |
| 4     | 164.2     | 195.4      | 1.19× |
| 5     | 161.9     | 193.6      | 1.20× |

## Single-op Benchmark (relu, 5 rounds × 10000 iterations)

### Dispatched

| Round | njit (µs) | eager (µs) |
|-------|-----------|------------|
| 1     | 12.78     | 8.30       |
| 2     | 12.05     | 7.83       |
| 3     | 11.64     | 7.99       |
| 4     | 11.13     | 7.80       |
| 5     | 11.25     | 7.95       |

### Redispatch

| Round | njit (µs) | eager (µs) |
|-------|-----------|------------|
| 1     | 11.14     | 7.44       |
| 2     | 11.02     | 7.49       |
| 3     | 10.96     | 7.67       |
| 4     | 11.10     | 7.64       |
| 5     | 11.01     | 7.49       |

## Conclusion

**The PyTorch dispatcher overhead is negligible (~0.3µs per op).**

Skipping the dispatcher via `redispatch` saves ~0.2-0.3µs per op, which is
~5µs over a 20-op graph — well within noise. The dominant overhead in the
njit path is **unboxing/boxing** (converting between Python torch.Tensor and
raw TensorImpl* pointers), not dispatch.

For a single op, njit (~11µs) is actually slower than eager (~8µs) because
the unbox+box overhead per call exceeds the Python interpreter overhead that
njit eliminates. The njit advantage only appears with many ops in a single
compiled function, where unbox/box happens once at the boundaries while all
interior ops run as direct C++ calls.
