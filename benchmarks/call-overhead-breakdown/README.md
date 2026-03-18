# Overhead breakdown of Inductor's generated `call()` function

Inductor's generated `call(args)` runs on every forward pass and orchestrates
GPU kernels from Python.  This benchmark measures each category of CPython
interpreter overhead *in isolation* to quantify how much each component
contributes to the total per-kernel dispatch cost.

## Components

| Component | What it represents |
|-----------|-------------------|
| **Frame allocation** | Cost of entering and leaving a plain Python function — the baseline interpreter tax on every `call()` invocation. |
| **Buffer allocation** | One `torch.empty_strided(shape, stride, device='cuda', dtype=...)` per intermediate tensor.  Constructs Python tuples for shape/stride, resolves keyword arguments, and crosses the Python/C++ boundary into ATen. |
| **Grid computation** | One `grid(n)` call per kernel — a Python function that computes `ceil(n / BLOCK_SIZE)` and boxes the result as a Python integer. |
| **Triton launcher** | One `kernel[(grid,)](args...)` per kernel — `JITFunction.__call__` inspects argument types and shapes, looks up a cached specialization, packs a `void*` argument array in Python, and crosses into a `ctypes`-wrapped C function to call `cuLaunchKernelEx`.  This single call traverses four to six Python frames. |

## Methodology

Each component is measured with 500 warmup iterations followed by 5 000
timed iterations; the **median** per-call time is reported.

- **Frame allocation** and **grid computation** are measured by timing a
  no-op function and the `grid(n)` helper respectively — both CPU-only.
- **Buffer allocation** is measured by timing
  `torch.empty_strided((32, 64), (64, 1), device='cpu', dtype=torch.float32)`.
  The CPU-device call exercises the same Python-level path as the CUDA call
  (tuple construction, kwarg resolution, C++/Python boundary crossing); the
  actual allocation is handled by the caching allocator in both cases and is
  not the bottleneck.
- **Triton launcher** cannot be measured in isolation without a CUDA GPU.
  Its cost is therefore inferred: `t_launcher = k_compile − t_frame − t_grid − t_alloc`,
  where `k_compile = 5.4 µs/kernel` is the per-kernel slope measured in the
  [inductor-vs-njit benchmark](../inductor-vs-njit/README.md).

The reference total (`k_compile = 5.4 µs/kernel`) comes from fitting a linear
model to the inductor-vs-njit wall-clock data (64 data points, 1 000
iterations each, 2σ outlier removal).

## Results

| Component | Time (µs) | % of total |
|-----------|-----------|-----------|
| Frame allocation | 0.087 | 1.6% |
| Grid computation | 0.089 | 1.6% |
| Buffer allocation (`empty_strided`) | 1.639 | 30.4% |
| Triton launcher (inferred) | 3.585 | 66.4% |
| **Total** | **5.400** | **100%** |

Reference: per-kernel slope of `torch.compile` (inductor-vs-njit) = **5.4 µs/kernel**;
per-kernel slope of `@njit` wrapper = **1.9 µs/kernel**.

Applying `@numba.njit` to `call()` eliminates all four components above — there
is no Python stack frame, no Python-level tuple construction, no boxed-integer
grid result, and no `JITFunction.__call__` path.  The remaining 1.9 µs
represents the Numba dispatcher entry and LLVM-generated machine-code overhead
for the compiled body.  The overall reduction is **1.9 / 5.4 = 65%** of
per-kernel Python dispatch cost.

## Benchmark environment

Frame and grid measurements were taken on the same machine that ran the
inductor-vs-njit benchmark.  Environment for the reference total:

| Component | Details |
|-----------|---------|
| CPU | aarch64 |
| GPU | NVIDIA GB200 |
| CUDA | 13.2 |
| Driver | 580.65.06 |
| Python | 3.12.3 |
| PyTorch | 2.11.0a0+a6c236b9fd.nvinternal.main.45821058 |
| Numba | 0.64.0 |
| Triton | (bundled with PyTorch) |
| OS | Linux-6.14.0-1007-nvidia-64k-aarch64-with-glibc2.39 |

## Running

To regenerate results on a CUDA-capable machine:

```bash
PYTHONPATH=src python benchmarks/call-overhead-breakdown/run.py
```
