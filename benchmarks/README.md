# Benchmarks

## CPU overhead: njit vs eager PyTorch on CUDA

`bench_cpu_overhead.py` measures the CPU-side dispatch overhead of a 20-op
computation graph on tiny (4x4) CUDA tensors. No `cudaDeviceSynchronize` is
called — only wall-clock time spent on the CPU launching kernels is measured.

The graph exercises a variety of op types:

| Stage | Ops | Count |
|-------|-----|-------|
| Layer 1 | `matmul`, `add`, `relu` | 3 |
| Layer 2 | `matmul`, `add`, `sigmoid` | 3 |
| Trig | `sin`, `cos`, `mul`, `tan`, `add` | 4 |
| Nonlinear | `exp`, `abs`, `sqrt`, `sub` | 4 |
| Final | `add`, `div`, `tanh` | 3 |
| Reduce | `sum`, `mean`, `add` | 3 |
| **Total** | | **20** |

### Results

```
20-op graph on CUDA tiny tensors (4×4), 1000 iterations
  njit:    168.69 µs/call
  eager:   185.73 µs/call
  ratio:     1.10× (eager / njit)
```

### Machine configuration

| Component | Details |
|-----------|---------|
| CPU | ARM Neoverse-V2, 144 cores, aarch64 |
| GPU | 4x NVIDIA GB200 (185 GB each) |
| CUDA | 13.2 |
| Driver | 580.65.06 |
| Python | 3.12.3 |
| PyTorch | 2.11.0a0+a6c236b9fd (built from source) |
| Numba | 0.64.0 |
| OS | Linux 6.14.0-1007-nvidia-64k |

### Running

```bash
PYTHONPATH=src python benchmarks/bench_cpu_overhead.py
```
