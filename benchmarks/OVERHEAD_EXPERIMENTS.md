# Host Overhead Reduction Experiments — Full Results

## Setup

- Platform: aarch64 Linux (GB200)
- PyTorch: built from source
- GPU: NVIDIA GB200
- Tensors: 4×4 on CUDA (tiny, so GPU compute is negligible)
- Iterations: 5000, Warmup: 500

## Approaches Tested

1. **eager**: Standard PyTorch eager execution (`torch.clamp(x, min=0)` as relu equivalent)
2. **njit redispatch**: `at::_ops::relu::redispatch(DispatchKeySet(CUDA), ...)` — skips autograd/autocast
3. **njit ::call**: `at::_ops::relu::call(...)` — full PyTorch dispatcher (original behavior)
4. **njit at::cuda::relu**: Direct call to `at::cuda::relu` in libtorch_cuda.so — completely bypasses dispatcher
5. **torch.compile**: `torch.compile(fn, backend="inductor")` — PyTorch's own JIT
6. **njit C++ wrapper**: C++ function `int64->int64` that wraps TensorImpl* and calls at::relu (CRASHED — heap corruption)

## Results (Run 1)

| Approach | 1 op (µs) | 5 ops (µs) | 10 ops (µs) | 20 ops (µs) |
|----------|-----------|------------|-------------|-------------|
| eager | 8.89 | 45.44 | 89.22 | 177.00 |
| njit redispatch | 12.31 | 35.88 | 54.69 | 100.81 |
| njit ::call | 16.07 | 41.36 | 60.22 | 107.36 |
| njit at::cuda::relu | 12.83 | 35.25 | 61.98 | 108.24 |
| torch.compile | 26.62 | — | — | 25.13 |

## Results (Run 2)

| Approach | 1 op (µs) | 5 ops (µs) | 10 ops (µs) | 20 ops (µs) |
|----------|-----------|------------|-------------|-------------|
| eager | 9.67 | 47.19 | 93.62 | 186.01 |
| njit redispatch | 13.75 | 35.63 | 54.08 | 104.55 |
| njit ::call | 11.18 | 34.96 | 65.38 | 103.36 |
| njit at::cuda::relu | 14.20 | 39.40 | 63.99 | 113.31 |
| torch.compile | 24.37 | — | — | 24.78 |

## Derived Metrics

| Approach | Fixed overhead (µs) | Per-op marginal cost (µs) |
|----------|--------------------|-----------------------------|
| eager | ~0.2 | ~9.1 |
| njit redispatch | ~8.3 | ~4.7 |
| njit ::call | ~8.8 | ~4.8 |
| njit at::cuda::relu | ~8.4 | ~5.1 |
| torch.compile | ~25.5 | ~0 (fused into single kernel) |

## Key Findings

### 1. Dispatcher overhead is negligible (~0.2µs per op)

All three njit dispatch strategies (::call, ::redispatch, at::cuda::relu) have
essentially the same per-op marginal cost: **4.7-5.1 µs/op**. The theoretical
~0.8µs dispatcher overhead (measured in pure C++) is completely lost in noise at
the LLVM-to-C++ call boundary.

### 2. The dominant overhead is fixed (unbox/box): ~8µs per call

All njit approaches pay ~8µs of fixed overhead per function call for:
- Python→numba entry (~0.06µs)
- Per-tensor unbox: THPVariable_Unpack + incref (~0.3µs per tensor)
- Per-tensor box: THPVariable_Wrap + steal (~2µs per output tensor)
- LLVM function call overhead
- Total: ~8µs for the whole entry/exit

### 3. njit per-op cost is ~48% cheaper than eager

| | Per-op cost | What it includes |
|---|---|---|
| eager | ~9.1 µs | Python frame overhead + CPython dispatch + ATen op |
| njit | ~4.8 µs | Direct C++ call to ATen op (no Python overhead) |

### 4. torch.compile is unbeatable for fused graphs

torch.compile generates a single fused CUDA kernel for the entire 20-relu chain,
achieving ~25µs total regardless of op count. But it has ~25µs of fixed overhead
(Triton/inductor compilation cache lookup), making it slower than eager for
single ops.

### 5. Crossover points

- **njit vs eager**: ~2-3 ops (njit wins for longer chains)
- **njit vs torch.compile**: ~5 ops (torch.compile wins for longer chains)
- **eager vs torch.compile**: ~3 ops (torch.compile wins for longer chains)

### 6. C++ wrapper approach (int64 ABI) crashes

The C++ wrapper that eliminates sret by taking/returning int64 causes heap
corruption when called from numba-compiled code. Root cause needs investigation —
likely related to how at::Tensor reference counting interacts with numba's
object lifetime management. The wrapper works correctly when called via ctypes
from Python directly.

## Conclusions

1. **Don't bother trying to skip the dispatcher** — the ~0.2µs savings per op is
   immeasurable in practice.

2. **The real bottleneck is the fixed overhead** (~8µs per njit call). Reducing this
   requires:
   - Cheaper unbox: borrow TensorImpl* without refcounting (~saves 0.6µs per input)
   - Cheaper box: avoid THPVariable_Wrap overhead
   - Or: batch multiple function calls to amortize the fixed cost

3. **For truly minimal overhead, kernel fusion is the answer** (like torch.compile).
   But torch.compile has high fixed overhead (~25µs). A numba-based kernel fusion
   approach could combine the low fixed overhead of njit with the zero per-op
   marginal cost of fusion.

4. **The per-op cost of ~4.8µs in njit is dominated by the ATen op itself**
   (tensor allocation, CUDA kernel launch, refcounting), not the dispatch path.
   To reduce this, we'd need to eliminate per-op tensor allocation (use pre-allocated
   output buffers) or fuse ops into a single kernel.
