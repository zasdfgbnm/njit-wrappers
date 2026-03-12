# Host Overhead Reduction Experiments — Full Results

## Setup

- Platform: aarch64 Linux (GB200)
- PyTorch: built from source
- GPU: NVIDIA GB200
- Tensors: 4×4 on CUDA (tiny, so GPU compute is negligible)
- Iterations: 5000, Warmup: 500

## Approaches Tested

| # | Approach | What it does | Symbol used |
|---|----------|-------------|-------------|
| 1 | **eager** | Standard PyTorch eager | `torch.clamp(x, min=0)` (relu equivalent) |
| 2 | **njit ::call** | Full dispatcher, original behavior | `at::_ops::relu::call(const Tensor&)` |
| 3 | **njit ::redispatch** | Skip autograd/autocast | `at::_ops::relu::redispatch(DispatchKeySet(CUDA), const Tensor&)` |
| 4 | **njit at::cuda::relu** | Bypass dispatcher entirely | `at::cuda::relu(const Tensor&)` from libtorch_cuda.so |
| 5 | **torch.compile** | PyTorch inductor JIT | Fuses entire graph into one CUDA kernel |
| 6 | **C++ wrapper (int64 ABI)** | Eliminate sret | `njit_relu_wrapper(int64) -> int64` in _bridge.cpp |

## Results (averaged over 2 runs)

### Raw timings (µs per call)

| Approach | 1 op | 5 ops | 10 ops | 20 ops |
|----------|------|-------|--------|--------|
| eager | 9.3 | 46.4 | 90.5 | 181.6 |
| njit ::call (full dispatch) | 17.2 | 36.0 | 60.4 | 102.7 |
| njit ::redispatch (skip autograd) | 12.7 | 36.0 | 56.9 | 100.8 |
| njit at::cuda::relu (direct) | 11.1 | 35.4 | 60.9 | 105.5 |
| torch.compile (inductor) | 26.2 | — | — | 25.9 |
| C++ wrapper (int64 ABI) | CRASHED (memory leak amplification) |

### Derived metrics

| Approach | Fixed overhead (µs) | Per-op marginal cost (µs) |
|----------|--------------------|-----------------------------|
| eager | ~0.3 | **~9.1** |
| njit ::call | ~12.7 | **~4.5** |
| njit ::redispatch | ~8.1 | **~4.6** |
| njit at::cuda::relu | ~6.1 | **~5.0** |
| torch.compile | ~26.3 | **~0.0** (fused kernel) |

## Key Findings

### 1. Per-op cost is the same regardless of dispatch method

All three njit dispatch strategies have essentially the same per-op marginal
cost: **4.5-5.0 µs/op**. The theoretical ~0.8µs dispatcher overhead (measured
in pure C++) is lost in noise at the LLVM→C++ call boundary.

| Strategy | Per-op cost (µs) | Theoretical savings vs ::call |
|----------|------------------|-----------------------------|
| ::call (full dispatch) | 4.5 | baseline |
| ::redispatch (skip autograd) | 4.6 | ~0µs (noise) |
| at::cuda::relu (no dispatcher) | 5.0 | ~0µs (noise) |

### 2. Fixed overhead differs between approaches

| Strategy | Fixed overhead (µs) | Why |
|----------|--------------------|----|
| eager | ~0.3 | Python function call overhead only |
| njit ::call | ~12.7 | Unbox+box + numba entry, varies with JIT state |
| njit ::redispatch | ~8.1 | Same unbox+box, slightly less JIT overhead |
| njit at::cuda::relu | ~6.1 | Same unbox+box, simplest call convention |
| torch.compile | ~26.3 | Triton cache lookup, guard checks |

### 3. torch.compile is unbeatable for long chains

torch.compile generates a single fused CUDA kernel, achieving ~0µs per
additional op. But it has ~26µs of fixed overhead per call.

### 4. Crossover analysis

| Comparison | Crossover point |
|-----------|----------------|
| njit vs eager | ~2 ops |
| torch.compile vs eager | ~3 ops |
| torch.compile vs njit | ~4-5 ops |

### 5. C++ wrapper approach failed

The C++ wrapper that eliminates sret by taking/returning int64 causes heap
corruption after many iterations. Root cause: each intermediate tensor created
by `tensor_from_impl()` + `impl_from_tensor()` inside the wrapper creates an
extra owned reference that is never released (the same "known limitation" of
intermediate tensor leaks, but amplified). After ~100-200 iterations of a
20-op chain, the accumulated leaked TensorImpls corrupt the CUDA allocator.

### 6. Where the per-op 4.8µs goes

The per-op cost breakdown (estimated):
- CUDA kernel launch (async): ~2-3µs
- Output tensor allocation (TensorImpl + storage): ~1-2µs
- Refcount manipulation: ~0.1µs
- Dispatcher (when present): ~0.3µs
- Stack slot + bitcast (LLVM): ~0.05µs

The ATen op itself (allocation + launch) dominates. The dispatcher is a
small fraction.

## Additional Approaches

### CUDA Graphs

CUDA Graphs capture the kernel launch sequence and replay it with minimal CPU
overhead. This eliminates per-op dispatch, tensor allocation, and refcounting
on replay.

| Ops | CUDA Graph (µs) | njit redispatch (µs) | eager (µs) |
|-----|-----------------|---------------------|------------|
| 1   | 3.28            | 12.7                | 9.3        |
| 20  | 18.13           | 100.8               | 181.6      |

**Derived: 3.28µs fixed + 0.78µs per op.**

This is ~6x faster per-op than njit and ~12x faster than eager. The per-op
cost of 0.78µs is essentially just the CUDA graph replay overhead per node.

### torch.ops.aten.relu.default (Python-level dispatcher access)

Calling `torch.ops.aten.relu.default(tensor)` from Python has ~12µs overhead
for a single call and ~9.4µs per op — comparable to or slightly worse than
regular eager `torch.relu()`.

### torch.jit.script

TorchScript JIT compilation of 20-relu chain is extremely slow (>5 minutes),
making it impractical for benchmarking. For a single relu, it costs ~14µs.

## Complete Performance Hierarchy (20 ops, CUDA 4x4)

| Rank | Approach | 20-op time (µs) | Per-op (µs) | Fixed (µs) |
|------|----------|-----------------|-------------|------------|
| 1 | CUDA Graph | 18.1 | 0.78 | 3.3 |
| 2 | torch.compile | 25.9 | ~0 | 25.9 |
| 3 | njit (any dispatch) | 100-108 | 4.5-5.0 | 6-13 |
| 4 | eager | 181.6 | 9.1 | 0.3 |



### Pre-allocated output tensors
Would eliminate per-op allocation overhead (~1-2µs savings), but requires
fundamental changes to the ATen calling convention (use `_out` variants
which take a pre-allocated output tensor). Not compatible with the current
sret-based approach.

### Batch unbox
Would reduce fixed overhead by unboxing multiple tensors in a single C call.
Savings: ~0.3µs per additional tensor input. Negligible for most functions
(1-2 tensor inputs).

### Direct offset read (borrow TensorImpl*)
Read TensorImpl* at offset 16 from PyObject without incref. Savings: ~0.6µs
total for unbox+cleanup. Would reduce fixed overhead from ~8µs to ~7.4µs.
Requires guaranteeing Python object lifetime, which holds for synchronous
njit functions.

### Kernel fusion (numba.cuda)
Generate a single fused CUDA kernel for element-wise op chains. Would match
torch.compile performance (~0µs per additional op) with much lower fixed
overhead. Requires a fundamentally different approach — not just a dispatch
optimization.

## Conclusions

1. **The dispatcher is NOT the bottleneck.** Skipping it saves <0.5µs per op,
   unmeasurable in practice.

2. **The per-op cost (~4.8µs) is dominated by ATen op execution** (tensor
   allocation + CUDA kernel launch). No dispatch-level optimization can
   reduce this.

3. **The fixed overhead (~8µs) is dominated by unbox/box**. Small improvements
   are possible (borrow optimization, batch unbox) but the impact is limited.

4. **CUDA Graphs are the fastest approach** (0.78µs per op, 3.3µs fixed).
   They eliminate per-op dispatch, allocation, and refcounting by replaying
   a captured kernel launch sequence. However, they require static tensor
   shapes and sizes.

5. **torch.compile is fastest for fused computation** (~0µs per additional op)
   but has high fixed overhead (26µs). Best for long chains of fusible ops.

6. **For truly low overhead with flexibility, a hybrid approach is needed:**
   numba's low fixed cost (~8µs) + CUDA graph's low per-op cost (~0.78µs).
   This could be achieved by having numba capture a CUDA graph on first call,
   then replay on subsequent calls.
