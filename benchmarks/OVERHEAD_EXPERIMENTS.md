# Host Overhead Reduction Experiments

## Goal

Systematically try every approach to reduce CPU-side overhead for njit ATen calls
on CUDA tensors. Measure each approach and compare to baseline.

## Setup

- Platform: aarch64 Linux (GH200/GB200)
- PyTorch: built from source
- GPU: NVIDIA GB200
- Tensors: 4×4 on CUDA (tiny, so GPU compute is negligible)

---

## Experiment 0: Baseline Characterization

### Scaling behavior (njit vs eager, relu chains)

| # ops | njit (µs) | eager (µs) | njit/eager |
|-------|-----------|------------|------------|
| 1     | 12.4      | 8.8        | 1.41× slower |
| 5     | 36.6      | 41.3       | 0.89× (njit wins) |
| 20    | 110.0     | 163.5      | 0.67× (njit wins) |

### Linear regression (from detailed chain measurements)

| Metric | njit (redispatch) | eager |
|--------|-------------------|-------|
| Fixed overhead per call | ~3.1 µs | ~0 µs |
| Marginal cost per op | ~6.1 µs | ~6.7 µs |
| Crossover point | ~3 ops | - |

### Unbox/box cost breakdown (via ctypes, includes Python→C overhead)

| Operation | Cost |
|-----------|------|
| `njit_extract_impl` (THPVariable_Unpack + incref) | ~305 ns |
| `njit_release_impl` (decref) | ~282 ns |
| `njit_wrap_impl` (steal + THPVariable_Wrap) | N/A (ctypes unsafe) |
| `extract + release` round-trip | ~764 ns |
| Direct memory read at offset 16 (borrow, no refcount) | ~190 ns |

### Numba call overhead

| Function | Cost |
|----------|------|
| `noop()` (no args, returns int) | 0.06 µs |
| `identity(tensor)` (just unbox+box) | ~3.1 µs |

### Dispatcher overhead (pure C++, no Python)

| Method | Cost per relu call (C++) | Overhead vs direct |
|--------|--------------------------|-------------------|
| `at::_ops::relu::call` (full dispatch) | ~5,100 ns | ~780 ns |
| `at::_ops::relu::redispatch(CUDA)` | ~4,300 ns | ~0 ns |
| `at::cuda::relu` (direct CUDA) | ~4,300 ns | baseline |

The redispatch approach **does** skip the dispatcher overhead (~780ns). This is
consistent with the ~0.3µs per-op improvement we saw in Python-level benchmarks
(the rest is amortized over measurement noise).

---

## Experiment 1: redispatch vs call (already done)

See REDISPATCH_RESULTS.md. Summary: ~0.3µs/op improvement, masked by noise in
the 20-op benchmark.

---

## Experiment 2: Direct `at::cuda::*` calls via libtorch_cuda.so

### Discovery

`libtorch_cuda.so` exports `at::cuda::{op}` functions that call the CUDA kernel
directly without going through the dispatcher at all. These have the same ABI as
`at::_ops::{op}::call` (return Tensor via sret, take const Tensor& args) but
with zero dispatch overhead.

Example: `at::cuda::relu` at `_ZN2at4cuda4reluERKNS_6TensorE`

### Approach

Load `libtorch_cuda.so`, resolve `at::cuda::*` mangled names, register with LLVM.
Same calling convention as before but routed to the CUDA-specific entry point.

### Status: TESTED BELOW

---

## Experiment 3: Borrow optimization (skip refcount on unbox)

### Discovery

`TensorImpl*` lives at a fixed offset of 16 bytes from the PyObject start:
- Bytes 0-7: `ob_refcnt`
- Bytes 8-15: `ob_type`
- Bytes 16-23: `TensorImpl*` (the intrusive_ptr raw pointer)

### Approach

Instead of calling `njit_extract_impl` (which does THPVariable_Unpack + incref),
directly load `*(int64_t*)(pyobj + 16)` in LLVM IR. This is safe because:
1. The njit function executes synchronously (caller holds GIL)
2. The Python object outlives the compiled function call
3. No cleanup needed (no decref since we never incref'd)

Expected savings: eliminate ~600ns of extract+release per input tensor.

### Status: TESTED BELOW

---

## Experiment 4: C++ wrapper calling native kernel directly

### Approach

Write a C++ wrapper in `_bridge.cpp` that:
1. Takes raw `TensorImpl*` args
2. Wraps them in `at::Tensor` (just pointer assignment, no refcount)
3. Calls `at::cuda::relu` directly
4. Returns the result `TensorImpl*`

This eliminates the LLVM→libtorch ABI complexity and can use compiler
optimizations.

### Status: TESTED BELOW

---

## Experiment 5: torch.compile comparison

### Approach

Use `torch.compile` with the inductor backend as a reference point for what
PyTorch's own JIT can achieve.

### Status: TESTED BELOW

---

## Experiment 6: Cached dispatch table lookup

### Approach

Instead of going through the dispatcher, resolve the function pointer for the
CUDA kernel once at import time by walking the dispatch table, then call it
directly.

### Status: TESTED BELOW

---

## Experiment 7: Eliminate sret (return in register)

### Approach

Write a C wrapper that calls the ATen op and returns the `TensorImpl*` as a
plain `int64_t` in a register, avoiding the sret pointer dance.

### Status: TESTED BELOW

---

## Results

(Updated as experiments complete)
