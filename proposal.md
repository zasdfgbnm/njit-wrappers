# Reducing Host Latency with `numba.njit` in TorchInductor

**Author:** zasdfgbnm
**Status:** RFC / Proposal
**Target:** PyTorch core / TorchInductor

---

## TL;DR

We present a prototype demonstrating that wrapping TorchInductor's generated orchestration code
with `numba.njit` reduces host-side dispatch latency by **2–5× without changing a single kernel**.
All computation continues to be performed by Triton and ATen kernels — only the Python
orchestration layer is replaced by Numba-compiled native code.

The prototype lives at [zasdfgbnm/njit-wrappers](https://github.com/zasdfgbnm/njit-wrappers)
and covers three benchmark scenarios:

**Benchmark 1 — Eager ATen ops ([source](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/eager-vs-njit)):**
`torch.relu` chain on 4×4 tensors, NVIDIA GB200.

![Eager vs njit overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/eager-vs-njit/overhead_vs_ops.png)

Per-op dispatch cost: **8.72 µs (eager) → 5.69 µs (njit), 35% reduction.**

**Benchmark 2 — Triton kernel launch ([source](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/triton-vs-njit)):**
Element-wise add kernel, 1024 elements, NVIDIA A100-SXM4-80GB.

![Triton vs njit kernel launch overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/triton-vs-njit/overhead_vs_kernels.png)

Per-launch cost: **13.98 µs (Python) → 2.94 µs (njit), 4.8× reduction.**

**Benchmark 3 — End-to-end inductor graph ([source](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/inductor-vs-njit)):**
`torch.softmax` chain on 32×64 tensors, NVIDIA GB200.

![Inductor vs njit orchestration overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/inductor-vs-njit/overhead_vs_kernels.png)

| Metric | torch.compile (inductor) | numba.njit orchestration | Speedup |
|---|---|---|---|
| Per-kernel dispatch cost | 5.43 µs/kernel | 1.93 µs/kernel | **2.8×** |
| Fixed call overhead | 46.6 µs | 18.9 µs | **2.5×** |
| 64-kernel graph wall time | 396.8 µs | 143.7 µs | **2.8×** |

The proposed integration surface is a single new flag:

```python
torch.compile(model, enable_numba=True)
```

This leaves all existing compilation paths untouched and adds Numba as an optional dependency.

---

## Motivation

### Host latency is a first-class bottleneck in modern LLM inference

As GPU hardware has become increasingly powerful, the CPU-side orchestration of kernel launches
has emerged as a significant — and growing — bottleneck in large language model inference.

**Decoding is fundamentally serial.** Each autoregressive step requires a full forward pass
before the next token can begin. For a model that takes 400 µs of wall time per step but spends
50 µs of that in Python dispatching kernels, eliminating that overhead is a 14% latency reduction
with zero algorithmic change.

**Speculative decoding amplifies the problem.** Speculative decoding pairs a small draft model
with a large target model, running the draft model in a tight loop to propose multiple tokens per
step. The MLCEngine team reports that "LLM engine overhead reduction becomes *extremely* important
in speculative decoding scenarios, as the draft model runs in a tight loop and can take a strong
hit from engine overhead." [1] Host overhead that is tolerable in single-model inference becomes
the critical path when a second model runs many times per target-model call.

**The existing mitigation — CUDA Graphs — has significant limitations.** PyTorch's
`reduce-overhead` mode and `max-autotune` both use CUDA Graphs to bypass per-kernel Python
dispatch. CUDA Graphs can deliver up to 10% latency reduction in multi-GPU inference [1], but
they impose hard constraints: static shapes, no dynamic control flow, no CPU ops in the captured
graph, and no graph breaks. In practice, real models with dynamic batch sizes, conditional logic,
or custom ops frequently cannot be captured into a single CUDA Graph [2][3]. When a graph break
occurs, the entire mechanism degrades silently to the slow Python path.

**Inference-time scaling makes latency more, not less, critical.** The emerging paradigm of
inference-time compute scaling — chain-of-thought, tree search, multi-step reasoning — multiplies
the number of forward passes per user request. Latency that was acceptable for a single pass
becomes unacceptable when the runtime is chain-of-thought steps × per-step latency [4].

In short: host-side dispatch latency is not an edge case. It is on the critical path for the
inference workloads that matter most today, and it will become more important as models grow more
complex.

---

## The Proposal and Feasibility Study

### Core idea

Replace TorchInductor's Python-level graph runner with a `numba.njit`-compiled function. The
runner is the code that allocates scratch buffers, launches Triton kernels, calls ATen extern
kernels, and assembles outputs. Today it is ordinary Python executed by CPython. We propose to
JIT-compile it with Numba's LLVM-based compiler, eliminating interpreter overhead on every call.

**What this is NOT:** We are not using Numba to generate GPU kernels. All GPU computation
continues to be produced by Triton (for element-wise and reduction ops) and ATen/cuBLAS (for
extern kernels such as GEMM). Numba is used purely for the *orchestration* layer — the host-side
loop that drives the GPU.

### Why `numba.njit` rather than, say, C++?

1. **Pythonic.** The orchestration code stays in Python. It is editable, debuggable, and
   inspectable with standard Python tooling. There is no C++ compilation step at runtime, no
   `ctypes` interop boilerplate, and no ABI concerns.

2. **Lightweight.** The core implementation requires adding `torch.Tensor` and Triton kernel
   objects as first-class types in Numba's type system. This is a well-defined extension point
   in Numba's architecture. It does not require a new compiler, a new IR, or changes to Triton
   internals.

3. **Composable with the existing stack.** Numba's `@njit` compiles via LLVM and can call
   arbitrary C symbols. ATen operators are already exposed as stable C++ symbols in
   `libtorch_cpu.so` and `libtorch_cuda.so`. Triton kernels are compiled to PTX/CUBIN and
   launched via `cuLaunchKernelEx`. Both are straightforwardly callable from LLVM IR without
   any Python interpreter involvement.

### Prototype: the `njit-wrappers` repository

To validate feasibility, we built a standalone prototype package —
[zasdfgbnm/njit-wrappers](https://github.com/zasdfgbnm/njit-wrappers) — that implements three
capabilities:

#### 1. `torch.Tensor` inside `@numba.njit` ([source](https://github.com/zasdfgbnm/njit-wrappers/blob/main/src/njit_wrappers/_tensor.py), [benchmark](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/eager-vs-njit))

We registered `torch.Tensor` as a native Numba type backed by a `TensorImpl*` pointer (stored
as `int64` inside compiled code). The unboxing step (Python → Numba) borrows the pointer without
touching the reference count; the Python object's lifetime guarantees the pointer remains valid
for the duration of the call. The boxing step (Numba → Python) wraps the pointer back into a
Python tensor.

ATen operators are called by resolving their Itanium-mangled C++ symbol from `libtorch_cpu.so`
at import time and emitting an LLVM intrinsic call. No runtime Python dispatch occurs. The
following calling conventions are implemented:

- **Unary:** `void(sret Tensor*, const Tensor&)` — covers `relu`, `exp`, `log`, `sigmoid`,
  `tanh`, `silu`, and ~15 more.
- **Binary:** `void(sret Tensor*, const Tensor&, const Tensor&)` — covers `+`, `-`, `*`, `/`,
  `@`, comparison ops.
- **Alpha:** `void(sret Tensor*, Tensor&, Tensor&, Scalar&)` — covers scalar-weighted add/sub.
- **Reduction:** `void(sret Tensor*, Tensor&, Optional<ScalarType>)` — covers `sum`, `mean`.

Benchmark (NVIDIA GB200, 4×4 tensors, no synchronization, 1000 iterations):

| Op count | eager (µs) | njit (µs) | Speedup |
|---|---|---|---|
| 1 | 9.79 | 14.67 | 0.67× |
| 4 | 34.90 | 29.08 | **1.20×** |
| 16 | 139.60 | 97.44 | **1.43×** |
| 64 | 524.21 | 391.75 | **1.34×** |

Linear fit: per-op cost drops from 8.72 µs (eager) to 5.69 µs (njit), a **35% reduction**.
The njit path carries a fixed overhead of ~6.3 µs (Numba dispatch), so the crossover is around
3–4 ops. For realistic inductor graphs (tens to hundreds of ops per forward pass), the njit path
is always faster.

#### 2. Triton kernel launch inside `@numba.njit` ([source](https://github.com/zasdfgbnm/njit-wrappers/blob/main/src/njit_wrappers/_triton.py), [benchmark](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/triton-vs-njit))

We implemented `NumbaTritonKernel`, which wraps a compiled Triton kernel and generates a C
trampoline that calls `cuLaunchKernelEx` directly. A runtime specialization check inspects
pointer alignment at call time; if all pointers are 16-byte aligned, it dispatches to a
pre-compiled variant that passes `tt.divisibility=16` hints, enabling 128-bit vectorized loads.

Benchmark (NVIDIA A100-SXM4-80GB, 1024-element add kernel, no synchronization, 1000 iterations):

| Launch count | eager (µs) | njit (µs) | Speedup |
|---|---|---|---|
| 1 | 14.13 | 8.54 | **1.65×** |
| 4 | 54.90 | 18.32 | **3.00×** |
| 16 | 221.53 | 53.50 | **4.14×** |
| 64 | 913.16 | 194.54 | **4.70×** |

Linear fit: per-launch cost drops from 13.98 µs (eager Python) to 2.94 µs (njit), a **4.8×
reduction**. The standard Python path through Triton's launcher involves multiple layers of
Python dispatch; `cuLaunchKernelEx` called directly from compiled code eliminates all of them.

#### 3. End-to-end inductor graph wrapping (`NjitInductorGraph`) ([source](https://github.com/zasdfgbnm/njit-wrappers/blob/main/src/njit_wrappers/_inductor.py), [benchmark](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/inductor-vs-njit))

`NjitInductorGraph` is a drop-in replacement for `torch.compile`. It runs the model through
`torch.compile(backend='inductor', fullgraph=True)`, captures the generated Python source, parses
it into an IR (buffer allocations, kernel launches, extern kernels, aliases, returns), and emits
a single `@numba.njit` function that performs the same computation without any Python interpreter
calls.

Benchmark (NVIDIA GB200, 32×64 tensors, `torch.softmax` chain, no synchronization, 1000
iterations):

| Kernel count | torch.compile (µs) | njit (µs) | Speedup |
|---|---|---|---|
| 1 | 37.43 | 14.70 | **2.55×** |
| 4 | 68.22 | 26.56 | **2.57×** |
| 16 | 133.56 | 49.71 | **2.69×** |
| 64 | 396.78 | 143.71 | **2.76×** |

Linear fit: per-kernel cost drops from 5.43 µs to 1.93 µs (**2.8×**); fixed overhead drops from
46.6 µs to 18.9 µs (**2.5×**).

### Interpretation

All three benchmarks converge on the same conclusion: the Python interpreter is spending 60–80%
of host-side dispatch time on overhead that is not intrinsic to the work being done. The overhead
is not in ATen, not in Triton, and not in CUDA — it is in CPython's per-call and per-attribute-
lookup machinery. Numba's LLVM compiler eliminates that machinery.

Unlike CUDA Graphs, this approach does not impose static-shape requirements and does not break
on dynamic control flow in the *model*. The Numba-compiled function is still a regular function
call from Python's perspective; it simply executes without interpreter overhead internally.

---

## Development Plan

We divide the work into three largely independent modules, with an explicit strategy of *close
the loop first, then expand coverage*.

### Module 1: `torch.Tensor` as a Numba type

**Goal:** Make `torch.Tensor` passable as an argument to `@numba.njit` functions, and support
calling a representative subset of torch operators from within those functions.

**Scope of initial implementation:**
- Register `TensorType` in Numba's type system.
- Implement unboxing (Python tensor → TensorImpl* pointer) and boxing (pointer → Python tensor).
- Lower one operator (e.g., `torch.relu`) to its ATen C++ symbol via LLVM intrinsic.

**Why one op first:** This is the hardest architectural step. Once the type registration,
pointer model, and LLVM intrinsic plumbing work for one op, adding the remaining ~30–40 common
ops is mechanical. We want to validate the architecture before investing in breadth.

**Expandability:** ATen exposes a stable set of C++ symbols with predictable Itanium-mangled
names. Adding a new op is a matter of registering its signature. The prototype already covers
34 ops; the final implementation should cover all ops that appear in inductor-generated graphs.

**Upstream home:** `torch/_inductor/numba_support/tensor.py` or equivalent. This module has no
dependency on Triton or the inductor code generator.

### Module 2: Triton kernel launch from `@numba.njit`

**Goal:** Allow a compiled Triton kernel to be launched from within a `@numba.njit` function
with zero Python overhead.

**Approach:** Generate a thin C trampoline per kernel signature that calls `cuLaunchKernelEx`
directly. The trampoline is compiled at `torch.compile` time (when the Triton kernel is
finalized) and its address is passed into the njit function as an integer constant.

**Triton coordination:** We plan to open a discussion with the Triton maintainers about the
cleanest integration point:
- **Preferred:** Triton exposes a supported API to retrieve the kernel's `CUfunction` handle
  and its expected calling convention, so the trampoline can be generated reliably.
- **Fallback:** If Triton prefers not to add this surface area, the trampoline can be
  generated from Triton's existing (private) kernel metadata. This is functional but fragile;
  we would maintain it off-tree until a stable API exists.

In either case, this module produces a `NumbaTritonKernel` wrapper object whose `.launch`
attribute is callable from `@numba.njit`.

**Upstream home:** If Triton accepts the integration, the C trampoline generation lives in
Triton. The Numba wrapper lives in PyTorch alongside Module 1.

### Module 3: Inductor runner wrapping

**Goal:** When `enable_numba=True`, TorchInductor's generated graph runner is a `@numba.njit`
function rather than plain Python.

**Approach:** After the existing inductor compilation pipeline produces a Python runner, a
post-processing pass:
1. Parses the runner source into a lightweight IR (buffer allocations, kernel launches, extern
   kernels, return values). This parsing is already implemented in the prototype (~500 lines).
2. Emits a `@numba.njit` function that performs the same operations using Modules 1 and 2.
3. Compiles the function (one-time cost at `torch.compile` time).

**Required refactors in Inductor:** The runner's generated Python must be structured in a way
the parser can consume reliably. We expect this to require modest, non-invasive changes to
Inductor's code generation — no new IR nodes, no changes to kernel compilation, no changes to
the Triton backend. The overall code complexity of Inductor should not increase materially.

**API surface:**

```python
# New flag, opt-in, no effect on existing code
compiled = torch.compile(model, enable_numba=True)

# Everything else is unchanged
output = compiled(input)
```

**Upstream home:** `torch/_inductor/codegen/numba_runner.py` or equivalent.

---

### Phased rollout

We believe strongly in the principle of *close the loop before expanding coverage*. The
development sequence reflects this:

#### Phase 0: Internal prototyping (current state)

The `njit-wrappers` repository demonstrates the full stack on a single benchmark model. All
three modules are implemented at prototype quality. The only correctness guarantee is that the
unit tests pass.

#### Phase 1: One op, end-to-end, in-tree

Upstream a minimal but complete version of Module 1 (one op), Module 2 (one kernel signature),
and Module 3 (the runner wrapper) into PyTorch behind a feature flag. The end-to-end compilation
path works for any model that uses only the one supported op. This establishes the architecture
and CI infrastructure.

Success criterion: `torch.compile(model, enable_numba=True)` produces correct results for a
one-op model, all existing tests pass, and the compilation path is exercised in CI.

#### Phase 2: Alpha (internal use only)

Expand op coverage in Module 1 to cover the ops most commonly generated by Inductor for
transformer models (attention, GEMM, activation functions, layer norm, etc.). Expand Module 2
to cover all Triton calling conventions Inductor generates. Fix correctness bugs as they
surface. No public announcement. The only advertised requirement is that unit tests pass.

#### Phase 3: Early adopter preview (opt-in, known sharp edges)

Announce the flag to users who are willing to accept potential bugs and are prepared to own
their own correctness validation. Document known limitations clearly. Collect real-world models
from early adopters to drive edge-case coverage. Users at this stage should expect bugs and
are responsible for verifying their own outputs.

Success criterion: The flag works correctly for the 10 most common Hugging Face transformer
models in inference mode.

#### Phase 4: General availability

Resolve all known correctness issues. Expand op coverage to match inductor's full op surface.
Handle common edge cases (dynamic shapes in the model graph, graph breaks, non-CUDA devices).
Announce to the general PyTorch community.

---

## Open Questions

1. **Numba as a dependency.** Numba is a mature, well-maintained package (MIT license, active
   development, pip-installable). We propose it as an optional dependency, imported only when
   `enable_numba=True` is passed. PyTorch already has optional dependencies (e.g., `triton` is
   optional on CPU). We believe this is the right precedent to follow.

2. **Triton integration surface.** As noted in Module 2, the cleanest implementation depends on
   Triton exposing a stable API for retrieving kernel handles. We would like to engage with
   Triton maintainers early. If a stable API is not feasible, the off-tree fallback is
   functional.

3. **Refcount management for intermediate tensors.** In the current prototype, tensors
   created inside a `@numba.njit` function and not returned to Python have their `TensorImpl`
   refcount incremented but never decremented — a memory leak for long-running programs. The
   fix (explicit free calls at the end of the njit function) is known and straightforward to
   implement, but should be addressed before Phase 3.

4. **Non-CUDA devices.** The current prototype is CUDA-only. CPU support would require
   replacing `cuLaunchKernelEx` trampolines with the equivalent C++ dispatch for Inductor's CPU
   codegen. This is left for a later phase.

---

## Summary

Host-side dispatch latency is a meaningful fraction of LLM inference time, and it is
disproportionately important for speculative decoding, inference-time scaling, and other
workloads that run many forward passes in tight loops. The existing mitigation — CUDA Graphs
— does not work for models with dynamic shapes, graph breaks, or CPU ops.

We propose a complementary approach: compile TorchInductor's orchestration code with
`numba.njit`. This requires Numba as an optional dependency, a modest set of changes to
Inductor's code generator, and no changes to kernel compilation or the Triton backend.
The prototype demonstrates 2–5× reductions in host-side dispatch latency across three
benchmark scenarios. The development plan is conservative: close the loop on one op first,
then expand coverage, with a phased rollout that keeps the flag purely opt-in until correctness
is well-established.

We welcome feedback on the approach, the integration points, and the scope of changes
required for upstreaming.

---

## References

[1] MLC Blog: *Optimizing and Characterizing High-Throughput Low-Latency LLM Inference in
MLCEngine* (2024). https://blog.mlc.ai/2024/10/10/optimizing-and-characterizing-high-throughput-low-latency-llm-inference

[2] PyTorch Docs: *CUDAGraph Trees* — dynamic shapes and re-recording overhead.
https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html

[3] F. Kong: *How CUDA Graph Works in torch.compile* (2025).
https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/

[4] S. Raschka: *The State of LLMs 2025: Progress, Progress, and Predictions*.
https://magazine.sebastianraschka.com/p/state-of-llms-2025

[5] vLLM Blog: *How Speculative Decoding Boosts vLLM Performance by up to 2.8×* (2024).
https://blog.vllm.ai/2024/10/17/spec-decode.html

[6] Snowflake Engineering: *Fastest Speculative Decoding in vLLM with Arctic Inference*.
https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/

[7] PyTorch Docs: *TorchInductor GPU Profiling*.
https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_inductor_profiling.html
