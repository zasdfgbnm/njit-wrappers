# Reducing Host Latency with `numba.njit` in TorchInductor

**Author:** zasdfgbnm
**Status:** RFC / Proposal
**Target:** PyTorch core / TorchInductor

---

## TL;DR

We present a prototype demonstrating that wrapping TorchInductor's generated orchestration code
with `numba.njit` reduces host-side dispatch latency by **2–5× without changing a single kernel**.
The compiled kernels, the operator semantics, and the overall graph structure are identical to
what Inductor produces today — we only `@njit` the orchestration layer that drives them.

The prototype lives at [zasdfgbnm/njit-wrappers](https://github.com/zasdfgbnm/njit-wrappers).
End-to-end benchmark ([source](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/inductor-vs-njit)):
a chain of `torch.softmax` calls where Inductor generates one Triton kernel per softmax;
host-side dispatch latency measured without `cudaDeviceSynchronize`.
The `@njit` orchestration is **2.8× faster** than `torch.compile` at this task, as shown in the following figure.

![Inductor vs njit orchestration overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/inductor-vs-njit/overhead_vs_kernels.png)

We propose making Numba an optional dependency and adding a new flag to `torch.compile`:

```python
torch.compile(model, enable_numba=True)
```

When set to `True`, TorchInductor's generated orchestration code is compiled with `numba.njit` instead of being executed as plain Python — reducing host-side dispatch latency with no changes to kernel compilation or the Triton backend. All existing compilation paths are untouched.

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
step. The [MLCEngine team reports](https://blog.mlc.ai/2024/10/10/optimizing-and-characterizing-high-throughput-low-latency-llm-inference)
that "LLM engine overhead reduction becomes *extremely* important in speculative decoding
scenarios, as the draft model runs in a tight loop and can take a strong hit from engine
overhead." Host overhead that is tolerable in single-model inference becomes the critical path
when a second model runs many times per target-model call. This is corroborated by vLLM's
[speculative decoding analysis](https://blog.vllm.ai/2024/10/17/spec-decode.html) and
Snowflake's [Arctic Inference work](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/),
both of which identify host-side coordination overhead as a key factor limiting speculative
decoding efficiency.

**The existing mitigation — CUDA Graphs — has significant limitations.** PyTorch's
`reduce-overhead` mode and `max-autotune` both use CUDA Graphs to bypass per-kernel Python
dispatch. CUDA Graphs can deliver [up to 10% latency reduction in multi-GPU inference](https://blog.mlc.ai/2024/10/10/optimizing-and-characterizing-high-throughput-low-latency-llm-inference),
but they impose hard constraints: static shapes, no dynamic control flow, no CPU ops in the
captured graph, and no graph breaks. In practice, real models with dynamic batch sizes,
conditional logic, or custom ops frequently cannot be captured into a single CUDA Graph — as
documented in the [PyTorch CUDAGraph Trees docs](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)
and analyzed in depth by [F. Kong (2025)](https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile/).
When a graph break occurs, the entire mechanism degrades silently to the slow Python path.

**Inference-time scaling makes latency more, not less, critical.** The emerging paradigm of
inference-time compute scaling — chain-of-thought, tree search, multi-step reasoning — multiplies
the number of forward passes per user request. As [Raschka (2025)](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
notes, latency that was acceptable for a single pass becomes unacceptable when the runtime is
chain-of-thought steps × per-step latency.

In short: host-side dispatch latency is not an edge case. It is on the critical path for the
inference workloads that matter most today, and it will become more important as models grow more
complex.

---

## The Proposal and Feasibility Study

### Core idea

TorchInductor's compilation pipeline (`torch/_inductor/codegen/wrapper.py`) generates a Python
source file for each compiled model. The heart of that file is a `call(args)` function — the
orchestration runner — that executes on every forward pass. A representative example:

```python
def call(args):
    primals_1, primals_2 = args
    with torch.cuda._DeviceGuard(0):
        s0 = torch.cuda.current_stream()
        buf0 = torch.empty_strided((1024,), (1,), dtype=torch.float32, device='cuda')
        triton_poi_fused_0.run(primals_1, primals_2, buf0, 1024,
                               grid=(grid(1024),), stream=s0)
        return (buf0,)
```

Every time this function is called, CPython pays a cascade of overhead costs:

- **Frame allocation.** A new Python stack frame is pushed and torn down on each call.
- **Buffer allocation via `torch.empty_strided`.** Each call constructs Python tuples for shape
  and stride, resolves keyword arguments, and crosses the Python/C++ boundary into ATen.
- **Grid computation.** `math.ceil(n / BLOCK_SIZE)` allocates a Python integer; wrapping it in
  a tuple `(gridX, gridY, gridZ)` allocates another Python object.
- **Triton's Python launcher.** `triton_poi_fused_0.run(...)` enters `JITFunction.__call__`,
  which inspects argument types and shapes, selects a specialization, builds a `void*` parameter
  array in Python, and finally crosses into a `ctypes`-wrapped C function to call
  `cuLaunchKernelEx`. That single kernel launch traverses four to six Python frames.

None of this work is intrinsic to dispatching a GPU kernel. It is interpreter bookkeeping.

We propose to replace the `call(args)` function with a `@numba.njit`-compiled equivalent.
The compiled function performs exactly the same operations — buffer allocation via
`aoti_torch_empty_strided` (a stable C ABI export from `libtorch`), grid arithmetic in compiled
integer code, and kernel launch via a thin C trampoline that calls `cuLaunchKernelEx` directly —
but without a Python interpreter in the loop. No Triton Python launcher, no `ctypes` overhead,
no frame allocation on the hot path.

**What this is NOT:** We are not using Numba to generate GPU kernels. All GPU computation
continues to be produced by Triton (for element-wise and reduction ops) and ATen/cuBLAS (for
extern kernels such as GEMM). Numba is used purely for the *orchestration* layer — the host-side
`call(args)` function that drives the GPU.

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

**Study.** Can `torch.Tensor` be made a first-class type in Numba's JIT, such that Python-level
tensor operations compile down to direct ATen C++ calls with no Python interpreter involvement?

<details>
<summary><strong>Experiment design.</strong> Tensor is stored as <code>i64</code> (TensorImpl*); each ATen op is an LLVM intrinsic that spills inputs to the stack and calls the C++ symbol directly — zero Python involvement in the compiled hot path.</summary>

The key insight is that `at::Tensor` is exactly 8 bytes — it contains
only a `TensorImpl*` — so it can be represented inside Numba's compiled code as a plain `i64`.
The Python-to-compiled-code boundary is crossed by two thin C shims (`njit_borrow_impl` and
`njit_wrap_impl`) exported from a small C++ extension. For each ATen operator, its Itanium-mangled
C++ symbol is resolved from `libtorch_cpu.so` at import time and registered with LLVM; Numba
then emits a direct `call` instruction with no Python involvement.

The full path from Python to LLVM IR is:

1. `@typeof_impl` — tells Numba that a `torch.Tensor` Python object has type `TensorType`
2. `@register_model(TensorType)` — tells Numba to represent it as `ir.IntType(64)` (the `TensorImpl*`)
3. `@unbox(TensorType)` — emits `%impl = call i64 @njit_borrow_impl(i8* %pyobj)` to cross the boundary
4. `@intrinsic` per op — emits spill-to-stack + direct ATen call + load result
5. `@box(TensorType)` — emits `%obj = call i8* @njit_wrap_impl(i64 %impl)` on the way back out

ATen's calling conventions (SysV x86-64, `sret` return) are handled by four intrinsic factories:

| Convention | Signature | Ops |
|---|---|---|
| Unary | `void(sret Tensor*, const Tensor&)` | `relu`, `exp`, `log`, `sigmoid`, `tanh`, `silu`, … |
| Binary | `void(sret Tensor*, const Tensor&, const Tensor&)` | `+`, `*`, `/`, `@`, `==`, `<`, … |
| Alpha | `void(sret Tensor*, const Tensor&, const Tensor&, const Scalar&)` | `+`, `-` (with `alpha=1`) |
| Reduction | `void(sret Tensor*, const Tensor&, i16 optional<ScalarType>)` | `sum`, `mean` |

The `sret` slot and every `const Tensor&` argument are stack-allocated `i64` slots; the alpha
case additionally builds a `c10::Scalar(1)` as a 32-byte `[4 x i64]` on the stack.

<details>
<summary>Example: two-layer MLP (Python → generated LLVM IR)</summary>

**Example.** The two-layer MLP from [docs/torch-ops.md](https://github.com/zasdfgbnm/njit-wrappers/blob/main/docs/torch-ops.md):

```python
@numba.njit
def mlp_forward(x, w1, b1, w2, b2):
    """Two-layer MLP: relu(x @ w1 + b1) @ w2 + b2."""
    h = torch.relu(x @ w1 + b1)
    return h @ w2 + b2
```

The core LLVM IR generated for this function (simplified: Numba bookkeeping omitted, faithful
to the actual ATen call patterns):

```llvm
define i8* @mlp_forward(i8* %x_obj, i8* %w1_obj, i8* %b1_obj,
                         i8* %w2_obj, i8* %b2_obj) {
entry:
  ; ── Unbox: borrow TensorImpl* without touching refcount ─────────────
  %x  = call i64 @njit_borrow_impl(i8* %x_obj)
  %w1 = call i64 @njit_borrow_impl(i8* %w1_obj)
  %b1 = call i64 @njit_borrow_impl(i8* %b1_obj)
  %w2 = call i64 @njit_borrow_impl(i8* %w2_obj)
  %b2 = call i64 @njit_borrow_impl(i8* %b2_obj)
  ; njit_borrow_impl resolves to _bridge.cpp: returns TensorImpl* as i64
  ; without touching the refcount — safe because Python keeps the objects alive

  ; ── x @ w1  (_ZN2at4_ops7matmul4callERKNS_6TensorES4_) ──────────────
  %xw1.out = alloca i64
  %x.slot  = alloca i64 ; stack slot IS the at::Tensor (sizeof == 8)
  store i64 %x,  i64* %x.slot
  %w1.slot = alloca i64
  store i64 %w1, i64* %w1.slot
  call void @_aten_matmul(i8* sret bitcast(i64* %xw1.out to i8*),
                            i8*      bitcast(i64* %x.slot  to i8*),
                            i8*      bitcast(i64* %w1.slot to i8*))
  %xw1 = load i64, i64* %xw1.out

  ; ── (x@w1) + b1  (_ZN2at4_ops10add_Tensor4callERKNS_6TensorES4_RKNS_6ScalarE) ─
  %hpre.out = alloca i64
  %xw1.slot = alloca i64
  store i64 %xw1, i64* %xw1.slot
  %b1.slot  = alloca i64
  store i64 %b1,  i64* %b1.slot
  %alpha    = alloca [4 x i64]        ; c10::Scalar(1): {value=1, pad=0, tag=HAS_i, pad=0}
  store i64 1, i64* getelementptr([4 x i64], [4 x i64]* %alpha, i32 0, i32 0)
  store i64 0, i64* getelementptr([4 x i64], [4 x i64]* %alpha, i32 0, i32 1)
  store i64 1, i64* getelementptr([4 x i64], [4 x i64]* %alpha, i32 0, i32 2)
  store i64 0, i64* getelementptr([4 x i64], [4 x i64]* %alpha, i32 0, i32 3)
  call void @_aten_add(i8* sret bitcast(i64* %hpre.out  to i8*),
                        i8*      bitcast(i64* %xw1.slot  to i8*),
                        i8*      bitcast(i64* %b1.slot   to i8*),
                        i8*      bitcast([4 x i64]* %alpha to i8*))
  %hpre = load i64, i64* %hpre.out

  ; ── relu(hpre)  (_ZN2at4_ops4relu4callERKNS_6TensorE) ────────────────
  %h.out    = alloca i64
  %hpre.slot = alloca i64
  store i64 %hpre, i64* %hpre.slot
  call void @_aten_relu(i8* sret bitcast(i64* %h.out    to i8*),
                         i8*      bitcast(i64* %hpre.slot to i8*))
  %h = load i64, i64* %h.out

  ; ── h @ w2  (same pattern as x @ w1) ────────────────────────────────
  ; ...  (identical matmul pattern, omitted for brevity)
  %hw2 = ...

  ; ── (h@w2) + b2  (same pattern as (x@w1) + b1) ──────────────────────
  ; ...
  %result = ...

  ; ── Box: wrap TensorImpl* back into a Python torch.Tensor ────────────
  %ret = call i8* @njit_wrap_impl(i64 %result)
  ret i8* %ret
}

; External declarations resolved from libtorch_cpu.so at import time:
declare void @_aten_matmul(i8* sret, i8*, i8*)
  ; → _ZN2at4_ops7matmul4callERKNS_6TensorES4_
declare void @_aten_add(i8* sret, i8*, i8*, i8*)
  ; → _ZN2at4_ops10add_Tensor4callERKNS_6TensorES4_RKNS_6ScalarE
declare void @_aten_relu(i8* sret, i8*)
  ; → _ZN2at4_ops4relu4callERKNS_6TensorE
declare i64  @njit_borrow_impl(i8*)   ; _bridge.cpp
declare i8*  @njit_wrap_impl(i64)     ; _bridge.cpp
```

</details>

</details>

<details>
<summary><strong>Results.</strong> Per-op cost drops 35% vs. eager (8.72 µs → 5.69 µs); njit is faster for ≥4 ops on NVIDIA GB200.</summary>

Benchmark on NVIDIA GB200, `torch.relu` chain on 4×4 tensors, 1000 iterations:

![Eager vs njit overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/eager-vs-njit/overhead_vs_ops.png)

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

</details>

#### 2. Triton kernel launch inside `@numba.njit` ([source](https://github.com/zasdfgbnm/njit-wrappers/blob/main/src/njit_wrappers/_triton.py), [benchmark](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/triton-vs-njit))

**Study.** Can a compiled Triton kernel be launched directly from `@numba.njit` code, bypassing
Triton's Python-based launcher entirely?

<details>
<summary><strong>Experiment design.</strong> NumbaTritonKernel compiles 2^K C trampolines that call <code>cuLaunchKernelEx</code> directly, selected at runtime by pointer/integer alignment checks — eliminating the entire Triton Python launcher path.</summary>

Triton's normal Python launcher path involves multiple layers of dispatch:
`kernel[grid](args)` → `JITFunction.__call__` → `CompiledKernel.run` → `driver.launch` →
`cuLaunchKernelEx`. Every step involves Python frame allocation, attribute lookups, and
`ctypes`/cffi overhead. The `NumbaTritonKernel` class short-circuits all of this.

The architecture has three steps:

1. **Compile.** `NumbaTritonKernel` compiles the Triton kernel for a fixed type signature via
   `triton_compile()`. Because constexprs are baked in, the kernel binary is fixed at
   `NumbaTritonKernel` construction time.

2. **Generate C trampolines.** For a kernel with K specializable arguments (pointers and
   integers), 2^K C trampolines are generated — one per alignment combination. Each trampoline
   is a plain C function that packs arguments into a `void*` array and calls `cuLaunchKernelEx`
   directly via `dlsym`. Trampolines are compiled to shared libraries with `compile_module_from_src`.

3. **Generate `@numba.njit` launcher.** A single `@numba.njit` function is generated that
   checks `arg % 16 == 0` for each specializable argument and dispatches to the matching
   trampoline. Tensor pointer arguments are extracted from `TensorImpl*` via `njit_data_ptr`
   (another direct ATen C call), so no Python is involved in the hot path.

<details>
<summary>Example: vector-add kernel (Python → generated C trampoline)</summary>

**Example.** The vector-add kernel from [docs/triton.md](https://github.com/zasdfgbnm/njit-wrappers/blob/main/docs/triton.md):

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask)
                            + tl.load(y_ptr + offs, mask=mask), mask=mask)

numba_add = NumbaTritonKernel(
    add_kernel,
    signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'n_elements': 'i32'},
    constexprs={'BLOCK_SIZE': 1024},
)
launch_add = numba_add.launch  # extract before using inside @njit

@numba.njit
def f(x, y, out, stream):
    n    = x.numel()
    grid = (n + 1023) // 1024
    launch_add(grid, 1, 1, stream, x, y, out, n)
```

The generated C trampoline for the all-aligned variant
(where `tt.divisibility=16` is hinted to Triton, enabling 128-bit vectorized loads):

```c
/* auto-generated by njit_wrappers._triton for add_kernel, variant 1111 (all aligned) */
int launch_add_kernel_1111(
    int gridX, int gridY, int gridZ,
    int num_warps, int num_ctas, int coop, int pdl, int shared_memory,
    uint64_t stream, uint64_t function,
    uint64_t arg0,   /* x_ptr   — passed as raw device pointer */
    uint64_t arg1,   /* y_ptr   */
    uint64_t arg2,   /* out_ptr */
    int32_t  arg3)   /* n_elements */
{
    if (gridX * gridY * gridZ <= 0) return 0;

    CUdeviceptr p0 = (CUdeviceptr)arg0;
    CUdeviceptr p1 = (CUdeviceptr)arg1;
    CUdeviceptr p2 = (CUdeviceptr)arg2;
    int32_t     v3 = arg3;
    CUdeviceptr scratch0 = 0, scratch1 = 0;   /* no scratch memory needed */

    void *params[6] = {&p0, &p1, &p2, &v3, &scratch0, &scratch1};

    CUlaunchConfig config;
    config.gridDimX      = gridX;
    config.gridDimY      = gridY;
    config.gridDimZ      = gridZ;
    config.blockDimX     = 32 * num_warps;   /* BLOCK_SIZE=1024 → num_warps=32 */
    config.blockDimY     = 1;
    config.blockDimZ     = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream       = (CUstream)stream;
    config.attrs         = NULL;
    config.numAttrs      = 0;

    cuLaunchKernelEx_t fn = get_launch_handle();  /* dlsym("cuLaunchKernelEx") once */
    return (int)fn(&config, (CUfunction)function, params, 0);
}
```

The 2^4 = 16 variants for this kernel's 4 specializable arguments are compiled once at
`NumbaTritonKernel` construction time. The `@numba.njit` dispatcher selects among them with
four integer comparisons per call — the only runtime cost beyond the `cuLaunchKernelEx` call
itself.

</details>

</details>

<details>
<summary><strong>Results.</strong> Per-launch cost drops 4.8× vs. eager (13.98 µs → 2.94 µs) on NVIDIA A100, with speedup scaling to 4.7× at 64 concurrent launches.</summary>

Benchmark on NVIDIA A100-SXM4-80GB, 1024-element add kernel, 1000 iterations:

![Triton vs njit kernel launch overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/triton-vs-njit/overhead_vs_kernels.png)

| Launch count | eager (µs) | njit (µs) | Speedup |
|---|---|---|---|
| 1 | 14.13 | 8.54 | **1.65×** |
| 4 | 54.90 | 18.32 | **3.00×** |
| 16 | 221.53 | 53.50 | **4.14×** |
| 64 | 913.16 | 194.54 | **4.70×** |

Linear fit: per-launch cost drops from 13.98 µs (eager Python) to 2.94 µs (njit), a **4.8×
reduction**. The standard Python path through Triton's launcher involves multiple layers of
Python dispatch; `cuLaunchKernelEx` called directly from compiled code eliminates all of them.

</details>

#### 3. End-to-end inductor graph wrapping (`NjitInductorGraph`) ([source](https://github.com/zasdfgbnm/njit-wrappers/blob/main/src/njit_wrappers/_inductor.py), [benchmark](https://github.com/zasdfgbnm/njit-wrappers/tree/main/benchmarks/inductor-vs-njit))

**Study.** Can TorchInductor's complete compiled graph runner — buffer allocation, Triton kernel
launches, extern ATen calls, and output assembly — be replaced end-to-end with a single
`@numba.njit` function, with no changes to Triton or the kernel compilation pipeline?

<details>
<summary><strong>Experiment design.</strong> NjitInductorGraph captures Inductor's generated Python, parses it into a typed op schedule, and synthesizes a single @numba.njit function covering buffer allocation, kernel launches, and output return — no changes to Triton or Inductor required.</summary>

`NjitInductorGraph` composes Modules 1 and 2 above into a full pipeline:

1. **Compile via Inductor.** The model is compiled through
   `torch.compile(backend='inductor', fullgraph=True)`. The generated Python source code is
   captured before execution.

2. **Parse into a schedule.** The generated Python is parsed with `ast` into a flat sequence of
   typed operations: `AllocOp` (buffer allocation), `KernelLaunchOp` (Triton kernel launch),
   `ExternKernelOp` (ATen mm/addmm), `AliasOp` (buffer reuse), and `ReturnOp`. This parser is
   ~500 lines and handles all patterns Inductor currently generates for CUDA graphs.

3. **Wrap Triton kernels.** Each `KernelLaunchOp` is reconstructed from its source and wrapped
   with `NumbaTritonKernel`. Triton kernel source is embedded in Inductor's generated Python;
   no additional compilation is needed.

4. **Emit `@numba.njit` runner.** A single `@numba.njit` function is synthesized that:
   allocates scratch buffers via `aoti_torch_empty_strided` (an LLVM intrinsic backed by a
   libtorch C export), launches each kernel via its `NumbaTritonKernel` trampoline, and returns
   the output tensors.

<details>
<summary>Example: relu(x + y) model (Python → inductor-generated runner → synthesized @njit function + LLVM IR)</summary>

**Example.** The model from [docs/inductor.md](https://github.com/zasdfgbnm/njit-wrappers/blob/main/docs/inductor.md):

```python
class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)

graph = NjitInductorGraph(Model().cuda(), (x, y))
out   = graph(x, y)
```

Inductor generates a Python runner that looks roughly like:

```python
# inductor-generated (captured by NjitInductorGraph)
def call(args):
    x, y = args
    buf0 = empty_strided_cuda((1024,), (1,), torch.float32)
    triton_poi_fused_add_relu_0.run(x, y, buf0, 1024,
        grid=grid(1024), stream=stream)
    return (buf0,)
```

`NjitInductorGraph` parses this and synthesizes the following `@numba.njit` function to replace it
(shown schematically; actual codegen uses Numba's `@intrinsic` factories):

```python
# synthesized by NjitInductorGraph — executes with zero Python overhead
@numba.njit
def njit_runner(x, y, stream):
    # AllocOp → aoti_torch_empty_strided intrinsic (all args baked as LLVM constants)
    buf0 = _alloc_buf0()   # shape=(1024,), stride=(1,), dtype=float32, device=cuda:0

    # KernelLaunchOp → NumbaTritonKernel trampoline (cuLaunchKernelEx directly)
    n    = 1024
    grid = (n + 255) // 256
    launch_triton_poi_fused_add_relu_0(grid, 1, 1, stream, x, y, buf0, n)

    # ReturnOp → box buf0 back to Python
    return buf0
```

The corresponding LLVM IR for the `_alloc_buf0` intrinsic (a zero-argument function that
allocates a `(1024,)` float32 CUDA tensor with all shape/stride/dtype constants baked in):

```llvm
; _alloc_buf0: allocate empty_strided((1024,), (1,), float32, cuda:0)
; all parameters are LLVM constants — zero runtime overhead for shape/dtype lookup
define i64 @_alloc_buf0() {
entry:
  %sizes       = alloca [1 x i64]
  %strides     = alloca [1 x i64]
  store i64 1024, i64* getelementptr([1 x i64], [1 x i64]* %sizes,   i32 0, i32 0)
  store i64    1, i64* getelementptr([1 x i64], [1 x i64]* %strides, i32 0, i32 0)
  %sizes_ptr   = bitcast [1 x i64]* %sizes   to i64*
  %strides_ptr = bitcast [1 x i64]* %strides to i64*
  %handle_slot = alloca i8*
  call i32 @aoti_torch_empty_strided(
    i64  1,              ; ndim
    i64* %sizes_ptr,
    i64* %strides_ptr,
    i32  6,              ; dtype = c10::ScalarType::Float (float32)
    i32  1,              ; device_type = CUDA
    i32  0,              ; device_index = 0
    i8*  bitcast(i8** %handle_slot to i8*))
  %handle  = load i8*,  i8** %handle_slot   ; AtenTensorHandle = at::Tensor*
  %impl_ptr = bitcast i8* %handle to i64*
  %impl    = load i64, i64* %impl_ptr       ; dereference to get TensorImpl*
  ret i64 %impl
}

declare i32 @aoti_torch_empty_strided(i64, i64*, i64*, i32, i32, i32, i8*)
  ; → aoti_torch_empty_strided from libtorch_cpu.so (stable C ABI)
```

</details>

</details>

<details>
<summary><strong>Results.</strong> Per-kernel cost drops 2.8× (5.43 µs → 1.93 µs) and fixed dispatch overhead drops 2.5× (46.6 µs → 18.9 µs) on NVIDIA GB200.</summary>

Benchmark on NVIDIA GB200, `torch.softmax` chain on 32×64 tensors, 1000 iterations:

![Inductor vs njit orchestration overhead](https://raw.githubusercontent.com/zasdfgbnm/njit-wrappers/main/benchmarks/inductor-vs-njit/overhead_vs_kernels.png)

| Kernel count | torch.compile (µs) | njit (µs) | Speedup |
|---|---|---|---|
| 1 | 37.43 | 14.70 | **2.55×** |
| 4 | 68.22 | 26.56 | **2.57×** |
| 16 | 133.56 | 49.71 | **2.69×** |
| 64 | 396.78 | 143.71 | **2.76×** |

Linear fit: per-kernel cost drops from 5.43 µs to 1.93 µs (**2.8×**); fixed overhead drops from
46.6 µs to 18.9 µs (**2.5×**).

</details>

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

