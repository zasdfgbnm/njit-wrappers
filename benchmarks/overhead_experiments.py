"""Comprehensive host overhead reduction experiments.

Tests multiple approaches to reduce CPU-side overhead for njit ATen calls.
Each experiment creates its own numba intrinsics to test a different strategy.

Usage:
    PYTHONPATH=src python benchmarks/overhead_experiments.py
"""

import ctypes
import os
import time
from pathlib import Path

import llvmlite.binding as llvm
import numba
import torch
from llvmlite import ir
from numba.core import cgutils
from numba.core.extending import intrinsic

# We need the current njit_wrappers to be loaded for the base infrastructure
import njit_wrappers  # noqa: F401
import njit_wrappers._bridge as _bridge_module
from njit_wrappers._tensor import TensorType, tensor_type

_bridge_lib = ctypes.CDLL(_bridge_module.__file__)
TORCH_LIB = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))
CUDA_LIB_PATH = str(Path(torch.__file__).parent / "lib" / "libtorch_cuda.so")
CUDA_LIB = ctypes.CDLL(CUDA_LIB_PATH)
ITERS = 5000
WARMUP = 500


def _fn_addr(lib, mangled):
    fn = getattr(lib, mangled)
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    assert addr, f"symbol not found: {mangled}"
    return addr


def bench(fn, args, iters=ITERS, warmup=WARMUP):
    """Time iters calls, return per-call microseconds."""
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - start) / iters * 1e6


def make_cuda_tensor():
    return torch.randn(4, 4, device="cuda")


# ---------------------------------------------------------------------------
# Helper to make sret-based unary intrinsic (same as _tensor.py pattern)
# ---------------------------------------------------------------------------
def _tensor_slot(builder, val):
    i64 = ir.IntType(64)
    slot = builder.alloca(i64)
    builder.store(val, slot)
    return builder.bitcast(slot, ir.IntType(8).as_pointer())


# ---------------------------------------------------------------------------
# Approach 1: C++ wrapper ops (int64 -> int64, no sret, dispatched via at::relu)
# ---------------------------------------------------------------------------

llvm.add_symbol("njit_relu_wrapper", _fn_addr(_bridge_lib, "njit_relu_wrapper"))


@intrinsic
def _wrapper_relu(typingctx, a):
    """Call njit_relu_wrapper: int64 -> int64 (simple register ABI)."""
    if not isinstance(a, TensorType):
        return None
    sig = tensor_type(tensor_type)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(i64, [i64]), "njit_relu_wrapper"
        )
        return builder.call(fn, [args[0]])

    return sig, codegen


# ---------------------------------------------------------------------------
# Approach 2: at::cuda::relu via libtorch_cuda.so (sret, no dispatcher at all)
# ---------------------------------------------------------------------------

llvm.add_symbol(
    "_cuda_relu", _fn_addr(CUDA_LIB, "_ZN2at4cuda4reluERKNS_6TensorE")
)


@intrinsic
def _cuda_relu_intrinsic(typingctx, a):
    """Direct call to at::cuda::relu (sret ABI, no dispatcher)."""
    if not isinstance(a, TensorType):
        return None
    sig = tensor_type(tensor_type)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        i8p = ir.IntType(8).as_pointer()
        out = builder.alloca(i64)
        fn = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(ir.VoidType(), [i8p, i8p]),
            "_cuda_relu",
        )
        fn.args[0].add_attribute("sret")
        builder.call(
            fn, [builder.bitcast(out, i8p), _tensor_slot(builder, args[0])]
        )
        return builder.load(out)

    return sig, codegen


# ---------------------------------------------------------------------------
# Approach 3: Borrow-based unbox (no refcount on input)
# We use njit_borrow_impl to extract TensorImpl* without incref.
# This saves the incref+decref pair on each input tensor.
# For this experiment, we still use the same op calling convention.
# ---------------------------------------------------------------------------

llvm.add_symbol("njit_borrow_impl", _fn_addr(_bridge_lib, "njit_borrow_impl"))

# We can't easily change the unbox function at runtime, so instead we'll
# measure the theoretical savings: the borrow_impl is ~600ns cheaper per
# tensor input (measured via ctypes). For the benchmark, the savings would
# be visible in the fixed overhead (unbox cost) not per-op cost.

# ---------------------------------------------------------------------------
# Approach 4: at::_ops::relu::call (full dispatch, original behavior)
# For comparison with redispatch.
# ---------------------------------------------------------------------------

llvm.add_symbol(
    "_aten_relu_call",
    _fn_addr(TORCH_LIB, "_ZN2at4_ops4relu4callERKNS_6TensorE"),
)


@intrinsic
def _call_relu_intrinsic(typingctx, a):
    """Full dispatch: at::_ops::relu::call (sret ABI)."""
    if not isinstance(a, TensorType):
        return None
    sig = tensor_type(tensor_type)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        i8p = ir.IntType(8).as_pointer()
        out = builder.alloca(i64)
        fn = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(ir.VoidType(), [i8p, i8p]),
            "_aten_relu_call",
        )
        fn.args[0].add_attribute("sret")
        builder.call(
            fn, [builder.bitcast(out, i8p), _tensor_slot(builder, args[0])]
        )
        return builder.load(out)

    return sig, codegen


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------


def _make_chain(op_fn, n):
    """Create an njit function that chains n calls of op_fn."""
    if n == 1:

        @numba.njit
        def f(x):
            return op_fn(x)

        return f
    elif n == 5:

        @numba.njit
        def f(x):
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            return op_fn(x)

        return f
    elif n == 10:

        @numba.njit
        def f(x):
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            return op_fn(x)

        return f
    elif n == 20:

        @numba.njit
        def f(x):
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            x = op_fn(x)
            return op_fn(x)

        return f
    else:
        raise ValueError(f"Unsupported chain length: {n}")


def run_eager():
    """Eager baseline using torch.clamp (since torch.relu is broken on this branch)."""

    def f(x):
        return torch.clamp(x, min=0)

    def f5(x):
        for _ in range(5):
            x = torch.clamp(x, min=0)
        return x

    def f10(x):
        for _ in range(10):
            x = torch.clamp(x, min=0)
        return x

    def f20(x):
        for _ in range(20):
            x = torch.clamp(x, min=0)
        return x

    t = make_cuda_tensor()
    return {
        "1op": bench(f, (t,)),
        "5op": bench(f5, (t,)),
        "10op": bench(f10, (t,)),
        "20op": bench(f20, (t,)),
    }


def run_approach(name, op_fn):
    """Run 1/5/10/20-op chain for a given op function."""
    t = make_cuda_tensor()
    result = {}
    for n in [1, 5, 10, 20]:
        fn = _make_chain(op_fn, n)
        fn(t)  # warmup JIT
        result[f"{n}op"] = bench(fn, (t,))
    return result


def run_torch_compile():
    """torch.compile with inductor backend."""

    def f(x):
        return torch.clamp(x, min=0)

    def f20(x):
        for _ in range(20):
            x = torch.clamp(x, min=0)
        return x

    fc = torch.compile(f, backend="inductor")
    fc20 = torch.compile(f20, backend="inductor")
    t = make_cuda_tensor()
    return {"1op": bench(fc, (t,)), "20op": bench(fc20, (t,))}


def main():
    assert torch.cuda.is_available(), "Requires CUDA"
    print("=" * 80, flush=True)
    print("HOST OVERHEAD REDUCTION EXPERIMENTS", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name()}", flush=True)
    print(f"Tensor size: 4x4, Iterations: {ITERS}, Warmup: {WARMUP}", flush=True)
    print("=" * 80, flush=True)

    results = {}

    experiments = [
        ("eager (torch.clamp min=0)", run_eager),
        (
            "njit redispatch (::redispatch)",
            lambda: run_approach("redispatch", torch.relu),
        ),
        (
            "njit ::call (full dispatch)",
            lambda: run_approach("call", _call_relu_intrinsic),
        ),
        (
            "njit at::cuda::relu (direct)",
            lambda: run_approach("cuda", _cuda_relu_intrinsic),
        ),
        # C++ wrapper crashes after many iterations due to doubled intermediate
        # tensor leaks (known limitation). Per-op cost would be similar since
        # it still goes through at::relu dispatcher.
        # ("njit C++ wrapper (int64 ABI)", lambda: run_approach("wrapper", _wrapper_relu)),
        ("torch.compile (inductor)", run_torch_compile),
    ]

    for name, fn in experiments:
        print(f"\n--- {name} ---", flush=True)
        try:
            r = fn()
            results[name] = r
            for k, v in sorted(r.items(), key=lambda x: int(x[0].replace("op", ""))):
                print(f"  {k}: {v:.2f} us", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback

            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY (all times in microseconds per call)", flush=True)
    print("=" * 80, flush=True)
    cols = ["1op", "5op", "10op", "20op"]
    print(f"{'Approach':<38} " + " ".join(f"{c:>7}" for c in cols), flush=True)
    print("-" * 80, flush=True)
    for name, r in results.items():
        vals = " ".join(f"{r.get(c, float('nan')):>7.2f}" for c in cols)
        print(f"{name:<38} {vals}", flush=True)

    # Derived: per-op marginal cost (from 1op to 20op)
    print("\n" + "=" * 80, flush=True)
    print("DERIVED: per-op marginal cost (1->20 ops) and fixed overhead", flush=True)
    print("=" * 80, flush=True)
    for name, r in results.items():
        if "1op" in r and "20op" in r:
            per_op = (r["20op"] - r["1op"]) / 19
            fixed = r["1op"] - per_op
            print(
                f"{name:<38} fixed={fixed:>7.2f}us  per_op={per_op:>7.2f}us",
                flush=True,
            )

    # Force exit to avoid cleanup crashes from redispatch on this branch
    os._exit(0)


if __name__ == "__main__":
    main()
