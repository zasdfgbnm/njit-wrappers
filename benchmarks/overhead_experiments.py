"""Comprehensive host overhead reduction experiments.

Tests multiple approaches to reduce CPU-side overhead for njit ATen calls.
Each experiment modifies how we call ATen ops and measures the impact.

Usage:
    PYTHONPATH=src python benchmarks/overhead_experiments.py
"""

import ctypes
import time
from pathlib import Path

import numba
import torch
from llvmlite import ir
from numba.core import cgutils
from numba.core.extending import intrinsic

# We need the current njit_wrappers to be loaded for the base infrastructure
import njit_wrappers  # noqa: F401
from njit_wrappers._tensor import TensorType, tensor_type

TORCH_LIB = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))
ITERS = 5000
WARMUP = 500


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
# BASELINE: current implementation (redispatch on this branch)
# ---------------------------------------------------------------------------


def run_baseline():
    """Current redispatch implementation."""

    @numba.njit
    def f(x):
        return torch.relu(x)

    @numba.njit
    def f5(x):
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        return torch.relu(x)

    @numba.njit
    def f20(x):
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        return torch.relu(x)

    t = make_cuda_tensor()
    return {
        "1op": bench(f, (t,)),
        "5op": bench(f5, (t,)),
        "20op": bench(f20, (t,)),
    }


def run_eager():
    """Eager baseline for comparison."""

    def f(x):
        return torch.relu(x)

    def f5(x):
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        return torch.relu(x)

    def f20(x):
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        return torch.relu(x)

    t = make_cuda_tensor()
    return {
        "1op": bench(f, (t,)),
        "5op": bench(f5, (t,)),
        "20op": bench(f20, (t,)),
    }


def main():
    assert torch.cuda.is_available(), "Requires CUDA"
    print("=" * 70)
    print("HOST OVERHEAD REDUCTION EXPERIMENTS")
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Tensor size: 4x4, Iterations: {ITERS}, Warmup: {WARMUP}")
    print("=" * 70)

    results = {}

    print("\n--- Eager baseline ---")
    r = run_eager()
    results["eager"] = r
    for k, v in r.items():
        print(f"  {k}: {v:.2f} us")

    print("\n--- njit baseline (current redispatch) ---")
    r = run_baseline()
    results["njit_redispatch"] = r
    for k, v in r.items():
        print(f"  {k}: {v:.2f} us")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Approach':<35} {'1op':>8} {'5op':>8} {'20op':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<35} {r['1op']:>7.2f}  {r['5op']:>7.2f}  {r['20op']:>7.2f}")


if __name__ == "__main__":
    main()
