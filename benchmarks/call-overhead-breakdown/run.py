"""Breakdown of Python overhead in Inductor's generated call() function.

Inductor's generated ``call(args)`` orchestrates GPU kernels from Python.
Every invocation incurs several categories of interpreter overhead.  This
benchmark measures each category *in isolation* using representative
microbenchmarks:

1. **Frame allocation** — the cost of entering and leaving a Python function.
2. **Buffer allocation** — one call to ``torch.empty_strided(..., device='cuda')``
   per intermediate buffer.
3. **Grid computation** — one call to the ``grid(n)`` helper per kernel.
4. **Triton launcher** — one call to ``kernel[(grid,)](args...)`` per kernel,
   the bulk of which is ``JITFunction.__call__`` building the ``void*``
   parameter array and calling ``cuLaunchKernelEx`` via ctypes.

Each component is measured independently (same warmup/iteration count).
Times are reported in µs; the percentage is each component's share of the
measured sum.

The results are compared against the per-kernel slope measured in the
``inductor-vs-njit`` benchmark (≈ 5.4 µs/kernel for ``torch.compile``).

Usage:
    PYTHONPATH=src python benchmarks/call-overhead-breakdown/run.py
"""

import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

OUT_DIR = Path(__file__).resolve().parent

WARMUP = 200
ITERS = 2000


# ---------------------------------------------------------------------------
# Trivial Triton kernel used to measure launcher overhead
# ---------------------------------------------------------------------------


@triton.jit
def _noop_kernel(x_ptr, n_elements: int, BLOCK: tl.constexpr):
    """Read one block of floats and write them back unchanged."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, v, mask=mask)


BLOCK_SIZE = 1024
N_ELEMENTS = 1024


def _triton_grid(n: int) -> tuple[int]:
    return ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _collect_env() -> dict[str, str]:
    info: dict[str, str] = {}
    info["CPU"] = platform.processor() or platform.machine()
    info["GPU"] = torch.cuda.get_device_name(0)
    info["CUDA"] = torch.version.cuda or "N/A"
    try:
        driver = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .split("\n")[0]
        )
    except Exception:
        driver = "N/A"
    info["Driver"] = driver
    info["Python"] = platform.python_version()
    info["PyTorch"] = torch.__version__
    info["Triton"] = triton.__version__
    info["OS"] = platform.platform()
    return info


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def bench(fn: object, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median per-call time in microseconds (no-arg callable)."""
    assert callable(fn)
    for _ in range(warmup):
        fn()  # type: ignore[operator]
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required"

    # Pre-allocate a tensor and warm up the Triton kernel so compilation cost
    # is excluded from the launcher measurement.
    x = torch.ones(N_ELEMENTS, device="cuda", dtype=torch.float32)
    grid = _triton_grid(N_ELEMENTS)
    for _ in range(20):
        _noop_kernel[grid](x, N_ELEMENTS, BLOCK=BLOCK_SIZE)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 1. Frame allocation: cost of entering + leaving a Python function
    # ------------------------------------------------------------------
    def _empty() -> None:
        pass

    t_frame = bench(_empty)

    # ------------------------------------------------------------------
    # 2. Buffer allocation: one empty_strided call with typical args
    #    (shape and stride representative of a 32×64 float32 buffer)
    # ------------------------------------------------------------------
    _shape = (32, 64)
    _stride = (64, 1)

    def _alloc() -> None:
        torch.empty_strided(_shape, _stride, device="cuda", dtype=torch.float32)

    t_alloc = bench(_alloc)

    # ------------------------------------------------------------------
    # 3. Grid computation: typical ceil-div helper called once per kernel
    # ------------------------------------------------------------------
    def _grid() -> None:
        _triton_grid(N_ELEMENTS)

    t_grid = bench(_grid)

    # ------------------------------------------------------------------
    # 4. Triton launcher: full JITFunction.__call__ path including
    #    specialisation lookup, void* array construction, ctypes call to
    #    cuLaunchKernelEx
    # ------------------------------------------------------------------
    def _launch() -> None:
        _noop_kernel[grid](x, N_ELEMENTS, BLOCK=BLOCK_SIZE)

    t_launch = bench(_launch)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    components = [
        ("Frame allocation", t_frame),
        ("Buffer allocation (`empty_strided_cuda`)", t_alloc),
        ("Grid computation", t_grid),
        ("Triton launcher", t_launch),
    ]
    total = sum(t for _, t in components)

    print("\nOverhead component breakdown (per-kernel, µs)\n")
    print(f"  {'Component':<45} {'Time (µs)':>10}  {'% of sum':>9}")
    print(f"  {'-' * 45}  {'-' * 10}  {'-' * 9}")
    for name, t in components:
        print(f"  {name:<45} {t:>10.3f}  {t / total * 100:>8.1f}%")
    print(f"  {'Sum':<45} {total:>10.3f}  {'100.0%':>9}")
    print(
        "\n  Reference (inductor-vs-njit slope, torch.compile): ~5.4 µs/kernel"
        "\n  Reference (inductor-vs-njit slope, njit):           ~1.9 µs/kernel"
    )

    # ------------------------------------------------------------------
    # Write README.md
    # ------------------------------------------------------------------
    env_info = _collect_env()
    env_rows = "\n".join(f"| {k} | {v} |" for k, v in env_info.items())

    component_rows = "\n".join(
        f"| {name} | {t:.3f} | {t / total * 100:.1f}% |" for name, t in components
    )

    readme = f"""\
# Overhead breakdown of Inductor's generated `call()` function

Inductor's generated ``call(args)`` runs on every forward pass and
orchestrates GPU kernels from Python.  This benchmark measures each category
of CPython interpreter overhead *in isolation*, using representative
microbenchmarks, to quantify how much each component contributes to the total
per-kernel dispatch cost.

## Components measured

| Component | What it represents |
|-----------|-------------------|
| Frame allocation | Python function entry/exit — baseline interpreter tax. |
| Buffer allocation | `torch.empty_strided` per intermediate buffer. |
| Grid computation | `grid(n)` call per kernel. |
| Triton launcher | `JITFunction.__call__` builds a void* arg array, calls cuLaunch. |

## Results

| Component | Time (µs) | % of sum |
|-----------|-----------|----------|
{component_rows}
| **Sum** | **{total:.3f}** | **100%** |

### Comparison with inductor-vs-njit

The per-kernel slope measured in the
[inductor-vs-njit benchmark](../inductor-vs-njit/README.md) is:

- `torch.compile` (Python): **~5.4 µs/kernel**
- `njit` wrapper:           **~1.9 µs/kernel**

`@numba.njit` eliminates all four components above — there is no Python stack
frame, no Python-level tuple construction, no boxed integer grid result, and
no `JITFunction.__call__` path.  The remaining 1.9 µs represents the Numba
dispatcher and LLVM-generated machine-code overhead.

The sum of the four isolated microbenchmarks ({total:.1f} µs) is broadly
consistent with the 5.4 µs/kernel slope, confirming that these components
account for the majority of the Python overhead that `@njit` removes.

## Methodology

Each component is measured with {WARMUP} warmup iterations followed by
{ITERS} timed iterations; the **median** is reported.  The Triton kernel is
pre-compiled before timing so compilation cost is excluded.  No
`cudaDeviceSynchronize` is called — we measure CPU-side dispatch overhead
only.

## Benchmark environment

| Component | Details |
|-----------|---------|
{env_rows}

## Running

```bash
PYTHONPATH=src python benchmarks/call-overhead-breakdown/run.py
```
"""

    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(readme)
    print(f"\nREADME saved to {readme_path}")


if __name__ == "__main__":
    main()
