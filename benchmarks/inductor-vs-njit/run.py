"""Benchmark CPU-side latency of inductor graph calls: njit vs torch.compile.

Measures wall-clock time of running an inductor-compiled graph through
``NjitInductorGraph`` vs the standard ``torch.compile`` Python wrapper.
Tensors are small (32×64) so GPU compute is negligible — what we measure
is the CPU orchestration overhead (buffer allocation, grid computation,
kernel launches).  No cudaDeviceSynchronize is called.

The independent variable is the **number of Triton kernels** in the graph.
We use ``torch.softmax`` with alternating dims so that inductor produces
exactly one kernel per op (softmax is a reduction that cannot be fused
across different dims).

Produces a plot (overhead_vs_kernels.png) and a README.md in the same
directory.

Usage:
    python benchmarks/inductor-vs-njit/run.py
"""

import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model that produces exactly N kernels
# ---------------------------------------------------------------------------

MIN_KERNELS = 1
MAX_KERNELS = 64


class _SoftmaxChain(torch.nn.Module):
    """Chain of N softmax ops with alternating dims → N kernels."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(self.n):
            out = torch.softmax(out, dim=i % 2)
        return out


# ---------------------------------------------------------------------------
# Environment collection
# ---------------------------------------------------------------------------


def _collect_env():
    """Collect benchmark environment info as an ordered dict."""
    import triton

    info = {}

    info["CPU"] = platform.processor() or platform.machine()
    info["GPU"] = torch.cuda.get_device_name(0)
    info["CUDA"] = torch.version.cuda or "N/A"

    try:
        driver = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
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
    info["Numba"] = numba.__version__
    info["Triton"] = triton.__version__
    info["OS"] = platform.platform()

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARMUP = 50
ITERS = 1000


def bench(fn, args, warmup=WARMUP, iters=ITERS):
    """Time *iters* calls of fn(*args), return per-call time in microseconds."""
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e6  # → µs


def _robust_polyfit(xs, ys, threshold=2.0):
    """Fit y = k*x + b, removing outliers whose residual exceeds *threshold* σ."""
    k, b = np.polyfit(xs, ys, 1)
    residuals = ys - (k * xs + b)
    std = np.std(residuals)
    mask = np.abs(residuals) <= threshold * std
    return np.polyfit(xs[mask], ys[mask], 1)


def _robust_fit_through_origin(xs, ys, threshold=2.0):
    """Fit y = k*x (forcing b=0), removing outliers beyond *threshold* σ."""

    def _fit(x, y):
        return float(np.sum(x * y) / np.sum(x * x))

    k = _fit(xs, ys)
    residuals = ys - k * xs
    std = np.std(residuals)
    mask = np.abs(residuals) <= threshold * std
    return _fit(xs[mask], ys[mask])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    from njit_wrappers import NjitInductorGraph

    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(32, 64, device=device)

    kernel_counts = []
    times_njit = []
    times_compile = []

    for n in range(MIN_KERNELS, MAX_KERNELS + 1):
        model = _SoftmaxChain(n).cuda()

        # Build NjitInductorGraph (resets dynamo internally)
        njit_graph = NjitInductorGraph(model, (x,))

        # Build standard torch.compile graph
        torch._dynamo.reset()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        compiled(x)  # trigger compilation

        t_njit = bench(njit_graph, (x,))
        t_compile = bench(compiled, (x,))

        kernel_counts.append(n)
        times_njit.append(t_njit)
        times_compile.append(t_compile)

        print(f"  kernels={n:3d}  njit={t_njit:8.2f} µs  compile={t_compile:8.2f} µs")

    # -- Linear fits (with outlier removal) --
    #   njit:    y = k*x + b  (free intercept — fixed njit overhead)
    #   compile: y = k*x + b  (free intercept — fixed torch.compile overhead)
    xs = np.array(kernel_counts, dtype=np.float64)
    ys_njit = np.array(times_njit, dtype=np.float64)
    ys_compile = np.array(times_compile, dtype=np.float64)

    k_njit, b_njit = _robust_polyfit(xs, ys_njit)
    k_compile, b_compile = _robust_polyfit(xs, ys_compile)

    print("\nLinear fit (outliers removed)")
    print(f"  njit:    y = {k_njit:.4f}x + {b_njit:.4f}  (µs)")
    print(f"  compile: y = {k_compile:.4f}x + {b_compile:.4f}  (µs)")

    # -- Console table --
    print(
        f"\nCPU latency for inductor graphs (32×64 CUDA tensors), {ITERS} iterations\n"
    )
    print(f"{'Kernels':>7}  {'njit (µs)':>10}  {'compile (µs)':>13}")
    print(f"{'-------':>7}  {'----------':>10}  {'-------------':>13}")
    for i, n in enumerate(kernel_counts):
        print(f"{n:7d}  {times_njit[i]:10.2f}  {times_compile[i]:13.2f}")

    # -- Plot --
    plot_name = "overhead_vs_kernels.png"
    plot_path = OUT_DIR / plot_name

    fit_xs = np.linspace(MIN_KERNELS, MAX_KERNELS, 200)
    fit_njit = k_njit * fit_xs + b_njit
    fit_compile = k_compile * fit_xs + b_compile

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        kernel_counts, times_compile, s=12, alpha=0.5, label="torch.compile (data)"
    )
    ax.scatter(kernel_counts, times_njit, s=12, alpha=0.5, label="njit (data)")
    ax.plot(
        fit_xs,
        fit_compile,
        "-",
        linewidth=2,
        label=f"compile fit: y = {k_compile:.2f}x + {b_compile:.2f}",
    )
    ax.plot(
        fit_xs,
        fit_njit,
        "-",
        linewidth=2,
        label=f"njit fit: y = {k_njit:.2f}x + {b_njit:.2f}",
    )
    ax.set_xlabel("Number of Triton kernels in graph")
    ax.set_ylabel("Latency (µs/call)")
    ax.set_title("CPU Latency: NjitInductorGraph vs torch.compile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # -- README.md --
    readme_path = OUT_DIR / "README.md"
    env_info = _collect_env()

    md_table_rows = []
    for i, n in enumerate(kernel_counts):
        md_table_rows.append(f"| {n} | {times_njit[i]:.2f} | {times_compile[i]:.2f} |")
    md_table = "\n".join(md_table_rows)

    env_table_rows = "\n".join(f"| {k} | {v} |" for k, v in env_info.items())

    readme = f"""\
# Inductor vs njit: CPU Latency by Number of Kernels

Measures wall-clock time of running an inductor-compiled graph through
`NjitInductorGraph` (the njit wrapper) vs the standard `torch.compile`
Python wrapper.  Tensors are small (32×64) so GPU compute is negligible —
only the CPU orchestration overhead is measured.  No `cudaDeviceSynchronize`
is called.

The independent variable is the number of Triton kernels in the graph.
Each graph is a chain of `torch.softmax` calls with alternating dims
(`dim=i%2`), which forces inductor to produce exactly one kernel per op.

## Results

![overhead_vs_kernels]({plot_name})

### Linear fit (outliers removed, 2σ threshold)

|         | model     | k (µs/kernel) | b (µs)  |
|---------|-----------|---------------|---------|
| njit    | y = kx+b  | {k_njit:.4f}        | {b_njit:.4f}  |
| compile | y = kx+b  | {k_compile:.4f}        | {b_compile:.4f}  |

- **k** (slope) is the **per-kernel cost** — the marginal time (in µs)
  added by each additional Triton kernel launch.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs)
  for entering and leaving the wrapper, independent of how many kernels
  are launched.

### Raw data

| Kernels | njit (µs) | compile (µs) |
|---------|-----------|--------------|
{md_table}

> {ITERS} iterations per data point, {WARMUP} warmup iterations.

## Why njit is faster

There are two independent sources of savings, matching the two model
parameters *k* and *b*.

### Per-kernel cost: {k_njit:.2f} vs {k_compile:.2f} µs/kernel

When `torch.compile` runs the inductor-generated Python wrapper, each
Triton kernel launch crosses the Python/C boundary multiple times:

1. Python grid lambda is evaluated for the grid dimensions
2. `CachingAutotuner.__call__` is invoked — a Python method that looks up
   the best config and packages arguments
3. The Python-side launcher calls into the Triton C extension to fire
   `cuLaunchKernelEx`

Inside a compiled **njit** function, this entire chain is replaced by
LLVM-compiled machine code.  The grid is a compile-time integer constant
(computed once during `NjitInductorGraph.__init__`), and each kernel fires
through a lightweight C trampoline (`_generate_launch_trampoline_src`) that
calls `cuLaunchKernelEx` directly — no Python frames, no argument-parsing,
no autotuner lookup.  The ~{k_compile - k_njit:.1f} µs/kernel savings
({k_compile:.2f} → {k_njit:.2f} µs/kernel) is the cost of Python's
per-launch interpreter overhead.

### Fixed overhead: {b_njit:.2f} vs {b_compile:.2f} µs

Every `torch.compile` call pays a fixed Python cost before the first
kernel even launches: the dynamo/inductor graph wrapper must check guards
(shape guards, device guards, etc.) in Python, unpack the argument tuple,
and resolve the cached compiled artifact.  This accounts for the
~{b_compile:.1f} µs baseline.

The njit wrapper's ~{b_njit:.1f} µs baseline comes from the Numba dispatcher
(one C-level function call) plus tensor unboxing — extracting the
`TensorImpl*` from each PyTorch tensor argument so it can be passed as
a raw pointer into compiled code.

### Break-even

The njit wrapper wins immediately: even at 1 kernel the lower fixed cost
more than compensates for any overhead.  The gap widens linearly with
graph size at ~{k_compile - k_njit:.1f} µs per additional kernel.

## Benchmark environment

| Component | Details |
|-----------|---------|
{env_table_rows}

## Running

```bash
PYTHONPATH=src python benchmarks/inductor-vs-njit/run.py
```
"""

    readme_path.write_text(readme)
    print(f"README saved to {readme_path}")


if __name__ == "__main__":
    main()
