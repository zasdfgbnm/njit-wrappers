"""Benchmark CPU-side overhead of njit-wrapped ATen calls on CUDA tensors.

Measures wall-clock time of launching GPU ops through njit vs plain Python.
All tensors are tiny (4×4) so GPU compute is negligible — what we measure
is the CPU dispatch overhead.  No cudaDeviceSynchronize is called.

Each graph is simply ``for i in range(N): x = torch.relu(x)`` so the only
variable is the number of ops.

Produces a plot (overhead_vs_ops.png) and a README.md in the same directory.

Usage:
    python benchmarks/eager-vs-njit/run.py
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

import njit_wrappers  # noqa: F401, E402 – registers TensorType

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Graph builders — ``for i in range(N): x = torch.relu(x)``
#
# Numba's njit does not support Python-level for-loops over torch tensors,
# so we generate a dedicated function for each op count.
# ---------------------------------------------------------------------------

MIN_OPS = 1
MAX_OPS = 64


def _make_graph(n_ops):
    """Return an eager function that applies torch.relu *n_ops* times."""
    body = "\n".join("    x = torch.relu(x)" for _ in range(n_ops))
    src = f"def _graph(x):\n{body}\n    return x"
    ns = {"torch": torch}
    exec(src, ns)  # noqa: S102
    return ns["_graph"]


# ---------------------------------------------------------------------------
# Environment collection
# ---------------------------------------------------------------------------


def _collect_env():
    """Collect benchmark environment info as an ordered dict."""
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
    """Fit y = k*x + b, removing outliers whose residual exceeds *threshold* σ.

    Performs an initial fit, computes residuals, discards points beyond
    *threshold* standard deviations, then refits on the clean data.
    """
    k, b = np.polyfit(xs, ys, 1)
    residuals = ys - (k * xs + b)
    std = np.std(residuals)
    mask = np.abs(residuals) <= threshold * std
    return np.polyfit(xs[mask], ys[mask], 1)


def _robust_fit_through_origin(xs, ys, threshold=2.0):
    """Fit y = k*x (forcing b=0), removing outliers beyond *threshold* σ.

    Uses least-squares with no intercept: k = Σ(x·y) / Σ(x²).
    """

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
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(4, 4, device=device)

    op_counts = []
    times_njit = []
    times_eager = []

    for n_ops in range(MIN_OPS, MAX_OPS + 1):
        graph_fn = _make_graph(n_ops)
        graph_fn_njit = numba.njit(graph_fn)
        # Trigger njit compilation (not timed)
        graph_fn_njit(x)

        t_njit = bench(graph_fn_njit, (x,))
        t_eager = bench(graph_fn, (x,))

        op_counts.append(n_ops)
        times_njit.append(t_njit)
        times_eager.append(t_eager)

        print(f"  ops={n_ops:3d}  njit={t_njit:8.2f} µs  eager={t_eager:8.2f} µs")

    # -- Linear fits (with outlier removal) --
    #   njit:  y = k*x + b  (free intercept — real fixed overhead from dispatcher)
    #   eager: y = k*x      (forced through origin — no fixed overhead)
    xs = np.array(op_counts, dtype=np.float64)
    ys_njit = np.array(times_njit, dtype=np.float64)
    ys_eager = np.array(times_eager, dtype=np.float64)

    k_njit, b_njit = _robust_polyfit(xs, ys_njit)
    k_eager = _robust_fit_through_origin(xs, ys_eager)

    print("\nLinear fit (outliers removed)")
    print(f"  njit:   y = {k_njit:.4f}x + {b_njit:.4f}  (µs)")
    print(f"  eager:  y = {k_eager:.4f}x           (µs, b forced to 0)")

    # -- Console table --
    print(f"\nCPU overhead on CUDA tiny tensors (4×4), {ITERS} iterations\n")
    print(f"{'Ops':>4}  {'njit (µs)':>10}  {'eager (µs)':>11}")
    print(f"{'----':>4}  {'----------':>10}  {'-----------':>11}")
    for i, n_ops in enumerate(op_counts):
        print(f"{n_ops:4d}  {times_njit[i]:10.2f}  {times_eager[i]:11.2f}")

    # -- Plot --
    plot_name = "overhead_vs_ops.png"
    plot_path = OUT_DIR / plot_name

    fit_xs = np.linspace(MIN_OPS, MAX_OPS, 200)
    fit_njit = k_njit * fit_xs + b_njit
    fit_eager = k_eager * fit_xs

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(op_counts, times_eager, s=12, alpha=0.5, label="eager (data)")
    ax.scatter(op_counts, times_njit, s=12, alpha=0.5, label="njit (data)")
    ax.plot(
        fit_xs,
        fit_eager,
        "-",
        linewidth=2,
        label=f"eager fit: y = {k_eager:.2f}x",
    )
    ax.plot(
        fit_xs,
        fit_njit,
        "-",
        linewidth=2,
        label=f"njit fit: y = {k_njit:.2f}x + {b_njit:.2f}",
    )
    ax.set_xlabel("Number of ops")
    ax.set_ylabel("Overhead (µs/call)")
    ax.set_title("CPU Dispatch Overhead: njit vs Eager PyTorch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # -- README.md --
    readme_path = OUT_DIR / "README.md"
    env_info = _collect_env()

    md_table_rows = []
    for i, n_ops in enumerate(op_counts):
        md_table_rows.append(
            f"| {n_ops} | {times_njit[i]:.2f} | {times_eager[i]:.2f} |"
        )
    md_table = "\n".join(md_table_rows)

    env_table_rows = "\n".join(f"| {k} | {v} |" for k, v in env_info.items())

    readme = f"""\
# Eager vs njit: CPU Dispatch Overhead

Measures wall-clock time of launching GPU ops through `numba.njit` vs plain
eager PyTorch.  All tensors are tiny (4×4) so GPU compute is negligible — only
the CPU dispatch overhead is measured.  No `cudaDeviceSynchronize` is called.

Each graph is simply `for i in range(N): x = torch.relu(x)`.

## Results

![overhead_vs_ops]({plot_name})

### Linear fit (outliers removed, 2σ threshold)

|       | model     | k (µs/op) | b (µs)  |
|-------|-----------|-----------|---------|
| njit  | y = kx+b  | {k_njit:.4f}    | {b_njit:.4f}  |
| eager | y = kx    | {k_eager:.4f}    | 0 (forced) |

- **k** (slope) is the **per-op cost** — the marginal time (in µs) added by
  each additional `torch.relu` call.  A smaller *k* means each op dispatches
  faster.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs) for
  entering and leaving the function, independent of how many ops it contains.
  For njit, this captures the Numba dispatcher + tensor borrow/wrap cost.
  For eager, *b* is forced to 0 because plain Python function calls have no
  meaningful fixed overhead beyond the per-op cost itself.

### Raw data

| Ops | njit (µs) | eager (µs) |
|-----|-----------|------------|
{md_table}

> {ITERS} iterations per data point, {WARMUP} warmup iterations.

## Why njit is faster per op

Each `torch.relu(x)` call in **eager** mode crosses the Python/C++ boundary:

1. Python attribute lookup (`torch.relu`)
2. Python function-call frame creation
3. pybind11 entry point + `PythonArgParser` argument parsing
4. ATen dispatcher → kernel

Inside a compiled **njit** function, after JIT compilation, each op lowers to a
direct call to `at::_ops::relu::call()` in LLVM-compiled machine code — the
Python interpreter is not involved per op.  The ~{k_eager - k_njit:.1f} µs/op savings
({k_eager:.2f} → {k_njit:.2f} µs/op) is the cost of PyTorch's Python dispatch layer.

The tradeoff is the ~{b_njit:.1f} µs **fixed** overhead that njit pays on every
function entry: the Numba dispatcher + tensor unboxing (borrowing the
`TensorImpl*` from the Python object).  Eager has negligible fixed overhead.
The break-even point is at roughly {b_njit / (k_eager - k_njit):.0f} ops;
beyond that njit wins.

## Benchmark environment

| Component | Details |
|-----------|---------|
{env_table_rows}

## Running

```bash
PYTHONPATH=src python benchmarks/eager-vs-njit/run.py
```
"""

    readme_path.write_text(readme)
    print(f"README saved to {readme_path}")


if __name__ == "__main__":
    main()
