"""Benchmark CPU-side overhead of inductor graph calls: njit vs torch.compile.

Measures wall-clock time of running an inductor-compiled graph through
``NjitInductorGraph`` vs the standard ``torch.compile`` Python wrapper.
All tensors are tiny (4×4) so GPU compute is negligible — what we measure
is the CPU orchestration overhead (buffer allocation, grid computation,
kernel launches).  No cudaDeviceSynchronize is called.

The independent variable is the number of fused pointwise ops per graph
(1 to 64 chained ``torch.relu`` calls).  Each op count produces a
distinct inductor graph with one or more Triton kernels.

Produces a plot (overhead_vs_ops.png) and a README.md in the same directory.

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
# Graph builders
# ---------------------------------------------------------------------------

MIN_OPS = 1
MAX_OPS = 64


class _PointwiseModel(torch.nn.Module):
    """Model that applies torch.relu *n_ops* times."""

    def __init__(self, n_ops):
        super().__init__()
        self.n_ops = n_ops

    def forward(self, x, y):
        out = x + y
        for _ in range(self.n_ops):
            out = torch.relu(out)
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
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)

    op_counts = []
    times_njit = []
    times_compile = []

    for n_ops in range(MIN_OPS, MAX_OPS + 1):
        model = _PointwiseModel(n_ops).cuda()

        # Build NjitInductorGraph (resets dynamo internally)
        njit_graph = NjitInductorGraph(model, (x, y))

        # Build standard torch.compile graph
        torch._dynamo.reset()
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        compiled(x, y)  # trigger compilation

        t_njit = bench(njit_graph, (x, y))
        t_compile = bench(compiled, (x, y))

        op_counts.append(n_ops)
        times_njit.append(t_njit)
        times_compile.append(t_compile)

        print(f"  ops={n_ops:3d}  njit={t_njit:8.2f} µs  compile={t_compile:8.2f} µs")

    # -- Linear fits (with outlier removal) --
    #   njit:    y = k*x + b  (free intercept — fixed njit overhead)
    #   compile: y = k*x      (forced through origin — baseline)
    xs = np.array(op_counts, dtype=np.float64)
    ys_njit = np.array(times_njit, dtype=np.float64)
    ys_compile = np.array(times_compile, dtype=np.float64)

    k_njit, b_njit = _robust_polyfit(xs, ys_njit)
    k_compile = _robust_fit_through_origin(xs, ys_compile)

    print("\nLinear fit (outliers removed)")
    print(f"  njit:    y = {k_njit:.4f}x + {b_njit:.4f}  (µs)")
    print(f"  compile: y = {k_compile:.4f}x           (µs, b forced to 0)")

    # -- Console table --
    print(
        f"\nCPU overhead for inductor graphs on tiny CUDA tensors (4×4),"
        f" {ITERS} iterations\n"
    )
    print(f"{'Ops':>4}  {'njit (µs)':>10}  {'compile (µs)':>13}")
    print(f"{'----':>4}  {'----------':>10}  {'-------------':>13}")
    for i, n_ops in enumerate(op_counts):
        print(f"{n_ops:4d}  {times_njit[i]:10.2f}  {times_compile[i]:13.2f}")

    # -- Plot --
    plot_name = "overhead_vs_ops.png"
    plot_path = OUT_DIR / plot_name

    fit_xs = np.linspace(MIN_OPS, MAX_OPS, 200)
    fit_njit = k_njit * fit_xs + b_njit
    fit_compile = k_compile * fit_xs

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(op_counts, times_compile, s=12, alpha=0.5, label="torch.compile (data)")
    ax.scatter(op_counts, times_njit, s=12, alpha=0.5, label="njit (data)")
    ax.plot(
        fit_xs,
        fit_compile,
        "-",
        linewidth=2,
        label=f"compile fit: y = {k_compile:.2f}x",
    )
    ax.plot(
        fit_xs,
        fit_njit,
        "-",
        linewidth=2,
        label=f"njit fit: y = {k_njit:.2f}x + {b_njit:.2f}",
    )
    ax.set_xlabel("Number of ops (fused into inductor graph)")
    ax.set_ylabel("Overhead (µs/call)")
    ax.set_title("CPU Overhead: NjitInductorGraph vs torch.compile")
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
            f"| {n_ops} | {times_njit[i]:.2f} | {times_compile[i]:.2f} |"
        )
    md_table = "\n".join(md_table_rows)

    env_table_rows = "\n".join(f"| {k} | {v} |" for k, v in env_info.items())

    readme = f"""\
# Inductor vs njit: CPU Graph-Call Overhead

Measures wall-clock time of running an inductor-compiled graph through
`NjitInductorGraph` (the njit wrapper) vs the standard `torch.compile`
Python wrapper.  All tensors are tiny (4×4) so GPU compute is negligible —
only the CPU orchestration overhead is measured (buffer allocation, grid
computation, kernel launches).  No `cudaDeviceSynchronize` is called.

The independent variable is the number of fused pointwise ops per graph
(1 to 64 chained `torch.relu` calls after an initial `x + y`).

## Results

![overhead_vs_ops]({plot_name})

### Linear fit (outliers removed, 2σ threshold)

|         | model     | k (µs/op) | b (µs)  |
|---------|-----------|-----------|---------|
| njit    | y = kx+b  | {k_njit:.4f}    | {b_njit:.4f}  |
| compile | y = kx    | {k_compile:.4f}    | 0 (forced) |

- **k** (slope) is the **per-op cost** — the marginal time (in µs) added by
  each additional fused op in the inductor graph.  For `torch.compile` this
  includes the Python wrapper overhead; for njit this is the compiled
  orchestration cost.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs)
  for entering and leaving the function.  For njit, this captures the Numba
  dispatcher + tensor borrow/wrap cost.

### Raw data

| Ops | njit (µs) | compile (µs) |
|-----|-----------|--------------|
{md_table}

> {ITERS} iterations per data point, {WARMUP} warmup iterations.

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
