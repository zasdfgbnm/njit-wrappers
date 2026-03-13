"""Benchmark CPU-side overhead of Triton kernel launches: njit vs eager Python.

Measures wall-clock time of launching Triton kernels through njit vs the
standard Python launch path.  All tensors are tiny (1024 elements) so GPU
compute is negligible — what we measure is the CPU launch overhead.
No cudaDeviceSynchronize is called.

Each benchmark simply calls ``launch_add`` N times, so the only variable
is the number of kernel launches.

Produces a plot (overhead_vs_kernels.png) and a README.md in the same
directory.

Usage:
    python benchmarks/triton-vs-njit/run.py
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
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from njit_wrappers import NumbaTritonKernel  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


BLOCK_SIZE = 1024

numba_add = NumbaTritonKernel(
    add_kernel,
    signature={
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n_elements": "i32",
    },
    constexprs={"BLOCK_SIZE": BLOCK_SIZE},
)
launch_add = numba_add.launch

# ---------------------------------------------------------------------------
# Graph builders — call launch_add N times
# ---------------------------------------------------------------------------

MIN_LAUNCHES = 1
MAX_LAUNCHES = 64


def _make_eager_graph(n_launches):
    """Return a function that eagerly launches add_kernel *n_launches* times."""
    lines = []
    for _ in range(n_launches):
        lines.append("    add_kernel[(grid,)](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)")
    body = "\n".join(lines)
    src = f"def _graph(x, y, out, n, grid):\n{body}"
    ns = {"add_kernel": add_kernel, "BLOCK_SIZE": BLOCK_SIZE}
    exec(src, ns)  # noqa: S102
    return ns["_graph"]


def _make_njit_graph(n_launches):
    """Return an @njit function that calls launch_add *n_launches* times."""
    lines = []
    for _ in range(n_launches):
        lines.append("    launch_add(grid, 1, 1, stream, x, y, out, n)")
    body = "\n".join(lines)
    src = f"def _graph(x, y, out, n, grid, stream):\n{body}"
    ns = {"launch_add": launch_add}
    exec(src, ns)  # noqa: S102
    return numba.njit(ns["_graph"])


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
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = torch.device("cuda")
    torch.manual_seed(0)

    n_elements = 1024
    x = torch.randn(n_elements, device=device, dtype=torch.float32)
    y = torch.randn(n_elements, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    stream = torch.cuda.current_stream().cuda_stream

    launch_counts = []
    times_njit = []
    times_eager = []

    for n in range(MIN_LAUNCHES, MAX_LAUNCHES + 1):
        eager_fn = _make_eager_graph(n)
        njit_fn = _make_njit_graph(n)
        # Trigger njit compilation (not timed)
        njit_fn(x, y, out, n_elements, grid, stream)

        t_njit = bench(njit_fn, (x, y, out, n_elements, grid, stream))
        t_eager = bench(eager_fn, (x, y, out, n_elements, grid))

        launch_counts.append(n)
        times_njit.append(t_njit)
        times_eager.append(t_eager)

        print(f"  launches={n:3d}  njit={t_njit:8.2f} µs  eager={t_eager:8.2f} µs")

    # -- Linear fits (with outlier removal) --
    xs = np.array(launch_counts, dtype=np.float64)
    ys_njit = np.array(times_njit, dtype=np.float64)
    ys_eager = np.array(times_eager, dtype=np.float64)

    k_njit, b_njit = _robust_polyfit(xs, ys_njit)
    k_eager = _robust_fit_through_origin(xs, ys_eager)

    print("\nLinear fit (outliers removed)")
    print(f"  njit:   y = {k_njit:.4f}x + {b_njit:.4f}  (µs)")
    print(f"  eager:  y = {k_eager:.4f}x           (µs, b forced to 0)")

    # -- Console table --
    print(
        f"\nCPU launch overhead on CUDA tiny tensors (1024 elements),"
        f" {ITERS} iterations\n"
    )
    print(f"{'Launches':>8}  {'njit (µs)':>10}  {'eager (µs)':>11}")
    print(f"{'--------':>8}  {'----------':>10}  {'-----------':>11}")
    for i, n in enumerate(launch_counts):
        print(f"{n:8d}  {times_njit[i]:10.2f}  {times_eager[i]:11.2f}")

    # -- Plot --
    plot_name = "overhead_vs_kernels.png"
    plot_path = OUT_DIR / plot_name

    fit_xs = np.linspace(MIN_LAUNCHES, MAX_LAUNCHES, 200)
    fit_njit = k_njit * fit_xs + b_njit
    fit_eager = k_eager * fit_xs

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(launch_counts, times_eager, s=12, alpha=0.5, label="eager (data)")
    ax.scatter(launch_counts, times_njit, s=12, alpha=0.5, label="njit (data)")
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
    ax.set_xlabel("Number of kernel launches")
    ax.set_ylabel("Overhead (µs/call)")
    ax.set_title("CPU Launch Overhead: njit vs Eager Triton")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # -- README.md --
    readme_path = OUT_DIR / "README.md"
    env_info = _collect_env()

    md_table_rows = []
    for i, n in enumerate(launch_counts):
        md_table_rows.append(f"| {n} | {times_njit[i]:.2f} | {times_eager[i]:.2f} |")
    md_table = "\n".join(md_table_rows)

    env_table_rows = "\n".join(f"| {k} | {v} |" for k, v in env_info.items())

    readme = f"""\
# Triton vs njit: CPU Launch Overhead

Measures wall-clock time of launching Triton kernels through `numba.njit` vs
the standard Python Triton launch path.  All tensors are tiny (1024 elements)
so GPU compute is negligible — only the CPU launch overhead is measured.
No `cudaDeviceSynchronize` is called.

Each benchmark simply calls `add_kernel` N times.

## Results

![overhead_vs_kernels]({plot_name})

### Linear fit (outliers removed, 2σ threshold)

|       | model     | k (µs/launch) | b (µs)  |
|-------|-----------|----------------|---------|
| njit  | y = kx+b  | {k_njit:.4f}         | {b_njit:.4f}  |
| eager | y = kx    | {k_eager:.4f}         | 0 (forced) |

- **k** (slope) is the **per-launch cost** — the marginal time (in µs) added
  by each additional Triton kernel launch.
- **b** (intercept) is the **fixed overhead** — the baseline time (in µs) for
  entering and leaving the function, independent of how many kernels are
  launched.

### Raw data

| Launches | njit (µs) | eager (µs) |
|----------|-----------|------------|
{md_table}

> {ITERS} iterations per data point, {WARMUP} warmup iterations.

## Benchmark environment

| Component | Details |
|-----------|---------|
{env_table_rows}

## Running

```bash
PYTHONPATH=src python benchmarks/triton-vs-njit/run.py
```
"""

    readme_path.write_text(readme)
    print(f"README saved to {readme_path}")


if __name__ == "__main__":
    main()
