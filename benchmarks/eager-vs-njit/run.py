"""Benchmark CPU-side overhead of njit-wrapped ATen calls on CUDA tensors.

Measures wall-clock time of launching GPU ops through njit vs plain Python.
All tensors are tiny (size 4) so GPU compute is negligible — what we measure
is the CPU dispatch overhead.  No cudaDeviceSynchronize is called.

Produces a table and a plot (saved to benchmarks/eager-vs-njit/overhead_vs_ops.png)
showing overhead (µs/call) as a function of the number of ops for both njit and
eager.

Usage:
    python benchmarks/eager-vs-njit/run.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import torch  # noqa: E402

import njit_wrappers  # noqa: F401, E402 – registers TensorType

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Computation graphs at varying op counts
#
# Each function builds on the previous one, adding more ops.
# The op counts correspond to cumulative stages:
#
#   3 ops:  layer 1  — relu(x @ w1 + b1)
#   6 ops:  + layer 2 — sigmoid(h @ w2 + b2)
#  10 ops:  + trig    — sin(h) * cos(h) + tan(h)
#  14 ops:  + nonlin  — exp(t) - sqrt(abs(t))
#  17 ops:  + final   — tanh(u / (u + u))
#  20 ops:  + reduce  — sum(o) + mean(o)
# ---------------------------------------------------------------------------


def graph_3(x, w1, b1, w2, b2):
    return torch.relu(x @ w1 + b1)


def graph_6(x, w1, b1, w2, b2):
    h = torch.relu(x @ w1 + b1)
    return torch.sigmoid(h @ w2 + b2)


def graph_10(x, w1, b1, w2, b2):
    h = torch.relu(x @ w1 + b1)
    h = torch.sigmoid(h @ w2 + b2)
    return torch.sin(h) * torch.cos(h) + torch.tan(h)


def graph_14(x, w1, b1, w2, b2):
    h = torch.relu(x @ w1 + b1)
    h = torch.sigmoid(h @ w2 + b2)
    t = torch.sin(h) * torch.cos(h) + torch.tan(h)
    return torch.exp(t) - torch.sqrt(torch.abs(t))


def graph_17(x, w1, b1, w2, b2):
    h = torch.relu(x @ w1 + b1)
    h = torch.sigmoid(h @ w2 + b2)
    t = torch.sin(h) * torch.cos(h) + torch.tan(h)
    u = torch.exp(t) - torch.sqrt(torch.abs(t))
    return torch.tanh(u / (u + u))


def graph_20(x, w1, b1, w2, b2):
    h = torch.relu(x @ w1 + b1)
    h = torch.sigmoid(h @ w2 + b2)
    t = torch.sin(h) * torch.cos(h) + torch.tan(h)
    u = torch.exp(t) - torch.sqrt(torch.abs(t))
    o = torch.tanh(u / (u + u))
    return torch.sum(o) + torch.mean(o)


GRAPHS = [
    (3, graph_3),
    (6, graph_6),
    (10, graph_10),
    (14, graph_14),
    (17, graph_17),
    (20, graph_20),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARMUP = 50
ITERS = 1000


def make_inputs(device: torch.device):
    """Create tiny tensors for the benchmark graph."""
    torch.manual_seed(0)
    x = torch.randn(4, 4, device=device)
    w1 = torch.randn(4, 4, device=device)
    b1 = torch.randn(4, device=device)
    w2 = torch.randn(4, 4, device=device)
    b2 = torch.randn(4, device=device)
    return x, w1, b1, w2, b2


def bench(fn, args, warmup=WARMUP, iters=ITERS):
    """Time *iters* calls of fn(*args), return per-call time in microseconds."""
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e6  # → µs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = torch.device("cuda")
    args = make_inputs(device)

    op_counts = []
    times_njit = []
    times_eager = []

    for n_ops, graph_fn in GRAPHS:
        graph_fn_njit = numba.njit(graph_fn)
        # Trigger njit compilation (not timed)
        graph_fn_njit(*args)

        t_njit = bench(graph_fn_njit, args)
        t_eager = bench(graph_fn, args)

        op_counts.append(n_ops)
        times_njit.append(t_njit)
        times_eager.append(t_eager)

    # -- Table --
    table_lines = []
    table_lines.append(f"CPU overhead on CUDA tiny tensors (4×4), {ITERS} iterations")
    table_lines.append("")
    table_lines.append(
        f"{'Ops':>4}  {'njit (µs)':>10}  {'eager (µs)':>11}  {'ratio':>6}"
    )
    table_lines.append(
        f"{'----':>4}  {'----------':>10}  {'-----------':>11}  {'------':>6}"
    )
    for i, n_ops in enumerate(op_counts):
        ratio = times_eager[i] / times_njit[i]
        table_lines.append(
            f"{n_ops:4d}  {times_njit[i]:10.2f}  {times_eager[i]:11.2f}  {ratio:6.2f}×"
        )

    table_text = "\n".join(table_lines)
    print(table_text)

    table_path = OUT_DIR / "results.txt"
    table_path.write_text(table_text + "\n")
    print(f"\nTable saved to {table_path}")

    # -- Plot --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(op_counts, times_eager, "o-", label="eager", linewidth=2, markersize=6)
    ax.plot(op_counts, times_njit, "s-", label="njit", linewidth=2, markersize=6)
    ax.set_xlabel("Number of ops")
    ax.set_ylabel("Overhead (µs/call)")
    ax.set_title("CPU Dispatch Overhead: njit vs Eager PyTorch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(op_counts)

    plot_path = OUT_DIR / "overhead_vs_ops.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
