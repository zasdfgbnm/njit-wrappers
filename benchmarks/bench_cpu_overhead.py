"""Benchmark CPU-side overhead of njit-wrapped ATen calls on CUDA tensors.

Measures wall-clock time of launching GPU ops through njit vs plain Python.
All tensors are tiny (size 4) so GPU compute is negligible — what we measure
is the CPU dispatch overhead.  No cudaDeviceSynchronize is called.

Usage:
    python benchmarks/bench_cpu_overhead.py
"""

import time

import numba
import torch

import njit_wrappers  # noqa: F401 – registers TensorType

# ---------------------------------------------------------------------------
# Computation graph: a 20-op chain that exercises many op types
#
#   layer 1:  h = relu(x @ w1 + b1)            3 ops
#   layer 2:  h = sigmoid(h @ w2 + b2)         3 ops
#   trig:     t = sin(h) * cos(h) + tan(h)     4 ops
#   nonlin:   u = exp(t) - sqrt(abs(t))         4 ops
#   final:    o = tanh(u / (u + u))             3 ops
#   reduce:   s = sum(o) + mean(o)              3 ops  → 20 ops total
# ---------------------------------------------------------------------------


def graph_eager(x, w1, b1, w2, b2):
    h = torch.relu(x @ w1 + b1)
    h = torch.sigmoid(h @ w2 + b2)
    t = torch.sin(h) * torch.cos(h) + torch.tan(h)
    u = torch.exp(t) - torch.sqrt(torch.abs(t))
    o = torch.tanh(u / (u + u))
    return torch.sum(o) + torch.mean(o)


graph_njit = numba.njit(graph_eager)


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

    # Trigger njit compilation (not timed)
    graph_njit(*args)

    t_njit = bench(graph_njit, args)
    t_eager = bench(graph_eager, args)

    print(f"20-op graph on CUDA tiny tensors (4×4), {ITERS} iterations")
    print(f"  njit:  {t_njit:8.2f} µs/call")
    print(f"  eager: {t_eager:8.2f} µs/call")
    print(f"  ratio: {t_eager / t_njit:8.2f}× (eager / njit)")


if __name__ == "__main__":
    main()
