"""Additional overhead experiments: CUDA graphs, TorchScript, ops API.

Usage:
    PYTHONPATH=src python benchmarks/additional_experiments.py
"""

import os
import time

import torch

t = torch.randn(4, 4, device="cuda")
ITERS = 5000
WARMUP = 500


def bench(fn, args, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - start) / iters * 1e6


# ---- CUDA Graphs ----
print("=== CUDA Graphs ===", flush=True)

# 1 relu
si1 = torch.randn(4, 4, device="cuda")
for _ in range(3):
    torch.clamp(si1, min=0)
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    so1 = torch.clamp(si1, min=0)

t_g1 = bench(g1.replay, ())
print(f"  1 relu: {t_g1:.2f} us", flush=True)

# 20 relus
si20 = torch.randn(4, 4, device="cuda")
for _ in range(3):
    x = si20
    for _ in range(20):
        x = torch.clamp(x, min=0)

g20 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g20):
    x = si20
    for _ in range(20):
        x = torch.clamp(x, min=0)
    so20 = x

t_g20 = bench(g20.replay, ())
print(f"  20 relus: {t_g20:.2f} us", flush=True)


# ---- torch.jit.script ----
print("\n=== torch.jit.script ===", flush=True)


@torch.jit.script
def jit_relu_1(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@torch.jit.script
def jit_relu_20(x: torch.Tensor) -> torch.Tensor:
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
    x = torch.relu(x)
    return x


t_j1 = bench(jit_relu_1, (t,))
print(f"  1 relu: {t_j1:.2f} us", flush=True)

t_j20 = bench(jit_relu_20, (t,))
print(f"  20 relus: {t_j20:.2f} us", flush=True)


# ---- torch.ops.aten.relu.default ----
print("\n=== torch.ops.aten.relu.default ===", flush=True)

relu_op = torch.ops.aten.relu.default

t_o1 = bench(relu_op, (t,))
print(f"  1 relu: {t_o1:.2f} us", flush=True)


def ops_relu_20(x):
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    x = relu_op(x)
    return relu_op(x)


t_o20 = bench(ops_relu_20, (t,))
print(f"  20 relus: {t_o20:.2f} us", flush=True)


# ---- torch.clamp eager (for reference) ----
print("\n=== eager (torch.clamp) ===", flush=True)


def eager_1(x):
    return torch.clamp(x, min=0)


def eager_20(x):
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    x = torch.clamp(x, min=0)
    return torch.clamp(x, min=0)


t_e1 = bench(eager_1, (t,))
print(f"  1 relu: {t_e1:.2f} us", flush=True)

t_e20 = bench(eager_20, (t,))
print(f"  20 relus: {t_e20:.2f} us", flush=True)


# ---- Summary ----
print("\n" + "=" * 72, flush=True)
print("SUMMARY (all times in µs)", flush=True)
print("=" * 72, flush=True)
print(f"{'Approach':<40} {'1op':>8} {'20op':>8} {'per_op':>8}", flush=True)
print("-" * 72, flush=True)
for name, v1, v20 in [
    ("CUDA Graph replay", t_g1, t_g20),
    ("torch.jit.script", t_j1, t_j20),
    ("torch.ops.aten.relu.default", t_o1, t_o20),
    ("eager (torch.clamp)", t_e1, t_e20),
]:
    per_op = (v20 - v1) / 19
    print(f"{name:<40} {v1:>7.2f}  {v20:>7.2f}  {per_op:>7.2f}", flush=True)

os._exit(0)
