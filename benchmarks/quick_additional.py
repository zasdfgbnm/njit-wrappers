"""Quick additional experiments: CUDA graphs, ops API.

Usage:
    python benchmarks/quick_additional.py
"""

import os
import time

import torch

ITERS = 5000
WARMUP = 50


def bench(fn, args):
    for _ in range(WARMUP):
        fn(*args)
    start = time.perf_counter()
    for _ in range(ITERS):
        fn(*args)
    return (time.perf_counter() - start) / ITERS * 1e6


t = torch.randn(4, 4, device="cuda")

# ---- CUDA Graph ----
print("=== CUDA Graph ===", flush=True)
si1 = torch.randn(4, 4, device="cuda")
for _ in range(3):
    torch.clamp(si1, min=0)
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    so1 = torch.clamp(si1, min=0)
t_g1 = bench(g1.replay, ())
print(f"  1 relu: {t_g1:.2f} us", flush=True)

si20 = torch.randn(4, 4, device="cuda")
x = si20
for _ in range(3):
    x = si20
    for i in range(20):
        x = torch.clamp(x, min=0)
g20 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g20):
    x = si20
    for i in range(20):
        x = torch.clamp(x, min=0)
    so20 = x
t_g20 = bench(g20.replay, ())
print(f"  20 relus: {t_g20:.2f} us", flush=True)

# ---- torch.ops.aten ----
print("\n=== torch.ops.aten.relu.default ===", flush=True)
relu_op = torch.ops.aten.relu.default
t_o1 = bench(relu_op, (t,))
print(f"  1 relu: {t_o1:.2f} us", flush=True)


def ops_20(x):
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


t_o20 = bench(ops_20, (t,))
print(f"  20 relus: {t_o20:.2f} us", flush=True)

# ---- Eager ----
print("\n=== eager (torch.clamp) ===", flush=True)
t_e1 = bench(lambda x: torch.clamp(x, min=0), (t,))
print(f"  1 relu: {t_e1:.2f} us", flush=True)


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


t_e20 = bench(eager_20, (t,))
print(f"  20 relus: {t_e20:.2f} us", flush=True)

# ---- Summary ----
print("\n" + "=" * 72, flush=True)
header = f"{'Approach':<40} {'1op':>8} {'20op':>8} {'per_op':>8}"
print(header, flush=True)
print("-" * 72, flush=True)
for name, v1, v20 in [
    ("CUDA Graph replay", t_g1, t_g20),
    ("torch.ops.aten.relu.default", t_o1, t_o20),
    ("eager (torch.clamp)", t_e1, t_e20),
]:
    per_op = (v20 - v1) / 19
    line = f"{name:<40} {v1:>7.2f}  {v20:>7.2f}  {per_op:>7.2f}"
    print(line, flush=True)

os._exit(0)
