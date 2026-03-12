# Combined example: torch ops + Triton kernel

Mix regular torch operations with custom Triton kernels inside a single
`@numba.njit` function — zero Python overhead for the entire pipeline.

```python
import numba
import torch
import triton
import triton.language as tl
from njit_wrappers import NumbaTritonKernel


@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x), mask=mask)


numba_silu = NumbaTritonKernel(
    silu_kernel,
    signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
    constexprs={"BLOCK_SIZE": 1024},
)
launch_silu = numba_silu.launch


@numba.njit
def linear_silu(x, w, b, out, stream):
    h = x @ w + b
    n = h.numel()
    grid = (n + 1023) // 1024
    launch_silu(grid, 1, 1, stream, h, out, n)


n, d_in, d_out = 4, 8, 16
torch.manual_seed(0)
x = torch.randn(n, d_in, device="cuda", dtype=torch.float32)
w = torch.randn(d_in, d_out, device="cuda", dtype=torch.float32)
b = torch.zeros(d_out, device="cuda", dtype=torch.float32)
out = torch.empty(n, d_out, device="cuda", dtype=torch.float32)
stream = torch.cuda.current_stream().cuda_stream

linear_silu(x, w, b, out, stream)
torch.cuda.synchronize()

expected = torch.nn.functional.silu(x @ w + b)
assert torch.allclose(out, expected, atol=1e-5)
```

The `linear_silu` function performs a matrix multiply + bias (`x @ w + b`)
using torch ops, then applies a SiLU activation via a custom Triton kernel.
Both steps run with no Python interpreter involvement.
