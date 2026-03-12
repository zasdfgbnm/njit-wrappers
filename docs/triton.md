# Triton kernel support

Launch [Triton](https://triton-lang.org/) kernels from `@numba.njit`
with zero Python overhead.  `NumbaTritonKernel` compiles a Triton kernel
for a fixed type signature, generates a thin C trampoline that calls
`cuLaunchKernelEx` directly, and wraps it in an `@numba.njit` function.

```python notest
import numba
import torch
import triton
import triton.language as tl
from njit_wrappers import NumbaTritonKernel

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

numba_add = NumbaTritonKernel(
    add_kernel,
    signature={
        'x_ptr': '*fp32', 'y_ptr': '*fp32',
        'out_ptr': '*fp32', 'n_elements': 'i32',
    },
    constexprs={'BLOCK_SIZE': 1024},
)
launch_add = numba_add.launch  # extract the @njit function

@numba.njit
def f(x, y, out, stream):
    n = x.numel()
    grid = (n + 1023) // 1024
    launch_add(grid, 1, 1, stream, x, y, out, n)

n = 1024
x = torch.randn(n, device='cuda', dtype=torch.float32)
y = torch.randn(n, device='cuda', dtype=torch.float32)
out = torch.empty_like(x)
stream = torch.cuda.current_stream().cuda_stream

f(x, y, out, stream)  # pass tensors directly
torch.cuda.synchronize()
assert torch.allclose(out, x + y)
```

The `launch` function signature is
`launch(gridX, gridY, gridZ, stream, arg0, arg1, ..., argN)`.
Grid dimensions and CUDA stream are explicit; kernel arguments follow
in the same order as the `signature` dict (minus constexprs).

Pointer arguments (`*fp32`, `*fp64`, etc.) accept `torch.Tensor`
directly — the device pointer is extracted automatically inside the
compiled function.  Non-pointer arguments (scalars) must be passed as
their corresponding C types.

**Note:** The launch function must be extracted into a module-level
variable (e.g. `launch_add = numba_add.launch`) before being used
inside `@numba.njit`.  Numba cannot resolve attribute access on custom
Python objects within compiled code.

## Runtime specialization

For each pointer and integer argument, the launcher checks alignment
(`% 16 == 0`) at call time and dispatches to the appropriate
pre-compiled variant with `tt.divisibility=16` hints.  This enables
vectorized 128-bit loads for aligned data, matching the behavior of
Triton's normal autotuning path.

## Limitations

- NVIDIA only (CUDA driver API)
- No scratch memory support
- Stream must be passed explicitly as `uint64`
