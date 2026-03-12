# njit-wrappers

Use `torch.Tensor` inside `@numba.njit` functions.

## Installation

```bash
pip install njit-wrappers
```

## Quick start

```python
import numba
import torch
import njit_wrappers

@numba.njit
def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

a = torch.ones(1024)
b = torch.ones(1024)
c = add(a, b)
```

Importing `njit_wrappers` is all it takes.  After that, any
`@numba.njit` function can accept and return `torch.Tensor` arguments
as if they were native types.

## Two-layer MLP example

The following example shows how multiple ops compose inside a single compiled function.

```python
import numba
import torch
import njit_wrappers

@numba.njit
def mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
) -> torch.Tensor:
    """Two-layer MLP: relu(x @ w1 + b1) @ w2 + b2."""
    h = torch.relu(x @ w1 + b1)
    return h @ w2 + b2

# 8 samples, 16 features → 32 hidden → 8 outputs
torch.manual_seed(0)
x  = torch.randn(8, 16)
w1 = torch.randn(16, 32)
b1 = torch.zeros(32)
w2 = torch.randn(32, 8)
b2 = torch.zeros(8)

out = mlp_forward(x, w1, b1, w2, b2)
# out.shape == (8, 8)
```

## Supported operations

### Arithmetic operators

`a + b`, `a - b`, `a * b`, `a / b`, `a @ b`, `-a`, `abs(a)`

### Comparison operators (return bool tensor)

`a == b`, `a != b`, `a < b`, `a <= b`, `a > b`, `a >= b`

### `torch.*` functions

`torch.exp`, `torch.log`, `torch.sqrt`, `torch.sin`, `torch.cos`,
`torch.tan`, `torch.abs`, `torch.relu`, `torch.sigmoid`, `torch.tanh`,
`torch.nn.functional.silu`, `torch.sum`, `torch.mean`

## Known limitation

Intermediate tensors that are computed inside an njit function but
**not** returned will leak their `TensorImpl` refcount.

## Triton kernel support

Launch [Triton](https://triton-lang.org/) kernels from `@numba.njit`
with zero Python overhead.  `NumbaTritonKernel` compiles a Triton kernel
for a fixed type signature, generates a thin C trampoline that calls
`cuLaunchKernelEx` directly, and wraps it in an `@numba.njit` function.

```python
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
def f(x_ptr, y_ptr, out_ptr, n, stream):
    grid = (n + 1023) // 1024
    launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)
```

The `launch` function signature is
`launch(gridX, gridY, gridZ, stream, arg0, arg1, ..., argN)`.
Grid dimensions and CUDA stream are explicit; kernel arguments follow
in the same order as the `signature` dict (minus constexprs).

**Note:** The launch function must be extracted into a module-level
variable (e.g. `launch_add = numba_add.launch`) before being used
inside `@numba.njit`.  Numba cannot resolve attribute access on custom
Python objects within compiled code.

### Runtime specialization

For each pointer and integer argument, the launcher checks alignment
(`% 16 == 0`) at call time and dispatches to the appropriate
pre-compiled variant with `tt.divisibility=16` hints.  This enables
vectorized 128-bit loads for aligned data, matching the behavior of
Triton's normal autotuning path.

### Limitations

- NVIDIA only (CUDA driver API)
- No scratch memory support
- Stream must be passed explicitly as `uint64`
