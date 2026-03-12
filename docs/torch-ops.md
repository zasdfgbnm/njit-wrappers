# Torch op example

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
