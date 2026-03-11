# njit-wrappers

Use `torch.Tensor` inside `@numba.njit` functions.

## Installation

```bash
pip install njit-wrappers
```

## Example

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

Under the hood, `a + b` compiles to a direct call to
`at::add(a, b)` — no Python overhead inside the compiled function.
