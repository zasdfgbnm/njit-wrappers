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

Under the hood, `a + b` compiles to a direct call to
`at::add(a, b)` — no Python overhead inside the compiled function.

## Two-layer MLP example

The following example shows how **multiple ops** compose inside a
single compiled function.  The entire forward pass — two matrix
multiplications, two additions, and a ReLU activation — is lowered to
five consecutive ATen calls with no interpreter re-entry.

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

| Expression    | ATen op           |
|---------------|-------------------|
| `a + b`       | `at::add`         |
| `a - b`       | `at::sub`         |
| `a * b`       | `at::mul`         |
| `a / b`       | `at::div`         |
| `a @ b`       | `at::matmul`      |
| `-a`          | `at::neg`         |
| `abs(a)`      | `at::abs`         |

### Comparison operators (return bool tensor)

`a == b`, `a != b`, `a < b`, `a <= b`, `a > b`, `a >= b`

### `torch.*` functions

| Call                   | ATen op            |
|------------------------|--------------------|
| `torch.exp(a)`         | `at::exp`          |
| `torch.log(a)`         | `at::log`          |
| `torch.sqrt(a)`        | `at::sqrt`         |
| `torch.sin(a)`         | `at::sin`          |
| `torch.cos(a)`         | `at::cos`          |
| `torch.tan(a)`         | `at::tan`          |
| `torch.abs(a)`         | `at::abs`          |
| `torch.relu(a)`        | `at::relu`         |
| `torch.sigmoid(a)`     | `at::sigmoid`      |
| `torch.tanh(a)`        | `at::tanh`         |
| `torch.nn.functional.silu(a)` | `at::silu` |
| `torch.sum(a)`         | `at::sum`          |
| `torch.mean(a)`        | `at::mean`         |

## How it works

### No hard-coded mangled C++ names, no wrapper call overhead

Naïve approaches either hard-code the mangled C++ symbol name
(`_ZN2at4_ops10add_Tensor4call…`) in Python source — brittle across
PyTorch versions — or introduce a C wrapper function that the JIT code
calls instead of ATen directly, adding an extra call per tensor op.

`njit-wrappers` does neither.  `_bridge.so` exposes a tiny **address
getter** per op:

```cpp
// Called ONCE at Python import time, never again.
extern "C" int64_t njit_addr_relu() {
    return (int64_t)(void*)&at::_ops::relu::call;
}
```

At import time, Python calls each getter, receives the real ATen
function address, and registers it with LLVM under a short stable name:

```python
addr = _bridge_lib.njit_addr_relu()      # resolve once
llvm.add_symbol("_aten_relu", addr)      # LLVM sees the real address
```

JIT-compiled LLVM IR then calls `_aten_relu` directly.  That name
resolves to `at::_ops::relu::call` — the real ATen dispatch entry — with
**zero extra function calls** at runtime.  The C++ compiler in the
build step handles all name mangling; the Python layer never touches a
mangled name.

### Tensor representation

Inside compiled functions, a `torch.Tensor` is represented as an
`int64` holding a `TensorImpl*` **with an owned reference**.

- **Unboxing** (Python → compiled): `njit_extract_impl()` increments
  the `TensorImpl` refcount.
- **Boxing** (compiled → Python): `njit_wrap_impl()` steals the owned
  reference into a fresh `torch.Tensor`.
- **ATen ops**: LLVM IR spills each `int64` handle onto an 8-byte stack
  slot and passes its address as `const at::Tensor&` (valid because
  `sizeof(at::Tensor) == 8` and the only field is the `TensorImpl*`).
  The result is returned via the sret convention and read back as an
  `int64` owned reference.

### Known limitation

Intermediate tensors that are computed inside an njit function but
**not** returned will leak their `TensorImpl` refcount.  This will be
addressed in a future iteration.
