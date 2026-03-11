# njit-wrappers

`torch.Tensor` support inside `@numba.njit`.

**User documentation:** https://zasdfgbnm.github.io/njit-wrappers/

## How it works

- `import njit_wrappers` registers `torch.Tensor` as a Numba type.
- Inside a compiled function, tensor operations lower directly to ATen
  C++ symbols — no intermediate Python calls, no wrapper overhead.
- Boxing/unboxing (converting between Python tensors and the compiled
  representation) happens only at the function boundary.

## Installation

```bash
# Install PyTorch first (see https://pytorch.org/get-started)
pip install torch

# Build and install njit-wrappers
pip install .
```

The package contains a C++ extension that is compiled against your
installed PyTorch during `pip install`.

## Toolchain

| Tool | Purpose |
|------|---------|
| [setuptools](https://setuptools.pypa.io/) + [torch.utils.cpp_extension](https://pytorch.org/docs/stable/cpp_extension.html) | Build backend (C++ extension) |
| [ruff](https://docs.astral.sh/ruff/) | Linter & formatter |
| [pytest](https://pytest.org/) | Test framework |

## Development

```bash
# Build the C++ extension in-place
python setup.py build_ext --inplace

# Run tests
PYTHONPATH=src pytest

# Lint & format check
ruff check .
ruff format --check .
```

## Project Structure

```
njit-wrappers/
├── setup.py                    # C++ extension build
├── pyproject.toml              # project metadata & tool config
├── src/
│   └── njit_wrappers/
│       ├── __init__.py         # registers TensorType on import
│       ├── _tensor.py          # Numba type, box/unbox, @intrinsic, overloads
│       └── csrc/
│           └── _bridge.cpp     # C++: extract/release/wrap TensorImpl*
└── tests/
    └── test_tensor.py
```

## Known limitations

- Intermediate tensors created inside an njit function (i.e. results of
  ops that are not the final return value) currently leak their
  refcount.  Functions with a single operation are unaffected.
- Only `operator.add` (`+`) is implemented so far.  More ops will be
  added incrementally.
