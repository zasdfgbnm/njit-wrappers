# Inductor graph support

Replace [torch-inductor](https://pytorch.org/docs/stable/torch.compiler.html)'s
Python wrapper with a single `@numba.njit` function.  Inductor compiles your
model into optimised Triton kernels; `NjitInductorGraph` keeps those kernels
and replaces the Python orchestration layer (buffer allocation, grid
computation, kernel launches) with compiled machine code.

```python
import pytest, torch
if not torch.cuda.is_available():
    pytest.skip("Requires CUDA and triton", allow_module_level=True)

from njit_wrappers import NjitInductorGraph

class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)

model = Model().cuda()
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

graph = NjitInductorGraph(model, (x, y))
out = graph(x, y)
torch.cuda.synchronize()
assert torch.allclose(out, torch.relu(x + y), atol=1e-5)
```

## How it works

1. **Inductor compilation** — the model is compiled through
   `torch.compile(backend='inductor', fullgraph=True)` and the generated
   Python source code is captured.
2. **Source parsing** — the generated source is parsed with `ast` to extract
   buffer allocations, Triton kernel launches, and extern kernel calls into
   a flat schedule of operations.
3. **Kernel wrapping** — each Triton kernel is reconstructed from its source
   and wrapped with `NumbaTritonKernel` for zero-overhead launching.
4. **@njit generation** — a single `@numba.njit` function is generated that
   allocates buffers (via `aoti_torch_empty_strided`), launches kernels, and
   returns the output tensors.

## API

### `NjitInductorGraph(model, example_inputs, *, fullgraph=True)`

**Parameters:**

- `model` — a `torch.nn.Module` or callable
- `example_inputs` — a tuple of CUDA tensors matching the model's input
  signature
- `fullgraph` — must be `True` (graph breaks are not supported)

**Returns:** An object that can be called like the original model.

### Calling the graph

```
out = graph(*inputs)
```

Inputs must be CUDA tensors with the same shapes and dtypes as the
example inputs used during construction.

### Inspecting internals

- `graph.source_code` — the inductor-generated Python source
- `graph.schedule` — the parsed `InductorSchedule` with all operations

## Limitations

- **GPU/CUDA only** — CPU inductor graphs are not supported.
- **`fullgraph=True` required** — graph breaks are not supported.
- **Static shapes only** — input shapes must match those used at
  compile time.  Dynamic/symbolic shapes are not supported.
- **Limited extern kernel support** — only `mm` and `addmm` are mapped
  to ATen intrinsics.  Other extern kernels (e.g. convolutions) will
  raise `NotImplementedError`.
- **`reinterpret_tensor` not supported** — models that generate view
  operations in the inductor wrapper will raise `NotImplementedError`.
- **Intermediate tensor leaks** — tensors allocated inside the `@njit`
  function but not returned will leak their `TensorImpl` refcount
  (same limitation as the base tensor support).
