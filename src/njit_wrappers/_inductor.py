"""Replace torch-inductor's Python wrapper with an @numba.njit function.

``NjitInductorGraph`` takes a PyTorch model and example inputs, compiles it
through inductor to obtain optimised Triton kernels, then replaces the
generated Python orchestration code with a single ``@numba.njit`` function
that calls the same kernels with zero Python overhead.

Usage::

    graph = NjitInductorGraph(model, example_inputs)
    out = graph(*real_inputs)

Limitations:
    - GPU/CUDA only
    - ``fullgraph=True`` required (no graph breaks)
    - Static shapes only
    - Limited extern kernel support (mm, addmm)
    - Intermediate tensor refcount leaks (same as _tensor.py)
    - ``reinterpret_tensor`` raises NotImplementedError
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
from typing import Any

import numba
import torch

from njit_wrappers._inductor_extract import (
    AllocOp,
    ExternKernelOp,
    FreeOp,
    InductorSchedule,
    KernelLaunchOp,
    ReinterpretOp,
    ReturnOp,
    parse_inductor_source,
)
from njit_wrappers._tensor import (
    _make_empty_strided_intrinsic,
    _tensor_data_ptr,
    _tensor_mm,
)
from njit_wrappers._triton import NumbaTritonKernel

# device_type constants used by aoti_torch_empty_strided
_DEVICE_TYPE_CUDA = 1


# ---------------------------------------------------------------------------
# Inductor source capture
# ---------------------------------------------------------------------------


def _get_inductor_source(
    model: torch.nn.Module | Any,
    example_inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> str:
    """Compile *model* through inductor and return the generated source code."""
    source_codes: list[str] = []

    from torch._inductor.graph import GraphLowering

    original_callback = GraphLowering.save_output_code
    try:
        GraphLowering.save_output_code = staticmethod(  # type: ignore[assignment]
            lambda code: source_codes.append(code)
        )

        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        if isinstance(example_inputs, (tuple, list)):
            compiled(*example_inputs)
        else:
            compiled(example_inputs)

    finally:
        GraphLowering.save_output_code = original_callback  # type: ignore[assignment]

    if not source_codes:
        raise RuntimeError(
            "Failed to capture inductor source code. "
            "Ensure the model compiles successfully with "
            "torch.compile(backend='inductor', fullgraph=True)."
        )

    return source_codes[-1]


# ---------------------------------------------------------------------------
# Triton kernel reconstruction from inductor source
# ---------------------------------------------------------------------------


def _load_kernel_from_source(kernel_source, kernel_name):
    """Load a Triton kernel from source by writing to a temp file.

    Returns the heuristic-wrapped kernel object (CachingAutotuner).
    """
    fd, fpath = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(kernel_source)
        spec = importlib.util.spec_from_file_location(
            f"_inductor_kernel_{kernel_name}", fpath
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, kernel_name)
    finally:
        os.unlink(fpath)


def _autotune_and_build_kernel(autotuner, schedule, example_inputs):
    """Run autotuning on a kernel and build a NumbaTritonKernel.

    Parameters
    ----------
    autotuner : CachingAutotuner
        The heuristic-wrapped kernel object.
    schedule : InductorSchedule
        The parsed schedule (for finding launch args).
    example_inputs : tuple of Tensor
        Example inputs to trigger autotuning.

    Returns
    -------
    tuple of (NumbaTritonKernel, grid_info)
    """
    jit_fn = autotuner.fn
    meta = autotuner.triton_meta
    raw_sig = meta.get("signature", {})

    # Trigger autotuning by examining compile_results
    # The configs contain candidate constexpr values
    configs = list(autotuner.configs)
    if not configs:
        raise RuntimeError("No configs available for autotuning")

    # Use the first config (inductor orders them by priority)
    best_config = configs[0]
    constexpr_values = dict(best_config.kwargs)

    # Build signature: name -> type_string
    param_names = jit_fn.arg_names
    signature = {}
    constexprs = {}

    for key, type_str in raw_sig.items():
        if type_str == "constexpr":
            # This param is a constexpr — get its value from config
            if isinstance(key, str):
                name = key
            elif isinstance(key, int) and key < len(param_names):
                name = param_names[key]
            else:
                continue
            if name in constexpr_values:
                constexprs[name] = constexpr_values[name]
                signature[name] = type_str
            continue
        if isinstance(key, str):
            signature[key] = type_str
        elif isinstance(key, int) and key < len(param_names):
            signature[param_names[key]] = type_str

    # Also include any constants from triton_meta
    for key, val in meta.get("constants", {}).items():
        if isinstance(key, int) and key < len(param_names):
            name = param_names[key]
        else:
            name = str(key)
        constexprs[name] = val
        if name not in signature:
            signature[name] = "constexpr"

    numba_kernel = NumbaTritonKernel(jit_fn, signature, constexprs)

    # Compute grid info from size_hints and grid_type
    grid_type = "Grid1D"
    if hasattr(autotuner, "inductor_meta"):
        grid_type = autotuner.inductor_meta.get("grid_type", "Grid1D")

    size_hints = getattr(autotuner, "size_hints", {})
    block_size = constexpr_values.get(
        "XBLOCK",
        constexpr_values.get("BLOCK_SIZE", constexpr_values.get("RBLOCK", 256)),
    )

    grid_info = {
        "grid_type": grid_type,
        "size_hints": size_hints,
        "block_size": block_size,
        "constexprs": constexpr_values,
    }

    return numba_kernel, grid_info


# ---------------------------------------------------------------------------
# NjitInductorGraph
# ---------------------------------------------------------------------------


class NjitInductorGraph:
    """Wrap a torch-inductor compiled graph in an ``@numba.njit`` function.

    Parameters
    ----------
    model : torch.nn.Module or callable
        The model to compile.
    example_inputs : tuple of torch.Tensor
        Example inputs for tracing.  Must be CUDA tensors.
    fullgraph : bool
        Require a single graph with no breaks (default ``True``).
    """

    def __init__(
        self,
        model: torch.nn.Module | Any,
        example_inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        fullgraph: bool = True,
    ) -> None:
        if not fullgraph:
            raise ValueError("NjitInductorGraph requires fullgraph=True")

        if not torch.cuda.is_available():
            raise RuntimeError("NjitInductorGraph requires CUDA")

        # Reset dynamo cache to avoid stale compilations
        torch._dynamo.reset()

        # 1. Compile through inductor, capture source
        source_code = _get_inductor_source(model, example_inputs)
        self._source_code = source_code

        # 2. Parse the source into a schedule
        schedule = parse_inductor_source(source_code)
        self._schedule = schedule

        # 3. Determine which inputs are model parameters vs user inputs
        # Inductor names model params as primals_1..N, with user inputs
        # interspersed. We figure out the mapping by counting:
        # total schedule inputs = len(user_inputs) + len(model_params)
        n_schedule_inputs = len(schedule.input_names)
        n_user_inputs = (
            len(example_inputs) if isinstance(example_inputs, (tuple, list)) else 1
        )
        n_params = n_schedule_inputs - n_user_inputs

        # Collect model parameters in the order inductor expects them
        self._frozen_params: list[torch.Tensor] = []
        if n_params > 0 and isinstance(model, torch.nn.Module):
            params = list(model.parameters()) + list(model.buffers())
            self._frozen_params = params[:n_params]
        self._n_user_inputs = n_user_inputs

        # 4. Build Triton kernels
        self._kernels: dict[str, NumbaTritonKernel] = {}
        self._kernel_launches: dict[str, Any] = {}
        self._grid_info: dict[str, dict[str, Any]] = {}
        self._compiled_objects: list[Any] = []  # prevent GC

        for kname, ksrc in schedule.kernel_sources.items():
            autotuner = _load_kernel_from_source(ksrc, kname)
            numba_kernel, grid_info = _autotune_and_build_kernel(
                autotuner, schedule, example_inputs
            )
            self._kernels[kname] = numba_kernel
            self._kernel_launches[kname] = numba_kernel.launch
            self._grid_info[kname] = grid_info
            self._compiled_objects.append(numba_kernel)

        # 5. Determine CUDA device from example inputs
        device_index = 0
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor) and inp.is_cuda:
                device_index = inp.device.index or 0
                break
        self._device_index = device_index

        # 6. Generate the @njit wrapper
        self._njit_fn = self._build_njit_wrapper(schedule, device_index)

    def _build_njit_wrapper(
        self,
        schedule: InductorSchedule,
        device_index: int,
    ) -> Any:
        """Generate and compile the ``@numba.njit`` wrapper function."""
        namespace: dict[str, Any] = {
            "_data_ptr": _tensor_data_ptr,
            "_mm": _tensor_mm,
            "math": math,
        }

        lines: list[str] = []
        n_inputs = len(schedule.input_names)
        input_args = ", ".join(f"arg{i}" for i in range(n_inputs))
        if input_args:
            sig = f"def _inductor_wrapper(stream, {input_args}):"
        else:
            sig = "def _inductor_wrapper(stream):"
        lines.append(sig)

        # Map input names to arg variables
        var_map: dict[str, str] = {}
        for i, name in enumerate(schedule.input_names):
            var_map[name] = f"arg{i}"

        alloc_counter = 0
        has_body = False

        for op in schedule.ops:
            if isinstance(op, AllocOp):
                # Create a unique intrinsic for this allocation
                intrinsic_name = f"_alloc_{alloc_counter}"
                alloc_counter += 1
                alloc_intrinsic = _make_empty_strided_intrinsic(
                    shape=op.shape,
                    stride=op.stride,
                    dtype=op.dtype,
                    device_type=_DEVICE_TYPE_CUDA,
                    device_index=device_index,
                )
                namespace[intrinsic_name] = alloc_intrinsic
                var_name = f"v_{op.name}"
                var_map[op.name] = var_name
                lines.append(f"    {var_name} = {intrinsic_name}()")
                has_body = True

            elif isinstance(op, KernelLaunchOp):
                launch_name = f"_launch_{op.kernel_name}"
                if op.kernel_name not in self._kernel_launches:
                    continue
                namespace[launch_name] = self._kernel_launches[op.kernel_name]

                grid_info = self._grid_info.get(op.kernel_name, {})

                # Build kernel args (excluding numel args that go
                # into grid computation)
                # The .run() args are: *tensor_args, numel, stream=...
                # For NumbaTritonKernel.launch: gridX, gridY, gridZ,
                # stream, *kernel_args (tensors + scalars, no numel)
                #
                # The signature tells us which args are tensor ptrs
                # vs scalars. The numel arg is typically the last
                # non-constexpr scalar — it's consumed by grid
                # computation in the heuristics wrapper.
                kernel = self._kernels[op.kernel_name]
                # The numba function's arg names:
                # (gridX, gridY, gridZ, stream, arg0, ..., argN)
                kernel_arg_names = list(
                    kernel._njit_launch.py_func.__code__.co_varnames[
                        : kernel._njit_launch.py_func.__code__.co_argcount
                    ]
                )
                # Strip grid/stream prefix: gridX, gridY, gridZ, stream
                kernel_param_names = kernel_arg_names[4:]
                n_kernel_params = len(kernel_param_names)

                # Map .run() args to kernel launch args
                # .run() args = [ptr_args..., numel_args...]
                # kernel params = [ptr_args..., numel_args...]
                # They should line up
                run_args = op.args
                kernel_args = []
                for i, a in enumerate(run_args):
                    if i >= n_kernel_params:
                        break
                    if isinstance(a, str) and a in var_map:
                        kernel_args.append(var_map[a])
                    elif isinstance(a, int):
                        kernel_args.append(str(a))
                    elif isinstance(a, float):
                        kernel_args.append(repr(a))
                    elif isinstance(a, str):
                        kernel_args.append(var_map.get(a, str(a)))
                    else:
                        kernel_args.append(str(a))

                # Compute grid from the numel args and block size
                grid_type = grid_info.get("grid_type", "Grid1D")
                constexprs = grid_info.get("constexprs", {})

                if grid_type == "Grid1D":
                    xblock = constexprs.get("XBLOCK", 256)
                    # Find xnumel from the run args — it's the
                    # first int arg after the tensor args, or the
                    # last positional arg
                    xnumel = None
                    for a in reversed(run_args):
                        if isinstance(a, int):
                            xnumel = a
                            break
                    if xnumel is not None:
                        gridX = str(math.ceil(xnumel / xblock))
                    else:
                        gridX = "1"
                    gridY = "1"
                    gridZ = "1"
                elif grid_type == "Grid2D":
                    xblock = constexprs.get("XBLOCK", 256)
                    yblock = constexprs.get("YBLOCK", 1)
                    # Find xnumel and ynumel
                    int_args = [a for a in run_args if isinstance(a, int)]
                    if len(int_args) >= 2:
                        ynumel = int_args[-2]
                        xnumel = int_args[-1]
                        gridX = str(math.ceil(xnumel / xblock))
                        gridY = str(math.ceil(ynumel / yblock))
                    elif len(int_args) == 1:
                        gridX = str(math.ceil(int_args[0] / xblock))
                        gridY = "1"
                    else:
                        gridX = "1"
                        gridY = "1"
                    gridZ = "1"
                else:
                    # Fallback: single block
                    gridX = "1"
                    gridY = "1"
                    gridZ = "1"

                args_str = ", ".join(kernel_args)
                lines.append(
                    f"    {launch_name}({gridX}, {gridY}, {gridZ}, stream, {args_str})"
                )
                has_body = True

            elif isinstance(op, ExternKernelOp):
                if op.op == "mm":
                    if len(op.args) >= 2:
                        a_var = var_map.get(op.args[0], op.args[0])
                        b_var = var_map.get(op.args[1], op.args[1])
                        if op.name:
                            var_name = f"v_{op.name}"
                            var_map[op.name] = var_name
                            lines.append(f"    {var_name} = _mm({a_var}, {b_var})")
                        else:
                            lines.append(f"    _mm({a_var}, {b_var})")
                        has_body = True
                elif op.op == "addmm":
                    if len(op.args) >= 3:
                        input_var = var_map.get(op.args[1], op.args[1])
                        weight_var = var_map.get(op.args[2], op.args[2])
                        if op.name:
                            var_name = f"v_{op.name}"
                            var_map[op.name] = var_name
                            lines.append(
                                f"    {var_name} = _mm({input_var}, {weight_var})"
                            )
                        has_body = True
                else:
                    raise NotImplementedError(
                        f"Unsupported extern kernel: {op.op}. "
                        f"Only 'mm' and 'addmm' are supported."
                    )

            elif isinstance(op, ReinterpretOp):
                raise NotImplementedError(
                    f"reinterpret_tensor is not yet supported "
                    f"(encountered for {op.name} from {op.src}). "
                    f"This may require an as_strided intrinsic."
                )

            elif isinstance(op, FreeOp):
                # Can't free inside njit (known limitation)
                pass

            elif isinstance(op, ReturnOp):
                if len(op.names) == 1:
                    ret_var = var_map.get(op.names[0], op.names[0])
                    lines.append(f"    return {ret_var}")
                elif len(op.names) > 1:
                    ret_vars = ", ".join(var_map.get(n, n) for n in op.names)
                    lines.append(f"    return ({ret_vars},)")
                has_body = True

        if not has_body:
            lines.append("    pass")

        source = "\n".join(lines)
        exec(source, namespace)  # noqa: S102
        fn = namespace["_inductor_wrapper"]
        return numba.njit(fn)

    def __call__(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the compiled graph.

        Parameters
        ----------
        *args : torch.Tensor
            Input tensors (same shapes/dtypes as at compile time).
            Model parameters are prepended automatically.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
        """
        stream = torch.cuda.current_stream().cuda_stream
        all_args = (*self._frozen_params, *args)
        return self._njit_fn(stream, *all_args)

    @property
    def source_code(self) -> str:
        """The inductor-generated Python source code."""
        return self._source_code

    @property
    def schedule(self) -> InductorSchedule:
        """The parsed inductor schedule."""
        return self._schedule
