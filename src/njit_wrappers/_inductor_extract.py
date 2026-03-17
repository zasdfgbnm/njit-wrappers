"""AST-based parser for torch-inductor generated GPU wrapper code.

Parses the Python source that inductor produces (containing Triton kernel
definitions and a ``call(args)`` wrapper function) into an
``InductorSchedule`` — a flat list of operations that can be translated
into an ``@numba.njit`` function.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Schedule operation dataclasses
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


@dataclass
class AllocOp:
    """Allocate a buffer via ``empty_strided_cuda(shape, stride, dtype)``."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class KernelLaunchOp:
    """Launch a Triton kernel."""

    kernel_name: str
    args: list[str | int | float]
    grid: list[str | int]
    stream_var: str


@dataclass
class ReinterpretOp:
    """Reinterpret a tensor with new shape/stride/offset."""

    name: str
    src: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int


@dataclass
class FreeOp:
    """Free intermediate buffers (``del`` statement)."""

    names: list[str]


@dataclass
class ReturnOp:
    """Return output tensors."""

    names: list[str]


@dataclass
class AliasOp:
    """Alias one buffer to another (``buf5 = buf2``, buffer reuse)."""

    name: str
    src: str


@dataclass
class ExternKernelOp:
    """Call an external kernel (e.g. ``extern_kernels.mm``)."""

    name: str  # output variable name (empty string if no assignment)
    op: str  # e.g. "mm", "addmm"
    args: list[str]


@dataclass
class InductorSchedule:
    """Parsed representation of an inductor-generated wrapper."""

    input_names: list[str]
    ops: list[
        AllocOp
        | KernelLaunchOp
        | ReinterpretOp
        | FreeOp
        | ReturnOp
        | AliasOp
        | ExternKernelOp
    ]
    kernel_sources: dict[str, str] = field(default_factory=dict)
    triton_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _const_value(node: ast.expr) -> Any:
    """Extract a Python literal from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_value(node.operand)
        if isinstance(v, (int, float)):
            return -v
    if isinstance(node, ast.Attribute):
        # e.g. torch.float32
        parts: list[str] = []
        n: ast.expr = node
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(_const_value(e) for e in node.elts)
    return None


def _resolve_dtype(val: Any) -> torch.dtype:
    """Convert a string or torch.dtype to torch.dtype."""
    if isinstance(val, torch.dtype):
        return val
    if isinstance(val, str) and val in _DTYPE_MAP:
        return _DTYPE_MAP[val]
    raise ValueError(f"Cannot resolve dtype: {val!r}")


def _get_dotted_name(node: ast.expr) -> str | None:
    """Get dotted attribute name, e.g. 'extern_kernels.mm'."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


# ---------------------------------------------------------------------------
# Module-level extraction
# ---------------------------------------------------------------------------


def _extract_kernel_sources(tree: ast.Module) -> dict[str, str]:
    """Extract Triton kernel source strings from async_compile.triton() calls."""
    sources: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func_name = _get_dotted_name(call.func)
        if func_name == "async_compile.triton":
            # async_compile.triton('name', source_string, ...)
            if len(call.args) >= 2:
                src_val = _const_value(call.args[1])
                if isinstance(src_val, str):
                    sources[name] = src_val
    return sources


def _extract_triton_meta(source_code: str) -> dict[str, dict[str, Any]]:
    """Extract triton_meta dicts from the source using regex.

    Inductor generates lines like:
        triton_meta0 = {signature: {0: '*fp32', ...}, ...}
    We parse these as Python dict literals.
    """
    meta: dict[str, dict[str, Any]] = {}
    # Match triton_meta assignments in the kernel source strings
    pattern = re.compile(r"triton_meta\d*\s*=\s*\{")
    for m in pattern.finditer(source_code):
        # Find the start of the dict
        start = m.start()
        # Find the variable name
        line_start = source_code.rfind("\n", 0, start) + 1
        line = source_code[line_start : source_code.find("\n", start)]
        eq_pos = line.find("=")
        if eq_pos == -1:
            continue
        var_name = line[:eq_pos].strip()

        # Find matching closing brace
        brace_start = m.end() - 1
        depth = 0
        pos = brace_start
        while pos < len(source_code):
            ch = source_code[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            elif ch in ("'", '"'):
                # Skip strings
                quote = ch
                pos += 1
                while pos < len(source_code) and source_code[pos] != quote:
                    if source_code[pos] == "\\":
                        pos += 1
                    pos += 1
            pos += 1

        dict_str = source_code[brace_start : pos + 1]
        try:
            # Replace device() calls and common patterns for eval
            dict_str_clean = dict_str.replace("device(", "dict(type=")
            parsed = eval(dict_str_clean, {"__builtins__": {}})  # noqa: S307
            if isinstance(parsed, dict):
                meta[var_name] = parsed
        except Exception:
            pass
    return meta


# ---------------------------------------------------------------------------
# call() function body parsing
# ---------------------------------------------------------------------------


def _parse_call_function(  # noqa: C901
    func_def: ast.FunctionDef,
) -> tuple[
    list[str],
    list[
        AllocOp
        | KernelLaunchOp
        | ReinterpretOp
        | FreeOp
        | ReturnOp
        | AliasOp
        | ExternKernelOp
    ],
]:
    """Parse the body of an inductor ``call(args)`` function."""
    input_names: list[str] = []
    ops: list[
        AllocOp
        | KernelLaunchOp
        | ReinterpretOp
        | FreeOp
        | ReturnOp
        | AliasOp
        | ExternKernelOp
    ] = []

    for stmt in func_def.body:
        # Input unpacking: arg0, arg1, ... = args
        # or: primals_1, primals_2 = args  (inductor naming)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Name):
                if stmt.value.id == "args":
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            input_names.append(elt.id)
                    continue

        # Also handle: assert_size_stride, check patterns — skip
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fname = _get_dotted_name(stmt.value.func)
            if fname and (
                "assert" in fname
                or fname == "torch._check"
                or fname.startswith("assert_size_stride")
            ):
                continue

        # Assignment statements
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                name = target.id
                call = stmt.value
                func_name = _get_dotted_name(call.func)

                # empty_strided_cuda(shape, stride, dtype)
                if func_name in (
                    "empty_strided_cuda",
                    "empty_strided",
                    "torch.empty_strided",
                ):
                    if len(call.args) >= 3:
                        shape = _const_value(call.args[0])
                        stride = _const_value(call.args[1])
                        dtype = _resolve_dtype(_const_value(call.args[2]))
                        if isinstance(shape, tuple) and isinstance(stride, tuple):
                            ops.append(AllocOp(name, shape, stride, dtype))
                    continue

                # reinterpret_tensor(src, shape, stride, offset)
                if func_name == "reinterpret_tensor":
                    if len(call.args) >= 4:
                        src = _const_value(call.args[0])
                        shape = _const_value(call.args[1])
                        stride = _const_value(call.args[2])
                        offset = _const_value(call.args[3])
                        if isinstance(src, str) and isinstance(shape, tuple):
                            stride = stride if isinstance(stride, tuple) else ()
                            offset = offset if isinstance(offset, int) else 0
                            ops.append(ReinterpretOp(name, src, shape, stride, offset))
                    continue

                # extern_kernels.mm(a, b, out=c)
                if func_name and func_name.startswith("extern_kernels."):
                    op_name = func_name.split(".")[-1]
                    args = [
                        _const_value(a) for a in call.args if isinstance(a, ast.expr)
                    ]
                    str_args = [str(a) for a in args if a is not None]
                    # Check for out= kwarg
                    for kw in call.keywords:
                        if kw.arg == "out":
                            v = _const_value(kw.value)
                            if v is not None:
                                str_args.append(str(v))
                    ops.append(ExternKernelOp(name, op_name, str_args))
                    continue

            # Skip stream assignment: s0 = ... or with ...
            if isinstance(target, ast.Name) and target.id.startswith("s"):
                continue

            # Simple alias: buf5 = buf2  (buffer reuse)
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Name):
                ops.append(AliasOp(target.id, stmt.value.id))
                continue

        # Kernel launch via .run(...) — most common inductor pattern
        # e.g.: triton_poi_fused_0.run(buf0, buf1, 1024,
        #        grid=grid(1024), stream=stream0)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func_name = _get_dotted_name(call.func)
            if func_name and ".run" in func_name:
                kernel_name = func_name.replace(".run", "")
                args: list[str | int | float] = []
                grid: list[str | int] = []
                stream_var = "stream0"
                for a in call.args:
                    v = _const_value(a)
                    if v is not None:
                        args.append(v)
                for kw in call.keywords:
                    if kw.arg == "grid":
                        gv = _const_value(kw.value)
                        if isinstance(gv, tuple):
                            grid = list(gv)
                        elif isinstance(gv, (int, str)):
                            grid = [gv]
                        # grid=grid(N) — call expression
                        elif isinstance(kw.value, ast.Call):
                            for ga in kw.value.args:
                                gval = _const_value(ga)
                                if gval is not None:
                                    grid.append(gval)
                    elif kw.arg == "stream":
                        sv = _const_value(kw.value)
                        if isinstance(sv, str):
                            stream_var = sv
                ops.append(KernelLaunchOp(kernel_name, args, grid, stream_var))
                continue

            # extern_kernels.mm(...) as expression (no assignment)
            if func_name and func_name.startswith("extern_kernels."):
                op_name = func_name.split(".")[-1]
                ext_args = [
                    _const_value(a) for a in call.args if isinstance(a, ast.expr)
                ]
                str_args = [str(a) for a in ext_args if a is not None]
                for kw in call.keywords:
                    if kw.arg == "out":
                        v = _const_value(kw.value)
                        if v is not None:
                            str_args.append(str(v))
                ops.append(ExternKernelOp("", op_name, str_args))
                continue

        # del statements
        if isinstance(stmt, ast.Delete):
            names: list[str] = []
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
            if names:
                ops.append(FreeOp(names))
            continue

        # return statement
        if isinstance(stmt, ast.Return):
            ret_names: list[str] = []
            if stmt.value is not None:
                if isinstance(stmt.value, ast.Tuple):
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            ret_names.append(elt.id)
                elif isinstance(stmt.value, ast.Name):
                    ret_names.append(stmt.value.id)
                elif isinstance(stmt.value, ast.List):
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            ret_names.append(elt.id)
            ops.append(ReturnOp(ret_names))
            continue

        # with torch.cuda._DeviceGuard(0): — enter the body
        if isinstance(stmt, ast.With):
            inner_inputs, inner_ops = _parse_call_function(
                ast.FunctionDef(
                    name="__inner__",
                    args=func_def.args,
                    body=stmt.body,
                    decorator_list=[],
                    returns=None,
                    lineno=0,
                    col_offset=0,
                )
            )
            if inner_inputs and not input_names:
                input_names = inner_inputs
            ops.extend(inner_ops)
            continue

    return input_names, ops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_inductor_source(source_code: str) -> InductorSchedule:
    """Parse inductor-generated Python source into an ``InductorSchedule``.

    Parameters
    ----------
    source_code : str
        The full Python source code generated by torch.inductor.

    Returns
    -------
    InductorSchedule
        A parsed schedule of operations.
    """
    tree = ast.parse(source_code)
    kernel_sources = _extract_kernel_sources(tree)
    triton_meta = _extract_triton_meta(source_code)

    # Find the call() function
    call_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            call_func = node
            break

    if call_func is None:
        raise ValueError("No 'call' function found in inductor source")

    input_names, ops = _parse_call_function(call_func)

    return InductorSchedule(
        input_names=input_names,
        ops=ops,
        kernel_sources=kernel_sources,
        triton_meta=triton_meta,
    )
