"""Numba nopython support for torch.Tensor.

Importing this module registers torch.Tensor as a Numba type so that
@numba.njit functions can accept and return tensors.  Operations on
tensors inside the compiled function lower directly to ATen C++ functions
via stable C-named wrappers – no hard-coded C++ mangled names anywhere.

Lifetime model
--------------
Internally a tensor is represented as an int64 holding a TensorImpl*
**with an owned reference** (refcount bumped by 1).

- Unboxing  : njit_extract_impl() increments the TensorImpl refcount.
              Numba's NativeValue cleanup calls njit_release_impl() when
              the value is no longer needed.
- Boxing    : njit_wrap_impl() steals the owned reference into a fresh
              Python torch.Tensor object.
- ATen ops  : called through njit_aten_<name>() C wrappers compiled into
              the _bridge extension.  Symbol names are plain C identifiers,
              so there is nothing to mangle or hard-code.

Known limitation
----------------
Intermediate tensor values produced inside an njit function that are
*not* the final return value will leak their TensorImpl refcount.
This will be addressed in a future iteration.
"""

import ctypes
import operator

import llvmlite.binding as llvm
import torch
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
    NativeValue,
    box,
    intrinsic,
    overload,
    register_model,
    typeof_impl,
    unbox,
)

# ---------------------------------------------------------------------------
# Load the C bridge extension and register all needed symbols with LLVM JIT
# ---------------------------------------------------------------------------
import njit_wrappers._bridge as _bridge_module  # noqa: F401 – side-effect load

_bridge_lib = ctypes.CDLL(_bridge_module.__file__)

_bridge_lib.njit_extract_impl.restype = ctypes.c_int64
_bridge_lib.njit_extract_impl.argtypes = [ctypes.py_object]

_bridge_lib.njit_release_impl.restype = None
_bridge_lib.njit_release_impl.argtypes = [ctypes.c_int64]

_bridge_lib.njit_wrap_impl.restype = ctypes.py_object
_bridge_lib.njit_wrap_impl.argtypes = [ctypes.c_int64]


def _sym_addr(func) -> int:
    addr = ctypes.cast(func, ctypes.c_void_p).value
    assert addr is not None, f"could not resolve address for {func}"
    return addr


llvm.add_symbol("njit_extract_impl", _sym_addr(_bridge_lib.njit_extract_impl))
llvm.add_symbol("njit_release_impl", _sym_addr(_bridge_lib.njit_release_impl))
llvm.add_symbol("njit_wrap_impl", _sym_addr(_bridge_lib.njit_wrap_impl))

# ---------------------------------------------------------------------------
# Register ATen wrapper symbols (stable C names – no C++ mangling required).
#
# The _bridge extension exports njit_aten_<name>(i64[, i64]) -> i64 for every
# supported operation.  We look up each symbol's address once at import time
# and register it with LLVM so generated IR can reference it directly.
# ---------------------------------------------------------------------------

_BINARY_OPS: list[str] = [
    "add", "sub", "mul", "div", "pow",
    "matmul", "mm",
    "eq", "ne", "lt", "le", "gt", "ge",
]
_UNARY_OPS: list[str] = [
    "neg", "abs", "exp", "log", "sqrt", "sin", "cos", "tan",
    "relu", "sigmoid", "tanh", "silu",
    "sum", "mean",
]

for _op in _BINARY_OPS:
    _sym = f"njit_aten_{_op}"
    _fn = getattr(_bridge_lib, _sym)
    _fn.restype = ctypes.c_int64
    _fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    llvm.add_symbol(_sym, _sym_addr(_fn))

for _op in _UNARY_OPS:
    _sym = f"njit_aten_{_op}"
    _fn = getattr(_bridge_lib, _sym)
    _fn.restype = ctypes.c_int64
    _fn.argtypes = [ctypes.c_int64]
    llvm.add_symbol(_sym, _sym_addr(_fn))

# ---------------------------------------------------------------------------
# Numba type
# ---------------------------------------------------------------------------


class TensorType(types.Type):
    def __init__(self):
        super().__init__(name="TorchTensor")


tensor_type = TensorType()


@typeof_impl.register(torch.Tensor)
def typeof_tensor(val, c):
    return tensor_type


# ---------------------------------------------------------------------------
# Data model: int64 holding TensorImpl* (owned ref)
# ---------------------------------------------------------------------------


@register_model(TensorType)
class TensorModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(64))


# ---------------------------------------------------------------------------
# Unboxing: Python torch.Tensor -> int64 (TensorImpl*, owned ref)
# ---------------------------------------------------------------------------


@unbox(TensorType)
def unbox_tensor(typ, obj, c):
    i64 = ir.IntType(64)

    extract_fn = cgutils.get_or_insert_function(
        c.builder.module,
        ir.FunctionType(i64, [ir.IntType(8).as_pointer()]),
        "njit_extract_impl",
    )
    impl = c.builder.call(extract_fn, [obj])

    release_fn = cgutils.get_or_insert_function(
        c.builder.module,
        ir.FunctionType(ir.VoidType(), [i64]),
        "njit_release_impl",
    )

    def cleanup():
        c.builder.call(release_fn, [impl])

    return NativeValue(impl, cleanup=cleanup)


# ---------------------------------------------------------------------------
# Boxing: int64 (TensorImpl*, owned ref) -> Python torch.Tensor
# ---------------------------------------------------------------------------


@box(TensorType)
def box_tensor(typ, val, c):
    i64 = ir.IntType(64)
    i8p = ir.IntType(8).as_pointer()

    wrap_fn = cgutils.get_or_insert_function(
        c.builder.module,
        ir.FunctionType(i8p, [i64]),
        "njit_wrap_impl",
    )
    return c.builder.call(wrap_fn, [val])


# ---------------------------------------------------------------------------
# Intrinsic factories
#
# Rather than copy-pasting one @intrinsic per operation, these factories
# capture the symbol name in a closure.  The generated LLVM IR is a simple
# i64 -> i64 (unary) or (i64, i64) -> i64 (binary) call – far simpler than
# the original sret / c10::Scalar boilerplate that was needed when calling
# C++ ATen symbols directly.
# ---------------------------------------------------------------------------


def _make_binary_intrinsic(sym_name: str):
    """Return a Numba @intrinsic that calls njit_aten_<op>(i64, i64) -> i64."""

    @intrinsic
    def _op(typingctx, a, b):
        if not (isinstance(a, TensorType) and isinstance(b, TensorType)):
            return None
        sig = tensor_type(tensor_type, tensor_type)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            fn_type = ir.FunctionType(i64, [i64, i64])
            fn = cgutils.get_or_insert_function(builder.module, fn_type, sym_name)
            return builder.call(fn, args)

        return sig, codegen

    return _op


def _make_unary_intrinsic(sym_name: str):
    """Return a Numba @intrinsic that calls njit_aten_<op>(i64) -> i64."""

    @intrinsic
    def _op(typingctx, a):
        if not isinstance(a, TensorType):
            return None
        sig = tensor_type(tensor_type)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            fn_type = ir.FunctionType(i64, [i64])
            fn = cgutils.get_or_insert_function(builder.module, fn_type, sym_name)
            return builder.call(fn, args)

        return sig, codegen

    return _op


# ---------------------------------------------------------------------------
# Intrinsic instances (one per ATen op)
# ---------------------------------------------------------------------------

_tensor_add = _make_binary_intrinsic("njit_aten_add")
_tensor_sub = _make_binary_intrinsic("njit_aten_sub")
_tensor_mul = _make_binary_intrinsic("njit_aten_mul")
_tensor_div = _make_binary_intrinsic("njit_aten_div")
_tensor_pow = _make_binary_intrinsic("njit_aten_pow")
_tensor_matmul = _make_binary_intrinsic("njit_aten_matmul")
_tensor_mm = _make_binary_intrinsic("njit_aten_mm")

_tensor_eq = _make_binary_intrinsic("njit_aten_eq")
_tensor_ne = _make_binary_intrinsic("njit_aten_ne")
_tensor_lt = _make_binary_intrinsic("njit_aten_lt")
_tensor_le = _make_binary_intrinsic("njit_aten_le")
_tensor_gt = _make_binary_intrinsic("njit_aten_gt")
_tensor_ge = _make_binary_intrinsic("njit_aten_ge")

_tensor_neg = _make_unary_intrinsic("njit_aten_neg")
_tensor_abs = _make_unary_intrinsic("njit_aten_abs")
_tensor_exp = _make_unary_intrinsic("njit_aten_exp")
_tensor_log = _make_unary_intrinsic("njit_aten_log")
_tensor_sqrt = _make_unary_intrinsic("njit_aten_sqrt")
_tensor_sin = _make_unary_intrinsic("njit_aten_sin")
_tensor_cos = _make_unary_intrinsic("njit_aten_cos")
_tensor_tan = _make_unary_intrinsic("njit_aten_tan")
_tensor_relu = _make_unary_intrinsic("njit_aten_relu")
_tensor_sigmoid = _make_unary_intrinsic("njit_aten_sigmoid")
_tensor_tanh = _make_unary_intrinsic("njit_aten_tanh")
_tensor_silu = _make_unary_intrinsic("njit_aten_silu")
_tensor_sum = _make_unary_intrinsic("njit_aten_sum")
_tensor_mean = _make_unary_intrinsic("njit_aten_mean")

# ---------------------------------------------------------------------------
# Operator overloads (Python operators -> intrinsics)
# ---------------------------------------------------------------------------


@overload(operator.add)
def overload_tensor_add(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_add(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.sub)
def overload_tensor_sub(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_sub(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.mul)
def overload_tensor_mul(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_mul(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.truediv)
def overload_tensor_div(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_div(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.pow)
def overload_tensor_pow(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_pow(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.matmul)
def overload_tensor_matmul(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_matmul(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.neg)
def overload_tensor_neg(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_neg(a)  # type: ignore[call-arg]

        return impl


@overload(abs)
def overload_tensor_abs(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_abs(a)  # type: ignore[call-arg]

        return impl


@overload(operator.eq)
def overload_tensor_eq(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_eq(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.ne)
def overload_tensor_ne(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_ne(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.lt)
def overload_tensor_lt(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_lt(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.le)
def overload_tensor_le(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_le(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.gt)
def overload_tensor_gt(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_gt(a, b)  # type: ignore[call-arg]

        return impl


@overload(operator.ge)
def overload_tensor_ge(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):

        def impl(a, b):
            return _tensor_ge(a, b)  # type: ignore[call-arg]

        return impl


# ---------------------------------------------------------------------------
# torch.* function overloads
# ---------------------------------------------------------------------------


@overload(torch.exp)
def overload_torch_exp(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_exp(a)  # type: ignore[call-arg]

        return impl


@overload(torch.log)
def overload_torch_log(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_log(a)  # type: ignore[call-arg]

        return impl


@overload(torch.sqrt)
def overload_torch_sqrt(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_sqrt(a)  # type: ignore[call-arg]

        return impl


@overload(torch.sin)
def overload_torch_sin(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_sin(a)  # type: ignore[call-arg]

        return impl


@overload(torch.cos)
def overload_torch_cos(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_cos(a)  # type: ignore[call-arg]

        return impl


@overload(torch.tan)
def overload_torch_tan(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_tan(a)  # type: ignore[call-arg]

        return impl


@overload(torch.abs)
def overload_torch_abs(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_abs(a)  # type: ignore[call-arg]

        return impl


@overload(torch.relu)
def overload_torch_relu(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_relu(a)  # type: ignore[call-arg]

        return impl


@overload(torch.sigmoid)
def overload_torch_sigmoid(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_sigmoid(a)  # type: ignore[call-arg]

        return impl


@overload(torch.tanh)
def overload_torch_tanh(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_tanh(a)  # type: ignore[call-arg]

        return impl


@overload(torch.nn.functional.silu)
def overload_torch_silu(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_silu(a)  # type: ignore[call-arg]

        return impl


@overload(torch.sum)
def overload_torch_sum(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_sum(a)  # type: ignore[call-arg]

        return impl


@overload(torch.mean)
def overload_torch_mean(a):
    if isinstance(a, TensorType):

        def impl(a):
            return _tensor_mean(a)  # type: ignore[call-arg]

        return impl
