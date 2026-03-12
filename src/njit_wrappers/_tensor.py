"""Numba nopython support for torch.Tensor.

Importing this module registers torch.Tensor as a Numba type so that
@numba.njit functions can accept and return tensors.  Operations inside
the compiled function lower **directly** to ATen functions with no extra
wrapper call at runtime.

How symbol resolution works
---------------------------
ATen functions exported by libtorch_cpu.so use Itanium C++ name mangling.
We compute the mangled name in Python using the regular structure of ATen
function signatures (all in namespace ``at``, arguments drawn from a small
set of types), then look up the address via ctypes and register it with
LLVM.  No hard-coded mangled strings in the source, no C++ address-getter
helper functions.

Example for ``at::relu(const at::Tensor&)``:

    _mangle_aten("relu", _ARGS_UNARY)
    → "_ZN2at4reluERKNS_6TensorE"

ATen calling conventions (SysV x86-64, all ops return at::Tensor via sret)
---------------------------------------------------------------------------
  UNARY      void(Tensor* sret, const Tensor& self)
  BINARY     void(Tensor* sret, const Tensor& self, const Tensor& other)
  ALPHA      void(Tensor* sret, const Tensor& self, const Tensor& other,
                  const Scalar& alpha)          ← add, sub
  REDUCTION  void(Tensor* sret, const Tensor& self,
                  optional<ScalarType> dtype)   ← i16 by value (trivially
                                                  copyable 2-byte struct)

Tensor representation inside compiled functions
-----------------------------------------------
An ``at::Tensor`` is 8 bytes (just a ``TensorImpl*``).  We store it as
``int64`` holding that pointer with an **owned reference** (refcount ≥ 1).
Allocating an ``i64`` slot on the stack and passing its address as
``const Tensor&`` is valid because ``sizeof(at::Tensor) == 8`` and the
pointer is the first (and only) field.

Known limitation
----------------
Intermediate tensors that are produced inside an njit function but not
returned will leak their TensorImpl refcount.
"""

import ctypes
import operator
from pathlib import Path

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

import njit_wrappers._bridge as _bridge_module  # noqa: F401 – side-effect load

_bridge_lib = ctypes.CDLL(_bridge_module.__file__)

# ---------------------------------------------------------------------------
# Reference-management symbols (bridge extension)
# ---------------------------------------------------------------------------

_bridge_lib.njit_extract_impl.restype = ctypes.c_int64
_bridge_lib.njit_extract_impl.argtypes = [ctypes.py_object]
_bridge_lib.njit_release_impl.restype = None
_bridge_lib.njit_release_impl.argtypes = [ctypes.c_int64]
_bridge_lib.njit_wrap_impl.restype = ctypes.py_object
_bridge_lib.njit_wrap_impl.argtypes = [ctypes.c_int64]


def _fn_addr(lib: ctypes.CDLL, mangled: str) -> int:
    fn = getattr(lib, mangled)
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    assert addr, f"symbol not found: {mangled}"
    return addr


llvm.add_symbol("njit_extract_impl", _fn_addr(_bridge_lib, "njit_extract_impl"))
llvm.add_symbol("njit_release_impl", _fn_addr(_bridge_lib, "njit_release_impl"))
llvm.add_symbol("njit_wrap_impl", _fn_addr(_bridge_lib, "njit_wrap_impl"))

# ---------------------------------------------------------------------------
# ATen symbol resolution via Itanium C++ name mangling computed in Python.
#
# We call at::_ops::{name}::call, which is always exported from
# libtorch_cpu.so (unlike at::{name}, which is an inline wrapper).
#
# Mangled name pattern for at::_ops::{op}::call({args}):
#
#   _ZN  2at  4_ops  {len(op)}{op}  4call  E  {arg_suffix}
#
# Substitution context after encoding "N 2at 4_ops {n}{op} 4call E":
#   S_  = at::
#   S0_ = at::_ops
#   S1_ = at::_ops::{op}   (the generated struct)
# After encoding the first "const at::Tensor &" (RKNS_6TensorE):
#   S2_ = at::Tensor
#   S3_ = const at::Tensor
#   S4_ = const at::Tensor&   ← used as S4_ for second Tensor arg
#
# For reduction ops the second argument is std::optional<c10::ScalarType>
# (c10::optional = std::optional via using-declaration in modern PyTorch).
# St8optional = std::optional, N3c1010ScalarTypeE = c10::ScalarType.
# _bridge.cpp has a static_assert confirming this type is 2 bytes and
# trivially copyable, hence passed as i16 in the LLVM IR.
# ---------------------------------------------------------------------------

_TORCH_LIB = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))

# Argument suffix constants (substitutions as annotated above)
_ARGS_UNARY = "RKNS_6TensorE"  # (const Tensor&)
_ARGS_BINARY = "RKNS_6TensorES4_"  # (Tensor&, Tensor&)
_ARGS_ALPHA = "RKNS_6TensorES4_RKN3c106ScalarE"  # (Tensor&, Tensor&, Scalar&)
_ARGS_REDUCE = (  # (Tensor&, optional<ScalarType>)
    "RKNS_6TensorESt8optionalIN3c1010ScalarTypeEE"
)


def _mangle_aten(op: str, arg_suffix: str) -> str:
    """Compute the Itanium mangled name for at::_ops::{op}::call(...)."""
    return f"_ZN2at4_ops{len(op)}{op}4callE{arg_suffix}"


# (op_name in at::_ops, arg_suffix, llvm_sym, calling_convention)
_ATEN_OPS: list[tuple[str, str, str, str]] = [
    # UNARY:     void(sret Tensor*, const Tensor&)
    ("neg", _ARGS_UNARY, "_aten_neg", "unary"),
    ("abs", _ARGS_UNARY, "_aten_abs", "unary"),
    ("exp", _ARGS_UNARY, "_aten_exp", "unary"),
    ("log", _ARGS_UNARY, "_aten_log", "unary"),
    ("sqrt", _ARGS_UNARY, "_aten_sqrt", "unary"),
    ("sin", _ARGS_UNARY, "_aten_sin", "unary"),
    ("cos", _ARGS_UNARY, "_aten_cos", "unary"),
    ("tan", _ARGS_UNARY, "_aten_tan", "unary"),
    ("relu", _ARGS_UNARY, "_aten_relu", "unary"),
    ("sigmoid", _ARGS_UNARY, "_aten_sigmoid", "unary"),
    ("tanh", _ARGS_UNARY, "_aten_tanh", "unary"),
    ("silu", _ARGS_UNARY, "_aten_silu", "unary"),
    # REDUCTION: void(sret Tensor*, const Tensor&, i16 optional<ScalarType>)
    ("sum", _ARGS_REDUCE, "_aten_sum", "reduction"),
    ("mean", _ARGS_REDUCE, "_aten_mean", "reduction"),
    # ALPHA:     void(sret Tensor*, const Tensor&, const Tensor&, const Scalar&)
    ("add_Tensor", _ARGS_ALPHA, "_aten_add", "alpha"),
    ("sub_Tensor", _ARGS_ALPHA, "_aten_sub", "alpha"),
    # BINARY:    void(sret Tensor*, const Tensor&, const Tensor&)
    ("mul_Tensor", _ARGS_BINARY, "_aten_mul", "binary"),
    ("div_Tensor", _ARGS_BINARY, "_aten_div", "binary"),
    ("matmul", _ARGS_BINARY, "_aten_matmul", "binary"),
    ("mm", _ARGS_BINARY, "_aten_mm", "binary"),
    ("pow_Tensor_Tensor", _ARGS_BINARY, "_aten_pow", "binary"),
    ("eq_Tensor", _ARGS_BINARY, "_aten_eq", "binary"),
    ("ne_Tensor", _ARGS_BINARY, "_aten_ne", "binary"),
    ("lt_Tensor", _ARGS_BINARY, "_aten_lt", "binary"),
    ("le_Tensor", _ARGS_BINARY, "_aten_le", "binary"),
    ("gt_Tensor", _ARGS_BINARY, "_aten_gt", "binary"),
    ("ge_Tensor", _ARGS_BINARY, "_aten_ge", "binary"),
]

for _name, _args, _llvm_sym, _cc in _ATEN_OPS:
    llvm.add_symbol(_llvm_sym, _fn_addr(_TORCH_LIB, _mangle_aten(_name, _args)))

# ---------------------------------------------------------------------------
# Numba type system
# ---------------------------------------------------------------------------


class TensorType(types.Type):
    def __init__(self):
        super().__init__(name="TorchTensor")


tensor_type = TensorType()


@typeof_impl.register(torch.Tensor)
def typeof_tensor(val, c):
    return tensor_type


@register_model(TensorType)
class TensorModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(64))


# ---------------------------------------------------------------------------
# Unboxing / boxing
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
# LLVM IR helpers
# ---------------------------------------------------------------------------


def _tensor_slot(builder, val):
    """Spill an i64 (TensorImpl*) onto the stack, return i8* to the slot.

    ATen receives ``const at::Tensor&`` which is a pointer to the 8-byte
    Tensor struct.  Since ``at::Tensor`` is just a ``TensorImpl*`` field,
    the stack slot IS the at::Tensor in memory.
    """
    i64 = ir.IntType(64)
    slot = builder.alloca(i64)
    builder.store(val, slot)
    return builder.bitcast(slot, ir.IntType(8).as_pointer())


def _alpha_scalar_one(builder):
    """Build a c10::Scalar(1) on the stack, return i8* to it.

    c10::Scalar layout (32 bytes / 4 × i64):
      [0]  int64 = 1     (union .i – integer value)
      [1]  int64 = 0     (upper half of the 16-byte union, unused)
      [2]  int64 = 1     (Tag::HAS_i)
      [3]  int64 = 0     (padding)
    """
    i64 = ir.IntType(64)
    i32 = ir.IntType(32)
    arr = builder.alloca(ir.ArrayType(i64, 4))
    zero32 = ir.Constant(i32, 0)
    for idx, value in enumerate([1, 0, 1, 0]):
        ptr = builder.gep(arr, [zero32, ir.Constant(i32, idx)])
        builder.store(ir.Constant(i64, value), ptr)
    return builder.bitcast(arr, ir.IntType(8).as_pointer())


# ---------------------------------------------------------------------------
# Intrinsic factories – one per ATen calling convention
# ---------------------------------------------------------------------------


def _make_unary_intrinsic(sym: str):
    """Direct call: void _aten_op(sret Tensor*, const Tensor& self)."""

    @intrinsic
    def _op(typingctx, a):
        if not isinstance(a, TensorType):
            return None
        sig = tensor_type(tensor_type)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            i8p = ir.IntType(8).as_pointer()
            out = builder.alloca(i64)
            fn = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.VoidType(), [i8p, i8p]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn, [builder.bitcast(out, i8p), _tensor_slot(builder, args[0])]
            )
            return builder.load(out)

        return sig, codegen

    return _op


def _make_binary_intrinsic(sym: str):
    """Direct call: void _aten_op(sret Tensor*, const Tensor&, const Tensor&)."""

    @intrinsic
    def _op(typingctx, a, b):
        if not (isinstance(a, TensorType) and isinstance(b, TensorType)):
            return None
        sig = tensor_type(tensor_type, tensor_type)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            i8p = ir.IntType(8).as_pointer()
            out = builder.alloca(i64)
            fn = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.VoidType(), [i8p, i8p, i8p]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    _tensor_slot(builder, args[0]),
                    _tensor_slot(builder, args[1]),
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


def _make_alpha_intrinsic(sym: str):
    """Direct call: void _aten_op(sret Tensor*, Tensor&, Tensor&, Scalar& alpha=1)."""

    @intrinsic
    def _op(typingctx, a, b):
        if not (isinstance(a, TensorType) and isinstance(b, TensorType)):
            return None
        sig = tensor_type(tensor_type, tensor_type)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            i8p = ir.IntType(8).as_pointer()
            out = builder.alloca(i64)
            fn = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.VoidType(), [i8p, i8p, i8p, i8p]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    _tensor_slot(builder, args[0]),
                    _tensor_slot(builder, args[1]),
                    _alpha_scalar_one(builder),
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


def _make_reduction_intrinsic(sym: str):
    """Direct call: void _aten_op(sret Tensor*, Tensor&, i16 dtype=nullopt).

    c10::optional<ScalarType> is 2 bytes, trivially copyable, so SysV ABI
    passes it by value as i16.  We pass 0x0000 (engaged=false = nullopt).
    """

    @intrinsic
    def _op(typingctx, a):
        if not isinstance(a, TensorType):
            return None
        sig = tensor_type(tensor_type)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            i16 = ir.IntType(16)
            i8p = ir.IntType(8).as_pointer()
            out = builder.alloca(i64)
            fn = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.VoidType(), [i8p, i8p, i16]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    _tensor_slot(builder, args[0]),
                    ir.Constant(i16, 0),  # nullopt: engaged byte = 0
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


# ---------------------------------------------------------------------------
# Intrinsic instances
# ---------------------------------------------------------------------------

_tensor_neg = _make_unary_intrinsic("_aten_neg")
_tensor_abs = _make_unary_intrinsic("_aten_abs")
_tensor_exp = _make_unary_intrinsic("_aten_exp")
_tensor_log = _make_unary_intrinsic("_aten_log")
_tensor_sqrt = _make_unary_intrinsic("_aten_sqrt")
_tensor_sin = _make_unary_intrinsic("_aten_sin")
_tensor_cos = _make_unary_intrinsic("_aten_cos")
_tensor_tan = _make_unary_intrinsic("_aten_tan")
_tensor_relu = _make_unary_intrinsic("_aten_relu")
_tensor_sigmoid = _make_unary_intrinsic("_aten_sigmoid")
_tensor_tanh = _make_unary_intrinsic("_aten_tanh")
_tensor_silu = _make_unary_intrinsic("_aten_silu")

_tensor_sum = _make_reduction_intrinsic("_aten_sum")
_tensor_mean = _make_reduction_intrinsic("_aten_mean")

_tensor_add = _make_alpha_intrinsic("_aten_add")
_tensor_sub = _make_alpha_intrinsic("_aten_sub")

_tensor_mul = _make_binary_intrinsic("_aten_mul")
_tensor_div = _make_binary_intrinsic("_aten_div")
_tensor_matmul = _make_binary_intrinsic("_aten_matmul")
_tensor_mm = _make_binary_intrinsic("_aten_mm")
_tensor_pow = _make_binary_intrinsic("_aten_pow")
_tensor_eq = _make_binary_intrinsic("_aten_eq")
_tensor_ne = _make_binary_intrinsic("_aten_ne")
_tensor_lt = _make_binary_intrinsic("_aten_lt")
_tensor_le = _make_binary_intrinsic("_aten_le")
_tensor_gt = _make_binary_intrinsic("_aten_gt")
_tensor_ge = _make_binary_intrinsic("_aten_ge")

# ---------------------------------------------------------------------------
# Operator overloads
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
