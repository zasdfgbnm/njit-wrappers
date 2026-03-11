"""Numba nopython support for torch.Tensor.

Importing this module registers torch.Tensor as a Numba type so that
@numba.njit functions can accept and return tensors.  Operations inside
the compiled function lower **directly** to ATen dispatch functions with
no extra wrapper call at runtime.

How symbol resolution works (no hard-coded mangled C++ names)
--------------------------------------------------------------
1. At Python import time we call small C getter functions in _bridge.so,
   e.g. ``njit_addr_relu()``.  Each getter returns the address of the
   real ATen dispatch entry, e.g. ``at::_ops::relu::call``.  The C++
   compiler resolved the C++ mangled name when building _bridge.so.
2. We register each address with LLVM under a short, stable name
   (``_aten_relu``, …) via ``llvm.add_symbol()``.
3. JIT-compiled LLVM IR calls ``_aten_relu`` directly, which IS the ATen
   function.  No extra indirection at tensor-operation time.

ATen calling conventions (SysV x86-64, all ops return at::Tensor via sret)
---------------------------------------------------------------------------
  UNARY      void(Tensor* sret, const Tensor& self)
  BINARY     void(Tensor* sret, const Tensor& self, const Tensor& other)
  ALPHA      void(Tensor* sret, const Tensor& self, const Tensor& other,
                  const Scalar& alpha)          ← add, sub
  REDUCTION  void(Tensor* sret, const Tensor& self,
                  optional<ScalarType> dtype)   ← i16 by value (trivially
                                                  copyable 2-byte type)

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
# Reference-management symbols (always needed)
# ---------------------------------------------------------------------------

_bridge_lib.njit_extract_impl.restype = ctypes.c_int64
_bridge_lib.njit_extract_impl.argtypes = [ctypes.py_object]
_bridge_lib.njit_release_impl.restype = None
_bridge_lib.njit_release_impl.argtypes = [ctypes.c_int64]
_bridge_lib.njit_wrap_impl.restype = ctypes.py_object
_bridge_lib.njit_wrap_impl.argtypes = [ctypes.c_int64]


def _sym_addr(func) -> int:
    addr = ctypes.cast(func, ctypes.c_void_p).value
    assert addr is not None
    return addr


llvm.add_symbol("njit_extract_impl", _sym_addr(_bridge_lib.njit_extract_impl))
llvm.add_symbol("njit_release_impl", _sym_addr(_bridge_lib.njit_release_impl))
llvm.add_symbol("njit_wrap_impl", _sym_addr(_bridge_lib.njit_wrap_impl))

# ---------------------------------------------------------------------------
# Resolve real ATen dispatch addresses once at import time.
#
# The getter functions in _bridge.so were compiled against the ATen headers;
# the C++ compiler resolved the mangled names.  We just call the getter once
# and hand the resulting address to LLVM.  After this point, every occurrence
# of "_aten_relu" etc. in JIT-compiled IR calls the ATen function directly.
# ---------------------------------------------------------------------------

# (op_suffix, llvm_sym,       calling_convention)
_ATEN_OPS: list[tuple[str, str, str]] = [
    # --- UNARY: void(sret Tensor*, const Tensor&) ---
    ("neg",     "_aten_neg",     "unary"),
    ("abs",     "_aten_abs",     "unary"),
    ("exp",     "_aten_exp",     "unary"),
    ("log",     "_aten_log",     "unary"),
    ("sqrt",    "_aten_sqrt",    "unary"),
    ("sin",     "_aten_sin",     "unary"),
    ("cos",     "_aten_cos",     "unary"),
    ("tan",     "_aten_tan",     "unary"),
    ("relu",    "_aten_relu",    "unary"),
    ("sigmoid", "_aten_sigmoid", "unary"),
    ("tanh",    "_aten_tanh",    "unary"),
    ("silu",    "_aten_silu",    "unary"),
    # --- REDUCTION: void(sret Tensor*, const Tensor&, i16 dtype) ---
    ("sum",     "_aten_sum",     "reduction"),
    ("mean",    "_aten_mean",    "reduction"),
    # --- ALPHA: void(sret Tensor*, const Tensor&, const Tensor&, const Scalar&) ---
    ("add",     "_aten_add",     "alpha"),
    ("sub",     "_aten_sub",     "alpha"),
    # --- BINARY: void(sret Tensor*, const Tensor&, const Tensor&) ---
    ("mul",     "_aten_mul",     "binary"),
    ("div",     "_aten_div",     "binary"),
    ("matmul",  "_aten_matmul",  "binary"),
    ("mm",      "_aten_mm",      "binary"),
    ("pow",     "_aten_pow",     "binary"),
    ("eq",      "_aten_eq",      "binary"),
    ("ne",      "_aten_ne",      "binary"),
    ("lt",      "_aten_lt",      "binary"),
    ("le",      "_aten_le",      "binary"),
    ("gt",      "_aten_gt",      "binary"),
    ("ge",      "_aten_ge",      "binary"),
]

for _op, _llvm_sym, _cc in _ATEN_OPS:
    _getter = getattr(_bridge_lib, f"njit_addr_{_op}")
    _getter.restype = ctypes.c_int64
    _getter.argtypes = []
    llvm.add_symbol(_llvm_sym, _getter())

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
            builder.call(fn, [builder.bitcast(out, i8p), _tensor_slot(builder, args[0])])
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
            builder.call(fn, [
                builder.bitcast(out, i8p),
                _tensor_slot(builder, args[0]),
                _tensor_slot(builder, args[1]),
            ])
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
            builder.call(fn, [
                builder.bitcast(out, i8p),
                _tensor_slot(builder, args[0]),
                _tensor_slot(builder, args[1]),
                _alpha_scalar_one(builder),
            ])
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
            builder.call(fn, [
                builder.bitcast(out, i8p),
                _tensor_slot(builder, args[0]),
                ir.Constant(i16, 0),  # nullopt: engaged byte = 0
            ])
            return builder.load(out)

        return sig, codegen

    return _op


# ---------------------------------------------------------------------------
# Intrinsic instances
# ---------------------------------------------------------------------------

_tensor_neg     = _make_unary_intrinsic("_aten_neg")
_tensor_abs     = _make_unary_intrinsic("_aten_abs")
_tensor_exp     = _make_unary_intrinsic("_aten_exp")
_tensor_log     = _make_unary_intrinsic("_aten_log")
_tensor_sqrt    = _make_unary_intrinsic("_aten_sqrt")
_tensor_sin     = _make_unary_intrinsic("_aten_sin")
_tensor_cos     = _make_unary_intrinsic("_aten_cos")
_tensor_tan     = _make_unary_intrinsic("_aten_tan")
_tensor_relu    = _make_unary_intrinsic("_aten_relu")
_tensor_sigmoid = _make_unary_intrinsic("_aten_sigmoid")
_tensor_tanh    = _make_unary_intrinsic("_aten_tanh")
_tensor_silu    = _make_unary_intrinsic("_aten_silu")

_tensor_sum     = _make_reduction_intrinsic("_aten_sum")
_tensor_mean    = _make_reduction_intrinsic("_aten_mean")

_tensor_add     = _make_alpha_intrinsic("_aten_add")
_tensor_sub     = _make_alpha_intrinsic("_aten_sub")

_tensor_mul     = _make_binary_intrinsic("_aten_mul")
_tensor_div     = _make_binary_intrinsic("_aten_div")
_tensor_matmul  = _make_binary_intrinsic("_aten_matmul")
_tensor_mm      = _make_binary_intrinsic("_aten_mm")
_tensor_pow     = _make_binary_intrinsic("_aten_pow")
_tensor_eq      = _make_binary_intrinsic("_aten_eq")
_tensor_ne      = _make_binary_intrinsic("_aten_ne")
_tensor_lt      = _make_binary_intrinsic("_aten_lt")
_tensor_le      = _make_binary_intrinsic("_aten_le")
_tensor_gt      = _make_binary_intrinsic("_aten_gt")
_tensor_ge      = _make_binary_intrinsic("_aten_ge")

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
