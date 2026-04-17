"""Numba nopython support for torch.Tensor.

Importing this module registers torch.Tensor as a Numba type so that
@numba.njit functions can accept and return tensors.  Operations inside
the compiled function lower **directly** to ATen redispatch functions,
bypassing the full PyTorch dispatcher and jumping straight to the CUDA
kernel.  No extra wrapper call at runtime.

How symbol resolution works
---------------------------
ATen functions exported by libtorch_cpu.so use Itanium C++ name mangling.
We compute the mangled name in Python using the regular structure of ATen
function signatures (all in namespace ``at``, arguments drawn from a small
set of types), then look up the address via ctypes and register it with
LLVM.  No hard-coded mangled strings in the source, no C++ address-getter
helper functions.

We use ``at::_ops::{name}::redispatch`` instead of ``::call`` to skip
the dispatcher.  The first argument is a ``c10::DispatchKeySet`` (uint64)
that tells the dispatcher which backend to jump to directly.

Example for ``at::_ops::relu::redispatch(c10::DispatchKeySet, const at::Tensor&)``:

    _mangle_aten("relu", _ARGS_UNARY)
    → "_ZN2at4_ops4relu10redispatchEN3c1014DispatchKeySetERKNS_6TensorE"

ATen calling conventions (aarch64 AAPCS64 / SysV x86-64, all ops return
at::Tensor via sret, all take DispatchKeySet as first non-sret arg)
---------------------------------------------------------------------------
  UNARY      void(Tensor* sret, uint64 keyset, const Tensor& self)
  BINARY     void(Tensor* sret, uint64 keyset, const Tensor& self,
                  const Tensor& other)
  ALPHA      void(Tensor* sret, uint64 keyset, const Tensor& self,
                  const Tensor& other, const Scalar& alpha)
  REDUCTION  void(Tensor* sret, uint64 keyset, const Tensor& self,
                  optional<ScalarType> dtype)   ← i16 by value

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
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import njit_wrappers._bridge as _bridge_module  # noqa: F401 – side-effect load

_bridge_lib = ctypes.CDLL(_bridge_module.__file__)

# ---------------------------------------------------------------------------
# Reference-management symbols (bridge extension)
# ---------------------------------------------------------------------------

_bridge_lib.njit_borrow_impl.restype = ctypes.c_int64
_bridge_lib.njit_borrow_impl.argtypes = [ctypes.py_object]
_bridge_lib.njit_wrap_impl.restype = ctypes.py_object
_bridge_lib.njit_wrap_impl.argtypes = [ctypes.c_int64]


def _fn_addr(lib: ctypes.CDLL, mangled: str) -> int:
    fn = getattr(lib, mangled)
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    assert addr, f"symbol not found: {mangled}"
    return addr


llvm.add_symbol("njit_borrow_impl", _fn_addr(_bridge_lib, "njit_borrow_impl"))
llvm.add_symbol("njit_wrap_impl", _fn_addr(_bridge_lib, "njit_wrap_impl"))

_bridge_lib.njit_data_ptr.restype = ctypes.c_uint64
_bridge_lib.njit_data_ptr.argtypes = [ctypes.c_int64]
llvm.add_symbol("njit_data_ptr", _fn_addr(_bridge_lib, "njit_data_ptr"))

# ---------------------------------------------------------------------------
# ATen symbol resolution via Itanium C++ name mangling computed in Python.
#
# We call at::_ops::{name}::redispatch, which takes a c10::DispatchKeySet
# (uint64_t) as the first argument and skips dispatch layers above the
# target backend.  We pass DispatchKeySet(CUDA) = 0x10002 to jump straight
# to the CUDA kernel, bypassing autograd, autocast, and other middleware.
#
# Mangled name pattern for at::_ops::{op}::redispatch(DispatchKeySet, {args}):
#
#   _ZN  2at  4_ops  {len(op)}{op}  10redispatch  E  {arg_suffix}
#
# Substitution context after encoding "N 2at 4_ops {n}{op} 10redispatch E":
#   S_  = at::
#   S0_ = at::_ops
#   S1_ = at::_ops::{op}   (the generated struct)
# After encoding c10::DispatchKeySet (N3c1014DispatchKeySetE):
#   S2_ = c10::
#   S3_ = c10::DispatchKeySet
# After encoding the first "const at::Tensor &" (RKNS_6TensorE):
#   S4_ = at::Tensor
#   S5_ = const at::Tensor
#   S6_ = const at::Tensor&   ← used as S6_ for second Tensor arg
#
# For reduction ops the second argument is std::optional<c10::ScalarType>
# (c10::optional = std::optional via using-declaration in modern PyTorch).
# St8optional = std::optional, N3c1010ScalarTypeE = c10::ScalarType.
# _bridge.cpp has a static_assert confirming this type is 2 bytes and
# trivially copyable, hence passed as i16 in the LLVM IR.
# ---------------------------------------------------------------------------

_TORCH_LIB = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))

# DispatchKeySet(CUDA) = Dense | CUDA backend = 0x10002
_CUDA_DISPATCH_KEYSET = 0x10002

# Argument suffix constants for redispatch (substitutions as annotated above).
# The DispatchKeySet first arg shifts all substitution indices by +2 vs ::call.
_ARGS_UNARY = "N3c1014DispatchKeySetERKNS_6TensorE"
_ARGS_BINARY = "N3c1014DispatchKeySetERKNS_6TensorES6_"
_ARGS_ALPHA = "N3c1014DispatchKeySetERKNS_6TensorES6_RKNS2_6ScalarE"
_ARGS_REDUCE = (
    "N3c1014DispatchKeySetERKNS_6TensorESt8optionalINS2_10ScalarTypeEE"
)


def _mangle_aten(op: str, arg_suffix: str) -> str:
    """Compute the Itanium mangled name for at::_ops::{op}::redispatch(...)."""
    return f"_ZN2at4_ops{len(op)}{op}10redispatchE{arg_suffix}"


# (op_name in at::_ops, arg_suffix, llvm_sym, calling_convention)
_ATEN_OPS: list[tuple[str, str, str, str]] = [
    # UNARY:     void(sret Tensor*, i64 keyset, const Tensor&)
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
    # REDUCTION: void(sret Tensor*, i64 keyset, const Tensor&, i16 optional<ScalarType>)
    ("sum", _ARGS_REDUCE, "_aten_sum", "reduction"),
    ("mean", _ARGS_REDUCE, "_aten_mean", "reduction"),
    # ALPHA:     void(sret Tensor*, i64 keyset, const Tensor&, const Tensor&, const Scalar&)
    ("add_Tensor", _ARGS_ALPHA, "_aten_add", "alpha"),
    ("sub_Tensor", _ARGS_ALPHA, "_aten_sub", "alpha"),
    # BINARY:    void(sret Tensor*, i64 keyset, const Tensor&, const Tensor&)
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

# aoti_torch_get_numel is a C function exported by libtorch_cpu.so:
#   int32_t aoti_torch_get_numel(AtenTensorHandle tensor, int64_t* ret_numel)
# AtenTensorHandle is an opaque pointer compatible with TensorImpl*.
llvm.add_symbol("aoti_torch_get_numel", _fn_addr(_TORCH_LIB, "aoti_torch_get_numel"))

# aoti_torch_empty_strided is a C function exported by libtorch_cpu.so:
#   int32_t aoti_torch_empty_strided(int64_t ndim, const int64_t* sizes,
#       const int64_t* strides, int32_t dtype, int32_t device_type,
#       int32_t device_index, AtenTensorHandle* ret)
llvm.add_symbol(
    "aoti_torch_empty_strided",
    _fn_addr(_TORCH_LIB, "aoti_torch_empty_strided"),
)

# Mapping from torch.dtype to c10::ScalarType integer values.
_DTYPE_TO_SCALAR_TYPE: dict[torch.dtype, int] = {
    torch.float32: 6,
    torch.float64: 7,
    torch.float16: 5,
    torch.bfloat16: 15,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.uint8: 0,
    torch.bool: 11,
    torch.complex64: 9,
    torch.complex128: 10,
}

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
    # Borrow the TensorImpl* without touching the refcount.  The Python
    # object is kept alive by the caller for the duration of the njit call,
    # so a borrowed reference is safe and avoids the cost of an atomic
    # incref + decref pair on every invocation.
    borrow_fn = cgutils.get_or_insert_function(
        c.builder.module,
        ir.FunctionType(i64, [ir.IntType(8).as_pointer()]),
        "njit_borrow_impl",
    )
    impl = c.builder.call(borrow_fn, [obj])
    return NativeValue(impl)


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
    """Redispatch: void _aten_op(sret Tensor*, i64 keyset, const Tensor& self)."""

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
                ir.FunctionType(ir.VoidType(), [i8p, i64, i8p]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    ir.Constant(i64, _CUDA_DISPATCH_KEYSET),
                    _tensor_slot(builder, args[0]),
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


def _make_binary_intrinsic(sym: str):
    """Redispatch: void _aten_op(sret Tensor*, i64 keyset, Tensor&, Tensor&)."""

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
                ir.FunctionType(ir.VoidType(), [i8p, i64, i8p, i8p]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    ir.Constant(i64, _CUDA_DISPATCH_KEYSET),
                    _tensor_slot(builder, args[0]),
                    _tensor_slot(builder, args[1]),
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


def _make_alpha_intrinsic(sym: str):
    """Redispatch: void _aten_op(sret Tensor*, i64 keyset, Tensor&, Tensor&, Scalar&)."""

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
                ir.FunctionType(ir.VoidType(), [i8p, i64, i8p, i8p, i8p]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    ir.Constant(i64, _CUDA_DISPATCH_KEYSET),
                    _tensor_slot(builder, args[0]),
                    _tensor_slot(builder, args[1]),
                    _alpha_scalar_one(builder),
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


def _make_reduction_intrinsic(sym: str):
    """Redispatch: void _aten_op(sret Tensor*, i64 keyset, Tensor&, i16 dtype).

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
                ir.FunctionType(ir.VoidType(), [i8p, i64, i8p, i16]),
                sym,
            )
            fn.args[0].add_attribute("sret")
            builder.call(
                fn,
                [
                    builder.bitcast(out, i8p),
                    ir.Constant(i64, _CUDA_DISPATCH_KEYSET),
                    _tensor_slot(builder, args[0]),
                    ir.Constant(i16, 0),  # nullopt: engaged byte = 0
                ],
            )
            return builder.load(out)

        return sig, codegen

    return _op


# ---------------------------------------------------------------------------
# empty_strided intrinsic factory (for inductor buffer allocation)
# ---------------------------------------------------------------------------


def _make_empty_strided_intrinsic(shape, stride, dtype, device_type, device_index):
    """Return a no-arg @intrinsic that allocates a tensor via aoti_torch_empty_strided.

    All parameters are baked into the LLVM IR as constants so the intrinsic
    takes zero arguments and returns a TensorType (i64 holding TensorImpl*).

    ``aoti_torch_empty_strided`` writes a heap-allocated ``at::Tensor*``
    (``AtenTensorHandle``) into the output slot.  Since ``at::Tensor`` is
    just 8 bytes holding ``TensorImpl*``, we dereference the handle once
    to obtain the raw ``TensorImpl*`` we store as our i64 representation.
    The heap ``at::Tensor`` is leaked (same known limitation as other
    intermediates produced inside ``@njit``).
    """
    ndim = len(shape)
    dtype_int = _DTYPE_TO_SCALAR_TYPE[dtype]

    @intrinsic
    def _alloc(typingctx):
        sig = tensor_type()

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            i32 = ir.IntType(32)
            i8p = ir.IntType(8).as_pointer()

            # Build constant arrays on the stack for sizes and strides
            sizes_arr = builder.alloca(ir.ArrayType(i64, ndim))
            strides_arr = builder.alloca(ir.ArrayType(i64, ndim))
            zero32 = ir.Constant(i32, 0)
            for i in range(ndim):
                ptr_s = builder.gep(sizes_arr, [zero32, ir.Constant(i32, i)])
                builder.store(ir.Constant(i64, shape[i]), ptr_s)
                ptr_st = builder.gep(strides_arr, [zero32, ir.Constant(i32, i)])
                builder.store(ir.Constant(i64, stride[i]), ptr_st)

            sizes_ptr = builder.bitcast(sizes_arr, i64.as_pointer())
            strides_ptr = builder.bitcast(strides_arr, i64.as_pointer())

            # Output slot: receives AtenTensorHandle (= at::Tensor*)
            # which is a pointer-sized value.
            handle_slot = builder.alloca(i8p)

            fn = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(
                    i32,
                    [i64, i64.as_pointer(), i64.as_pointer(), i32, i32, i32, i8p],
                ),
                "aoti_torch_empty_strided",
            )
            # Pass &handle_slot as AtenTensorHandle*
            builder.call(
                fn,
                [
                    ir.Constant(i64, ndim),
                    sizes_ptr,
                    strides_ptr,
                    ir.Constant(i32, dtype_int),
                    ir.Constant(i32, device_type),
                    ir.Constant(i32, device_index),
                    builder.bitcast(handle_slot, i8p),
                ],
            )
            # Load the AtenTensorHandle (at::Tensor*) from the slot
            handle = builder.load(handle_slot)
            # Dereference at::Tensor* to get TensorImpl* (first 8 bytes)
            impl_ptr = builder.bitcast(handle, i64.as_pointer())
            return builder.load(impl_ptr)

        return sig, codegen

    return _alloc


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
# Tensor data pointer extraction (for Triton kernel launches)
# ---------------------------------------------------------------------------


@intrinsic
def _tensor_data_ptr(typingctx, a):
    """Extract the raw device pointer from a TensorType as uint64."""
    if not isinstance(a, TensorType):
        return None
    sig = types.uint64(tensor_type)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(i64, [i64]),
            "njit_data_ptr",
        )
        return builder.call(fn, [args[0]])

    return sig, codegen


# ---------------------------------------------------------------------------
# Tensor numel (via aoti_torch_get_numel from libtorch_cpu.so)
# ---------------------------------------------------------------------------


@intrinsic
def _tensor_numel(typingctx, a):
    """Return the number of elements in a TensorType as int64."""
    if not isinstance(a, TensorType):
        return None
    sig = types.int64(tensor_type)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        i32 = ir.IntType(32)
        i8p = ir.IntType(8).as_pointer()
        # aoti_torch_get_numel(AtenTensorHandle tensor, int64_t* ret_numel)
        # AtenTensorHandle is at::Tensor* (pointer to 8-byte TensorImpl*),
        # so we spill the i64 handle onto the stack and pass its address.
        fn = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(i32, [i8p, i64.as_pointer()]),
            "aoti_torch_get_numel",
        )
        out = builder.alloca(i64)
        builder.call(fn, [_tensor_slot(builder, args[0]), out])
        return builder.load(out)

    return sig, codegen


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


# ---------------------------------------------------------------------------
# Tensor method overloads
# ---------------------------------------------------------------------------


@overload_method(TensorType, "numel")
def overload_tensor_numel(self):
    def impl(self):
        return _tensor_numel(self)  # type: ignore[call-arg]

    return impl
