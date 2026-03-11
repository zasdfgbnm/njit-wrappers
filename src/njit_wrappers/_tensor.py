"""Numba nopython support for torch.Tensor.

Importing this module registers torch.Tensor as a Numba type so that
@numba.njit functions can accept and return tensors.  Operations on
tensors inside the compiled function lower directly to the corresponding
ATen C++ symbols – no extra Python-level wrapper calls.

Lifetime model
--------------
Internally a tensor is represented as an int64 holding a TensorImpl*
**with an owned reference** (refcount bumped by 1).

- Unboxing  : njit_extract_impl() increments the TensorImpl refcount.
              Numba's NativeValue cleanup calls njit_release_impl() when
              the value is no longer needed.
- Boxing    : njit_wrap_impl() steals the owned reference into a fresh
              Python torch.Tensor object.
- ATen ops  : called via direct LLVM symbol references (zero wrapper
              overhead).  The result's refcount starts at 1 (owned).

Known limitation
----------------
Intermediate tensor values produced inside an njit function that are
*not* the final return value will leak their TensorImpl refcount.
This will be addressed in a future iteration.
"""

import ctypes
import operator
from pathlib import Path

import torch
from llvmlite import ir
import llvmlite.binding as llvm
import numba
from numba import types
from numba.core import cgutils
from numba.core.extending import (
    NativeValue,
    box,
    intrinsic,
    overload,
    register_model,
    typeof_impl,
    unbox,
)
from numba.core.datamodel import models

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
    return ctypes.cast(func, ctypes.c_void_p).value

llvm.add_symbol("njit_extract_impl", _sym_addr(_bridge_lib.njit_extract_impl))
llvm.add_symbol("njit_release_impl", _sym_addr(_bridge_lib.njit_release_impl))
llvm.add_symbol("njit_wrap_impl",    _sym_addr(_bridge_lib.njit_wrap_impl))

# Register the ATen add symbol so the @intrinsic can reference it directly.
_TORCH_LIB_PATH = Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"
_torch_lib = ctypes.CDLL(str(_TORCH_LIB_PATH))

_ATEN_ADD_SYM = (
    "_ZN2at4_ops10add_Tensor4callERKNS_6TensorES4_RKN3c106ScalarE"
)
llvm.add_symbol(_ATEN_ADD_SYM, _sym_addr(getattr(_torch_lib, _ATEN_ADD_SYM)))

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
# @intrinsic: direct zero-overhead call to at::_ops::add_Tensor::call
#
# Calling convention (x86-64 System V / Itanium C++ ABI):
#   rdi = sret pointer  (caller-allocated 8-byte slot for the returned Tensor)
#   rsi = &self         (const at::Tensor& — pointer to 8-byte slot)
#   rdx = &other        (const at::Tensor& — pointer to 8-byte slot)
#   rcx = &alpha        (const c10::Scalar& — pointer to 32-byte slot)
#
# c10::Scalar(1) layout (32 bytes):
#   [0 ]  int64  value = 1      (union field .i)
#   [8 ]  int64  unused = 0     (upper half of the 16-byte union)
#   [16]  int64  tag   = 1      (Tag::HAS_i)
#   [24]  int64  pad   = 0
# ---------------------------------------------------------------------------

@intrinsic
def _tensor_add(typingctx, a, b):
    if not (isinstance(a, TensorType) and isinstance(b, TensorType)):
        return None

    sig = tensor_type(tensor_type, tensor_type)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        i32 = ir.IntType(32)
        void = ir.VoidType()
        scalar_ty = ir.ArrayType(i64, 4)  # 32 bytes

        a_val, b_val = args

        # Allocate output slot (sret)
        sret = builder.alloca(i64, name="sret")

        # Allocate and populate at::Tensor slots for the two inputs
        self_slot = builder.alloca(i64, name="self_slot")
        builder.store(a_val, self_slot)

        other_slot = builder.alloca(i64, name="other_slot")
        builder.store(b_val, other_slot)

        # Construct c10::Scalar(1) on the stack
        alpha_slot = builder.alloca(scalar_ty, name="alpha")
        zero = ir.Constant(i64, 0)
        one  = ir.Constant(i64, 1)
        for idx, val in enumerate([one, zero, one, zero]):
            ptr = builder.gep(
                alpha_slot,
                [ir.Constant(i32, 0), ir.Constant(i32, idx)],
            )
            builder.store(val, ptr)

        # Declare the ATen function (void-returning, sret convention)
        fn_type = ir.FunctionType(void, [
            ir.PointerType(i64),       # sret: at::Tensor*
            ir.PointerType(i64),       # const at::Tensor& self
            ir.PointerType(i64),       # const at::Tensor& other
            ir.PointerType(scalar_ty), # const c10::Scalar& alpha
        ])
        add_fn = cgutils.get_or_insert_function(
            builder.module, fn_type, _ATEN_ADD_SYM
        )
        add_fn.args[0].attributes.add("sret")

        builder.call(add_fn, [sret, self_slot, other_slot, alpha_slot])

        return builder.load(sret)

    return sig, codegen


# ---------------------------------------------------------------------------
# Operator overloading
# ---------------------------------------------------------------------------

@overload(operator.add)
def overload_tensor_add(a, b):
    if isinstance(a, TensorType) and isinstance(b, TensorType):
        def impl(a, b):
            return _tensor_add(a, b)
        return impl
