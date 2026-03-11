#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Per-op dispatch-table headers expose at::_ops::{name}::call (the real ATen
// dispatch entry points).  The C++ compiler resolves their mangled names at
// build time; Python only ever sees the stable C getter names below.
#include <ATen/ops/abs.h>
#include <ATen/ops/add.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/div.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/le.h>
#include <ATen/ops/log.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tanh.h>

#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/python_variable.h>

// c10::optional<ScalarType> must be trivially copyable so that it is passed
// by value (as a 16-bit integer) under the SysV x86-64 ABI.  The LLVM IR in
// _tensor.py depends on this layout.
static_assert(
    std::is_trivially_copyable<c10::optional<c10::ScalarType>>::value,
    "c10::optional<ScalarType> must be trivially copyable (i16 by-value ABI)");
static_assert(
    sizeof(c10::optional<c10::ScalarType>) == 2,
    "c10::optional<ScalarType> must be exactly 2 bytes");

extern "C" {

// ---------------------------------------------------------------------------
// Reference management – called from compiled code via LLVM symbols
// ---------------------------------------------------------------------------

// Extract TensorImpl* from a Python torch.Tensor and increment its refcount.
int64_t njit_extract_impl(PyObject* obj) {
    const at::Tensor& tensor = THPVariable_Unpack(obj);
    c10::TensorImpl* impl = tensor.unsafeGetTensorImpl();
    c10::raw::intrusive_ptr::incref(impl);
    return reinterpret_cast<int64_t>(impl);
}

// Decrement the refcount of an owned TensorImpl* handle.
void njit_release_impl(int64_t impl_int) {
    auto* impl = reinterpret_cast<c10::TensorImpl*>(impl_int);
    c10::raw::intrusive_ptr::decref(impl);
}

// Wrap an owned TensorImpl* handle into a Python torch.Tensor, stealing it.
PyObject* njit_wrap_impl(int64_t impl_int) {
    auto* impl = reinterpret_cast<c10::TensorImpl*>(impl_int);
    auto ptr = c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>
        ::unsafe_steal_from_new(impl);
    at::Tensor t(std::move(ptr));
    return THPVariable_Wrap(std::move(t));
}

// ---------------------------------------------------------------------------
// ATen address getters
//
// Each function is called EXACTLY ONCE at Python import time to resolve the
// real address of an ATen dispatch function.  That address is then registered
// with LLVM under a stable name so that JIT-compiled code can call it
// DIRECTLY – with zero extra indirection at tensor-operation time.
//
// Calling conventions (all return via sret because at::Tensor is non-trivially
// copyable; actual ABI args follow the hidden sret pointer):
//
//   UNARY       void(Tensor* sret, const Tensor& self)
//   BINARY      void(Tensor* sret, const Tensor& self, const Tensor& other)
//   ALPHA       void(Tensor* sret, const Tensor& self, const Tensor& other,
//                    const Scalar& alpha)
//   REDUCTION   void(Tensor* sret, const Tensor& self,
//                    optional<ScalarType> dtype)   ← passed as i16 by value
// ---------------------------------------------------------------------------

#define NJIT_ADDR_GETTER(cname, fn_expr) \
    int64_t cname() { return (int64_t)(void*)(fn_expr); }

// --- UNARY ---
NJIT_ADDR_GETTER(njit_addr_neg,     &at::_ops::neg::call)
NJIT_ADDR_GETTER(njit_addr_abs,     &at::_ops::abs::call)
NJIT_ADDR_GETTER(njit_addr_exp,     &at::_ops::exp::call)
NJIT_ADDR_GETTER(njit_addr_log,     &at::_ops::log::call)
NJIT_ADDR_GETTER(njit_addr_sqrt,    &at::_ops::sqrt::call)
NJIT_ADDR_GETTER(njit_addr_sin,     &at::_ops::sin::call)
NJIT_ADDR_GETTER(njit_addr_cos,     &at::_ops::cos::call)
NJIT_ADDR_GETTER(njit_addr_tan,     &at::_ops::tan::call)
NJIT_ADDR_GETTER(njit_addr_relu,    &at::_ops::relu::call)
NJIT_ADDR_GETTER(njit_addr_sigmoid, &at::_ops::sigmoid::call)
NJIT_ADDR_GETTER(njit_addr_tanh,    &at::_ops::tanh::call)
NJIT_ADDR_GETTER(njit_addr_silu,    &at::_ops::silu::call)

// --- REDUCTION (sret + Tensor + optional<ScalarType> as i16) ---
NJIT_ADDR_GETTER(njit_addr_sum,     &at::_ops::sum::call)
NJIT_ADDR_GETTER(njit_addr_mean,    &at::_ops::mean::call)

// --- ALPHA (sret + Tensor + Tensor + Scalar) ---
NJIT_ADDR_GETTER(njit_addr_add,     &at::_ops::add_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_sub,     &at::_ops::sub_Tensor::call)

// --- BINARY (sret + Tensor + Tensor) ---
NJIT_ADDR_GETTER(njit_addr_mul,     &at::_ops::mul_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_div,     &at::_ops::div_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_matmul,  &at::_ops::matmul::call)
NJIT_ADDR_GETTER(njit_addr_mm,      &at::_ops::mm::call)
NJIT_ADDR_GETTER(njit_addr_pow,     &at::_ops::pow_Tensor_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_eq,      &at::_ops::eq_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_ne,      &at::_ops::ne_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_lt,      &at::_ops::lt_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_le,      &at::_ops::le_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_gt,      &at::_ops::gt_Tensor::call)
NJIT_ADDR_GETTER(njit_addr_ge,      &at::_ops::ge_Tensor::call)

}  // extern "C"

// ---------------------------------------------------------------------------
// Minimal Python extension module boilerplate
// ---------------------------------------------------------------------------
static PyMethodDef BridgeMethods[] = {{nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_bridge", nullptr, -1, BridgeMethods
};

PyMODINIT_FUNC PyInit__bridge() {
    return PyModule_Create(&moduledef);
}
