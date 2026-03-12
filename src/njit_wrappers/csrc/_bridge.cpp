#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/ops/relu.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/python_variable.h>

// Verify the ABI assumption relied on by _make_reduction_intrinsic in
// _tensor.py: c10::optional<c10::ScalarType> must be trivially copyable
// and exactly 2 bytes so that SysV x86-64 passes it by value as i16.
static_assert(std::is_trivially_copyable<c10::optional<c10::ScalarType>>::value,
              "c10::optional<ScalarType> must be trivially copyable (i16 by-value ABI)");
static_assert(sizeof(c10::optional<c10::ScalarType>) == 2,
              "c10::optional<ScalarType> must be exactly 2 bytes");

// ---------------------------------------------------------------------------
// Helper: create at::Tensor from raw TensorImpl* with proper refcounting.
// The resulting Tensor is a new owning reference (incref on construction).
// When it destructs, it will decref. Net effect on the TensorImpl: none
// (if the original reference is also released).
// ---------------------------------------------------------------------------
static inline at::Tensor tensor_from_impl(int64_t impl_int) {
  c10::TensorImpl* impl = reinterpret_cast<c10::TensorImpl*>(impl_int);
  // incref to create a new owning reference for this Tensor
  c10::raw::intrusive_ptr::incref(impl);
  auto ptr =
      c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::unsafe_steal_from_new(impl);
  return at::Tensor(std::move(ptr));
}

// Helper: extract TensorImpl* from at::Tensor result, returning owned ref.
// Takes ownership of the tensor (rvalue ref) — no copy overhead.
static inline int64_t impl_from_tensor(at::Tensor&& t) {
  c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  c10::raw::intrusive_ptr::incref(impl);
  return reinterpret_cast<int64_t>(impl);
  // t destructs here (moved-from state), decrefs. Net: caller has one owned ref.
}

extern "C" {

// Extract TensorImpl* from a Python torch.Tensor and increment its refcount.
// The caller owns the returned reference and must eventually call
// njit_release_impl() or transfer ownership via njit_wrap_impl().
int64_t njit_extract_impl(PyObject* obj) {
  const at::Tensor& tensor = THPVariable_Unpack(obj);
  c10::TensorImpl* impl = tensor.unsafeGetTensorImpl();
  c10::raw::intrusive_ptr::incref(impl);
  return (int64_t)(void*)impl;
}

// Decrement the refcount of a TensorImpl*.
// Use this to release an owned reference that will not be boxed.
void njit_release_impl(int64_t impl_int) {
  c10::TensorImpl* impl = (c10::TensorImpl*)(void*)impl_int;
  c10::raw::intrusive_ptr::decref(impl);
}

// Wrap a TensorImpl* (owned ref) into a Python torch.Tensor, stealing the ref.
// After this call, the int64 handle must not be used again.
PyObject* njit_wrap_impl(int64_t impl_int) {
  c10::TensorImpl* impl = (c10::TensorImpl*)(void*)impl_int;
  // unsafe_steal_from_new wraps the raw pointer without incrementing refcount,
  // which is exactly what we want: steal the ref we already own.
  auto ptr =
      c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::unsafe_steal_from_new(impl);
  at::Tensor t(std::move(ptr));
  return THPVariable_Wrap(std::move(t));
}

// ---------------------------------------------------------------------------
// Experiment: C++ wrapper ops that return int64_t (TensorImpl*) directly.
// These avoid the sret ABI complexity — the result is in a register.
// Each function takes TensorImpl* args as int64_t, calls the ATen op,
// and returns the result TensorImpl* as int64_t (owned ref).
// ---------------------------------------------------------------------------

// Unary op: takes one TensorImpl*, returns one TensorImpl* (owned)
int64_t njit_relu_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::relu(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_sigmoid_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::sigmoid(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_tanh_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::tanh(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_neg_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::neg(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_abs_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::abs(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_exp_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::exp(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_log_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::log(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_sqrt_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::sqrt(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_sin_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::sin(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_cos_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::cos(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_tan_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::tan(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_silu_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::silu(self);
  return impl_from_tensor(std::move(result));
}

// Binary ops
int64_t njit_mul_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::mul(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_div_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::div(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_matmul_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::matmul(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_mm_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::mm(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_pow_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::pow(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_eq_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::eq(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_ne_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::ne(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_lt_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::lt(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_le_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::le(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_gt_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::gt(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_ge_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::ge(a, b);
  return impl_from_tensor(std::move(result));
}

// Alpha ops (add, sub with alpha=1)
int64_t njit_add_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::add(a, b);
  return impl_from_tensor(std::move(result));
}

int64_t njit_sub_wrapper(int64_t a_int, int64_t b_int) {
  at::Tensor a = tensor_from_impl(a_int);
  at::Tensor b = tensor_from_impl(b_int);
  at::Tensor result = at::sub(a, b);
  return impl_from_tensor(std::move(result));
}

// Reduction ops
int64_t njit_sum_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::sum(self);
  return impl_from_tensor(std::move(result));
}

int64_t njit_mean_wrapper(int64_t self_int) {
  at::Tensor self = tensor_from_impl(self_int);
  at::Tensor result = at::mean(self);
  return impl_from_tensor(std::move(result));
}

// ---------------------------------------------------------------------------
// Experiment: Borrow-based unbox (no refcount manipulation)
// ---------------------------------------------------------------------------

// Extract TensorImpl* from a Python torch.Tensor WITHOUT incrementing refcount.
// The returned pointer is a BORROW — the caller must not release it, and must
// ensure the Python object outlives the usage of this pointer.
int64_t njit_borrow_impl(PyObject* obj) {
  const at::Tensor& tensor = THPVariable_Unpack(obj);
  c10::TensorImpl* impl = tensor.unsafeGetTensorImpl();
  return (int64_t)(void*)impl;
}

}  // extern "C"

// Minimal Python extension module boilerplate so this can be imported.
static PyMethodDef BridgeMethods[] = {{nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_bridge", nullptr, -1,
                                       BridgeMethods};

PyMODINIT_FUNC PyInit__bridge() { return PyModule_Create(&moduledef); }
