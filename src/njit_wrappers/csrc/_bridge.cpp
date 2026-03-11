#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ATen/ATen.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/python_variable.h>

// ---------------------------------------------------------------------------
// Helpers: convert between int64_t (TensorImpl* with owned ref) and at::Tensor
// ---------------------------------------------------------------------------

// Borrow: reconstruct an at::Tensor from an owned int64 handle without
// disturbing the caller's ownership.  The temporary incref is undone when
// the returned Tensor is destroyed.
static at::Tensor impl_to_tensor(int64_t impl_int) {
    auto* impl = reinterpret_cast<c10::TensorImpl*>(impl_int);
    c10::raw::intrusive_ptr::incref(impl);
    return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>
            ::unsafe_steal_from_new(impl));
}

// Take ownership: convert a freshly-created result Tensor into an int64 handle
// with an owned reference.  The extra incref lets the local Tensor destruct
// without dropping the ref we return.
static int64_t tensor_to_impl(at::Tensor t) {
    c10::TensorImpl* impl = t.unsafeGetTensorImpl();
    c10::raw::intrusive_ptr::incref(impl);
    return reinterpret_cast<int64_t>(impl);
}

// ---------------------------------------------------------------------------
// Reference management – called from compiled code via LLVM symbols
// ---------------------------------------------------------------------------

extern "C" {

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
// ATen operation wrappers with stable C names (no C++ name mangling needed).
//
// Each wrapper takes int64_t handles (owned TensorImpl* refs) and returns
// a new int64_t handle for the result.  Calling code in _tensor.py only
// needs to know these simple C identifiers – the C++ mangled symbols for
// the underlying ATen dispatch entries are entirely hidden here.
// ---------------------------------------------------------------------------

// Macro for binary tensor×tensor ops (self, other) -> result
#define NJIT_BINARY_OP(name, expr)                                    \
int64_t njit_aten_##name(int64_t self_int, int64_t other_int) {       \
    at::Tensor self  = impl_to_tensor(self_int);                      \
    at::Tensor other = impl_to_tensor(other_int);                     \
    return tensor_to_impl(expr);                                      \
}

// Macro for unary tensor -> result ops
#define NJIT_UNARY_OP(name, expr)                                     \
int64_t njit_aten_##name(int64_t self_int) {                          \
    at::Tensor self = impl_to_tensor(self_int);                       \
    return tensor_to_impl(expr);                                      \
}

// --- Element-wise arithmetic ---
NJIT_BINARY_OP(add,    at::add(self, other))
NJIT_BINARY_OP(sub,    at::sub(self, other))
NJIT_BINARY_OP(mul,    at::mul(self, other))
NJIT_BINARY_OP(div,    at::div(self, other))
NJIT_BINARY_OP(pow,    at::pow(self, other))

// --- Linear algebra ---
NJIT_BINARY_OP(matmul, at::matmul(self, other))
NJIT_BINARY_OP(mm,     at::mm(self, other))

// --- Comparison (result is a bool tensor) ---
NJIT_BINARY_OP(eq, at::eq(self, other))
NJIT_BINARY_OP(ne, at::ne(self, other))
NJIT_BINARY_OP(lt, at::lt(self, other))
NJIT_BINARY_OP(le, at::le(self, other))
NJIT_BINARY_OP(gt, at::gt(self, other))
NJIT_BINARY_OP(ge, at::ge(self, other))

// --- Unary element-wise math ---
NJIT_UNARY_OP(neg,     at::neg(self))
NJIT_UNARY_OP(abs,     at::abs(self))
NJIT_UNARY_OP(exp,     at::exp(self))
NJIT_UNARY_OP(log,     at::log(self))
NJIT_UNARY_OP(sqrt,    at::sqrt(self))
NJIT_UNARY_OP(sin,     at::sin(self))
NJIT_UNARY_OP(cos,     at::cos(self))
NJIT_UNARY_OP(tan,     at::tan(self))

// --- Activations ---
NJIT_UNARY_OP(relu,    at::relu(self))
NJIT_UNARY_OP(sigmoid, at::sigmoid(self))
NJIT_UNARY_OP(tanh,    at::tanh(self))
NJIT_UNARY_OP(silu,    at::silu(self))

// --- Reductions (returns a scalar-valued 0-dim tensor) ---
NJIT_UNARY_OP(sum,  at::sum(self))
NJIT_UNARY_OP(mean, at::mean(self))

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
