#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/python_variable.h>

// Verify the ABI assumption relied on by _make_reduction_intrinsic in
// _tensor.py: c10::optional<c10::ScalarType> must be trivially copyable
// and exactly 2 bytes so that SysV x86-64 passes it by value as i16.
static_assert(
    std::is_trivially_copyable<c10::optional<c10::ScalarType>>::value,
    "c10::optional<ScalarType> must be trivially copyable (i16 by-value ABI)");
static_assert(
    sizeof(c10::optional<c10::ScalarType>) == 2,
    "c10::optional<ScalarType> must be exactly 2 bytes");

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

}  // extern "C"

static PyMethodDef BridgeMethods[] = {{nullptr, nullptr, 0, nullptr}};
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_bridge", nullptr, -1, BridgeMethods
};
PyMODINIT_FUNC PyInit__bridge() { return PyModule_Create(&moduledef); }
