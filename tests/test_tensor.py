"""Tests for torch.Tensor support in numba.njit.

Covers:
- All supported element-wise binary ops (+, -, *, /, @)
- All supported unary ops (neg, abs, exp, log, sqrt, sin, cos, relu, …)
- Reduction ops (sum, mean)
- Comparison ops (==, !=, <, <=, >, >=)
- Complex multi-op computation graphs
"""

import numba
import pytest
import torch

import njit_wrappers  # noqa: F401 – registers TensorType

# ---------------------------------------------------------------------------
# Simple njit kernels used across multiple tests
# ---------------------------------------------------------------------------


@numba.njit
def add(a, b):
    return a + b


@numba.njit
def sub(a, b):
    return a - b


@numba.njit
def mul(a, b):
    return a * b


@numba.njit
def div(a, b):
    return a / b


@numba.njit
def matmul(a, b):
    return a @ b


@numba.njit
def neg(a):
    return -a


@numba.njit
def absolute(a):
    return abs(a)


@numba.njit
def exp(a):
    return torch.exp(a)


@numba.njit
def log(a):
    return torch.log(a)


@numba.njit
def sqrt(a):
    return torch.sqrt(a)


@numba.njit
def sin(a):
    return torch.sin(a)


@numba.njit
def cos(a):
    return torch.cos(a)


@numba.njit
def relu(a):
    return torch.relu(a)


@numba.njit
def sigmoid(a):
    return torch.sigmoid(a)


@numba.njit
def tanh_fn(a):
    return torch.tanh(a)


@numba.njit
def tensor_sum(a):
    return torch.sum(a)


@numba.njit
def tensor_mean(a):
    return torch.mean(a)


# ---------------------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------------------


def test_add_basic():
    a = torch.ones(3)
    b = torch.ones(3)
    c = add(a, b)
    assert isinstance(c, torch.Tensor)
    assert c.shape == (3,)
    torch.testing.assert_close(c, torch.full((3,), 2.0))


def test_add_2d():
    a = torch.ones(2, 3)
    b = torch.ones(2, 3)
    torch.testing.assert_close(add(a, b), torch.full((2, 3), 2.0))


def test_add_float64():
    a = torch.ones(4, dtype=torch.float64)
    b = torch.ones(4, dtype=torch.float64)
    c = add(a, b)
    assert c.dtype == torch.float64
    torch.testing.assert_close(c, torch.full((4,), 2.0, dtype=torch.float64))


def test_add_preserves_grad():
    a = torch.ones(3, requires_grad=True)
    b = torch.ones(3)
    c = add(a, b)
    assert c.requires_grad


def test_add_reuse_jit():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    torch.testing.assert_close(add(a, b), add(b, a))


def test_sub():
    a = torch.tensor([5.0, 6.0, 7.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(sub(a, b), torch.tensor([4.0, 4.0, 4.0]))


def test_mul():
    a = torch.tensor([2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0])
    torch.testing.assert_close(mul(a, b), torch.tensor([10.0, 18.0, 28.0]))


def test_div():
    a = torch.tensor([10.0, 9.0, 8.0])
    b = torch.tensor([2.0, 3.0, 4.0])
    torch.testing.assert_close(div(a, b), torch.tensor([5.0, 3.0, 2.0]))


def test_matmul_2d():
    a = torch.eye(3)
    b = torch.arange(9.0).reshape(3, 3)
    torch.testing.assert_close(matmul(a, b), b)


def test_matmul_shapes():
    a = torch.randn(4, 8)
    b = torch.randn(8, 6)
    c = matmul(a, b)
    assert c.shape == (4, 6)
    torch.testing.assert_close(c, a @ b)


# ---------------------------------------------------------------------------
# Comparison ops
# ---------------------------------------------------------------------------


@numba.njit
def eq_fn(a, b):
    return a == b


@numba.njit
def ne_fn(a, b):
    return a != b


@numba.njit
def lt_fn(a, b):
    return a < b


@numba.njit
def le_fn(a, b):
    return a <= b


@numba.njit
def gt_fn(a, b):
    return a > b


@numba.njit
def ge_fn(a, b):
    return a >= b


def test_eq():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 0.0, 3.0])
    torch.testing.assert_close(eq_fn(a, b), a == b)


def test_ne():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 0.0, 3.0])
    torch.testing.assert_close(ne_fn(a, b), a != b)


def test_lt():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    torch.testing.assert_close(lt_fn(a, b), a < b)


def test_le():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    torch.testing.assert_close(le_fn(a, b), a <= b)


def test_gt():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    torch.testing.assert_close(gt_fn(a, b), a > b)


def test_ge():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    torch.testing.assert_close(ge_fn(a, b), a >= b)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


def test_neg():
    a = torch.tensor([1.0, -2.0, 3.0])
    torch.testing.assert_close(neg(a), -a)


def test_abs():
    a = torch.tensor([-1.0, 2.0, -3.0])
    torch.testing.assert_close(absolute(a), a.abs())


def test_exp():
    a = torch.tensor([0.0, 1.0, 2.0])
    torch.testing.assert_close(exp(a), a.exp())


def test_log():
    a = torch.tensor([1.0, 2.0, 4.0])
    torch.testing.assert_close(log(a), a.log())


def test_sqrt():
    a = torch.tensor([1.0, 4.0, 9.0])
    torch.testing.assert_close(sqrt(a), a.sqrt())


def test_sin():
    a = torch.linspace(0.0, 1.0, 8)
    torch.testing.assert_close(sin(a), a.sin())


def test_cos():
    a = torch.linspace(0.0, 1.0, 8)
    torch.testing.assert_close(cos(a), a.cos())


def test_relu():
    a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    torch.testing.assert_close(relu(a), a.relu())


def test_sigmoid():
    a = torch.tensor([-2.0, 0.0, 2.0])
    torch.testing.assert_close(sigmoid(a), a.sigmoid())


def test_tanh():
    a = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(tanh_fn(a), a.tanh())


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_sum():
    a = torch.ones(5)
    result = tensor_sum(a)
    assert result.shape == ()
    torch.testing.assert_close(result, torch.tensor(5.0))


def test_mean():
    a = torch.tensor([2.0, 4.0, 6.0])
    result = tensor_mean(a)
    assert result.shape == ()
    torch.testing.assert_close(result, torch.tensor(4.0))


# ---------------------------------------------------------------------------
# Complex multi-op computation graphs
# ---------------------------------------------------------------------------


@numba.njit
def diff_of_squares(a, b):
    """(a + b) * (a - b)  ==  a² - b²   — 3-op graph."""
    return (a + b) * (a - b)


def test_diff_of_squares():
    a = torch.tensor([3.0, 4.0, 5.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    got = diff_of_squares(a, b)
    expected = a * a - b * b
    torch.testing.assert_close(got, expected)


@numba.njit
def normalize_center(x):
    """Subtract the mean: x - mean(x) — uses reduction + broadcast sub."""
    return x - torch.mean(x)


def test_normalize_center():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    got = normalize_center(x)
    expected = x - x.mean()
    torch.testing.assert_close(got, expected)
    # Result should have zero mean
    torch.testing.assert_close(got.mean(), torch.zeros(()))


@numba.njit
def relu_linear(x, w, b):
    """relu(x @ w + b) — 3-op graph spanning matmul, add, relu."""
    return torch.relu(x @ w + b)


def test_relu_linear():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    w = torch.randn(8, 16)
    b = torch.zeros(16)
    got = relu_linear(x, w, b)
    expected = torch.relu(x @ w + b)
    assert got.shape == (4, 16)
    torch.testing.assert_close(got, expected)


@numba.njit
def mlp_forward(x, w1, b1, w2, b2):
    """Two-layer MLP: relu(x @ w1 + b1) @ w2 + b2 — 5-op graph."""
    h = torch.relu(x @ w1 + b1)
    return h @ w2 + b2


def test_mlp_forward():
    torch.manual_seed(42)
    x = torch.randn(8, 16)
    w1 = torch.randn(16, 32)
    b1 = torch.zeros(32)
    w2 = torch.randn(32, 8)
    b2 = torch.zeros(8)

    got = mlp_forward(x, w1, b1, w2, b2)
    expected = torch.relu(x @ w1 + b1) @ w2 + b2

    assert got.shape == (8, 8)
    torch.testing.assert_close(got, expected)


@numba.njit
def elementwise_pipeline(a, b):
    """Chain: exp(sin(a)) + sqrt(abs(b)) — 5 unary + 1 binary op."""
    return torch.exp(torch.sin(a)) + torch.sqrt(torch.abs(b))


def test_elementwise_pipeline():
    a = torch.linspace(0.1, 1.0, 6)
    b = torch.tensor([-4.0, -1.0, 0.25, 1.0, 4.0, 9.0])
    got = elementwise_pipeline(a, b)
    expected = torch.exp(torch.sin(a)) + torch.sqrt(torch.abs(b))
    torch.testing.assert_close(got, expected)


@numba.njit
def attention_scores(q, k):
    """Dot-product attention: (q @ k^T) scaled by the RMS of q — 4-op graph.

    Uses matmul, mul (element-wise), sum, sqrt.
    Note: ^T is achieved by passing k transposed from Python.
    """
    # Scale = sqrt(sum(q * q))  – proxy for the head dimension
    scale = torch.sqrt(torch.sum(q * q))
    return (q @ k) / scale


def test_attention_scores():
    torch.manual_seed(7)
    q = torch.randn(4, 8)
    k = torch.randn(8, 4)  # pre-transposed
    got = attention_scores(q, k)
    scale = torch.sqrt(torch.sum(q * q))
    expected = (q @ k) / scale
    assert got.shape == (4, 4)
    torch.testing.assert_close(got, expected)


# ---------------------------------------------------------------------------
# Regression: same njit function callable multiple times
# ---------------------------------------------------------------------------


def test_reuse_complex_jit():
    """Verify repeated calls to the same njit function are stable."""
    torch.manual_seed(0)
    x1 = torch.randn(3, 4)
    w1 = torch.randn(4, 8)
    b1 = torch.zeros(8)
    w2 = torch.randn(8, 3)
    b2 = torch.zeros(3)

    out1 = mlp_forward(x1, w1, b1, w2, b2)
    out2 = mlp_forward(x1, w1, b1, w2, b2)
    torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# dtype / shape preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_unary_dtype_preserved(dtype):
    a = torch.ones(4, dtype=dtype)
    result = relu(a)
    assert result.dtype == dtype


@pytest.mark.parametrize(
    "shape",
    [(3,), (2, 3), (2, 3, 4)],
    ids=["1d", "2d", "3d"],
)
def test_elementwise_shape_preserved(shape):
    a = torch.randn(*shape)
    b = torch.randn(*shape)
    assert add(a, b).shape == shape
    assert mul(a, b).shape == shape


# ---------------------------------------------------------------------------
# Tensor methods
# ---------------------------------------------------------------------------


@numba.njit
def numel(a):
    return a.numel()


def test_numel_1d():
    a = torch.ones(7)
    assert numel(a) == 7


def test_numel_2d():
    a = torch.zeros(3, 4)
    assert numel(a) == 12


def test_numel_scalar():
    a = torch.tensor(1.0)
    assert numel(a) == 1
