"""Tests for torch.Tensor support in numba.njit."""

import numba
import torch

import njit_wrappers  # noqa: F401 – registers TensorType


@numba.njit
def add(a, b):
    return a + b


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
    c = add(a, b)
    torch.testing.assert_close(c, torch.full((2, 3), 2.0))


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
    """Calling the same njit function twice should work correctly."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    c1 = add(a, b)
    c2 = add(b, a)
    torch.testing.assert_close(c1, c2)
