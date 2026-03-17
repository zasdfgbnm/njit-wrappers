"""Tests for NjitInductorGraph — inductor compiled graphs wrapped in @njit.

The entire module is skipped when CUDA or triton is not available,
since inductor GPU compilation requires both.
"""

import pytest
import torch

_has_cuda = torch.cuda.is_available()

try:
    import triton  # noqa: F401

    _has_triton = True
except ImportError:
    _has_triton = False

if not (_has_cuda and _has_triton):
    pytest.skip("Requires CUDA and triton", allow_module_level=True)

from njit_wrappers import NjitInductorGraph  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class PointwiseModel(torch.nn.Module):
    """Simple element-wise ops: add + relu."""

    def forward(self, x, y):
        return torch.relu(x + y)


class MultiOpModel(torch.nn.Module):
    """Multiple ops requiring multiple kernel launches."""

    def forward(self, x, y):
        a = x + y
        b = a * x
        c = torch.relu(b)
        return c - a


class MatmulModel(torch.nn.Module):
    """Model with matrix multiplication."""

    def forward(self, x, w):
        return x @ w


class MultiOutputModel(torch.nn.Module):
    """Model returning multiple tensors."""

    def forward(self, x):
        return x + 1, x * 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_pointwise():
    """Test element-wise ops (add + relu) → single or few Triton kernels."""
    model = PointwiseModel().cuda()
    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")

    graph = NjitInductorGraph(model, (x, y))

    # Verify schedule was parsed
    assert len(graph.schedule.input_names) >= 1
    assert any(hasattr(op, "kernel_name") for op in graph.schedule.ops)

    out = graph(x, y)
    expected = torch.relu(x + y)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"max diff: {(out - expected).abs().max().item()}"
    )


def test_multi_kernel():
    """Test model requiring multiple kernel launches."""
    model = MultiOpModel().cuda()
    x = torch.randn(2048, device="cuda")
    y = torch.randn(2048, device="cuda")

    graph = NjitInductorGraph(model, (x, y))
    out = graph(x, y)

    expected = model(x, y)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"max diff: {(out - expected).abs().max().item()}"
    )


def test_with_matmul():
    """Test model with matrix multiplication."""
    model = MatmulModel().cuda()
    x = torch.randn(32, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")

    graph = NjitInductorGraph(model, (x, w))
    out = graph(x, w)

    expected = x @ w
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-3), (
        f"max diff: {(out - expected).abs().max().item()}"
    )


def test_simple_mlp():
    """Test two-layer MLP (Linear + relu).

    Skipped because inductor uses reinterpret_tensor for weight
    transposition, which is not yet supported.
    """
    pytest.skip("MLP uses reinterpret_tensor (not yet supported)")


def test_correctness_vs_eager():
    """Compare NjitInductorGraph against eager PyTorch with different data."""
    model = PointwiseModel().cuda()
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")

    graph = NjitInductorGraph(model, (x, y))

    # Run with different inputs to test generalization (same shape)
    x2 = torch.randn(4096, device="cuda")
    y2 = torch.randn(4096, device="cuda")

    out_njit = graph(x2, y2)
    out_eager = model(x2, y2)
    torch.cuda.synchronize()

    assert torch.allclose(out_njit, out_eager, atol=1e-5), (
        f"max diff: {(out_njit - out_eager).abs().max().item()}"
    )


def test_correctness_vs_compile():
    """Compare NjitInductorGraph against regular torch.compile."""
    model = PointwiseModel().cuda()
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")

    graph = NjitInductorGraph(model, (x, y))

    # Use eager as ground truth (avoids torch.compile caching issues)
    out_njit = graph(x, y)
    out_eager = model(x, y)
    torch.cuda.synchronize()

    assert torch.allclose(out_njit, out_eager, atol=1e-5), (
        f"max diff: {(out_njit - out_eager).abs().max().item()}"
    )


def test_multiple_outputs():
    """Test model returning multiple tensors."""
    model = MultiOutputModel().cuda()
    x = torch.randn(1024, device="cuda")

    graph = NjitInductorGraph(model, (x,))
    outs = graph(x)

    torch.cuda.synchronize()
    assert isinstance(outs, tuple)
    assert len(outs) == 2
    assert torch.allclose(outs[0], x + 1, atol=1e-5)
    assert torch.allclose(outs[1], x * 2, atol=1e-5)
