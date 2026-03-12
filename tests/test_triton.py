"""Tests for Triton + Numba integration via NumbaTritonKernel."""

import pytest
import torch
import triton
import triton.language as tl

from njit_wrappers import NumbaTritonKernel


def _has_cuda():
    return torch.cuda.is_available()


def _has_numba():
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def _has_triton():
    try:
        import triton as _triton  # noqa: F811

        del _triton
        return True
    except ImportError:
        return False


requires_cuda_numba_triton = pytest.mark.skipif(
    not (_has_cuda() and _has_numba() and _has_triton()),
    reason="Requires CUDA, numba, and triton",
)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scale, mask=mask)


@requires_cuda_numba_triton
def test_vector_add_basic():
    """Test basic vector add via numba.njit launch."""
    import numba

    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_add = numba_add.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

    f(x.data_ptr(), y.data_ptr(), out.data_ptr(), n, stream)
    torch.cuda.synchronize()

    assert torch.allclose(out, x + y), f"max diff: {(out - x - y).abs().max().item()}"


@requires_cuda_numba_triton
def test_vector_add_large():
    """Test vector add with multiple blocks."""
    import numba

    n = 100000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_add = numba_add.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

    f(x.data_ptr(), y.data_ptr(), out.data_ptr(), n, stream)
    torch.cuda.synchronize()

    assert torch.allclose(out, x + y, atol=1e-5), (
        f"max diff: {(out - x - y).abs().max().item()}"
    )


@requires_cuda_numba_triton
def test_mixed_types():
    """Test kernel with mixed pointer and scalar types."""
    import numba

    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    numba_scale = NumbaTritonKernel(
        scale_kernel,
        signature={
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "scale": "fp32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_scale = numba_scale.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, out_ptr, n, scale, stream):
        grid = (n + 1023) // 1024
        launch_scale(grid, 1, 1, stream, x_ptr, out_ptr, n, scale)

    scale_val = 2.5
    f(x.data_ptr(), out.data_ptr(), n, scale_val, stream)
    torch.cuda.synchronize()

    expected = x * scale_val
    assert torch.allclose(out, expected, atol=1e-5), (
        f"max diff: {(out - expected).abs().max().item()}"
    )


@requires_cuda_numba_triton
def test_numba_kernel_matches_expected():
    """Verify numba launch produces correct results matching expected output."""
    import numba

    n = 4096
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    expected = x + y

    out_numba = torch.empty_like(x)
    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_add = numba_add.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

    f(x.data_ptr(), y.data_ptr(), out_numba.data_ptr(), n, stream)
    torch.cuda.synchronize()

    assert torch.allclose(out_numba, expected), (
        f"max diff: {(out_numba - expected).abs().max().item()}"
    )


@requires_cuda_numba_triton
def test_empty_grid():
    """Test that an empty grid (n=0) doesn't crash."""
    import numba

    n = 0
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    y = torch.empty(0, device="cuda", dtype=torch.float32)
    out = torch.empty(0, device="cuda", dtype=torch.float32)

    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_add = numba_add.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

    # Should not crash even with grid=0
    f(x.data_ptr(), y.data_ptr(), out.data_ptr(), n, stream)
    torch.cuda.synchronize()


@requires_cuda_numba_triton
def test_specialization_variant_count():
    """Verify that 2^K variants are compiled for K specializable args."""
    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )

    # 4 specializable args (3 pointers + 1 int) -> 16 variants
    fn_handles = numba_add._fn_handles
    assert len(fn_handles) == 16, f"Expected 16 variants, got {len(fn_handles)}"

    # All-divisible (mask=15) and none-divisible (mask=0) should differ
    assert fn_handles[0] != fn_handles[15], (
        "All-divisible and no-divisible variants should have different "
        "CUfunction handles"
    )


@requires_cuda_numba_triton
def test_specialization_correctness_aligned():
    """Test that the aligned variant produces correct results."""
    import numba

    # Size divisible by 16 to trigger the aligned path
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_add = numba_add.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

    f(x.data_ptr(), y.data_ptr(), out.data_ptr(), n, stream)
    torch.cuda.synchronize()

    assert torch.allclose(out, x + y), f"max diff: {(out - x - y).abs().max().item()}"


@requires_cuda_numba_triton
def test_specialization_correctness_unaligned():
    """Test that the unaligned variant produces correct results."""
    import numba

    # n=1023 is NOT divisible by 16
    n = 1023
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    numba_add = NumbaTritonKernel(
        add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )
    launch_add = numba_add.launch

    stream = torch.cuda.current_stream().cuda_stream

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

    f(x.data_ptr(), y.data_ptr(), out.data_ptr(), n, stream)
    torch.cuda.synchronize()

    assert torch.allclose(out, x + y), f"max diff: {(out - x - y).abs().max().item()}"


@requires_cuda_numba_triton
def test_non_specializable_float_arg():
    """Test that float args are not included in specialization bitmask."""
    numba_scale = NumbaTritonKernel(
        scale_kernel,
        signature={
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "scale": "fp32",
        },
        constexprs={"BLOCK_SIZE": 1024},
    )

    # 3 specializable args (2 pointers + 1 int), float not specializable
    fn_handles = numba_scale._fn_handles
    assert len(fn_handles) == 8, f"Expected 8 variants, got {len(fn_handles)}"
