"""njit-wrappers: torch.Tensor support inside numba.njit."""

import njit_wrappers._tensor  # noqa: F401  # pyright: ignore[reportUnusedImport]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    if name == "NumbaTritonKernel":
        from njit_wrappers._triton import NumbaTritonKernel

        return NumbaTritonKernel
    if name == "NjitInductorGraph":
        from njit_wrappers._inductor import NjitInductorGraph

        return NjitInductorGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
