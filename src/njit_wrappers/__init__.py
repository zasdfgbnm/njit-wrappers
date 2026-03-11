"""njit-wrappers: torch.Tensor support inside numba.njit."""

import njit_wrappers._tensor  # noqa: F401 – registers TensorType with Numba

__version__ = "0.1.0"
