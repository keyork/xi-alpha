"""后端自动选择（MVP: NumPy，后续: JAX/CuPy）。"""

from __future__ import annotations

from .base import BackendBase
from .numpy_backend import NumPyBackend


def get_backend() -> BackendBase:
    return NumPyBackend()
