from __future__ import annotations

from ._native import BackendRequiredError, backend_info, has_native, reload_backend, require_backend
from .io import load_obj
from .materials import Material
from .mesh import SurfaceMesh, TetraMesh
from .post import SolveResult
from .problem import Problem

__all__ = [
    "SurfaceMesh",
    "TetraMesh",
    "Material",
    "Problem",
    "SolveResult",
    "load_obj",
    "has_native",
    "backend_info",
    "reload_backend",
    "require_backend",
    "BackendRequiredError",
]
