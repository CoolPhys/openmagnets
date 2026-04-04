from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

_DOUBLE = ctypes.c_double
_INT = ctypes.c_int
_DOUBLE_PTR = ctypes.POINTER(_DOUBLE)
_INT_PTR = ctypes.POINTER(_INT)
_LAST_LOAD_ERROR: str | None = None
_BUILD_HINT = "Build the native backend with: python scripts/build_native.py"


class BackendRequiredError(RuntimeError):
    pass


def _as_points(points: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float64)


def _as_int_matrix(values: np.ndarray | list[list[int]], width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.size == 0:
        return np.empty((0, width), dtype=np.int32)
    return np.ascontiguousarray(arr.reshape(-1, width), dtype=np.int32)


def _as_int_vector(values: np.ndarray | list[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int32)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.int32)


def _library_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    if sys.platform.startswith("win"):
        names = ["openmagnets_native.dll", "libopenmagnets_native.dll"]
    elif sys.platform == "darwin":
        names = ["libopenmagnets_native.dylib", "openmagnets_native.dylib"]
    else:
        names = ["libopenmagnets_native.so", "openmagnets_native.so"]
    return [here / name for name in names]


@dataclass(slots=True)
class NativeBackend:
    lib: Any
    path: Path

    @classmethod
    def load(cls) -> "NativeBackend | None":
        global _LAST_LOAD_ERROR
        errors: list[str] = []
        for candidate in _library_candidates():
            if not candidate.exists():
                continue
            try:
                lib = ctypes.CDLL(str(candidate))
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
                continue

            lib.om_solve_c.argtypes = [
                _INT, _INT, _INT, _INT,
                _DOUBLE_PTR, _INT_PTR, _INT_PTR, _INT_PTR, _INT_PTR,
                _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
                _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
                _INT_PTR,
            ]
            lib.om_solve_c.restype = None

            _LAST_LOAD_ERROR = None
            return cls(lib=lib, path=candidate)

        _LAST_LOAD_ERROR = None if not errors else " | ".join(errors)
        return None

    def solve_problem(
        self,
        vertices: np.ndarray,
        tets: np.ndarray,
        region_ids: np.ndarray,
        face_nodes: np.ndarray,
        face_owners: np.ndarray,
        mu_r: np.ndarray,
        current_density: np.ndarray,
        magnetization: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices_arr = _as_points(vertices)
        tets_arr = _as_int_matrix(tets, 4)
        region_ids_arr = _as_int_vector(region_ids)
        face_nodes_arr = _as_int_matrix(face_nodes, 3)
        face_owners_arr = _as_int_matrix(face_owners, 2)
        mu_r_arr = np.ascontiguousarray(np.asarray(mu_r, dtype=np.float64).reshape(-1), dtype=np.float64)
        current_density_arr = np.ascontiguousarray(np.asarray(current_density, dtype=np.float64).reshape(-1, 3), dtype=np.float64)
        magnetization_arr = np.ascontiguousarray(np.asarray(magnetization, dtype=np.float64).reshape(-1, 3), dtype=np.float64)

        if tets_arr.shape[0] != region_ids_arr.shape[0]:
            raise ValueError("region_ids must match cell count")
        if face_nodes_arr.shape[0] != face_owners_arr.shape[0]:
            raise ValueError("face_nodes and face_owners must match face count")
        if mu_r_arr.shape[0] != current_density_arr.shape[0] or mu_r_arr.shape[0] != magnetization_arr.shape[0]:
            raise ValueError("region property arrays must have the same length")
        if np.any(region_ids_arr < 0):
            raise ValueError("region_ids must be non-negative")

        a_out = np.zeros((vertices_arr.shape[0], 3), dtype=np.float64)
        cell_b = np.zeros((tets_arr.shape[0], 3), dtype=np.float64)
        cell_h = np.zeros((tets_arr.shape[0], 3), dtype=np.float64)
        status = _INT(-1)

        self.lib.om_solve_c(
            _INT(vertices_arr.shape[0]),
            _INT(tets_arr.shape[0]),
            _INT(face_nodes_arr.shape[0]),
            _INT(mu_r_arr.shape[0]),
            vertices_arr.ctypes.data_as(_DOUBLE_PTR),
            tets_arr.ctypes.data_as(_INT_PTR),
            region_ids_arr.ctypes.data_as(_INT_PTR),
            face_nodes_arr.ctypes.data_as(_INT_PTR),
            face_owners_arr.ctypes.data_as(_INT_PTR),
            mu_r_arr.ctypes.data_as(_DOUBLE_PTR),
            current_density_arr.ctypes.data_as(_DOUBLE_PTR),
            magnetization_arr.ctypes.data_as(_DOUBLE_PTR),
            a_out.ctypes.data_as(_DOUBLE_PTR),
            cell_b.ctypes.data_as(_DOUBLE_PTR),
            cell_h.ctypes.data_as(_DOUBLE_PTR),
            ctypes.byref(status),
        )

        if status.value != 0:
            raise RuntimeError(f"native solve failed with status {status.value}")
        return a_out, cell_b, cell_h


BACKEND = NativeBackend.load()


def reload_backend() -> NativeBackend | None:
    global BACKEND
    BACKEND = NativeBackend.load()
    return BACKEND


def has_native() -> bool:
    global BACKEND
    if BACKEND is None:
        BACKEND = NativeBackend.load()
    return BACKEND is not None


def backend_info() -> dict[str, str | bool | None]:
    global BACKEND
    if BACKEND is None:
        BACKEND = NativeBackend.load()
    return {
        "native_available": BACKEND is not None,
        "native_path": None if BACKEND is None else str(BACKEND.path),
        "load_error": _LAST_LOAD_ERROR,
        "build_hint": _BUILD_HINT,
    }


def require_backend() -> NativeBackend:
    global BACKEND
    if BACKEND is None:
        BACKEND = NativeBackend.load()
    if BACKEND is None:
        parts = ["OpenMagnets requires the compiled Fortran backend, but no native library was found."]
        if _LAST_LOAD_ERROR:
            parts.append(f"Load error: {_LAST_LOAD_ERROR}")
        parts.append(_BUILD_HINT)
        raise BackendRequiredError(" ".join(parts))
    return BACKEND
