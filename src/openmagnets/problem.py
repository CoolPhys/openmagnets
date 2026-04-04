from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._native import require_backend
from .materials import Material
from .mesh import TetraMesh
from .post import SolveResult

_DEFAULT_MAX_DENSE_MATRIX_BYTES = 1_500_000_000
_FLOAT64_BYTES = np.dtype(np.float64).itemsize


def _normalize_region_ids(region_ids: int | list[int] | tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(region_ids, dtype=int)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("target_region_ids must not be empty")
    if np.any(arr < 0):
        raise ValueError("target_region_ids must be non-negative")
    return tuple(sorted({int(v) for v in arr.tolist()}))


def _copy_material(material: Material) -> Material:
    return Material(
        name=str(material.name),
        mu_r=float(material.mu_r),
        current_density=np.array(material.current_density, dtype=float, copy=True),
        magnetization=np.array(material.magnetization, dtype=float, copy=True),
    )


def _build_face_data(mesh: TetraMesh) -> tuple[np.ndarray, np.ndarray]:
    face_nodes: list[list[int]] = []
    face_owners: list[list[int]] = []
    for key, owners in mesh.face_adjacency().items():
        face_nodes.append([int(v) for v in key])
        if len(owners) == 1:
            face_owners.append([int(owners[0]) + 1, 0])
        elif len(owners) == 2:
            face_owners.append([int(owners[0]) + 1, int(owners[1]) + 1])
        else:
            raise ValueError("non-manifold tet face ownership is not supported by the native backend")
    if not face_nodes:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    return np.asarray(face_nodes, dtype=np.int32), np.asarray(face_owners, dtype=np.int32)


def _build_region_tables(
    mesh: TetraMesh,
    materials: dict[int, Material],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    material_map = {int(region_id): material for region_id, material in materials.items()}
    if any(region_id < 0 for region_id in material_map):
        raise ValueError("material region ids must be non-negative")

    used_region_ids = sorted(int(region_id) for region_id in np.unique(mesh.region_ids))
    missing = [region_id for region_id in used_region_ids if region_id not in material_map]
    if missing:
        raise KeyError(f"no material assigned for region ids {missing}")

    compact_index = {region_id: idx for idx, region_id in enumerate(used_region_ids)}
    compact_region_ids = np.asarray([compact_index[int(region_id)] for region_id in mesh.region_ids], dtype=np.int32)

    n_regions = len(used_region_ids)
    mu_r = np.zeros((n_regions,), dtype=np.float64)
    current_density = np.zeros((n_regions, 3), dtype=np.float64)
    magnetization = np.zeros((n_regions, 3), dtype=np.float64)

    for region_id in used_region_ids:
        idx = compact_index[region_id]
        material = material_map[region_id]
        mu_r[idx] = float(material.mu_r)
        current_density[idx] = np.asarray(material.current_density, dtype=np.float64).reshape(3)
        magnetization[idx] = np.asarray(material.magnetization, dtype=np.float64).reshape(3)

    return compact_region_ids, mu_r, current_density, magnetization


@dataclass(slots=True)
class Problem:
    mesh: TetraMesh
    materials: dict[int, Material] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.materials = {int(region_id): material for region_id, material in self.materials.items()}
        if any(region_id < 0 for region_id in self.materials):
            raise ValueError("material region ids must be non-negative")

    def material_for_region(self, region_id: int) -> Material:
        region_key = int(region_id)
        if region_key not in self.materials:
            raise KeyError(f"no material assigned for region {region_key}")
        return self.materials[region_key]

    def assign(self, region_id: int, material: Material) -> None:
        region_key = int(region_id)
        if region_key < 0:
            raise ValueError("material region ids must be non-negative")
        self.materials[region_key] = material

    def estimate_dense_matrix_bytes(self) -> int:
        n_nodes = int(self.mesh.n_nodes)
        n_cells = int(self.mesh.n_cells)

        dense_global = n_nodes * n_nodes * _FLOAT64_BYTES
        dense_solver_copy = n_nodes * n_nodes * _FLOAT64_BYTES

        per_node_workspace = (
            n_nodes * 3 * _FLOAT64_BYTES +
            n_nodes * _FLOAT64_BYTES +
            n_nodes * _FLOAT64_BYTES +
            n_nodes * _FLOAT64_BYTES +
            n_nodes * _FLOAT64_BYTES +
            n_nodes * 3 * _FLOAT64_BYTES
        )

        per_cell_workspace = (
            4 * 3 * n_cells * _FLOAT64_BYTES +
            3 * n_cells * _FLOAT64_BYTES +
            3 * n_cells * _FLOAT64_BYTES +
            3 * n_cells * _FLOAT64_BYTES
        )

        return dense_global + dense_solver_copy + per_node_workspace + per_cell_workspace

    def _validated_target_region_ids(self, target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
        target_ids = _normalize_region_ids(target_region_ids)
        missing_materials = [region_id for region_id in target_ids if region_id not in self.materials]
        if missing_materials:
            raise KeyError(f"no material assigned for target region ids {missing_materials}")
        if not np.any(np.isin(self.mesh.region_ids, np.asarray(target_ids, dtype=int))):
            raise ValueError(f"no cells were found in target_region_ids={list(target_ids)}")
        return target_ids

    def _target_cell_indices(self, target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        target_ids = self._validated_target_region_ids(target_region_ids)
        return np.flatnonzero(np.isin(self.mesh.region_ids, np.asarray(target_ids, dtype=int))).astype(int)

    def solve(self, *, max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES) -> SolveResult:
        matrix_bytes = self.estimate_dense_matrix_bytes()
        if max_dense_matrix_bytes is not None and matrix_bytes > int(max_dense_matrix_bytes):
            gib = matrix_bytes / float(1024 ** 3)
            raise MemoryError(
                "Problem is too large for the current dense solver "
                f"(estimated peak native workspace: {gib:.2f} GiB). "
                "Use a smaller mesh or implement a sparse/iterative backend."
            )

        backend = require_backend()
        face_nodes, face_owners = _build_face_data(self.mesh)
        compact_region_ids, mu_r, current_density, magnetization = _build_region_tables(self.mesh, self.materials)

        a_nodes, cell_b, cell_h = backend.solve_problem(
            vertices=np.asarray(self.mesh.vertices, dtype=np.float64),
            tets=np.asarray(self.mesh.tets, dtype=np.int32) + 1,
            region_ids=compact_region_ids,
            face_nodes=face_nodes + 1 if face_nodes.size else face_nodes,
            face_owners=face_owners,
            mu_r=mu_r,
            current_density=current_density,
            magnetization=magnetization,
        )
        return SolveResult(mesh=self.mesh, vector_potential=a_nodes, cell_B=cell_b, cell_H=cell_h)

    def without_target_sources(
        self,
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
    ) -> "Problem":

        target_ids = self._validated_target_region_ids(target_region_ids)
        zero = np.zeros(3, dtype=float)
        new_materials: dict[int, Material] = {}

        for region_id, material in self.materials.items():
            copied = _copy_material(material)
            if int(region_id) in target_ids:
                copied.current_density = zero.copy()
                copied.magnetization = zero.copy()
            new_materials[int(region_id)] = copied

        return Problem(mesh=self.mesh, materials=new_materials)

    def self_excluded_problem(
        self,
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
    ) -> "Problem":
        return self.without_target_sources(target_region_ids)

    def solve_without_target_sources(
        self,
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES,
    ) -> SolveResult:
        return self.without_target_sources(target_region_ids).solve(max_dense_matrix_bytes=max_dense_matrix_bytes)

    def solve_self_excluded(
        self,
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES,
    ) -> SolveResult:
        return self.solve_without_target_sources(target_region_ids, max_dense_matrix_bytes=max_dense_matrix_bytes)

    def external_grad_B_on_target(
        self,
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES,
    ) -> np.ndarray:
        target_ids = self._validated_target_region_ids(target_region_ids)
        result = self.solve_without_target_sources(target_ids, max_dense_matrix_bytes=max_dense_matrix_bytes)
        return result.external_cell_grad_B(target_ids)

    def self_excluded_external_cell_grad_B(
        self,
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES,
    ) -> np.ndarray:
        return self.external_grad_B_on_target(target_region_ids, max_dense_matrix_bytes=max_dense_matrix_bytes)

    def sample_external_grad_B_on_target(
        self,
        points: np.ndarray | list[list[float]],
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        outside: str = "raise",
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES,
    ) -> np.ndarray:
        if outside not in {"raise", "nan", "nearest"}:
            raise ValueError("outside must be one of 'raise', 'nan', or 'nearest'")

        target_ids = self._validated_target_region_ids(target_region_ids)
        result = self.solve_without_target_sources(target_ids, max_dense_matrix_bytes=max_dense_matrix_bytes)

        try:
            return result.sample_external_grad_B(points, target_ids, outside=outside)
        except ValueError as exc:
            if outside == "raise" and "outside the allowed sampling domain" in str(exc):
                raise ValueError(str(exc).replace("allowed sampling domain", "target region domain")) from None
            raise

    def sample_self_excluded_external_grad_B(
        self,
        points: np.ndarray | list[list[float]],
        target_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        outside: str = "raise",
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_DENSE_MATRIX_BYTES,
    ) -> np.ndarray:
        return self.sample_external_grad_B_on_target(
            points,
            target_region_ids,
            outside=outside,
            max_dense_matrix_bytes=max_dense_matrix_bytes,
        )
