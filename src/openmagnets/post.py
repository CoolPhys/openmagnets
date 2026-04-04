from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import TetraMesh


def _normalize_region_ids(region_ids: int | list[int] | tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(region_ids, dtype=int)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("external_region_ids must not be empty")
    if np.any(arr < 0):
        raise ValueError("external_region_ids must be non-negative")
    return tuple(sorted({int(v) for v in arr.tolist()}))


@dataclass(slots=True)
class SolveResult:
    mesh: TetraMesh
    vector_potential: np.ndarray
    cell_B: np.ndarray
    cell_H: np.ndarray
    _cell_centers_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _cell_bounds_cache: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False)
    _cell_neighbors_cache: list[list[int]] | None = field(default=None, init=False, repr=False)
    _cell_grad_B_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _cell_grad_H_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _external_grad_B_cache: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _external_grad_H_cache: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def cell_centers(self) -> np.ndarray:
        if self._cell_centers_cache is None:
            self._cell_centers_cache = self.mesh.cell_centers()
        return self._cell_centers_cache

    def _cell_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cell_bounds_cache is None:
            tet_vertices = self.mesh.vertices[self.mesh.tets]
            self._cell_bounds_cache = (tet_vertices.min(axis=1), tet_vertices.max(axis=1))
        return self._cell_bounds_cache

    def _candidate_cells_for_point(
        self,
        point: np.ndarray,
        allowed_cell_ids: np.ndarray,
        *,
        tol: float = 1.0e-9,
    ) -> np.ndarray:
        mins, maxs = self._cell_bounds()
        local_mins = mins[allowed_cell_ids]
        local_maxs = maxs[allowed_cell_ids]
        mask = np.all(point >= local_mins - tol, axis=1) & np.all(point <= local_maxs + tol, axis=1)
        return allowed_cell_ids[mask]

    def _cell_neighbors(self) -> list[list[int]]:
        if self._cell_neighbors_cache is None:
            neighbors: list[set[int]] = [set() for _ in range(self.mesh.n_cells)]
            for owners in self.mesh.face_adjacency().values():
                if len(owners) == 2:
                    a, b = int(owners[0]), int(owners[1])
                    neighbors[a].add(b)
                    neighbors[b].add(a)
            self._cell_neighbors_cache = [sorted(group) for group in neighbors]
        return self._cell_neighbors_cache

    def _allowed_cell_indices_for_regions(self, region_ids: tuple[int, ...]) -> np.ndarray:
        allowed = np.flatnonzero(np.isin(self.mesh.region_ids, np.asarray(region_ids, dtype=int)))
        if allowed.size == 0:
            raise ValueError(f"no cells were found in external_region_ids={list(region_ids)}")
        return allowed.astype(int)

    def _cell_gradient_order(self, cell_id: int, allowed_cell_ids: np.ndarray | None = None) -> list[int]:
        centers = self.cell_centers
        allowed_set = None
        if allowed_cell_ids is not None:
            allowed_set = {int(idx) for idx in np.asarray(allowed_cell_ids, dtype=int).reshape(-1)}
            allowed_set.discard(int(cell_id))

        primary = self._cell_neighbors()[cell_id]
        if allowed_set is not None:
            primary = [idx for idx in primary if idx in allowed_set]
        remaining = [
            idx
            for idx in range(self.mesh.n_cells)
            if idx != cell_id and idx not in primary and (allowed_set is None or idx in allowed_set)
        ]

        primary = sorted(primary, key=lambda idx: float(np.linalg.norm(centers[idx] - centers[cell_id])))
        remaining = sorted(remaining, key=lambda idx: float(np.linalg.norm(centers[idx] - centers[cell_id])))
        return primary + remaining

    def _reconstruct_cell_gradients(self, values: np.ndarray, allowed_cell_ids: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(self.mesh.n_cells, 3)
        centers = self.cell_centers

        if allowed_cell_ids is None:
            allowed = np.arange(self.mesh.n_cells, dtype=int)
            gradients = np.zeros((self.mesh.n_cells, 3, 3), dtype=float)
        else:
            allowed = np.asarray(allowed_cell_ids, dtype=int).reshape(-1)
            gradients = np.full((self.mesh.n_cells, 3, 3), np.nan, dtype=float)

        if allowed.size == 0:
            return gradients
        if allowed.size == 1:
            gradients[int(allowed[0])] = 0.0
            return gradients

        for cell_id in (int(idx) for idx in allowed):
            candidate_ids: list[int] = []
            for other_id in self._cell_gradient_order(cell_id, allowed):
                candidate_ids.append(int(other_id))
                deltas = centers[candidate_ids] - centers[cell_id]
                if len(candidate_ids) >= 3 and np.linalg.matrix_rank(deltas) == 3:
                    break

            if len(candidate_ids) < 3:
                gradients[cell_id] = 0.0
                continue

            deltas = centers[candidate_ids] - centers[cell_id]
            if np.linalg.matrix_rank(deltas) < 3:
                gradients[cell_id] = 0.0
                continue

            for comp in range(3):
                rhs = values[candidate_ids, comp] - values[cell_id, comp]
                grad, *_ = np.linalg.lstsq(deltas, rhs, rcond=None)
                gradients[cell_id, comp, :] = grad

        return gradients

    @property
    def cell_grad_B(self) -> np.ndarray:
        if self._cell_grad_B_cache is None:
            self._cell_grad_B_cache = self._reconstruct_cell_gradients(self.cell_B)
        return self._cell_grad_B_cache

    @property
    def cell_grad_H(self) -> np.ndarray:
        if self._cell_grad_H_cache is None:
            self._cell_grad_H_cache = self._reconstruct_cell_gradients(self.cell_H)
        return self._cell_grad_H_cache

    def external_cell_grad_B(self, external_region_ids: int | list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        key = _normalize_region_ids(external_region_ids)
        if key not in self._external_grad_B_cache:
            allowed = self._allowed_cell_indices_for_regions(key)
            self._external_grad_B_cache[key] = self._reconstruct_cell_gradients(self.cell_B, allowed)
        return self._external_grad_B_cache[key]

    def external_cell_grad_H(self, external_region_ids: int | list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        key = _normalize_region_ids(external_region_ids)
        if key not in self._external_grad_H_cache:
            allowed = self._allowed_cell_indices_for_regions(key)
            self._external_grad_H_cache[key] = self._reconstruct_cell_gradients(self.cell_H, allowed)
        return self._external_grad_H_cache[key]

    def _sample_cell_values(
        self,
        values: np.ndarray,
        points: np.ndarray | list[list[float]],
        *,
        outside: str = "raise",
        allowed_cell_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if outside not in {"raise", "nan", "nearest"}:
            raise ValueError("outside must be one of 'raise', 'nan', or 'nearest'")

        values_arr = np.asarray(values, dtype=float)
        if values_arr.shape[0] != self.mesh.n_cells:
            raise ValueError("values must have one entry per cell")

        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        centers = self.cell_centers
        allowed = np.arange(self.mesh.n_cells, dtype=int) if allowed_cell_ids is None else np.asarray(allowed_cell_ids, dtype=int).reshape(-1)
        if allowed.size == 0:
            raise ValueError("no allowed cells available for sampling")

        out = np.zeros((pts.shape[0],) + tuple(values_arr.shape[1:]), dtype=float)
        if outside == "nan":
            out[:] = np.nan

        for i, point in enumerate(pts):
            candidates = self._candidate_cells_for_point(point, allowed)
            found = False
            for cell_id in (int(idx) for idx in candidates):
                if self.mesh.tet_contains_point(cell_id, point):
                    out[i] = values_arr[cell_id]
                    found = True
                    break
            if found:
                continue
            if outside == "nearest":
                local = centers[allowed]
                j = int(allowed[int(np.argmin(np.linalg.norm(local - point, axis=1)))])
                out[i] = values_arr[j]
            elif outside == "raise":
                if allowed_cell_ids is None:
                    raise ValueError(f"point {point.tolist()} is outside the tetrahedral mesh")
                raise ValueError(f"point {point.tolist()} is outside the allowed sampling domain")

        return out

    def _sample_cell_vectors(
        self,
        values: np.ndarray,
        points: np.ndarray | list[list[float]],
        *,
        outside: str = "raise",
        allowed_cell_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._sample_cell_values(values, points, outside=outside, allowed_cell_ids=allowed_cell_ids)

    def _sample_cell_tensors(
        self,
        values: np.ndarray,
        points: np.ndarray | list[list[float]],
        *,
        outside: str = "raise",
        allowed_cell_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._sample_cell_values(values, points, outside=outside, allowed_cell_ids=allowed_cell_ids)

    def sample_B(self, points: np.ndarray | list[list[float]], *, outside: str = "raise") -> np.ndarray:
        return self._sample_cell_vectors(self.cell_B, points, outside=outside)

    def sample_H(self, points: np.ndarray | list[list[float]], *, outside: str = "raise") -> np.ndarray:
        return self._sample_cell_vectors(self.cell_H, points, outside=outside)

    def sample_grad_B(self, points: np.ndarray | list[list[float]], *, outside: str = "raise") -> np.ndarray:
        return self._sample_cell_tensors(self.cell_grad_B, points, outside=outside)

    def sample_grad_H(self, points: np.ndarray | list[list[float]], *, outside: str = "raise") -> np.ndarray:
        return self._sample_cell_tensors(self.cell_grad_H, points, outside=outside)

    def sample_external_grad_B(
        self,
        points: np.ndarray | list[list[float]],
        external_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        outside: str = "raise",
    ) -> np.ndarray:
        key = _normalize_region_ids(external_region_ids)
        allowed = self._allowed_cell_indices_for_regions(key)
        return self._sample_cell_tensors(self.external_cell_grad_B(key), points, outside=outside, allowed_cell_ids=allowed)

    def sample_external_grad_H(
        self,
        points: np.ndarray | list[list[float]],
        external_region_ids: int | list[int] | tuple[int, ...] | np.ndarray,
        *,
        outside: str = "raise",
    ) -> np.ndarray:
        key = _normalize_region_ids(external_region_ids)
        allowed = self._allowed_cell_indices_for_regions(key)
        return self._sample_cell_tensors(self.external_cell_grad_H(key), points, outside=outside, allowed_cell_ids=allowed)
