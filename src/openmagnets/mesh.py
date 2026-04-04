from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_vertices(value: np.ndarray | list[list[float]] | list[tuple[float, float, float]]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    return arr.reshape(-1, 3)


def _as_faces(value: np.ndarray | list[list[int]] | list[tuple[int, int, int]]) -> np.ndarray:
    arr = np.asarray(value, dtype=int)
    if arr.size == 0:
        return np.empty((0, 3), dtype=int)
    return arr.reshape(-1, 3)


def _as_padding(value: float | tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.repeat(float(arr), 3)
    arr = arr.reshape(3)
    if np.any(arr < 0.0):
        raise ValueError("padding must be non-negative")
    return arr


def _as_resolution(value: int | tuple[int, int, int] | list[int] | np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(value, dtype=int)
    if arr.ndim == 0:
        arr = np.repeat(int(arr), 3)
    arr = arr.reshape(3)
    if np.any(arr < 1):
        raise ValueError("resolution must be >= 1 in every axis")
    return int(arr[0]), int(arr[1]), int(arr[2])


def _tet_volume(verts: np.ndarray) -> float:
    a = verts[1] - verts[0]
    b = verts[2] - verts[0]
    c = verts[3] - verts[0]
    return abs(float(np.dot(a, np.cross(b, c)))) / 6.0


@dataclass(slots=True)
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = _as_vertices(self.vertices)
        self.faces = _as_faces(self.faces)
        if self.vertices.shape[0] == 0 or self.faces.shape[0] == 0:
            raise ValueError("surface mesh must contain vertices and faces")
        if self.vertices.shape[0] < 4:
            raise ValueError("surface mesh must contain at least 4 vertices")
        if np.any(self.faces < 0) or np.any(self.faces >= self.vertices.shape[0]):
            raise ValueError("face indices must be within the vertex range")
        if np.any([len(set(int(v) for v in face)) < 3 for face in self.faces]):
            raise ValueError("surface mesh faces must contain 3 distinct vertex indices")
        areas = np.linalg.norm(
            np.cross(
                self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
                self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]],
            ),
            axis=1,
        )
        if np.any(areas <= 0.0):
            raise ValueError("surface mesh faces must have positive area")

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def edge_counts(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for face in self.faces:
            a, b, c = (int(v) for v in face)
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def is_closed_manifold(self) -> bool:
        counts = self.edge_counts()
        return bool(counts) and all(count == 2 for count in counts.values())

    def oriented_faces(self) -> np.ndarray:
        faces = self.faces.copy()
        ctr = self.centroid
        for i, face in enumerate(faces):
            v0, v1, v2 = self.vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            face_ctr = (v0 + v1 + v2) / 3.0
            if float(np.dot(normal, ctr - face_ctr)) > 0.0:
                faces[i] = np.array([face[0], face[2], face[1]], dtype=int)
        return faces

    def is_convex(self, tol: float = 1.0e-9) -> bool:
        for face in self.oriented_faces():
            v0, v1, v2 = self.vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            n = float(np.linalg.norm(normal))
            if n == 0.0:
                return False
            signed = (self.vertices - v0) @ (normal / n)
            if np.any(signed > tol):
                return False
        return True

    def contains_points_convex(self, points: np.ndarray | list[list[float]], tol: float = 1.0e-9) -> np.ndarray:
        pts = _as_vertices(points)
        inside = np.ones((pts.shape[0],), dtype=bool)
        for face in self.oriented_faces():
            v0, v1, v2 = self.vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            n = float(np.linalg.norm(normal))
            if n == 0.0:
                continue
            signed = (pts - v0) @ (normal / n)
            inside &= signed <= tol
        return inside

    def voxel_tetrahedralize(
        self,
        *,
        padding: float | tuple[float, float, float] | list[float] | np.ndarray,
        resolution: int | tuple[int, int, int] | list[int] | np.ndarray,
        inside_region_id: int,
        outside_region_id: int,
        region_names: dict[int, str] | None = None,
    ) -> "TetraMesh":
        inside_region_id = int(inside_region_id)
        outside_region_id = int(outside_region_id)
        if inside_region_id == outside_region_id:
            raise ValueError("inside_region_id and outside_region_id must be different")
        if inside_region_id < 0 or outside_region_id < 0:
            raise ValueError("region ids must be non-negative")

        padding_vec = _as_padding(padding)
        if np.any(padding_vec <= 0.0):
            raise ValueError("padding must be strictly positive in every axis")
        nx_cells, ny_cells, nz_cells = _as_resolution(resolution)
        if min(nx_cells, ny_cells, nz_cells) < 2:
            raise ValueError("resolution must be >= 2 in every axis to avoid a fully constrained trivial mesh")
        if not self.is_closed_manifold():
            raise ValueError("voxel_tetrahedralize requires a closed manifold triangular surface mesh")
        if not self.is_convex():
            raise ValueError("voxel_tetrahedralize currently only supports convex closed surface meshes")

        lo, hi = self.bounds
        if np.any(hi <= lo):
            raise ValueError("surface mesh bounds must have positive extent in every axis")
        lo = lo - padding_vec
        hi = hi + padding_vec

        xs = np.linspace(lo[0], hi[0], nx_cells + 1)
        ys = np.linspace(lo[1], hi[1], ny_cells + 1)
        zs = np.linspace(lo[2], hi[2], nz_cells + 1)

        nx = nx_cells + 1
        ny = ny_cells + 1

        def node_index(i: int, j: int, k: int) -> int:
            return i + nx * (j + ny * k)

        vertices = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
        cube_tets = np.array(
            [
                [0, 1, 3, 7],
                [0, 3, 2, 7],
                [0, 2, 6, 7],
                [0, 6, 4, 7],
                [0, 4, 5, 7],
                [0, 5, 1, 7],
            ],
            dtype=int,
        )

        tets: list[list[int]] = []
        for k in range(nz_cells):
            for j in range(ny_cells):
                for i in range(nx_cells):
                    cube_nodes = np.array(
                        [
                            node_index(i, j, k),
                            node_index(i + 1, j, k),
                            node_index(i, j + 1, k),
                            node_index(i + 1, j + 1, k),
                            node_index(i, j, k + 1),
                            node_index(i + 1, j, k + 1),
                            node_index(i, j + 1, k + 1),
                            node_index(i + 1, j + 1, k + 1),
                        ],
                        dtype=int,
                    )
                    for local in cube_tets:
                        tets.append(cube_nodes[local].tolist())

        tet_array = np.asarray(tets, dtype=int)
        centers = vertices[tet_array].mean(axis=1)
        inside = self.contains_points_convex(centers)
        region_ids = np.where(inside, inside_region_id, outside_region_id).astype(int)

        resolved_region_names = {} if region_names is None else {int(k): str(v) for k, v in region_names.items()}

        return TetraMesh(
            vertices=vertices,
            tets=tet_array,
            region_ids=region_ids,
            region_names=resolved_region_names,
        )


@dataclass(slots=True)
class TetraMesh:
    vertices: np.ndarray
    tets: np.ndarray
    region_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    region_names: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = _as_vertices(self.vertices)
        self.tets = np.asarray(self.tets, dtype=int).reshape(-1, 4)
        if self.vertices.shape[0] == 0 or self.tets.shape[0] == 0:
            raise ValueError("tetra mesh must contain vertices and tetrahedra")
        if self.region_ids.size == 0:
            self.region_ids = np.zeros((self.tets.shape[0],), dtype=int)
        else:
            self.region_ids = np.asarray(self.region_ids, dtype=int).reshape(-1)
        if self.region_ids.shape[0] != self.tets.shape[0]:
            raise ValueError("region_ids must match the number of tetrahedra")
        if np.any(self.region_ids < 0):
            raise ValueError("region_ids must be non-negative")
        if np.any(self.tets < 0) or np.any(self.tets >= self.vertices.shape[0]):
            raise ValueError("tetrahedron node indices must be within the vertex range")
        if np.any([len(set(int(v) for v in tet)) < 4 for tet in self.tets]):
            raise ValueError("tetrahedra must contain 4 distinct vertex indices")
        if np.any(self.cell_volumes() <= 0.0):
            raise ValueError("tetrahedra must have positive volume")
        self.region_names = {int(k): str(v) for k, v in self.region_names.items()}

    @property
    def n_nodes(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.tets.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def cell_centers(self) -> np.ndarray:
        return self.vertices[self.tets].mean(axis=1)

    def cell_volumes(self) -> np.ndarray:
        return np.array([_tet_volume(self.vertices[tet]) for tet in self.tets], dtype=float)

    def total_volume(self, region_id: int | None = None) -> float:
        volumes = self.cell_volumes()
        if region_id is None:
            return float(volumes.sum())
        return float(volumes[self.region_ids == region_id].sum())

    def boundary_node_mask(self) -> np.ndarray:
        lo, hi = self.bounds
        mask = np.zeros((self.n_nodes,), dtype=bool)
        for axis in range(3):
            mask |= np.isclose(self.vertices[:, axis], lo[axis])
            mask |= np.isclose(self.vertices[:, axis], hi[axis])
        return mask

    def face_adjacency(self) -> dict[tuple[int, int, int], list[int]]:
        local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        mapping: dict[tuple[int, int, int], list[int]] = {}
        for cell_id, tet in enumerate(self.tets):
            for face in local_faces:
                key = tuple(sorted(int(tet[idx]) for idx in face))
                mapping.setdefault(key, []).append(cell_id)
        return mapping

    def region_label(self, region_id: int) -> str:
        return self.region_names.get(int(region_id), f"region_{int(region_id)}")

    def tet_contains_point(self, tet_index: int, point: np.ndarray) -> bool:
        tet = self.vertices[self.tets[tet_index]]
        p = np.asarray(point, dtype=float).reshape(3)
        m = np.vstack((tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0])).T
        det = float(np.linalg.det(m))
        if abs(det) < 1.0e-14:
            return False
        bary = np.linalg.solve(m, p - tet[0])
        l1, l2, l3 = bary
        l0 = 1.0 - l1 - l2 - l3
        w = np.array([l0, l1, l2, l3], dtype=float)
        return bool(np.all(w >= -1.0e-9) and np.all(w <= 1.0 + 1.0e-9))
