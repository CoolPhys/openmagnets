from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh import SurfaceMesh


def _parse_obj_index(token: str, n_vertices: int) -> int:
    idx = int(token)
    if idx > 0:
        return idx - 1
    if idx < 0:
        return n_vertices + idx
    raise ValueError("OBJ indices are 1-based and cannot be zero")


def load_obj(path: str | Path) -> SurfaceMesh:
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("v "):
            _, x, y, z, *rest = line.split()
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith("f "):
            parts = line.split()[1:]
            idxs: list[int] = []
            for part in parts:
                token = part.split("/")[0]
                idxs.append(_parse_obj_index(token, len(vertices)))
            if len(idxs) < 3:
                continue
            for i in range(1, len(idxs) - 1):
                faces.append([idxs[0], idxs[i], idxs[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"OBJ file {path} did not contain triangulated geometry")

    return SurfaceMesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int))
