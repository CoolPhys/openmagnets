from __future__ import annotations

import numpy as np
import pytest

import openmagnets as om
from openmagnets import Material, Problem, TetraMesh


def make_single_tet_mesh(region_id: int = 0) -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tets = np.array([[0, 1, 2, 3]], dtype=int)
    return TetraMesh(vertices=vertices, tets=tets, region_ids=np.array([region_id], dtype=int))


def test_native_solver_zero_source_smoke() -> None:
    if not om.has_native():
        pytest.skip("native backend not available")

    mesh = make_single_tet_mesh(region_id=0)
    problem = Problem(mesh=mesh, materials={0: Material(name="air", mu_r=1.0)})
    result = problem.solve()

    assert result.vector_potential.shape == (mesh.n_nodes, 3)
    assert result.cell_B.shape == (mesh.n_cells, 3)
    assert result.cell_H.shape == (mesh.n_cells, 3)
    assert np.isfinite(result.vector_potential).all()
    assert np.isfinite(result.cell_B).all()
    assert np.isfinite(result.cell_H).all()
    assert np.allclose(result.vector_potential, 0.0)
    assert np.allclose(result.cell_B, 0.0)
    assert np.allclose(result.cell_H, 0.0)
