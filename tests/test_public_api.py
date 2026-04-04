from __future__ import annotations

import numpy as np
import pytest

import openmagnets as om
from openmagnets import Material, Problem, SolveResult, SurfaceMesh, TetraMesh
from openmagnets.problem import _build_region_tables


def make_box_surface() -> SurfaceMesh:
    return SurfaceMesh(
        vertices=np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=int,
        ),
    )


def make_simple_tet_mesh(region_id: int = 0) -> TetraMesh:
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


def test_public_api_exposes_problem_and_not_shape_helpers() -> None:
    assert om.Problem is Problem
    assert not hasattr(om, "LowPolyMagnetostaticProblem")
    assert not hasattr(om, "make_box")
    assert not hasattr(om, "make_lowpoly_sphere")
    assert hasattr(om, "reload_backend")


def test_problem_exposes_natural_target_field_helpers() -> None:
    assert hasattr(Problem, "without_target_sources")
    assert hasattr(Problem, "solve_without_target_sources")
    assert hasattr(Problem, "external_grad_B_on_target")
    assert hasattr(Problem, "sample_external_grad_B_on_target")


def test_material_rejects_nonpositive_mu_r() -> None:
    with pytest.raises(ValueError, match="mu_r must be positive"):
        Material(name="bad", mu_r=0.0)


def test_surface_mesh_rejects_bad_face_indices() -> None:
    with pytest.raises(ValueError, match="face indices must be within the vertex range"):
        SurfaceMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 9]], dtype=int),
        )


def test_tetra_mesh_rejects_bad_node_indices() -> None:
    with pytest.raises(ValueError, match="tetrahedron node indices must be within the vertex range"):
        TetraMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            tets=np.array([[0, 1, 2, 4]], dtype=int),
            region_ids=np.array([0], dtype=int),
        )


def test_voxel_tetrahedralize_uses_explicit_inputs() -> None:
    surface = make_box_surface()
    tet_mesh = surface.voxel_tetrahedralize(
        padding=(0.5, 0.25, 0.75),
        resolution=(2, 3, 4),
        inside_region_id=5,
        outside_region_id=0,
        region_names={
            0: "air",
            5: "body",
        },
    )

    assert tet_mesh.n_nodes == (2 + 1) * (3 + 1) * (4 + 1)
    assert tet_mesh.n_cells == 2 * 3 * 4 * 6
    assert set(np.unique(tet_mesh.region_ids)).issubset({0, 5})
    assert tet_mesh.region_label(5) == "body"


def test_voxel_tetrahedralize_rejects_same_region_ids() -> None:
    surface = make_box_surface()
    with pytest.raises(ValueError, match="must be different"):
        surface.voxel_tetrahedralize(
            padding=0.5,
            resolution=2,
            inside_region_id=1,
            outside_region_id=1,
        )


def test_voxel_tetrahedralize_rejects_zero_padding() -> None:
    surface = make_box_surface()
    with pytest.raises(ValueError, match="strictly positive"):
        surface.voxel_tetrahedralize(
            padding=0.0,
            resolution=2,
            inside_region_id=1,
            outside_region_id=0,
        )


def test_voxel_tetrahedralize_rejects_resolution_one() -> None:
    surface = make_box_surface()
    with pytest.raises(ValueError, match="resolution must be >= 2"):
        surface.voxel_tetrahedralize(
            padding=0.5,
            resolution=1,
            inside_region_id=1,
            outside_region_id=0,
        )


def test_voxel_tetrahedralize_rejects_open_surface() -> None:
    surface = make_box_surface()
    open_surface = SurfaceMesh(vertices=surface.vertices, faces=surface.faces[:-1])
    with pytest.raises(ValueError, match="closed manifold"):
        open_surface.voxel_tetrahedralize(
            padding=0.5,
            resolution=2,
            inside_region_id=1,
            outside_region_id=0,
        )


def test_problem_compacts_sparse_region_ids() -> None:
    mesh = make_simple_tet_mesh(region_id=1000000)
    compact_region_ids, mu_r, current_density, magnetization = _build_region_tables(
        mesh,
        {1000000: Material(name="body", mu_r=2.0, current_density=[1.0, 2.0, 3.0], magnetization=[4.0, 5.0, 6.0])},
    )
    assert compact_region_ids.tolist() == [0]
    assert mu_r.tolist() == [2.0]
    assert np.allclose(current_density, [[1.0, 2.0, 3.0]])
    assert np.allclose(magnetization, [[4.0, 5.0, 6.0]])


def test_problem_rejects_negative_material_region_ids() -> None:
    mesh = make_simple_tet_mesh(region_id=0)
    with pytest.raises(ValueError, match="non-negative"):
        Problem(mesh=mesh, materials={-1: Material(name="bad")})


def test_problem_dense_estimate_counts_more_than_one_matrix() -> None:
    mesh = make_simple_tet_mesh(region_id=0)
    problem = Problem(mesh=mesh, materials={0: Material(name="air")})
    single_dense_matrix = mesh.n_nodes * mesh.n_nodes * np.dtype(np.float64).itemsize
    assert problem.estimate_dense_matrix_bytes() >= 2 * single_dense_matrix


def test_problem_dense_guard_trips_before_backend() -> None:
    mesh = make_simple_tet_mesh(region_id=0)
    problem = Problem(mesh=mesh, materials={0: Material(name="air")})
    with pytest.raises(MemoryError, match="too large for the current dense solver"):
        problem.solve(max_dense_matrix_bytes=0)


def test_sample_outside_defaults_to_raise() -> None:
    mesh = make_simple_tet_mesh(region_id=0)
    result = SolveResult(
        mesh=mesh,
        vector_potential=np.zeros((mesh.n_nodes, 3), dtype=float),
        cell_B=np.array([[1.0, 2.0, 3.0]], dtype=float),
        cell_H=np.array([[4.0, 5.0, 6.0]], dtype=float),
    )
    with pytest.raises(ValueError, match="outside the tetrahedral mesh"):
        result.sample_B([[2.0, 2.0, 2.0]])


def test_sample_outside_supports_nan_and_nearest() -> None:
    mesh = make_simple_tet_mesh(region_id=0)
    result = SolveResult(
        mesh=mesh,
        vector_potential=np.zeros((mesh.n_nodes, 3), dtype=float),
        cell_B=np.array([[1.0, 2.0, 3.0]], dtype=float),
        cell_H=np.array([[4.0, 5.0, 6.0]], dtype=float),
    )
    sampled_nan = result.sample_B([[2.0, 2.0, 2.0]], outside="nan")
    sampled_nearest = result.sample_H([[2.0, 2.0, 2.0]], outside="nearest")
    assert np.isnan(sampled_nan).all()
    assert np.allclose(sampled_nearest, [[4.0, 5.0, 6.0]])
