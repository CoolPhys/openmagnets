from __future__ import annotations

import numpy as np

from openmagnets import SolveResult, SurfaceMesh


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


def make_linear_gradient_result() -> tuple[SolveResult, np.ndarray, np.ndarray, np.ndarray]:
    mesh = make_box_surface().voxel_tetrahedralize(
        padding=0.5,
        resolution=(2, 2, 2),
        inside_region_id=1,
        outside_region_id=0,
    )
    centers = mesh.cell_centers()

    grad_B = np.array(
        [
            [1.5, -0.25, 0.5],
            [0.75, 2.0, -1.25],
            [-0.5, 0.25, 1.0],
        ],
        dtype=float,
    )
    offset_B = np.array([2.0, -1.0, 0.5], dtype=float)

    cell_B = centers @ grad_B.T + offset_B
    cell_H = np.zeros_like(cell_B)

    result = SolveResult(
        mesh=mesh,
        vector_potential=np.zeros((mesh.n_nodes, 3), dtype=float),
        cell_B=cell_B,
        cell_H=cell_H,
    )
    return result, grad_B, centers, mesh.region_ids


def test_external_B_gradient_reconstruction_matches_linear_external_field() -> None:
    result, grad_B, centers, region_ids = make_linear_gradient_result()
    external_grads = result.external_cell_grad_B(0)
    external_cells = np.flatnonzero(region_ids == 0)
    expected = np.broadcast_to(grad_B, (external_cells.size, 3, 3))
    assert np.allclose(external_grads[external_cells], expected, atol=1.0e-10)
    internal_cells = np.flatnonzero(region_ids == 1)
    assert np.isnan(external_grads[internal_cells]).all()


def test_sample_external_B_gradient_raises_on_internal_point() -> None:
    result, grad_B, centers, region_ids = make_linear_gradient_result()
    internal_point = centers[np.flatnonzero(region_ids == 1)[0]]
    try:
        result.sample_external_grad_B([internal_point], external_region_ids=0, outside="raise")
    except ValueError as exc:
        assert "outside the allowed sampling domain" in str(exc)
    else:
        raise AssertionError("expected ValueError for internal point sampling")


def test_sample_external_B_gradient_nearest_and_nan() -> None:
    result, grad_B, centers, region_ids = make_linear_gradient_result()
    point = [9.0, 9.0, 9.0]
    sampled_nearest = result.sample_external_grad_B([point], external_region_ids=0, outside="nearest")[0]
    sampled_nan = result.sample_external_grad_B([point], external_region_ids=0, outside="nan")
    assert np.allclose(sampled_nearest, grad_B, atol=1.0e-10)
    assert np.isnan(sampled_nan).all()
