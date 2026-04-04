from __future__ import annotations

import numpy as np

from openmagnets import SolveResult, SurfaceMesh, TetraMesh


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


def make_single_tet_mesh() -> TetraMesh:
    return TetraMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        tets=np.array([[0, 1, 2, 3]], dtype=int),
        region_ids=np.array([0], dtype=int),
    )


def make_linear_gradient_result() -> tuple[SolveResult, np.ndarray, np.ndarray]:
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
    grad_H = np.array(
        [
            [0.5, 0.0, -0.75],
            [1.25, -1.5, 0.25],
            [0.0, 0.5, 2.0],
        ],
        dtype=float,
    )
    offset_B = np.array([2.0, -1.0, 0.5], dtype=float)
    offset_H = np.array([-0.5, 3.0, 1.25], dtype=float)

    cell_B = centers @ grad_B.T + offset_B
    cell_H = centers @ grad_H.T + offset_H

    result = SolveResult(
        mesh=mesh,
        vector_potential=np.zeros((mesh.n_nodes, 3), dtype=float),
        cell_B=cell_B,
        cell_H=cell_H,
    )
    return result, grad_B, grad_H


def test_reconstructed_cell_gradients_match_linear_field() -> None:
    result, grad_B, grad_H = make_linear_gradient_result()
    expected_B = np.broadcast_to(grad_B, result.cell_grad_B.shape)
    expected_H = np.broadcast_to(grad_H, result.cell_grad_H.shape)
    assert np.allclose(result.cell_grad_B, expected_B, atol=1.0e-10)
    assert np.allclose(result.cell_grad_H, expected_H, atol=1.0e-10)


def test_single_cell_gradient_defaults_to_zero_when_reconstruction_is_underdetermined() -> None:
    mesh = make_single_tet_mesh()
    result = SolveResult(
        mesh=mesh,
        vector_potential=np.zeros((mesh.n_nodes, 3), dtype=float),
        cell_B=np.array([[1.0, 2.0, 3.0]], dtype=float),
        cell_H=np.array([[4.0, 5.0, 6.0]], dtype=float),
    )
    assert np.allclose(result.cell_grad_B, 0.0)
    assert np.allclose(result.cell_grad_H, 0.0)


def test_sample_gradients_inside_and_outside_mesh() -> None:
    result, grad_B, grad_H = make_linear_gradient_result()
    inside_point = result.cell_centers[0]

    sampled_inside = result.sample_grad_B([inside_point], outside="raise")[0]
    sampled_nearest = result.sample_grad_H([[9.0, 9.0, 9.0]], outside="nearest")[0]
    sampled_nan = result.sample_grad_B([[9.0, 9.0, 9.0]], outside="nan")

    assert np.allclose(sampled_inside, grad_B, atol=1.0e-10)
    assert np.allclose(sampled_nearest, grad_H, atol=1.0e-10)
    assert np.isnan(sampled_nan).all()
