from __future__ import annotations

import numpy as np
import pytest

from openmagnets import Material, Problem, SolveResult, SurfaceMesh


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


def make_problem() -> Problem:
    mesh = make_box_surface().voxel_tetrahedralize(
        padding=0.5,
        resolution=(2, 2, 2),
        inside_region_id=1,
        outside_region_id=0,
        region_names={0: "air", 1: "target"},
    )
    return Problem(
        mesh=mesh,
        materials={
            0: Material(name="air", mu_r=1.0),
            1: Material(name="target", mu_r=1.2, current_density=[1.0, 2.0, 3.0], magnetization=[4.0, 5.0, 6.0]),
        },
    )


def make_fake_result(problem: Problem) -> SolveResult:
    mesh = problem.mesh
    result = SolveResult(
        mesh=mesh,
        vector_potential=np.zeros((mesh.n_nodes, 3), dtype=float),
        cell_B=np.zeros((mesh.n_cells, 3), dtype=float),
        cell_H=np.zeros((mesh.n_cells, 3), dtype=float),
    )
    gradients = np.zeros((mesh.n_cells, 3, 3), dtype=float)
    for cell_id in range(mesh.n_cells):
        gradients[cell_id] = np.eye(3, dtype=float) * float(cell_id + 1)
    result._cell_grad_B_cache = gradients

    target_only_gradients = np.full((mesh.n_cells, 3, 3), np.nan, dtype=float)
    target_cells = np.flatnonzero(problem.mesh.region_ids == 1)
    target_only_gradients[target_cells] = gradients[target_cells]
    result._external_grad_B_cache[(1,)] = target_only_gradients
    return result


def test_without_target_sources_zeroes_target_sources_and_keeps_mu_r() -> None:
    problem = make_problem()
    excluded = problem.without_target_sources(1)

    assert excluded.materials[1].mu_r == pytest.approx(problem.materials[1].mu_r)
    assert np.allclose(excluded.materials[1].current_density, 0.0)
    assert np.allclose(excluded.materials[1].magnetization, 0.0)
    assert np.allclose(excluded.materials[0].current_density, problem.materials[0].current_density)
    assert np.allclose(excluded.materials[0].magnetization, problem.materials[0].magnetization)


def test_external_grad_B_on_target_returns_target_only_values(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = make_problem()
    fake_result = make_fake_result(problem)

    def fake_solve_without_target_sources(self: Problem, target_region_ids, *, max_dense_matrix_bytes=None):
        return fake_result

    monkeypatch.setattr(Problem, "solve_without_target_sources", fake_solve_without_target_sources)

    gradients = problem.external_grad_B_on_target(1)
    target_cells = np.flatnonzero(problem.mesh.region_ids == 1)
    air_cells = np.flatnonzero(problem.mesh.region_ids == 0)

    assert np.allclose(gradients[target_cells], fake_result.cell_grad_B[target_cells])
    assert np.isnan(gradients[air_cells]).all()


def test_sample_external_grad_B_on_target_restricts_to_target_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = make_problem()
    fake_result = make_fake_result(problem)

    def fake_solve_without_target_sources(self: Problem, target_region_ids, *, max_dense_matrix_bytes=None):
        return fake_result

    monkeypatch.setattr(Problem, "solve_without_target_sources", fake_solve_without_target_sources)

    target_cells = np.flatnonzero(problem.mesh.region_ids == 1)
    target_cell = int(target_cells[0])
    air_cell = int(np.flatnonzero(problem.mesh.region_ids == 0)[0])
    target_point = problem.mesh.cell_centers()[target_cell]
    air_point = problem.mesh.cell_centers()[air_cell]
    far_point = np.array([9.0, 9.0, 9.0], dtype=float)
    nearest_target_cell = int(target_cells[int(np.argmin(np.linalg.norm(problem.mesh.cell_centers()[target_cells] - far_point, axis=1)))])

    sampled_target = problem.sample_external_grad_B_on_target([target_point], 1, outside="raise")[0]
    sampled_nearest = problem.sample_external_grad_B_on_target([far_point], 1, outside="nearest")[0]
    sampled_nan = problem.sample_external_grad_B_on_target([far_point], 1, outside="nan")

    assert np.allclose(sampled_target, fake_result.cell_grad_B[target_cell])
    assert np.allclose(sampled_nearest, fake_result.cell_grad_B[nearest_target_cell])
    assert np.isnan(sampled_nan).all()

    with pytest.raises(ValueError, match="outside the target region domain"):
        problem.sample_external_grad_B_on_target([air_point], 1, outside="raise")


def test_legacy_self_excluded_names_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = make_problem()
    fake_result = make_fake_result(problem)

    def fake_solve_without_target_sources(self: Problem, target_region_ids, *, max_dense_matrix_bytes=None):
        return fake_result

    monkeypatch.setattr(Problem, "solve_without_target_sources", fake_solve_without_target_sources)

    new_grad = problem.external_grad_B_on_target(1)
    old_grad = problem.self_excluded_external_cell_grad_B(1)
    assert np.allclose(new_grad, old_grad, equal_nan=True)

    target_point = problem.mesh.cell_centers()[int(np.flatnonzero(problem.mesh.region_ids == 1)[0])]
    new_sample = problem.sample_external_grad_B_on_target([target_point], 1)
    old_sample = problem.sample_self_excluded_external_grad_B([target_point], 1)
    assert np.allclose(new_sample, old_sample, equal_nan=True)

    new_problem = problem.without_target_sources(1)
    old_problem = problem.self_excluded_problem(1)
    assert np.allclose(new_problem.materials[1].current_density, old_problem.materials[1].current_density)
    assert np.allclose(new_problem.materials[1].magnetization, old_problem.materials[1].magnetization)


def test_target_region_validation() -> None:
    problem = make_problem()
    with pytest.raises(KeyError, match="no material assigned"):
        problem.without_target_sources(99)
