module om_solve
  use om_kinds, only: dp
  use om_assemble, only: assemble_system
  use om_bc, only: apply_zero_dirichlet
  use om_dense, only: solve_dense_system_multiple
  use om_post, only: compute_cell_fields
  implicit none
contains
  subroutine solve_problem(vertices, tets, region_ids, face_nodes, face_owners, mu_r_table, current_density_table, magnetization_table, &
                           a_nodes, cell_b, cell_h, ok)
    real(dp), intent(in) :: vertices(:, :)
    integer, intent(in) :: tets(:, :)
    integer, intent(in) :: region_ids(:)
    integer, intent(in) :: face_nodes(:, :)
    integer, intent(in) :: face_owners(:, :)
    real(dp), intent(in) :: mu_r_table(:)
    real(dp), intent(in) :: current_density_table(:, :)
    real(dp), intent(in) :: magnetization_table(:, :)
    real(dp), intent(out) :: a_nodes(:, :)
    real(dp), intent(out) :: cell_b(:, :)
    real(dp), intent(out) :: cell_h(:, :)
    logical, intent(out) :: ok

    real(dp), allocatable :: k_global(:, :), f_global(:, :), cell_centers(:, :)
    real(dp), allocatable :: cell_grads(:, :, :)
    logical :: local_ok
    integer :: n_nodes, n_cells

    n_nodes = size(vertices, 2)
    n_cells = size(tets, 2)

    allocate(k_global(n_nodes, n_nodes), f_global(n_nodes, 3), cell_grads(4, 3, n_cells), cell_centers(3, n_cells))

    call assemble_system(vertices, tets, region_ids, face_nodes, face_owners, mu_r_table, current_density_table, magnetization_table, &
                         k_global, f_global, cell_grads, cell_centers, local_ok)
    if (.not. local_ok) then
      ok = .false.
      return
    end if

    call apply_zero_dirichlet(vertices, k_global, f_global)
    call solve_dense_system_multiple(k_global, f_global, a_nodes, local_ok)
    if (.not. local_ok) then
      ok = .false.
      return
    end if

    call compute_cell_fields(tets, region_ids, mu_r_table, current_density_table, magnetization_table, a_nodes, cell_grads, cell_b, cell_h, local_ok)
    if (.not. local_ok) then
      ok = .false.
      return
    end if

    ok = .true.
  end subroutine solve_problem
end module om_solve
