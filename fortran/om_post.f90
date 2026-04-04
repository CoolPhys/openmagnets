module om_post
  use om_kinds, only: dp, MU0
  use om_materials, only: material_props
  implicit none
contains
  subroutine compute_cell_fields(tets, region_ids, mu_r_table, current_density_table, magnetization_table, a_nodes, cell_grads, cell_b, cell_h, ok)
    integer, intent(in) :: tets(:, :)
    integer, intent(in) :: region_ids(:)
    real(dp), intent(in) :: mu_r_table(:)
    real(dp), intent(in) :: current_density_table(:, :)
    real(dp), intent(in) :: magnetization_table(:, :)
    real(dp), intent(in) :: a_nodes(:, :)
    real(dp), intent(in) :: cell_grads(:, :, :)
    real(dp), intent(out) :: cell_b(:, :)
    real(dp), intent(out) :: cell_h(:, :)
    logical, intent(out) :: ok

    integer :: cell, i, region_id
    real(dp) :: grad_ax(3), grad_ay(3), grad_az(3), mu_r, current_density(3), magnetization(3)
    logical :: local_ok

    ok = .false.
    cell_b = 0.0_dp
    cell_h = 0.0_dp

    do cell = 1, size(tets, 2)
      grad_ax = 0.0_dp
      grad_ay = 0.0_dp
      grad_az = 0.0_dp
      do i = 1, 4
        grad_ax = grad_ax + a_nodes(tets(i, cell), 1) * cell_grads(i, :, cell)
        grad_ay = grad_ay + a_nodes(tets(i, cell), 2) * cell_grads(i, :, cell)
        grad_az = grad_az + a_nodes(tets(i, cell), 3) * cell_grads(i, :, cell)
      end do

      cell_b(cell, 1) = grad_az(2) - grad_ay(3)
      cell_b(cell, 2) = grad_ax(3) - grad_az(1)
      cell_b(cell, 3) = grad_ay(1) - grad_ax(2)

      region_id = region_ids(cell)
      call material_props(region_id, mu_r_table, current_density_table, magnetization_table, mu_r, current_density, magnetization, local_ok)
      if (.not. local_ok) return
      cell_h(cell, :) = cell_b(cell, :) / (MU0 * mu_r) - magnetization
    end do

    ok = .true.
  end subroutine compute_cell_fields
end module om_post
