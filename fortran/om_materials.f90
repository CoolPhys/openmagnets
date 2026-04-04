module om_materials
  use om_kinds, only: dp
  implicit none
contains
  subroutine material_props(region_id, mu_r_table, current_density_table, magnetization_table, mu_r, current_density, magnetization, ok)
    integer, intent(in) :: region_id
    real(dp), intent(in) :: mu_r_table(:)
    real(dp), intent(in) :: current_density_table(:, :)
    real(dp), intent(in) :: magnetization_table(:, :)
    real(dp), intent(out) :: mu_r
    real(dp), intent(out) :: current_density(3)
    real(dp), intent(out) :: magnetization(3)
    logical, intent(out) :: ok
    integer :: idx

    idx = region_id + 1
    ok = .false.
    mu_r = 1.0_dp
    current_density = 0.0_dp
    magnetization = 0.0_dp
    if (idx < 1 .or. idx > size(mu_r_table)) return
    mu_r = mu_r_table(idx)
    if (mu_r <= 0.0_dp) return
    current_density = current_density_table(:, idx)
    magnetization = magnetization_table(:, idx)
    ok = .true.
  end subroutine material_props
end module om_materials
