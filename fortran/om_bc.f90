module om_bc
  use om_kinds, only: dp
  use om_mesh, only: build_boundary_mask
  implicit none
contains
  subroutine apply_zero_dirichlet(vertices, k_matrix, rhs)
    real(dp), intent(in) :: vertices(:, :)
    real(dp), intent(inout) :: k_matrix(:, :)
    real(dp), intent(inout) :: rhs(:, :)
    logical :: boundary_mask(size(rhs, 1))
    integer :: i, j, n

    n = size(rhs, 1)
    call build_boundary_mask(vertices, boundary_mask)

    do i = 1, n
      if (.not. boundary_mask(i)) cycle
      do j = 1, n
        k_matrix(i, j) = 0.0_dp
        k_matrix(j, i) = 0.0_dp
      end do
      k_matrix(i, i) = 1.0_dp
      rhs(i, :) = 0.0_dp
    end do
  end subroutine apply_zero_dirichlet
end module om_bc
