module om_mesh
  use om_kinds, only: dp
  implicit none
contains
  subroutine compute_bounds(vertices, lo, hi)
    real(dp), intent(in) :: vertices(:, :)
    real(dp), intent(out) :: lo(3), hi(3)
    lo(1) = minval(vertices(1, :))
    lo(2) = minval(vertices(2, :))
    lo(3) = minval(vertices(3, :))
    hi(1) = maxval(vertices(1, :))
    hi(2) = maxval(vertices(2, :))
    hi(3) = maxval(vertices(3, :))
  end subroutine compute_bounds

  subroutine build_boundary_mask(vertices, boundary_mask)
    real(dp), intent(in) :: vertices(:, :)
    logical, intent(out) :: boundary_mask(:)
    real(dp) :: lo(3), hi(3), extent, tol
    integer :: i

    call compute_bounds(vertices, lo, hi)
    extent = max(max(hi(1) - lo(1), hi(2) - lo(2)), hi(3) - lo(3))
    tol = max(1.0e-12_dp, 1.0e-12_dp * max(1.0_dp, extent))

    do i = 1, size(boundary_mask)
      boundary_mask(i) = abs(vertices(1, i) - lo(1)) <= tol .or. abs(vertices(1, i) - hi(1)) <= tol .or. &
                         abs(vertices(2, i) - lo(2)) <= tol .or. abs(vertices(2, i) - hi(2)) <= tol .or. &
                         abs(vertices(3, i) - lo(3)) <= tol .or. abs(vertices(3, i) - hi(3)) <= tol
    end do
  end subroutine build_boundary_mask

  subroutine compute_cell_centers(vertices, tets, centers)
    real(dp), intent(in) :: vertices(:, :)
    integer, intent(in) :: tets(:, :)
    real(dp), intent(out) :: centers(:, :)
    integer :: cell, i

    do cell = 1, size(tets, 2)
      centers(:, cell) = 0.0_dp
      do i = 1, 4
        centers(:, cell) = centers(:, cell) + vertices(:, tets(i, cell))
      end do
      centers(:, cell) = centers(:, cell) / 4.0_dp
    end do
  end subroutine compute_cell_centers
end module om_mesh
