module om_dense
  use om_kinds, only: dp
  implicit none
contains
  subroutine solve_dense_system_multiple(a_in, b_in, x_out, ok)
    real(dp), intent(in) :: a_in(:, :)
    real(dp), intent(in) :: b_in(:, :)
    real(dp), intent(out) :: x_out(:, :)
    logical, intent(out) :: ok

    real(dp), allocatable :: a(:, :), b(:, :), row_tmp(:), rhs_row_tmp(:)
    real(dp) :: factor, pivot, maxabs
    integer :: n, nrhs, i, k, pivot_row, rhs_idx

    n = size(a_in, 1)
    nrhs = size(b_in, 2)
    if (size(a_in, 2) /= n .or. size(b_in, 1) /= n) then
      ok = .false.
      x_out = 0.0_dp
      return
    end if
    if (size(x_out, 1) /= n .or. size(x_out, 2) /= nrhs) then
      ok = .false.
      x_out = 0.0_dp
      return
    end if

    allocate(a(n, n), b(n, nrhs), row_tmp(n), rhs_row_tmp(nrhs))
    a = a_in
    b = b_in
    x_out = 0.0_dp
    ok = .true.

    do k = 1, n
      pivot_row = k
      maxabs = abs(a(k, k))
      do i = k + 1, n
        if (abs(a(i, k)) > maxabs) then
          maxabs = abs(a(i, k))
          pivot_row = i
        end if
      end do

      if (maxabs < 1.0e-18_dp) then
        ok = .false.
        return
      end if

      if (pivot_row /= k) then
        row_tmp = a(k, :)
        a(k, :) = a(pivot_row, :)
        a(pivot_row, :) = row_tmp

        rhs_row_tmp = b(k, :)
        b(k, :) = b(pivot_row, :)
        b(pivot_row, :) = rhs_row_tmp
      end if

      pivot = a(k, k)
      do i = k + 1, n
        factor = a(i, k) / pivot
        if (abs(factor) > 0.0_dp) then
          a(i, k:n) = a(i, k:n) - factor * a(k, k:n)
          b(i, :) = b(i, :) - factor * b(k, :)
        end if
      end do
    end do

    x_out(n, :) = b(n, :) / a(n, n)
    do i = n - 1, 1, -1
      do rhs_idx = 1, nrhs
        x_out(i, rhs_idx) = (b(i, rhs_idx) - sum(a(i, i + 1:n) * x_out(i + 1:n, rhs_idx))) / a(i, i)
      end do
    end do
  end subroutine solve_dense_system_multiple

  subroutine solve_dense_system(a_in, b_in, x_out, ok)
    real(dp), intent(in) :: a_in(:, :)
    real(dp), intent(in) :: b_in(:)
    real(dp), intent(out) :: x_out(:)
    logical, intent(out) :: ok

    real(dp), allocatable :: b_matrix(:, :), x_matrix(:, :)
    integer :: n

    n = size(b_in)
    if (size(a_in, 1) /= n .or. size(a_in, 2) /= n .or. size(x_out) /= n) then
      ok = .false.
      x_out = 0.0_dp
      return
    end if

    allocate(b_matrix(n, 1), x_matrix(n, 1))
    b_matrix(:, 1) = b_in
    call solve_dense_system_multiple(a_in, b_matrix, x_matrix, ok)
    if (ok) then
      x_out = x_matrix(:, 1)
    else
      x_out = 0.0_dp
    end if
  end subroutine solve_dense_system
end module om_dense
