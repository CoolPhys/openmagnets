module om_kinds
  use iso_c_binding
  implicit none
  integer, parameter :: dp = c_double
  real(dp), parameter :: PI = 3.1415926535897932384626433832795_dp
  real(dp), parameter :: MU0 = 4.0e-7_dp * PI
contains
  pure function norm3(v) result(n)
    real(dp), intent(in) :: v(3)
    real(dp) :: n
    n = sqrt(max(0.0_dp, v(1)*v(1) + v(2)*v(2) + v(3)*v(3)))
  end function norm3

  pure function dot3(a, b) result(res)
    real(dp), intent(in) :: a(3), b(3)
    real(dp) :: res
    res = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
  end function dot3

  pure function cross3(a, b) result(c)
    real(dp), intent(in) :: a(3), b(3)
    real(dp) :: c(3)
    c(1) = a(2)*b(3) - a(3)*b(2)
    c(2) = a(3)*b(1) - a(1)*b(3)
    c(3) = a(1)*b(2) - a(2)*b(1)
  end function cross3
end module om_kinds
