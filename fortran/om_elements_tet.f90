module om_elements_tet
  use om_kinds, only: dp, norm3, dot3, cross3
  implicit none
contains
  subroutine swap_rows(mat, i, j)
    real(dp), intent(inout) :: mat(:, :)
    integer, intent(in) :: i, j
    real(dp) :: tmp(size(mat, 2))
    if (i == j) return
    tmp = mat(i, :)
    mat(i, :) = mat(j, :)
    mat(j, :) = tmp
  end subroutine swap_rows

  subroutine invert4(a, ainv, det_out, ok)
    real(dp), intent(in) :: a(4, 4)
    real(dp), intent(out) :: ainv(4, 4)
    real(dp), intent(out) :: det_out
    logical, intent(out) :: ok
    real(dp) :: aug(4, 8), pivot, factor
    integer :: i, k, pivot_row
    real(dp) :: maxabs
    integer :: det_sign

    aug(:, 1:4) = a
    aug(:, 5:8) = 0.0_dp
    do i = 1, 4
      aug(i, 4 + i) = 1.0_dp
    end do

    det_out = 1.0_dp
    det_sign = 1
    ok = .true.

    do k = 1, 4
      pivot_row = k
      maxabs = abs(aug(k, k))
      do i = k + 1, 4
        if (abs(aug(i, k)) > maxabs) then
          maxabs = abs(aug(i, k))
          pivot_row = i
        end if
      end do

      if (maxabs < 1.0e-18_dp) then
        ok = .false.
        det_out = 0.0_dp
        ainv = 0.0_dp
        return
      end if

      if (pivot_row /= k) then
        call swap_rows(aug, k, pivot_row)
        det_sign = -det_sign
      end if

      pivot = aug(k, k)
      det_out = det_out * pivot
      aug(k, :) = aug(k, :) / pivot

      do i = 1, 4
        if (i == k) cycle
        factor = aug(i, k)
        if (abs(factor) > 0.0_dp) then
          aug(i, :) = aug(i, :) - factor * aug(k, :)
        end if
      end do
    end do

    ainv = aug(:, 5:8)
    if (det_sign < 0) det_out = -det_out
  end subroutine invert4

  subroutine tet_shape_data(verts, volume, grads, ok)
    real(dp), intent(in) :: verts(3, 4)
    real(dp), intent(out) :: volume
    real(dp), intent(out) :: grads(4, 3)
    logical, intent(out) :: ok
    real(dp) :: m(4, 4), inv_m(4, 4), det_m
    integer :: i

    do i = 1, 4
      m(i, 1) = 1.0_dp
      m(i, 2) = verts(1, i)
      m(i, 3) = verts(2, i)
      m(i, 4) = verts(3, i)
    end do

    call invert4(m, inv_m, det_m, ok)
    if (.not. ok) then
      volume = 0.0_dp
      grads = 0.0_dp
      return
    end if

    volume = abs(det_m) / 6.0_dp
    do i = 1, 4
      grads(i, 1) = inv_m(2, i)
      grads(i, 2) = inv_m(3, i)
      grads(i, 3) = inv_m(4, i)
    end do
  end subroutine tet_shape_data

  subroutine local_stiffness(verts, nu, ke, volume, grads, ok)
    real(dp), intent(in) :: verts(3, 4)
    real(dp), intent(in) :: nu
    real(dp), intent(out) :: ke(4, 4)
    real(dp), intent(out) :: volume
    real(dp), intent(out) :: grads(4, 3)
    logical, intent(out) :: ok
    integer :: i, j

    call tet_shape_data(verts, volume, grads, ok)
    if (.not. ok) then
      ke = 0.0_dp
      return
    end if

    ke = 0.0_dp
    do i = 1, 4
      do j = 1, 4
        ke(i, j) = volume * nu * dot3(grads(i, :), grads(j, :))
      end do
    end do
  end subroutine local_stiffness

  function face_area(face_vertices) result(area)
    real(dp), intent(in) :: face_vertices(3, 3)
    real(dp) :: area
    area = 0.5_dp * norm3(cross3(face_vertices(:, 2) - face_vertices(:, 1), &
                                 face_vertices(:, 3) - face_vertices(:, 1)))
  end function face_area

  subroutine oriented_face_normal(face_vertices, owner_centroid, other_centroid, has_other, normal)
    real(dp), intent(in) :: face_vertices(3, 3)
    real(dp), intent(in) :: owner_centroid(3)
    real(dp), intent(in) :: other_centroid(3)
    logical, intent(in) :: has_other
    real(dp), intent(out) :: normal(3)
    real(dp) :: face_ctr(3), nmag

    normal = cross3(face_vertices(:, 2) - face_vertices(:, 1), face_vertices(:, 3) - face_vertices(:, 1))
    face_ctr = (face_vertices(:, 1) + face_vertices(:, 2) + face_vertices(:, 3)) / 3.0_dp

    if (.not. has_other) then
      if (dot3(normal, owner_centroid - face_ctr) > 0.0_dp) normal = -normal
    else
      if (dot3(normal, other_centroid - face_ctr) < 0.0_dp) normal = -normal
    end if

    nmag = norm3(normal)
    if (nmag <= 0.0_dp) then
      normal = 0.0_dp
    else
      normal = normal / nmag
    end if
  end subroutine oriented_face_normal
end module om_elements_tet
