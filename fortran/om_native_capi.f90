module om_native_capi
  use iso_c_binding
  use om_kinds, only: dp
  use om_solve, only: solve_problem
  implicit none
contains
  subroutine unpack_vec3_flat(flat, n, arr)
    integer, intent(in) :: n
    real(c_double), intent(in) :: flat(*)
    real(dp), intent(out) :: arr(3, n)
    integer :: i, base
    do i = 1, n
      base = 3 * (i - 1)
      arr(1, i) = flat(base + 1)
      arr(2, i) = flat(base + 2)
      arr(3, i) = flat(base + 3)
    end do
  end subroutine unpack_vec3_flat

  subroutine unpack_int4_flat(flat, n, arr)
    integer, intent(in) :: n
    integer(c_int), intent(in) :: flat(*)
    integer, intent(out) :: arr(4, n)
    integer :: i, base
    do i = 1, n
      base = 4 * (i - 1)
      arr(1, i) = flat(base + 1)
      arr(2, i) = flat(base + 2)
      arr(3, i) = flat(base + 3)
      arr(4, i) = flat(base + 4)
    end do
  end subroutine unpack_int4_flat

  subroutine unpack_int3_flat(flat, n, arr)
    integer, intent(in) :: n
    integer(c_int), intent(in) :: flat(*)
    integer, intent(out) :: arr(3, n)
    integer :: i, base
    do i = 1, n
      base = 3 * (i - 1)
      arr(1, i) = flat(base + 1)
      arr(2, i) = flat(base + 2)
      arr(3, i) = flat(base + 3)
    end do
  end subroutine unpack_int3_flat

  subroutine unpack_int2_flat(flat, n, arr)
    integer, intent(in) :: n
    integer(c_int), intent(in) :: flat(*)
    integer, intent(out) :: arr(2, n)
    integer :: i, base
    do i = 1, n
      base = 2 * (i - 1)
      arr(1, i) = flat(base + 1)
      arr(2, i) = flat(base + 2)
    end do
  end subroutine unpack_int2_flat

  subroutine unpack_region_tables(n_regions, mu_r_flat, current_density_flat, magnetization_flat, mu_r_table, current_density_table, magnetization_table)
    integer, intent(in) :: n_regions
    real(c_double), intent(in) :: mu_r_flat(*)
    real(c_double), intent(in) :: current_density_flat(*)
    real(c_double), intent(in) :: magnetization_flat(*)
    real(dp), intent(out) :: mu_r_table(n_regions)
    real(dp), intent(out) :: current_density_table(3, n_regions)
    real(dp), intent(out) :: magnetization_table(3, n_regions)
    integer :: i, base
    do i = 1, n_regions
      mu_r_table(i) = mu_r_flat(i)
      base = 3 * (i - 1)
      current_density_table(1, i) = current_density_flat(base + 1)
      current_density_table(2, i) = current_density_flat(base + 2)
      current_density_table(3, i) = current_density_flat(base + 3)
      magnetization_table(1, i) = magnetization_flat(base + 1)
      magnetization_table(2, i) = magnetization_flat(base + 2)
      magnetization_table(3, i) = magnetization_flat(base + 3)
    end do
  end subroutine unpack_region_tables

  subroutine pack_vec3_nodes(arr, n, flat)
    integer, intent(in) :: n
    real(dp), intent(in) :: arr(n, 3)
    real(c_double), intent(out) :: flat(*)
    integer :: i, base
    do i = 1, n
      base = 3 * (i - 1)
      flat(base + 1) = arr(i, 1)
      flat(base + 2) = arr(i, 2)
      flat(base + 3) = arr(i, 3)
    end do
  end subroutine pack_vec3_nodes

  subroutine om_solve_c(n_nodes, n_cells, n_faces, n_regions, &
                        vertices_flat, tets_flat, region_ids_in, face_nodes_flat, face_owners_flat, &
                        mu_r_flat, current_density_flat, magnetization_flat, &
                        a_out_flat, cell_b_flat, cell_h_flat, status_code) bind(c, name="om_solve_c")
    integer(c_int), value, intent(in) :: n_nodes, n_cells, n_faces, n_regions
    real(c_double), intent(in) :: vertices_flat(*)
    integer(c_int), intent(in) :: tets_flat(*)
    integer(c_int), intent(in) :: region_ids_in(*)
    integer(c_int), intent(in) :: face_nodes_flat(*)
    integer(c_int), intent(in) :: face_owners_flat(*)
    real(c_double), intent(in) :: mu_r_flat(*)
    real(c_double), intent(in) :: current_density_flat(*)
    real(c_double), intent(in) :: magnetization_flat(*)
    real(c_double), intent(out) :: a_out_flat(*)
    real(c_double), intent(out) :: cell_b_flat(*)
    real(c_double), intent(out) :: cell_h_flat(*)
    integer(c_int), intent(out) :: status_code

    real(dp), allocatable :: vertices(:, :)
    integer, allocatable :: tets(:, :), region_ids(:), face_nodes(:, :), face_owners(:, :)
    real(dp), allocatable :: mu_r_table(:), current_density_table(:, :), magnetization_table(:, :)
    real(dp), allocatable :: a_nodes(:, :), cell_b(:, :), cell_h(:, :)
    logical :: ok
    integer :: i

    status_code = 1_c_int
    if (n_nodes <= 0 .or. n_cells <= 0 .or. n_regions <= 0) then
      status_code = 10_c_int
      return
    end if

    allocate(vertices(3, n_nodes), tets(4, n_cells), region_ids(n_cells), a_nodes(n_nodes, 3), cell_b(n_cells, 3), cell_h(n_cells, 3))
    allocate(mu_r_table(n_regions), current_density_table(3, n_regions), magnetization_table(3, n_regions))

    call unpack_vec3_flat(vertices_flat, n_nodes, vertices)
    call unpack_int4_flat(tets_flat, n_cells, tets)

    do i = 1, n_cells
      region_ids(i) = region_ids_in(i)
    end do

    if (n_faces > 0) then
      allocate(face_nodes(3, n_faces), face_owners(2, n_faces))
      call unpack_int3_flat(face_nodes_flat, n_faces, face_nodes)
      call unpack_int2_flat(face_owners_flat, n_faces, face_owners)
    else
      allocate(face_nodes(3, 1), face_owners(2, 1))
      face_nodes = 0
      face_owners = 0
    end if

    call unpack_region_tables(n_regions, mu_r_flat, current_density_flat, magnetization_flat, mu_r_table, current_density_table, magnetization_table)

    call solve_problem(vertices, tets, region_ids, face_nodes, face_owners, mu_r_table, current_density_table, magnetization_table, &
                       a_nodes, cell_b, cell_h, ok)
    if (.not. ok) then
      status_code = 20_c_int
      return
    end if

    call pack_vec3_nodes(a_nodes, n_nodes, a_out_flat)
    call pack_vec3_nodes(cell_b, n_cells, cell_b_flat)
    call pack_vec3_nodes(cell_h, n_cells, cell_h_flat)
    status_code = 0_c_int
  end subroutine om_solve_c
end module om_native_capi
