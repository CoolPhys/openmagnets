module om_assemble
  use om_kinds, only: dp, MU0, norm3, cross3
  use om_materials, only: material_props
  use om_elements_tet, only: local_stiffness, face_area, oriented_face_normal
  use om_mesh, only: compute_cell_centers
  implicit none
contains
  subroutine assemble_system(vertices, tets, region_ids, face_nodes, face_owners, mu_r_table, current_density_table, magnetization_table, &
                             k_global, f_global, cell_grads, cell_centers, ok)
    real(dp), intent(in) :: vertices(:, :)
    integer, intent(in) :: tets(:, :)
    integer, intent(in) :: region_ids(:)
    integer, intent(in) :: face_nodes(:, :)
    integer, intent(in) :: face_owners(:, :)
    real(dp), intent(in) :: mu_r_table(:)
    real(dp), intent(in) :: current_density_table(:, :)
    real(dp), intent(in) :: magnetization_table(:, :)
    real(dp), intent(out) :: k_global(:, :)
    real(dp), intent(out) :: f_global(:, :)
    real(dp), intent(out) :: cell_grads(:, :, :)
    real(dp), intent(out) :: cell_centers(:, :)
    logical, intent(out) :: ok

    integer :: cell, i, j, face, owner_a, owner_b, region_a, region_b
    real(dp) :: verts(3, 4), ke(4, 4), volume, grads(4, 3), nu
    real(dp) :: mu_r, current_density(3), magnetization(3)
    real(dp) :: mu_r_b, current_density_b(3), magnetization_b(3)
    real(dp) :: face_vertices(3, 3), normal(3), m_jump(3), k_face(3), area
    integer :: node_ids(4), face_node_ids(3)
    logical :: local_ok

    ok = .false.
    k_global = 0.0_dp
    f_global = 0.0_dp
    cell_grads = 0.0_dp

    if (size(region_ids) /= size(tets, 2)) return
    if (size(face_nodes, 2) /= size(face_owners, 2)) return

    call compute_cell_centers(vertices, tets, cell_centers)

    do cell = 1, size(tets, 2)
      node_ids = tets(:, cell)
      if (any(node_ids < 1) .or. any(node_ids > size(vertices, 2))) return

      do i = 1, 4
        verts(:, i) = vertices(:, node_ids(i))
      end do

      call material_props(region_ids(cell), mu_r_table, current_density_table, magnetization_table, mu_r, current_density, magnetization, local_ok)
      if (.not. local_ok) return

      nu = 1.0_dp / (MU0 * mu_r)
      call local_stiffness(verts, nu, ke, volume, grads, local_ok)
      if (.not. local_ok) return
      cell_grads(:, :, cell) = grads

      do i = 1, 4
        do j = 1, 4
          k_global(node_ids(i), node_ids(j)) = k_global(node_ids(i), node_ids(j)) + ke(i, j)
        end do
        if (norm3(current_density) > 0.0_dp) then
          f_global(node_ids(i), :) = f_global(node_ids(i), :) + (volume / 4.0_dp) * current_density
        end if
      end do
    end do

    do face = 1, size(face_nodes, 2)
      owner_a = face_owners(1, face)
      owner_b = face_owners(2, face)
      if (owner_a <= 0 .or. owner_a > size(tets, 2)) return
      if (owner_b > size(tets, 2)) return

      face_node_ids = face_nodes(:, face)
      if (any(face_node_ids < 1) .or. any(face_node_ids > size(vertices, 2))) return

      do i = 1, 3
        face_vertices(:, i) = vertices(:, face_node_ids(i))
      end do

      region_a = region_ids(owner_a)
      call material_props(region_a, mu_r_table, current_density_table, magnetization_table, mu_r, current_density, magnetization, local_ok)
      if (.not. local_ok) return

      if (owner_b <= 0) then
        m_jump = magnetization
        if (norm3(m_jump) <= 0.0_dp) cycle
        call oriented_face_normal(face_vertices, cell_centers(:, owner_a), [0.0_dp, 0.0_dp, 0.0_dp], .false., normal)
      else
        region_b = region_ids(owner_b)
        call material_props(region_b, mu_r_table, current_density_table, magnetization_table, mu_r_b, current_density_b, magnetization_b, local_ok)
        if (.not. local_ok) return
        m_jump = magnetization - magnetization_b
        if (norm3(m_jump) <= 0.0_dp) cycle
        call oriented_face_normal(face_vertices, cell_centers(:, owner_a), cell_centers(:, owner_b), .true., normal)
      end if

      k_face = cross3(m_jump, normal)
      area = face_area(face_vertices)
      do i = 1, 3
        f_global(face_node_ids(i), :) = f_global(face_node_ids(i), :) + (area / 3.0_dp) * k_face
      end do
    end do

    ok = .true.
  end subroutine assemble_system
end module om_assemble
