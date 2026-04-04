project('openmagnets', 'fortran', version: '0.1.0')

fc = meson.get_compiler('fortran')
if fc.get_id() == 'gcc'
  add_project_arguments('-ffree-line-length-none', language : 'fortran')
endif

py = import('python').find_installation(pure: false)
py_pkg_dir = py.get_install_dir() / 'openmagnets'

py.install_sources(
  [
    'src/openmagnets/__init__.py',
    'src/openmagnets/_native.py',
    'src/openmagnets/io.py',
    'src/openmagnets/materials.py',
    'src/openmagnets/mesh.py',
    'src/openmagnets/post.py',
    'src/openmagnets/problem.py',
  ],
  subdir: 'openmagnets',
)

shared_library(
  'openmagnets_native',
  [
    'fortran/om_kinds.f90',
    'fortran/om_materials.f90',
    'fortran/om_mesh.f90',
    'fortran/om_elements_tet.f90',
    'fortran/om_assemble.f90',
    'fortran/om_bc.f90',
    'fortran/om_dense.f90',
    'fortran/om_post.f90',
    'fortran/om_solve.f90',
    'fortran/om_native_capi.f90',
  ],
  install: true,
  install_dir: py_pkg_dir,
)
