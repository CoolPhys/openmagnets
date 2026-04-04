# OpenMagnets

OpenMagnets is a magnetostatic solver with a Python API and a native Fortran core.

## public API

- `SurfaceMesh`
- `TetraMesh`
- `Material`
- `Problem`
- `SolveResult`
- `load_obj(...)`
- `has_native()`
- `backend_info()`
- `reload_backend()`
- `require_backend()`

## install

```bash
pip install openmagnets
```

### source build

- Python 3.11+
- `gfortran`
- `ninja`
- build isolation support through `pip`

## rebuild from source

```bash
python scripts/build_native.py
```

## quick start

```python
from openmagnets import Material, Problem, load_obj

surface = load_obj("part.obj")

mesh = surface.voxel_tetrahedralize(
    padding=0.01,
    resolution=(12, 12, 12),
    inside_region_id=10,
    outside_region_id=0,
    region_names={
        0: "air",
        10: "magnet",
    },
)

problem = Problem(
    mesh=mesh,
    materials={
        0: Material(name="air", mu_r=1.0),
        10: Material(name="magnet", mu_r=1.05, magnetization=[0.0, 0.0, 8.0e5]),
    },
)

result = problem.solve()
print(result.cell_B.shape)
```

## gradients

`SolveResult` reconstructs the cell centered gradients

```python
cell_grad_B = result.cell_grad_B
cell_grad_H = result.cell_grad_H

print(cell_grad_B.shape)  
print(result.sample_grad_B([[0.0, 0.0, 0.0]], outside="nearest")[0])
```
