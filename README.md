# peclet-pnm

**`peclet.pnm` — GPU pore-network extraction from SDF geometry.**

Given a signed-distance-field (SDF) description of a porous solid (negative inside the solid,
positive in the pore space), `peclet.pnm` extracts the pore network:

- **`SDFReader`** — pure-C++ VTI (VTK ImageData) reader for SDF volumes.
- **`extract_pores`** — pore detection: local maxima of the SDF + weighted centroids and radii.
- **`segment_volume`** — marker-controlled watershed segmentation of the pore space
  (marker init → union-find connected-component labelling → flood fill).
- **`extract_topology_gpu`** — pore-to-pore connectivity (throats) from boundary pairs between basins.
- **`extract_pore_network`** — the fused pipeline (SDF uploaded once, segmentation device-resident
  across all three stages): returns `(pores, segmentation, connections)` in one call.

The compute is [Kokkos](https://github.com/kokkos/kokkos) — the same source runs on **CUDA, HIP, and
OpenMP** backends, selected at build time by the install prefix. Part of the
[peclet](https://github.com/computational-chemical-engineering/peclet) suite; split out of
[peclet-flow](https://github.com/computational-chemical-engineering/peclet-flow) (its former
`peclet.flow.pnm` module — the repo's original "pnm_from_sdf" feature).

## Install / build

```bash
# From the peclet suite checkout (Kokkos prefix bootstrapped once by ../tools/bootstrap_deps.sh):
CMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" pip install .

# Or a dev cmake build (nanobind found via the active interpreter):
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j            # -> build/peclet/pnm/_pnm.*.so ; PYTHONPATH=$PWD/build to import
```

Without a Kokkos prefix on `CMAKE_PREFIX_PATH`, the build vendors Kokkos (OpenMP+Serial) via
FetchContent, so `pip install .` works standalone on any Linux with a C++20 toolchain.

## Usage

```python
import peclet.pnm as pnm

sdf_3d, origin_zyx, spacing_zyx = pnm.SDFReader.read_vti("packing.vti")  # (Nz,Ny,Nx) C-order
pores = pnm.extract_pores(sdf_3d, origin_zyx, spacing_zyx)               # Pore(x,y,z,radius) list
seg = pnm.segment_volume(sdf_3d, spacing_zyx)                            # flat per-voxel pore label
conns = pnm.extract_topology_gpu(seg, list(sdf_3d.shape))                # [(label_a, label_b), ...]

# or fused (SDF uploaded once, segmentation stays device-resident across stages):
pores, seg, conns = pnm.extract_pore_network(sdf_3d, origin_zyx, spacing_zyx)
```

Conventions: the SDF array is `(Nz, Ny, Nx)` C-order (x fastest), `origin`/`spacing` are z-y-x;
SDF sign is negative inside the solid — see the suite's `docs/CONVENTIONS.md`.

Smoke tests: `python scripts/test_extraction.py <sdf.vti>` and
`python scripts/verify_segmentation.py <sdf.vti>` (writes a labelled `.vti` + a pore-pair edge list).

## Network flow: throat flow rates + pore pressures from a DNS

`extract_network_flow` turns a converged [peclet-flow](https://github.com/computational-chemical-engineering/peclet-flow)
velocity/pressure field on the same grid into pore-network flow data — the method carried over
from the Voronoi-tessellation PNM of sphere packings (`pnm_voronoi`), where the throat flow was
∫u·n over the Voronoi facet and the pore pressure a trilinear sample at the pore center:

```python
s = peclet.flow.Solver(nx, ny, nz)
...; s.set_body_force(fx, 0, 0); s.set_solid(sdf_xyz, cutcell_pressure=True); ...steps...
net = pnm.extract_network_flow(
    sdf_zyx, origin_zyx, spacing_zyx,
    s.get_uf().T, s.get_vf().T, s.get_wf().T, s.get_p().T,   # zero-copy transposes to zyx
    s.get_ox().T, s.get_oy().T, s.get_oz().T,                # cut-cell face openness
    grad_p_zyx=[0, 0, -fx])                                  # body force f == -grad p_macro
net["throat_flow"]     # Q through each pore-pore interface (o·u·A summed over MAC faces)
net["pore_pressure"]   # periodic p interpolated at each pore center (basin SDF peak)
net["throat_dp"]       # total-pressure drop P_i - P_j (periodic parts + macro gradient
                       # along the throat-anchored min-image path)
net["pore_residual"]   # signed flux over each pore's whole boundary — ~ solver tolerance
```

On the voxel network the throat integral is exact: a throat is a set of grid-aligned MAC faces
and the openness-weighted face velocity is the discrete flux carrier, so per-pore mass balance
holds to the pressure-solve tolerance (`pore_residual` is the built-in check). Fluxes are
accumulated on **flow basins** (gradient-ascent assignment of *every* cell, including cut cells
whose center is inside the solid) — keyed on the segmentation labels alone, the near-wall
staircase flux would bypass the interface (measured 6% on a tube).

Validated in `scripts/verify_network_flow.py` (chamber-tube chain + asymmetric tube lattice,
DNS by peclet.flow): every throat carries the DNS flux to ~1e-11 relative, residuals ~1e-12·F,
g = Q/dp > 0 on all throats, and the dp sum around each loop equals the macroscopic drop.
`scripts/demo_network_flow_packing.py` runs the pipeline on a real sphere packing. Caveats: two
disjoint interfaces between the same two pores merge (pair-keyed throats); on loose packings
(porosity ≳ 0.6) intra-pore pressure variation is comparable to throat drops, so per-throat
g = Q/dp scatters — a property of the point-pressure PNM abstraction, not of the extraction.

## Distributed (MPI) extraction

Built with `-DPECLET_PNM_MPI=ON`, the module also runs the whole pipeline **multi-rank**: the SDF
is decomposed over ranks by the shared peclet-core ORB (the same deterministic partition flow/dem
use), every stage runs per-rank on a 1-cell ghost layer (core `GridHalo` exchange), and the result
is **bit-exact to the single-rank pipeline** — labels are global voxel ids, so the CCL fixpoint,
the watershed flood (Jacobi), the gradient-path pore basins, and the renumbering are all
decomposition-independent.

```python
# mpirun -np 4 python extract.py
import peclet.pnm as pnm
origin, shape = pnm.mpi_block(global_shape_zyx)      # this rank's ORB block of the global grid
local = sdf[origin[0]:origin[0]+shape[0], origin[1]:origin[1]+shape[1], origin[2]:origin[2]+shape[2]]
pores, seg, conns = pnm.extract_pore_network_mpi(local, global_shape_zyx, origin_zyx, spacing_zyx)
# pores: the pores whose peak this rank owns; seg: this rank's block; conns: global (identical everywhere)
```

Validated by `tests/kokkos_mpi` (ctest, np = 1, 2, 4, OpenMP + CUDA): per-voxel segmentation ids,
the pore set, and the connection list all match the single-rank oracle exactly (pore centroid
positions to 1e-5·spacing on GPU — FMA contraction noise; radii and everything integer bitwise).

## License

MIT.
