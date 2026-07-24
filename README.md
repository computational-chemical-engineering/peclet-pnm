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

## License

MIT.
