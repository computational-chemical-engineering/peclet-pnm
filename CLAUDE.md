# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`peclet.pnm` — pore-network extraction from SDF geometry, as an importable Python module. Split out
of `peclet-flow` (2026-07; it was `peclet.flow.pnm`, the original "pnm_from_sdf" feature). The
compute is **Kokkos** (CUDA / HIP / OpenMP — backend selected by the install prefix); the VTI reader
is pure C++. Part of the peclet suite (see `../CLAUDE.md` and `../docs/` for suite-wide conventions).

**Sources** (all under `src/`):
- `pore_extraction.hpp` — `namespace pnm`, header-only Kokkos compute: pore detection (local SDF
  maxima + weighted centroid), marker-controlled watershed segmentation (marker init → union-find
  CCL → flood fill), gradient-path pore basins, boundary-pair throat topology. Device kernels live
  in the `.hpp` compiled as C++ (never `.cu`).
- `pnm_bindings.cpp` — the nanobind module `peclet.pnm._pnm`: `SDFReader`, `extract_pores`,
  `segment_volume`, `extract_topology_gpu`, and the fused `extract_pore_network` (SDF uploaded once,
  segmentation device-resident across stages). Uses core's zero-copy View↔ndarray bridge.
- `sdf_reader.{h,cpp}` — pure-C++ VTI (VTK ImageData) reader, backend-free.
- `pore_extraction_mpi.hpp` — the **distributed** pipeline (gated `PECLET_PNM_MPI`): core ORB
  decomposition + g=1 `GridHalo` exchange; labels are GLOBAL voxel ids so every fixpoint is
  decomposition-independent → **bit-exact to single-rank**. Stage design: local union-find CCL +
  one-shot boundary-graph merge (allgathered surface pairs, host union-find); Jacobi flood
  (sweep-for-sweep = the single-rank Jacobi flood); gradient roots via hold-at-ghost pointer
  jumping with a `~root` finalization marker (NEVER store a remote mid-chain gid — that strands
  the chase outside the ghost ring and fragments basins; measured on plateau-heavy fields);
  renumber by global min-appearance gid (= single-rank first-encounter order).

## Build & test

```bash
# nanobind found via the active interpreter; Kokkos from the suite prefix
# (../extern/install/<backend>, built once by ../tools/bootstrap_deps.sh).
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j            # -> build/peclet/pnm/_pnm.*.so
PYTHONPATH=$PWD/build python scripts/test_extraction.py <sdf.vti>      # pore extraction smoke test
PYTHONPATH=$PWD/build python scripts/verify_segmentation.py <sdf.vti>  # watershed + topology
# Canonical install: CMAKE_PREFIX_PATH=... pip install .   (-> peclet.pnm)
```

Without a prefix, `cmake/PecletDeps.cmake` vendors Kokkos (OpenMP+Serial) + the peclet-core headers
via FetchContent (self-contained wheel path, `PECLET_VENDOR_DEPS=ON` in cibuildwheel). Keep the
`PECLET_*_TAG` pins in lockstep with `../tools/bootstrap_deps.sh`.

Multi-rank (MPI): add `-DPECLET_PNM_MPI=ON` to expose `mpi_rank`/`mpi_size`/`mpi_block` +
`extract_pore_network_mpi` (collective; see README). The C++ MPI ctests (np=1,2,4, distributed vs
single-rank oracle, bit-exact):
```bash
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun          # FORCE mpirun — ParaView's mpiexec runs singletons
cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
```
GPU pore-centroid caveat: nvcc FMA-contracts the centroid accumulation differently in the oracle
vs distributed kernels, so pore POSITIONS are compared to 1e-5·spacing on CUDA (bitwise on
OpenMP); seg ids, radii and connections are bitwise everywhere. The single-rank flood fill is
deliberately Jacobi (double-buffered) — deterministic AND what the distributed flood matches
sweep-for-sweep; don't "optimize" it back to in-place.

## Conventions

- SDF sign: **negative inside solid**, positive in the pore space (suite-wide).
- Python arrays are `(Nz, Ny, Nx)` C-order (x fastest — contiguous with the flat x-fastest layout);
  `origin`/`spacing` tuples are **z-y-x**. `segment_volume` returns a flat label vector.
- `Kokkos::initialize` happens at import; `Kokkos::finalize` is registered via `atexit`
  (REQUIRED on CUDA — see the comment in `pnm_bindings.cpp`).
- Test VTI inputs live in the suite (e.g. `../flow/data/packing_ring.vti`), not in this repo.
