# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`peclet.pnm` — pore-network extraction from SDF geometry, as an importable Python module. Split out
of `peclet-flow` (2026-07; it was `peclet.flow.pnm`, the original "pnm_from_sdf" feature). The
compute is **Kokkos** (CUDA / HIP / OpenMP — backend selected by the install prefix); the VTI reader
is pure C++. Part of the peclet suite (see `../CLAUDE.md` and `../docs/` for suite-wide conventions).

**Sources** (all under `src/`):
- `pore_kernels.hpp` — `namespace pnm::kernels`, the ONE set of stage kernels both pipelines run
  (QUALITY_PLAN G.3): pore detection, marker init, union-find CCL merge/flatten/fixpoint, Jacobi
  flood sweep/fixpoint, the steepest-neighbour stencil (+ the single-rank gradient walk, the
  forest init and the hold-at-ghost resolve round), boundary pairs, and the network-flow face
  stages (face init, face CCL merge, min-fid, core/leftover labelling, film attachment, throat
  flux, the throat-slot and pressure-drop host helpers). Every kernel is templated on a GEOMETRY
  POLICY (`GridGeo` = the whole periodic grid, single-rank; `BlockGeo` = a ghost-extended block,
  MPI) that supplies `cell()/gid()/owned()/inOwned()/localOf()/...` — same expressions, same
  evaluation order in both instantiations, which is what keeps the multi-rank result bit-exact.
  A stage-kernel fix is made HERE, once; neither pipeline file may re-inline a stage body.
  Also holds the shared types (`Pore`, `I3`, `Index`, `Exec/Mem`) and `uploadVec/downloadN`.
- `pore_extraction.hpp` — `namespace pnm`, the single-rank orchestration over those kernels:
  device-resident buffers, the fixpoint loops (trivial sync), the device scan renumbering, the
  per-pore network-flow kernels (peak/centre/pressure) and the host topology sort/unique. Device
  kernels live in `.hpp` compiled as C++ (never `.cu`).
- `pnm_bindings.cpp` — the nanobind module `peclet.pnm._pnm`: `SDFReader`, `Pore`, `extract_pores`
  (→ `list[Pore]`), `segment_volume` (→ int32 `(Nz,Ny,Nx)` ndarray), `extract_topology` (3-D or
  flat int32 array in, zero-copy → `(M,2)` int32 ndarray), the fused `extract_pore_network` (SDF
  uploaded once, segmentation device-resident across stages → `(pores, seg, connections)` with
  the same three types), `extract_network_flow` (→ dict: `pores` list, `throats` `(M,2)` int32,
  five float64 arrays), and under `PECLET_PNM_MPI` `mpi_block` (→ `(offset_zyx, shape_zyx)`, two
  int 3-tuples in VOXELS — not the physical `origin_zyx`) + `extract_pore_network_mpi` /
  `extract_network_flow_mpi`. The host vectors are moved into capsule-owned NumPy arrays
  (`vector_to_ndarray`, core's zero-copy bridge) — never `std::vector` → `list` boxing
  (QUALITY_PLAN F, 2026-09-10: a 256³ segmentation used to come back as 16.7 M Python ints).
  `mpi_rank()`/`mpi_size()` were removed at 1.0.0 (mpi4py's job). Every z-y-x triple argument
  carries the `_zyx` suffix (`origin_zyx`, `spacing_zyx`, `shape_zyx`, `global_shape_zyx`,
  `grad_p_zyx`; NAMING.md §1.7; `Pore.x/y/z` are self-named scalars, the recorded exception) —
  `extract_topology_gpu(shape=)` was removed at 1.0.0 (the `_gpu` suffix was a CUDA-era leftover).
  Precision: float32 SDF + geometry kernels, `origin_zyx`/`spacing_zyx` narrowed double→float32,
  float64 MAC fields in the network-flow extraction.
- `sdf_reader.{h,cpp}` — pure-C++ VTI (VTK ImageData) reader, backend-free.
- `pore_extraction.hpp` also holds **`extract_network_flow_k`** (binding `extract_network_flow`):
  throat flow rates + pore-center pressures from a peclet.flow MAC field — the method from the
  Voronoi PNM (`~/Codes/pnm_voronoi`). KEY FACTS: flow's `u(i,j,k)` is the **-x face** of cell
  (i,j,k) and the conserved flux is `ox·u·A` (openness-weighted; `cutcell_pressure=True`
  REQUIRED or all openness is 0); fluxes accumulate on **flow basins** (gradient ascent from
  EVERY cell — seg-keyed accumulation lets near-wall staircase flux bypass throats, measured 6%);
  throat dp uses a throat-anchored two-leg min-image (single min-image is ambiguous at L/2);
  per-pore residual ~ solver tolerance is the built-in correctness check. Validated:
  `scripts/verify_network_flow.py` (|Q|=F to 1e-11 on tube networks, g>0, Kirchhoff exact).
  On loose packings per-throat g=Q/dp scatters (intra-pore viscous variation ~ throat drops —
  real physics, not a bug; the p field's grid-scale roughness μ∇²u·h ≈ 20× the macro gradient).
  GHOST-CELL IBM (`set_ghost_projection`): pass flow's `get_ox_proj/...` (binary COUPLED
  openness); bookkeeping is truncation-accurate there (ghost IBM is NOT locally conservative at
  the wall — pore_residual = wall leak, 3.2e-2·F at r=4h, order ~2.7). MPI:
  `extract_network_flow_mpi` (pore_extraction_mpi.hpp) — flow-basin labels resolved by
  propagating the LABEL with the hold-at-ghost finalization trick; global network identical on
  every rank. GOTCHA: periodic-image decisions (face min-image + the throat dp image count) MUST
  anchor on integer peak-voxel coords / snapped-integer arithmetic — float-centroid anchoring
  flips images at exactly L/2 under CUDA FMA wobble (measured flaky np4 failures on symmetric
  lattices). THROATS ARE PER-PATCH (parallel throats resolved): face-CCL keyed by min global
  face id (fid = 3*gid+d); CORE tier = both cells FLUID-centered (an OPENNESS threshold cannot
  separate throats bridged by wall films — staircase faces reach opn ~0.7; the fluid-centered
  criterion is geometric and parameter-free); films attach to the min reachable core patch by
  Jacobi min-propagation (cannot bridge two cores), unreachable films form their own patches.
- `pore_extraction_mpi.hpp` — the **distributed** pipeline (gated `PECLET_PNM_MPI`): core ORB
  decomposition + g=1 `GridHalo` exchange around the SAME `pore_kernels.hpp` kernels on
  `BlockGeo`; this file holds only the halo exchanges, the ownership/merge orchestration
  (`detail_mpi::globalMerge/applyRemap`, `resolveToFixpoint`, the extended-field staging and
  scatters) and the distributed reductions (`minByKey`, allgathers, Allreduces). Labels are
  GLOBAL voxel ids so every fixpoint is decomposition-independent → **bit-exact to
  single-rank**. Stage design: local union-find CCL +
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

The CMake `project()` version is read from `pyproject.toml` (the one version source; the
`packaging/pyproject-cuda.toml`, `CITATION.cff` and `docs/Doxyfile` copies are checked against it
by the suite's release pre-flight). Without a prefix, `cmake/PecletDeps.cmake` vendors Kokkos (OpenMP+Serial) + the peclet-core headers
via FetchContent (self-contained wheel path, `PECLET_VENDOR_DEPS=ON` in cibuildwheel). Keep the
`PECLET_*_TAG` pins in lockstep with `../tools/bootstrap_deps.sh`.

Multi-rank (MPI): add `-DPECLET_PNM_MPI=ON` to expose `mpi_block` + `extract_pore_network_mpi` /
`extract_network_flow_mpi` (collective; see README).

**Tests** (`-DPECLET_PNM_BUILD_TESTS=ON`, OFF by default so wheel builds are unaffected; ON in CI).
One tree per backend — the test targets reuse the module's resolved Kokkos/MPI targets and core
include path (QUALITY_PLAN §3.D.3 done):
```bash
cmake -S . -B build_dev -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DPECLET_PNM_MPI=ON -DPECLET_PNM_BUILD_TESTS=ON \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun          # FORCE mpirun — ParaView's mpiexec runs singletons
cmake --build build_dev -j
OMP_NUM_THREADS=4 OMP_PROC_BIND=false ctest --test-dir build_dev --output-on-failure   # 9 tests
```
- `pnm_single_rank` (`tests/kokkos/test_pnm.cpp`): the extraction contract on synthetic SDFs with
  hand-derived exact counts — one spherical pore (1 pore, radius R at the integer centre, no
  throat), two overlapping pores (2 pores + throat (1,2)), the 2×2×2 periodic solid-sphere
  lattice (8 body-centre pores, 12 face throats, 8 solids, 76 connections) — plus staged == fused,
  two runs bitwise equal, origin/spacing mapping, empty input.
- `pnm_python_smoke` (`tests/python/test_smoke.py`): the same contracts through the binding
  (catches an (Nz,Ny,Nx)/z-y-x convention slip).
- `pnm_python_packing_ring` (`tests/python/test_packing_ring.py`): the 7199-pore / 53020-connection
  gate on `../flow/data/packing_ring.vti`. The file is suite data, not in this repo: the script
  exits 77 (`SKIP_RETURN_CODE`) when it is absent and ctest reports **Skipped**, never a silent
  pass. Override the path with `-DPECLET_PNM_PACKING_RING_VTI=`.
- `pnm_mpi_np{1,2,4}` / `pnm_flow_mpi_np{1,2,4}` (`tests/kokkos_mpi`, only with `PECLET_PNM_MPI`):
  distributed vs single-rank oracle, bit-exact. `-DMPIEXEC_PREFLAGS=--oversubscribe` for a small
  runner.
- `tests/regression/state_hash.py` (not a ctest; run by hand at OMP_NUM_THREADS=1, which it pins
  itself): the **byte gate** — SHA-256 of every output of every public entry path (pores, int32
  segmentation, `(M,2)` connections, the network-flow arrays; single-rank staged + fused, and
  np=2 via its own `mpirun` re-launch) on the sphere lattice and on packing_ring. Any refactor of
  the bindings or the kernels must reproduce its output line for line; the hashes at each gate are
  in the commit messages (first: the F commit of 2026-09-10).
The synthetic generators are shared in `tests/synthetic_sdf.hpp` (`pnm::test::`). Each
`tests/*/CMakeLists.txt` also still configures standalone (`cmake -S tests/kokkos_mpi -B …`, core
via `cmake/PecletDeps.cmake`, so it works outside the suite tree too).
GPU pore-centroid caveat: nvcc FMA-contracts the centroid accumulation differently in the oracle
vs distributed kernels, so pore POSITIONS are compared to 1e-5·spacing on CUDA (bitwise on
OpenMP); seg ids, radii and connections are bitwise everywhere. The single-rank flood fill is
deliberately Jacobi (double-buffered) — deterministic AND what the distributed flood matches
sweep-for-sweep; don't "optimize" it back to in-place.

## Conventions

- SDF sign: **negative inside solid**, positive in the pore space (suite-wide).
- Python arrays are `(Nz, Ny, Nx)` C-order (x fastest — contiguous with the flat x-fastest layout);
  `origin`/`spacing` tuples are **z-y-x**. `segment_volume` returns the labels as an int32 array of
  that shape (the kernels' flat vector re-shaped in place); `extract_topology` accepts it directly
  (or a flat vector plus `shape_zyx=`).
- `Kokkos::initialize` happens at import; `Kokkos::finalize` is registered via `atexit`
  (REQUIRED on CUDA — see the comment in `pnm_bindings.cpp`).
- **The library never prints** (QUALITY_PLAN §3.H.6): there is no `cout`/`cerr`/`fprintf` in
  `src/` and no verbosity flag to forget. Every failure path — a malformed/truncated VTI, a
  shape mismatch, a stage that does not reach its fixpoint — `throw`s `std::runtime_error`
  (a Python `RuntimeError`) with the offending sizes in the message. In the MPI pipeline the
  throw is COLLECTIVE: the condition is `MPI_Allreduce`d first so every rank raises together
  instead of the good ranks hanging in the next exchange — keep that pattern when adding a
  guard to `pore_extraction_mpi.hpp`.
- Test VTI inputs live in the suite (e.g. `../flow/data/packing_ring.vti`), not in this repo.
