# pnm playbook (overrides the cce-code-health seed by id)

Read the seed first (`plugins/cce-code-health/skills/refactor-loop/playbook-seed.md`, tool pin 00b1849), then this file.
Read `CLAUDE.md` (settled decisions) before touching any source file; a refactor never reverses one.
Gate: `PECLET_SUITE=<suite> codehealth gate`; bitwise = `state_hash` (stdout of `tests/regression/state_hash.py`:
single-rank staged + fused, network flow, np=2 MPI; lattice + packing_ring) on the host/OpenMP build at ONE thread.

## Keep-out (paper A7, session `pnm`; suite `../../docs/code_health/RULES.md`, umbrella branch code-health-docs)

A7's branch `junction-law` (merge-base 0ccf947) rebases onto every batch landed here; send A7 the commit list per
batch. Never touch `junction-law`, `suite/pnm-junction-law` or `suite/pnm-netflow-momentum`. Verbatim from RULES.md:

> Heavy-overlap files (rebase cost): src/pnm_bindings.cpp, src/pore_kernels.hpp, src/pore_extraction.hpp,
> src/pore_extraction_mpi.hpp, tests/kokkos_mpi/test_pnm_flow_mpi.cpp. In pore_kernels.hpp never rename/move the
> flow-mapping stage kernels, and never touch the extract_network_flow(_mpi) binding block, unless that is the
> commit's point; formatting-only passes over those files are fine as SEPARATE commits.

The flow-mapping stage kernels are the network-flow face stages of `pore_kernels.hpp` (face init, face CCL merge,
min-fid, core/leftover labelling, film attachment, throat flux incl. `faceVelocity`/`kThroatExt`, pore volume, the
throat-slot / pressure-drop / `throatExtras` / `poreVolumeStats` host helpers). They keep their names, their place in
the file and their signatures. The `extract_network_flow` / `extract_network_flow_mpi` blocks of `pnm_bindings.cpp`
are not edited at all.

## Rules for new code

1. Pass `Kokkos::View`s by value and capture them by value in `KOKKOS_LAMBDA`; never capture `this` in a device
   lambda (illegal on CUDA only, so the host build will not catch it; `build_cuda` does).
2. ONE set of stage kernels (`pore_kernels.hpp`, QUALITY_PLAN G.3): neither pipeline file may re-inline a stage
   body, and a moved kernel stays templated on the geometry policy (`GridGeo`/`BlockGeo`) with the same expressions
   in the same evaluation order in both instantiations. That is what keeps np>1 bit-exact to single-rank.
3. Never change the order of kernel launches, fences, halo exchanges or MPI calls, nor a fixpoint loop's sweep
   structure. The single-rank flood is deliberately Jacobi (double-buffered); never make it in-place.
4. Never introduce or reorder atomics or reductions (the pore counter's `atomic_fetch_add`, the network-flow
   `atomic_add` sums). state_hash runs at one thread and cannot see ordering changes that show only at more threads.
5. Labels stay GLOBAL voxel ids; the hold-at-ghost resolve keeps its `~root` finalization marker (never store a
   remote mid-chain gid); renumbering stays by global min-appearance gid.
6. Periodic-image decisions stay anchored on integer peak-voxel coords / snapped-integer arithmetic, never on a
   float centroid (CUDA FMA wobble flips images at L/2).
7. A moved kernel keeps its execution space and memory space; never move work to the host.
8. The library never prints. Every failure `throw`s `std::runtime_error` with the sizes; in the MPI pipeline the
   condition is `MPI_Allreduce`d first so the throw is collective. Keep both when moving a guard.
9. Host vectors leave through capsule-owned NumPy arrays (`vector_to_ndarray`), never `std::vector` -> `list`.
10. Name a thing by what it is, never where it runs (no `Kokkos`/`Device`/`Gpu` qualifiers); z-y-x Python
    triples keep the `_zyx` suffix (NAMING.md §1.7).
11. No environment variable may change a result. Keep `src/` as it is laid out (header-only kernels +
    `pnm_bindings.cpp` + `sdf_reader.cpp`); do not add a `.cpp`, never a `.cu`.
12. Format with the suite venv's clang-format 18.1.8 (pnm's CI format job blocks).

## Refactor recipes

| id | smell | when | when not | recipe |
|---|---|---|---|---|
| extract-orchestration-phase | long host orchestration (`pore_extraction*.hpp` pipeline functions) | a contiguous stage (upload, detect, CCL fixpoint, flood, renumber, topology) reads and writes a nameable set of Views and scalars | the stage is a flow-mapping stage (keep-out); it shares locals the next stage mutates; it needs 10+ inputs (group first) | move it verbatim into an `inline` function in the same header; same launches, same order, same fences, same Views; the caller reads as a sequence of stages |
| launch-params-struct | many parameters | the same 4+ Views/scalars travel together through several launchers and form one concept | the group is used by one function, or a member needs another memory space | a private aggregate holding the Views BY VALUE (trivially copyable, captured by value); defaults reproduce today's default arguments exactly |
| dedup-binding-glue | duplication in `pnm_bindings.cpp` (argument checks, ndarray conversion) | two entry points repeat the same validation or conversion | the block is `extract_network_flow(_mpi)` (keep-out); the duplicate differs in a message, dtype or shape | one `static` helper in the TU; Python names, signatures, defaults and docstrings unchanged |
| guard-clauses | (seed) | host-side setup code only | inside device code where it changes divergence or FP order | as seed |
| replace-temp-with-query | (seed) | host-side scalars only | device code (recomputing can change FP results) | as seed |

Evidence (filled by consolidation from the run log, never by hand):
- extract-orchestration-phase:
- launch-params-struct:
- dedup-binding-glue:

## Anti-patterns (review rejects)

- **Keep-out breach**: a rename, move or re-signature of a flow-mapping stage kernel, or any edit in the
  `extract_network_flow(_mpi)` binding block, inside a refactor step. Formatting-only passes on the heavy-overlap
  files are separate commits.
- **Pipeline divergence**: a stage body re-inlined into one pipeline, or a change that makes the `GridGeo` and
  `BlockGeo` instantiations evaluate differently.
- **Reordering for tidiness**: launch order, fence placement, halo/MPI call order, fixpoint structure, or lambda
  capture semantics, even when state_hash agrees.
- **Touching atomics or reductions** (type, order, fusion, split): never part of this campaign.
- **Space migration**: a moved kernel in another execution or memory space, or a `create_mirror_view`/`deep_copy`
  added or removed.
- **Settled-decision drift**: any change touching a CLAUDE.md fact (flow-basin accumulation not seg-keyed,
  per-patch throats with the fluid-centred CORE tier, two-leg min-image dp, Jacobi flood, global-id labels,
  `~root` marker, collective throws, never-prints).
- **Public surface change**: renaming or re-signing anything reachable from `src/pnm_bindings.cpp`.
- **A phase helper with a parameter list as long as the phase** (seed): group the inputs first.
- **Debt by lambda shuffling**: cpp@0 counts lambda bodies in the enclosing function; moving a kernel into a
  functor lowers "debt" without lowering complexity. Accept only when the reviewer agrees it reads better.

## User-facing round (WO-33, 2026-11-17; deltas from dem's, flow's and amr's recipes)
- **Keep-out by line range** (A7, RULES 11-17): before each commit, `git diff -U0 0ccf947 -- src/pnm_bindings.cpp`
  hunks must miss base lines 39-50, 96-186, 361-404, 456-496. The checks header is included after `using pnm::Pore;`.
- **Checks**: one `chk::` line at the top of each free lambda; `Pore.__init__` bound as `&chk::init_pore` (same
  signature). origin_zyx / spacing_zyx are narrowed to float32, so they are checked AS float32 (1e39 -> inf, 1e-50
  -> 0). Old RuntimeErrors stay (mpi_block length, extract_topology shape/rank); the new count check runs after them.
- **Found beyond the audit** (it probes only Pore): extract_pores(sdf, [], s) SIGSEGV; a length-2 triple read past
  the vector in every entry point; extract_topology(flat, shape_zyx=[-1, -1, n]) SIGABRT (the wrapped size_t
  product matched n); mpi_block([-1, 8, 8]) returned a block of -1 voxels. Not checked: SDF values (an O(N) scan,
  per rank, so not collective in the _mpi path) and origin_zyx's sign (a coordinate).
- **Gate tool**: --behaviour-only needs cce-code-health >= 1304e31 (pin 719092b lacks it; a scratch venv at b46d80d
  + lizard 1.26.0 works) and `--config .codehealth/config.toml` (0ccf947 has none). state_hash same_lines exempts
  the vendored-core line: the sibling head takes ../core silently, the base prints it for the same directory.
- **Wall clock (TRS2, OMP 1)**: incremental build 24-33 s; gate --behaviour-only 978 s (state_hash 880 s, base
  and head; ctest 11 tests 22 s; build_cuda 48 s; nb_signatures 27 s); pytest 85 cases 0.4 s; api audit 1 s.
