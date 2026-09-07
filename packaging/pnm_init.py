"""peclet.pnm — pore-network extraction from SDF pore geometry.

The "pnm_from_sdf" feature, split out of peclet-flow into its own package (the CFD solve lives in
:mod:`peclet.flow`). Everything is Kokkos (CUDA / HIP / OpenMP, ``execution_space`` says which):

- ``SDFReader.read_vti(path)`` -> ``(sdf, origin_zyx, spacing_zyx)``
- ``Pore`` (``x``, ``y``, ``z``, ``radius``)
- ``extract_pores(sdf, origin_zyx, spacing_zyx)`` -> ``list[Pore]``
- ``segment_volume(sdf, spacing_zyx)`` -> flat per-voxel labels (pores ``1, 2, …``, grains
  ``-1, -2, …``, ``0`` debris)
- ``extract_topology(segmentation, shape_zyx)`` -> sorted unique ``(a, b)`` label pairs
- ``extract_pore_network(sdf, origin_zyx, spacing_zyx)`` -> ``(pores, segmentation, connections)``,
  the fused pipeline (one SDF upload, segmentation device-resident across the stages)
- ``extract_network_flow(sdf, origin_zyx, spacing_zyx, u, v, w, p, ox=, oy=, oz=, grad_p_zyx=)``
  -> dict of per-pore pressures / residuals and per-throat flow rates from a peclet.flow MAC field
- built with ``PECLET_PNM_MPI``: ``mpi_rank()``, ``mpi_size()``, ``mpi_block(global_shape_zyx)``,
  ``extract_pore_network_mpi(...)`` and ``extract_network_flow_mpi(...)`` — the same pipelines
  distributed on the peclet-core ORB blocks, bit-exact to single-rank
- ``finalize()`` releases the Kokkos state (also registered at exit)

Conventions: arrays are ``(Nz, Ny, Nx)`` C-order and every triple describing them is z-y-x with
the ``_zyx`` suffix; SDF sign is negative inside the solid. Precision: the SDF is float32 and the
geometry kernels compute in float32 (``origin_zyx`` / ``spacing_zyx`` are narrowed from double,
so pore centres and radii are float32 in the input unit system); the network-flow MAC fields
(``u``, ``v``, ``w``, ``p``, openness) are float64.
"""

from ._pnm import *  # noqa: F401,F403

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown".
try:
    from importlib.metadata import PackageNotFoundError as _PNF, version as _dist_version
    try:
        __version__ = _dist_version("peclet-pnm")
    except _PNF:  # the CUDA wheel installs the same module under the -cu13 distribution name
        __version__ = _dist_version("peclet-pnm-cu13")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"
