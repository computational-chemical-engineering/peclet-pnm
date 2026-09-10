"""peclet.pnm — pore-network extraction from SDF pore geometry.

The "pnm_from_sdf" feature, split out of peclet-flow into its own package (the CFD solve lives in
:mod:`peclet.flow`). Everything is Kokkos (CUDA / HIP / OpenMP, ``execution_space`` says which):

- ``SDFReader.read_vti(path)`` -> ``(sdf, origin_zyx, spacing_zyx)``
- ``Pore`` (``x``, ``y``, ``z``, ``radius``)
- ``extract_pores(sdf, origin_zyx, spacing_zyx)`` -> ``list[Pore]``
- ``segment_volume(sdf, spacing_zyx)`` -> int32 ``(Nz, Ny, Nx)`` per-voxel labels (pores
  ``1, 2, …``, grains ``-1, -2, …``, ``0`` debris)
- ``extract_topology(segmentation, shape_zyx=None)`` -> sorted unique ``(a, b)`` label pairs as an
  ``(M, 2)`` int32 array (``shape_zyx`` only for a flat label vector)
- ``extract_pore_network(sdf, origin_zyx, spacing_zyx)`` -> ``(pores, segmentation, connections)``,
  the fused pipeline (one SDF upload, segmentation device-resident across the stages)
- ``extract_network_flow(sdf, origin_zyx, spacing_zyx, u, v, w, p, ox=, oy=, oz=, grad_p_zyx=)``
  -> dict of per-pore pressures / residuals and per-throat flow rates (float64 arrays, throats
  ``(M, 2)`` int32) from a peclet.flow MAC field — the openness must be the one the field was
  projected with (flow: ``set_solid(..., cutcell_pressure=True)``)
- built with ``PECLET_PNM_MPI``: ``mpi_block(global_shape_zyx)`` -> ``(offset_zyx, shape_zyx)``
  (this rank's ORB block, in voxels), ``extract_pore_network_mpi(...)`` and
  ``extract_network_flow_mpi(...)`` — the same pipelines distributed on the peclet-core ORB
  blocks, bit-exact to single-rank (rank / size come from mpi4py)
- ``finalize()`` releases the Kokkos state (also registered at exit)

Conventions: arrays are ``(Nz, Ny, Nx)`` C-order and every triple describing them is z-y-x with
the ``_zyx`` suffix (``Pore.x/y/z`` are three self-named scalars, the one exception); SDF sign is
negative inside the solid. Arrays in, arrays out: the segmentation, connection and throat lists
and the network-flow scalars are NumPy arrays over the kernels' own host buffers (no per-element
conversion); only the pores are a Python ``list[Pore]``. Precision: the SDF is float32 and the
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
