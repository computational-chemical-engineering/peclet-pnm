"""peclet.pnm — pore-network extraction from SDF pore geometry.

``SDFReader``, ``extract_pores``, ``segment_volume``, ``extract_topology_gpu`` — the "pnm_from_sdf"
feature, split out of peclet-flow into its own package (the CFD solve lives in :mod:`peclet.flow`).
"""

from ._pnm import *  # noqa: F401,F403

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown".
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("peclet-pnm")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"
