"""peclet.pnm — pore-network extraction from SDF pore geometry.

``SDFReader``, ``extract_pores``, ``segment_volume``, ``extract_topology_gpu`` — the "pnm_from_sdf"
feature, split out of peclet-flow into its own package (the CFD solve lives in :mod:`peclet.flow`).
"""

from ._pnm import *  # noqa: F401,F403
