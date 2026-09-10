"""The 7199-pore release gate on the suite's ring packing (flow/data/packing_ring.vti, 256^3).

Run:  PYTHONPATH=<pnm build dir> python tests/python/test_packing_ring.py <packing_ring.vti>
(registered as the `pnm_python_packing_ring` ctest under PECLET_PNM_BUILD_TESTS).

The data file lives in the suite (../flow/data), not in this repo, so a standalone checkout (CI)
cannot run it: the test then exits 77 = ctest SKIP_RETURN_CODE and is reported as *skipped*,
never as passed. The contract (measured, deterministic — the distributed path is bit-exact to
this): 7199 pores, 53020 connections, and the fused pipeline agrees with the staged one.
"""

import os
import sys

import numpy as np

EXPECTED_PORES = 7199
EXPECTED_CONNECTIONS = 53020
SKIP = 77


def main(path):
    if not os.path.isfile(path):
        print(f"SKIP: {path} not present (suite data file, not shipped with the repo)")
        return SKIP
    import peclet.pnm as pnm

    print("peclet.pnm on", pnm.execution_space)
    sdf, origin_zyx, spacing_zyx = pnm.SDFReader.read_vti(path)
    assert sdf.shape == (256, 256, 256) and sdf.dtype == np.float32, (sdf.shape, sdf.dtype)
    pores = pnm.extract_pores(sdf, origin_zyx, spacing_zyx)
    print(f"extract_pores: {len(pores)} pores")
    assert len(pores) == EXPECTED_PORES, len(pores)
    fused, seg, conns = pnm.extract_pore_network(sdf, origin_zyx, spacing_zyx)
    assert seg.shape == sdf.shape and seg.dtype == np.int32, (seg.shape, seg.dtype)
    assert conns.shape[1:] == (2,) and conns.dtype == np.int32, (conns.shape, conns.dtype)
    print(f"extract_pore_network: {len(fused)} pores, {len(conns)} connections, "
          f"{seg.max()} pore labels, {-seg.min()} solid labels")
    assert len(fused) == EXPECTED_PORES, len(fused)
    assert len(conns) == EXPECTED_CONNECTIONS, len(conns)
    assert seg.max() == EXPECTED_PORES, seg.max()
    # the pore set is identical between the staged and the fused path
    key = lambda p: (p.x, p.y, p.z, p.radius)  # noqa: E731
    assert sorted(map(key, pores)) == sorted(map(key, fused))
    print("packing_ring ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <packing_ring.vti>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
