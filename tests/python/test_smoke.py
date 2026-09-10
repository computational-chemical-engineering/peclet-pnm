"""peclet.pnm binding smoke: the synthetic-SDF contracts through the Python API.

Run:  PYTHONPATH=<pnm build dir> python tests/python/test_smoke.py
(registered as the `pnm_python_smoke` ctest under PECLET_PNM_BUILD_TESTS).

The same three geometries as tests/kokkos/test_pnm.cpp, with the same hand-derived counts, so a
(Nz, Ny, Nx) array / z-y-x origin+spacing convention slip in the binding shows up as a count or
position mismatch here while the C++ contract stays green.
"""

import sys

import numpy as np

import peclet.pnm as pnm


def grid(shape_xyz):
    nx, ny, nz = shape_xyz
    z, y, x = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        indexing="ij",
    )
    return x, y, z  # each (Nz, Ny, Nx), x fastest


def throats(conns):
    return conns[(conns[:, 0] > 0) & (conns[:, 1] > 0)].tolist()


def check_one_sphere():
    n, R = 16, 5.5
    x, y, z = grid((n, n, n))
    sdf = (R - np.sqrt((x - 8) ** 2 + (y - 8) ** 2 + (z - 8) ** 2)).astype(np.float32)
    pores, seg, conns = pnm.extract_pore_network(sdf, [0.0] * 3, [1.0] * 3)
    assert isinstance(seg, np.ndarray) and seg.dtype == np.int32 and seg.shape == sdf.shape
    assert isinstance(conns, np.ndarray) and conns.dtype == np.int32 and conns.shape == (1, 2)
    assert len(pores) == 1, pores
    p = pores[0]
    assert p.radius == np.float32(R), p
    assert (p.x, p.y, p.z) == (8.0, 8.0, 8.0), p
    assert seg.max() == 1 and seg.min() == -1 and not (seg == 0).any()
    assert conns.tolist() == [[-1, 1]], conns
    assert len(pnm.extract_pores(sdf, [0.0] * 3, [1.0] * 3)) == 1
    # staged == fused, and extract_topology reads the 3-D int32 array (or the flat vector + shape)
    seg2 = pnm.segment_volume(sdf, [1.0] * 3)
    assert seg2.shape == sdf.shape and np.array_equal(seg2, seg)
    assert np.array_equal(pnm.extract_topology(seg2), conns)
    assert np.array_equal(pnm.extract_topology(seg2.ravel(), shape_zyx=sdf.shape), conns)
    # z-y-x origin/spacing: the pore lands at origin + voxel*spacing per axis.
    (p2,), _, _ = pnm.extract_pore_network(sdf, [0.5, 2.0, -1.0], [2.0, 1.0, 0.5])
    assert abs(p2.x - (-1.0 + 8 * 0.5)) < 1e-5 and abs(p2.y - (2.0 + 8)) < 1e-5, p2
    assert abs(p2.z - (0.5 + 8 * 2.0)) < 1e-5, p2
    print("one_sphere: 1 pore, 0 throats, 1 connection  ok")


def check_two_spheres():
    R = 5.5
    x, y, z = grid((24, 16, 16))
    d1 = R - np.sqrt((x - 8) ** 2 + (y - 8) ** 2 + (z - 8) ** 2)
    d2 = R - np.sqrt((x - 16) ** 2 + (y - 8) ** 2 + (z - 8) ** 2)
    sdf = np.maximum(d1, d2).astype(np.float32)
    pores, seg, conns = pnm.extract_pore_network(sdf, [0.0] * 3, [1.0] * 3)
    assert len(pores) == 2, pores
    assert sorted((p.x, p.y, p.z) for p in pores) == [(8.0, 8.0, 8.0), (16.0, 8.0, 8.0)], pores
    assert conns.tolist() == [[-1, 1], [-1, 2], [1, 2]], conns
    assert throats(conns) == [[1, 2]]
    assert seg.max() == 2 and seg.min() == -1 and not (seg == 0).any()
    assert seg.shape == sdf.shape and seg[8, 8, 8] == 1 and seg[8, 8, 16] == 2  # (Nz,Ny,Nx)
    print("two_spheres: 2 pores, 1 throat, 3 connections  ok")


def check_sphere_lattice():
    # The tests/kokkos_mpi generator: 2x2x2 periodic lattice of solid spheres (min-image).
    gd = (36, 30, 24)
    R = np.float32(0.22 * min(gd))
    x, y, z = grid(gd)
    sdf = np.full(x.shape, 1e30, np.float32)
    for cz in (0, 1):
        for cy in (0, 1):
            for cx in (0, 1):
                cc = [gd[i] * (2 * c + 1) / 4.0 for i, c in enumerate((cx, cy, cz))]
                dx, dy, dz = np.abs(x - cc[0]), np.abs(y - cc[1]), np.abs(z - cc[2])
                dx = np.minimum(dx, gd[0] - dx)
                dy = np.minimum(dy, gd[1] - dy)
                dz = np.minimum(dz, gd[2] - dz)
                sdf = np.minimum(sdf, (np.sqrt(dx * dx + dy * dy + dz * dz) - R).astype(np.float32))
    pores, seg, conns = pnm.extract_pore_network(sdf, [0.0] * 3, [1.0] * 3)
    assert len(pores) == 8, len(pores)
    centres = sorted((round(p.x), round(p.y), round(p.z)) for p in pores)
    assert centres == [(px, py, pz) for px in (0, 18) for py in (0, 15) for pz in (0, 12)], centres
    assert all(abs(p.radius - sdf[0, 0, 0]) == 0 for p in pores), pores
    assert len(throats(conns)) == 12 and len(conns) == 76, (len(throats(conns)), len(conns))
    assert seg.max() == 8 and seg.min() == -8 and not (seg == 0).any()
    assert len(pnm.extract_pores(sdf, [0.0] * 3, [1.0] * 3)) == 8
    print("sphere_lattice: 8 pores, 12 throats, 76 connections  ok")


if __name__ == "__main__":
    print("peclet.pnm on", pnm.execution_space)
    check_one_sphere()
    check_two_spheres()
    check_sphere_lattice()
    print("peclet.pnm smoke ok")
    sys.exit(0)
