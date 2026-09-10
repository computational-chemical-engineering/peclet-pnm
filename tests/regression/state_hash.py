"""Byte gate for the peclet.pnm public surface: SHA-256 of every output of every entry path.

Run:  PYTHONPATH=<pnm build dir> python tests/regression/state_hash.py [packing_ring.vti]
(single-threaded: the script pins OMP_NUM_THREADS=1 before the module is imported; with an MPI
build it also re-launches itself under `mpirun -np 2` for the `*_mpi` collectives).

Two inputs: (i) the 2x2x2 periodic solid-sphere lattice of tests/synthetic_sdf.hpp
(regenerated in numpy, grid 36x30x24, non-unit origin/spacing/grad_p exactly as in
tests/kokkos_mpi/test_pnm_flow_mpi.cpp) and (ii) ../flow/data/packing_ring.vti (256^3; skipped
with a message when absent). The MAC fields fed to extract_network_flow are the deterministic
trigonometric fields of that MPI test — not a physical flow, the bookkeeping only needs
consistency, and the gate only needs reproducibility.

Hashed, per entry path: the pores as an (N, 4) float64 array [x, y, z, radius] sorted by
(z, y, x) (extract_pores returns device-completion order), the segmentation as int32 bytes,
the connections as an (M, 2) int32 array in the returned order, and every array of the
network-flow dict (pores in label order, throats (M, 2) int32, the five float64 arrays).
A refactor of the bindings or the kernels must reproduce every line of this output bitwise
(QUALITY_PLAN §6: "a byte-comparison of the outputs against the pre-refactor build, not just
a green battery"). Hashes recorded at each gate live in the commit messages.
"""

import hashlib
import os
import subprocess
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_PROC_BIND", "false")

import numpy as np  # noqa: E402

import peclet.pnm as pnm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PACKING_RING = os.path.join(HERE, "..", "..", "..", "flow", "data", "packing_ring.vti")

# The sphere-lattice case of tests/kokkos_mpi/test_pnm_flow_mpi.cpp, in x-y-z (C++) order …
LATTICE_GD_XYZ = (36, 30, 24)
LATTICE_ORG_XYZ = (0.5, -1.0, 2.0)
LATTICE_SPC_XYZ = (0.5, 1.0, 1.5)
LATTICE_GP_XYZ = (1.0e-3, -2.0e-3, 0.5e-3)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def grid_xyz(gd_xyz):
    nx, ny, nz = gd_xyz
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return x, y, z  # each (Nz, Ny, Nx), x fastest


def sphere_lattice_sdf(gd):
    """tests/synthetic_sdf.hpp sphereLatticeSdf, in float32 like the C++."""
    x, y, z = (a.astype(np.float32) for a in grid_xyz(gd))
    R = np.float32(0.22 * min(gd))
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
    return sdf


def mac_fields(gd, offset_xyz=(0, 0, 0), shape_xyz=None):
    """The seven deterministic float64 fields (u, v, w, p, ox, oy, oz) of test_pnm_flow_mpi.cpp
    on the global grid `gd`, evaluated on the block [offset, offset + shape) (x-y-z)."""
    shape_xyz = shape_xyz or gd
    x, y, z = grid_xyz(shape_xyz)
    gx, gy, gz = (a + o for a, o in zip((x, y, z), offset_xyz))

    def fld(c, gx):
        X, Y, Z = (2.0 * np.pi * g / n for g, n in zip((gx, gy, gz), gd))
        if c == 0:
            return np.sin(X) * np.cos(Y) + 0.3 * np.cos(2 * Z)
        if c == 1:
            return np.cos(X) * np.sin(Z) - 0.2 * np.sin(Y)
        if c == 2:
            return np.sin(Y) * np.sin(Z) + 0.1 * np.cos(X)
        if c == 3:
            return np.cos(X + Y) + 0.5 * np.sin(Z - X)
        s = 0.5 + 0.5 * np.sin(X + 2 * Y - Z)
        return np.where(s < 0.02, 0.0, s)

    out = [fld(c, gx) for c in range(4)] + [fld(4, gx + c) for c in (4, 5, 6)]
    return [np.ascontiguousarray(f, dtype=np.float64) for f in out]


def pores_array(pores, sort=True):
    a = np.array([(p.x, p.y, p.z, p.radius) for p in pores], np.float64).reshape(-1, 4)
    if sort and len(a):
        a = a[np.lexsort((a[:, 0], a[:, 1], a[:, 2]))]
    return a


def as_i32(x, shape):
    return np.ascontiguousarray(np.asarray(x, np.int32)).reshape(shape)


def report_network(tag, net):
    print(f"{tag + '.pores':<44}{sha(pores_array(net['pores'], sort=False))}  n={len(net['pores'])}")
    th = np.asarray(net["throats"], np.int32).reshape(-1, 2)
    assert isinstance(net["throats"], np.ndarray) and net["throats"].dtype == np.int32  # NEWAPI
    print(f"{tag + '.throats':<44}{sha(th)}  m={len(th)}")
    for k in ("pore_pressure", "pore_residual", "throat_flow", "throat_area", "throat_dp"):
        assert isinstance(net[k], np.ndarray) and net[k].dtype == np.float64  # NEWAPI
        print(f"{tag + '.' + k:<44}{sha(np.asarray(net[k], np.float64))}")


def run_case(tag, sdf, origin_zyx, spacing_zyx, grad_p_zyx):
    nz, ny, nx = sdf.shape
    fields = mac_fields((nx, ny, nz))
    pores = pnm.extract_pores(sdf, origin_zyx, spacing_zyx)
    print(f"{tag + '.extract_pores':<44}{sha(pores_array(pores))}  n={len(pores)}")
    seg = pnm.segment_volume(sdf, spacing_zyx)
    assert isinstance(seg, np.ndarray) and seg.shape == sdf.shape and seg.dtype == np.int32  # NEWAPI
    seg = as_i32(seg, sdf.shape)
    print(f"{tag + '.segment_volume':<44}{sha(seg)}")
    conns = pnm.extract_topology(seg, shape_zyx=sdf.shape)
    assert isinstance(conns, np.ndarray) and conns.dtype == np.int32 and conns.ndim == 2  # NEWAPI
    conns = as_i32(conns, (-1, 2))
    print(f"{tag + '.extract_topology':<44}{sha(conns)}  m={len(conns)}")
    fpores, fseg, fconns = pnm.extract_pore_network(sdf, origin_zyx, spacing_zyx)
    assert isinstance(fseg, np.ndarray) and fseg.shape == sdf.shape and fseg.dtype == np.int32  # NEWAPI
    assert isinstance(fconns, np.ndarray) and fconns.dtype == np.int32 and fconns.ndim == 2  # NEWAPI
    print(f"{tag + '.fused.pores':<44}{sha(pores_array(fpores))}  n={len(fpores)}")
    print(f"{tag + '.fused.seg':<44}{sha(as_i32(fseg, sdf.shape))}")
    print(f"{tag + '.fused.connections':<44}{sha(as_i32(fconns, (-1, 2)))}  m={len(fconns)}")
    net = pnm.extract_network_flow(sdf, origin_zyx, spacing_zyx, *fields[:4],
                                   ox=fields[4], oy=fields[5], oz=fields[6],
                                   grad_p_zyx=grad_p_zyx)
    report_network(f"{tag}.network_flow", net)
    net = pnm.extract_network_flow(sdf, origin_zyx, spacing_zyx, *fields[:4],
                                   grad_p_zyx=grad_p_zyx)
    report_network(f"{tag}.network_flow.open", net)


def run_case_mpi(tag, sdf, origin_zyx, spacing_zyx, grad_p_zyx):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    gshape = sdf.shape
    offset_zyx, shape_zyx = pnm.mpi_block(gshape)
    oz, oy, ox = offset_zyx
    sz, sy, sx = shape_zyx
    local = np.ascontiguousarray(sdf[oz:oz + sz, oy:oy + sy, ox:ox + sx])
    nz, ny, nx = gshape
    fields = mac_fields((nx, ny, nz), offset_xyz=(ox, oy, oz), shape_xyz=(sx, sy, sz))
    pores, seg, conns = pnm.extract_pore_network_mpi(local, gshape, origin_zyx, spacing_zyx)
    assert isinstance(seg, np.ndarray) and seg.shape == local.shape and seg.dtype == np.int32  # NEWAPI
    assert isinstance(conns, np.ndarray) and conns.dtype == np.int32 and conns.ndim == 2  # NEWAPI
    segs = comm.gather(as_i32(seg, local.shape).tobytes(), root=0)
    pores_all = comm.gather(pores_array(pores).tobytes(), root=0)
    net = pnm.extract_network_flow_mpi(local, gshape, origin_zyx, spacing_zyx, *fields[:4],
                                       ox=fields[4], oy=fields[5], oz=fields[6],
                                       grad_p_zyx=grad_p_zyx)
    if comm.rank == 0:
        np_ = comm.size
        print(f"{tag + f'.mpi{np_}.pores':<44}{hashlib.sha256(b''.join(pores_all)).hexdigest()}")
        print(f"{tag + f'.mpi{np_}.seg':<44}{hashlib.sha256(b''.join(segs)).hexdigest()}")
        print(f"{tag + f'.mpi{np_}.connections':<44}{sha(as_i32(conns, (-1, 2)))}  m={len(conns)}")
        report_network(f"{tag}.mpi{np_}.network_flow", net)


def cases(vti, quiet):
    yield ("lattice", sphere_lattice_sdf(LATTICE_GD_XYZ), list(LATTICE_ORG_XYZ[::-1]),
           list(LATTICE_SPC_XYZ[::-1]), list(LATTICE_GP_XYZ[::-1]))
    if os.path.isfile(vti):
        sdf, origin_zyx, spacing_zyx = pnm.SDFReader.read_vti(vti)
        yield ("packing_ring", sdf, list(origin_zyx), list(spacing_zyx), [0.0, 0.0, -1.0e-3])
    elif not quiet:
        print(f"packing_ring: SKIPPED ({vti} not present)")


def main(argv):
    paths = [a for a in argv if not a.startswith("--")]
    vti = os.path.abspath(paths[0]) if paths else os.path.normpath(PACKING_RING)
    if "--mpi" in argv:
        from mpi4py import MPI

        for c in cases(vti, quiet=MPI.COMM_WORLD.rank != 0):
            run_case_mpi(*c)
        return 0
    print(f"peclet.pnm on {pnm.execution_space}, OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    for c in cases(vti, quiet=False):
        run_case(*c)
    if hasattr(pnm, "mpi_block"):
        cmd = ["mpirun", "-np", "2", sys.executable, os.path.abspath(__file__), "--mpi", vti]
        sys.stdout.flush()
        return subprocess.call(cmd, env=os.environ)
    print("mpi: SKIPPED (module built without PECLET_PNM_MPI)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
