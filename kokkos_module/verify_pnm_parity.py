#!/usr/bin/env python3
# Parity test: the Kokkos pore-network extraction (pnm_kokkos) vs the CUDA pnm_backend on a synthetic
# sphere-packing SDF. The segmentation is deterministic (union-find -> min root, gradient path tie-broken
# on index), so the relabelled segmentation arrays and topology must match exactly; pores must match as a
# set (the atomic append order differs).
import sys, gc
import numpy as np

sys.path.insert(0, "build")            # CUDA pnm_backend
sys.path.insert(0, "build_module")     # Kokkos pnm_kokkos
import pnm_backend
import pnm_kokkos

print("pnm_kokkos execution space:", pnm_kokkos.execution_space)


def packing_sdf_zyx(N, radius_frac=0.18):
    R = N * radius_frac
    g = np.arange(N)
    cs = [(c + 0.5) * N / 2.0 for c in (0, 1)]
    # build in xyz then transpose to (Nz,Ny,Nx) C-order for the pnm convention
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    best = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx; dx -= N * np.round(dx / N)
                dy = Y - sy; dy -= N * np.round(dy / N)
                dz = Z - sz; dz -= N * np.round(dz / N)
                best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    return np.ascontiguousarray(best.transpose(2, 1, 0).astype(np.float32))  # (Nz,Ny,Nx)


def main():
    N = 48
    sdf = packing_sdf_zyx(N)
    origin_zyx = [0.0, 0.0, 0.0]
    spacing_zyx = [1.0, 1.0, 1.0]
    shape_zyx = list(sdf.shape)

    pc = pnm_backend.extract_pores(sdf, origin_zyx, spacing_zyx)
    pk = pnm_kokkos.extract_pores(sdf, origin_zyx, spacing_zyx)
    sc = np.asarray(pnm_backend.segment_volume(sdf, spacing_zyx))
    sk = np.asarray(pnm_kokkos.segment_volume(sdf, spacing_zyx))
    tc = pnm_backend.extract_topology_gpu(sc.tolist(), shape_zyx)
    tk = pnm_kokkos.extract_topology_gpu(sk.tolist(), shape_zyx)

    def pore_set(ps):
        return np.array(sorted((round(p.x, 4), round(p.y, 4), round(p.z, 4), round(p.radius, 4)) for p in ps))
    pc_s, pk_s = pore_set(pc), pore_set(pk)
    pores_ok = (len(pc) == len(pk)) and (pc_s.shape == pk_s.shape) and np.allclose(pc_s, pk_s, atol=1e-3)
    seg_ok = sc.shape == sk.shape and np.array_equal(sc, sk)
    topo_ok = (len(tc) == len(tk)) and all(tuple(a) == tuple(b) for a, b in zip(sorted(tc), sorted(tk)))

    print(f"  pores:        CUDA={len(pc):6d}  Kokkos={len(pk):6d}   match={pores_ok}")
    print(f"  segmentation: CUDA labels={len(np.unique(sc)):5d} Kokkos labels={len(np.unique(sk)):5d}  arrays_equal={seg_ok}")
    print(f"  topology:     CUDA pairs={len(tc):6d}  Kokkos pairs={len(tk):6d}   match={topo_ok}")
    ok = pores_ok and seg_ok and topo_ok
    print("PASS" if ok else "FAIL")
    del sdf; gc.collect()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
