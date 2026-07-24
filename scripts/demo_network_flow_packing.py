"""Network-flow extraction on a real sphere packing: DNS (peclet.flow Stokes) -> pore network.

Reads a packing SDF (VTI), solves body-force-driven Stokes flow through it, then extracts the
pore network with throat flow rates and pore pressures (pnm.extract_network_flow) and reports
network statistics + the built-in mass-balance check.

Run:  PYTHONPATH=<pnm/build>:<flow/build> python scripts/demo_network_flow_packing.py [sdf.vti]
"""
import sys
import time

import numpy as np

import peclet.flow
import peclet.pnm as pnm

vti = sys.argv[1] if len(sys.argv) > 1 else "../flow/data/packing_128.vti"
sdf_zyx, origin_zyx, spacing_zyx = pnm.SDFReader.read_vti(vti)
nz, ny, nx = sdf_zyx.shape
print(f"{vti}: grid {nx}x{ny}x{nz}, spacing {spacing_zyx[::-1]}, "
      f"porosity {(sdf_zyx > 0).mean():.3f}")

# ---- DNS: Stokes, body force in x, periodic --------------------------------------------------
MU, FX = 0.05, 1.0e-3
sdf_xyz = np.asfortranarray(sdf_zyx.T.astype(np.float64))
s = peclet.flow.Solver(nx, ny, nz)
s.set_rho(1.0)
s.set_mu(MU)
s.set_dt(100.0)
s.set_body_force(FX, 0.0, 0.0)
s.set_solid(sdf_xyz, cutcell_pressure=True)
t0 = time.time()
for _ in range(40):
    s.step()
u, v, w, p = s.get_uf(), s.get_vf(), s.get_wf(), s.get_p()
ox, oy, oz = s.get_ox(), s.get_oy(), s.get_oz()
F = np.array([(ox[i] * u[i]).sum() for i in range(nx)]) * spacing_zyx[1] * spacing_zyx[0]
print(f"DNS: {time.time()-t0:.1f} s, flux F = {F.mean():.4e} "
      f"(plane dev {np.abs(F-F.mean()).max()/abs(F.mean()):.1e})")

# ---- network extraction -----------------------------------------------------------------------
t0 = time.time()
net = pnm.extract_network_flow(
    sdf_zyx, list(origin_zyx), list(spacing_zyx),
    np.ascontiguousarray(u.T), np.ascontiguousarray(v.T), np.ascontiguousarray(w.T),
    np.ascontiguousarray(p.T), np.ascontiguousarray(ox.T), np.ascontiguousarray(oy.T),
    np.ascontiguousarray(oz.T), grad_p_zyx=[0.0, 0.0, -FX])
Q = np.array(net["throat_flow"])
dp = np.array(net["throat_dp"])
A = np.array(net["throat_area"])
res = np.array(net["pore_residual"])
g = Q / dp
print(f"extraction: {time.time()-t0:.1f} s")
print(f"network: {len(net['pores'])} pores, {len(Q)} throats")
print(f"mass residuals: max |res| = {np.abs(res).max():.3e} "
      f"({np.abs(res).max()/np.abs(Q).mean():.2e} of mean |Q|)")
print(f"throat |Q|: mean {np.abs(Q).mean():.3e}  max {np.abs(Q).max():.3e}")
print(f"throat dp: mean |dp| {np.abs(dp).mean():.3e}")
print(f"g = Q/dp > 0 on {100.0*(g > 0).mean():.1f}% of throats")
print("  NOTE: on loose packings (porosity ~0.6) the intra-pore viscous pressure variation is\n"
      "  comparable to the throat drops, so point-sampled pore dp does not resolve individual\n"
      "  throat resistances (g scatters; the Voronoi-PNM workflow fitted geometric conductances\n"
      "  for the same reason). On constricted geometry g>0 holds throat-by-throat — see\n"
      "  scripts/verify_network_flow.py (chamber-tube chain + asymmetric tube lattice).")

ok = np.abs(res).max() < 1e-6 * np.abs(Q).max() and np.isfinite(g).all()
print("PASS" if ok else "FAIL")
del s
sys.exit(0 if ok else 1)
