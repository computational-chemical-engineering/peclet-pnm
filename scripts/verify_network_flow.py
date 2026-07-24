"""Quantitative validation of pnm.extract_network_flow against a peclet.flow DNS.

Geometry: a periodic chamber-tube chain — two spherical chambers connected by a straight
cylindrical tube along x (also through the periodic boundary). The watershed segments this into
4 pores (2 chambers + 2 tube segments) in a single loop, so by mass conservation EVERY pore-pore
throat must carry the full DNS flux F (measured independently as the openness-weighted x-flux
through a grid plane). Checks:
  1. per-pore residual (signed flux over the whole pore boundary) ~ solver tolerance,
  2. |Q_throat| == F for every throat (small leakage through near-wall faces allowed),
  3. Q and the total-pressure drop dp agree in sign on every throat (g = Q/dp > 0),
  4. the DNS plane flux is x-independent (steady, converged).

Run:  PYTHONPATH=<pnm/build>:<flow/build> python scripts/verify_network_flow.py
"""
import sys

import numpy as np

import peclet.flow
import peclet.pnm as pnm

# ---- geometry: two chambers + periodic tube (pnm/flow convention: sdf > 0 in the PORE space) ---
NX, NY, NZ = 48, 32, 32
RC, RT = 9.0, 4.0
CY, CZ = NY / 2.0, NZ / 2.0
CENTERS_X = (12.0, 36.0)

x, y, z = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
x = x.astype(np.float64); y = y.astype(np.float64); z = z.astype(np.float64)
sdf = np.full((NX, NY, NZ), -1e30)
for cx in CENTERS_X:
    dx = np.abs(x - cx)
    dx = np.minimum(dx, NX - dx)  # periodic in x
    d = np.sqrt(dx**2 + (y - CY) ** 2 + (z - CZ) ** 2)
    sdf = np.maximum(sdf, RC - d)
sdf = np.maximum(sdf, RT - np.sqrt((y - CY) ** 2 + (z - CZ) ** 2))  # tube along x

# ---- DNS: Stokes flow driven by a body force in x (grad_p_macro = -f) ----------------------
MU, FX = 0.05, 1.0e-3
s = peclet.flow.Solver(NX, NY, NZ)
s.set_rho(1.0)
s.set_mu(MU)
s.set_dt(100.0)
s.set_body_force(FX, 0.0, 0.0)
s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)  # openness-weighted flux o*u*A
for _ in range(60):
    s.step()

u, v, w = s.get_uf(), s.get_vf(), s.get_wf()
p = s.get_p()
ox, oy, oz = s.get_ox(), s.get_oy(), s.get_oz()

# DNS flux through every x-plane (openness-weighted -x face fluxes of plane i)
F_planes = np.array([(ox[i, :, :] * u[i, :, :]).sum() for i in range(NX)])
F = F_planes.mean()
plane_dev = np.abs(F_planes - F).max() / abs(F)
print(f"DNS flux F = {F:.6e}  (max plane deviation {plane_dev:.2e})")

# ---- network extraction ------------------------------------------------------------------------
net = pnm.extract_network_flow(
    np.ascontiguousarray(sdf.T, dtype=np.float32), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
    np.ascontiguousarray(u.T), np.ascontiguousarray(v.T), np.ascontiguousarray(w.T),
    np.ascontiguousarray(p.T), np.ascontiguousarray(ox.T), np.ascontiguousarray(oy.T),
    np.ascontiguousarray(oz.T), grad_p_zyx=[0.0, 0.0, -FX])

pores = net["pores"]
Q = np.array(net["throat_flow"])
dp = np.array(net["throat_dp"])
res = np.array(net["pore_residual"])
print(f"pores: {len(pores)}  throats: {len(net['throats'])}")
for k, po in enumerate(pores):
    print(f"  pore {k+1}: x=({po.x:6.2f},{po.y:6.2f},{po.z:6.2f}) r={po.radius:5.2f} "
          f"p={net['pore_pressure'][k]: .4e} residual={res[k]: .3e}")
for t, (a, b) in enumerate(net["throats"]):
    print(f"  throat {a}-{b}: Q={Q[t]: .6e}  dp={dp[t]: .4e}  A={net['throat_area'][t]:7.2f}  "
          f"g=Q/dp={Q[t]/dp[t]: .4e}")

# ---- checks ------------------------------------------------------------------------------------
fails = 0
def check(name, ok, detail):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    fails += 0 if ok else 1

check("steady plane flux", plane_dev < 1e-6, f"max deviation {plane_dev:.2e}")
# Watershed topology: the chambers + the wrap tube segment (the mid-tube plateau drains into the
# chamber basins — only the wrap segment keeps a gid-seam peak). A single loop: #throats == #pores.
check("pore count", len(pores) == 3, f"{len(pores)} (expect 3: 2 chambers + wrap tube segment)")
check("single loop", len(net["throats"]) == len(pores),
      f"{len(net['throats'])} throats vs {len(pores)} pores")
check("mass residuals", np.abs(res).max() < 1e-4 * abs(F),
      f"max |residual|/F = {np.abs(res).max()/abs(F):.2e}")
check("throat flux = DNS flux", np.abs(np.abs(Q) - abs(F)).max() < 1e-3 * abs(F),
      f"max ||Q|-F|/F = {np.abs(np.abs(Q)-abs(F)).max()/abs(F):.2e}")
check("Q, dp sign-consistent (g>0)", bool(np.all(Q * dp > 0)),
      f"signs: {np.sign(Q*dp).astype(int).tolist()}")

# ================= case 2: asymmetric chamber lattice (a genuine network) ======================
# 4 chambers in a 2x2 (y,z) arrangement, connected along x by tubes of DIFFERENT radii through
# the periodic boundary — a multi-loop resistor network with real constrictions. On such
# geometry the throat data must behave like a network: g = Q/dp > 0 on every throat.
print("\n=== case 2: asymmetric tube lattice ===")
del s
NX2, NY2, NZ2 = 48, 48, 48
sdf2 = np.full((NX2, NY2, NZ2), -1e30)
x, y, z = np.meshgrid(*(np.arange(d, dtype=np.float64) for d in (NX2, NY2, NZ2)), indexing="ij")
tubes = []  # (cy, cz, radius) — one x-tube per chamber column, different radii
for (cy, cz, rt) in ((12.0, 12.0, 3.0), (12.0, 36.0, 4.0), (36.0, 12.0, 5.0), (36.0, 36.0, 6.0)):
    for cx in (12.0, 36.0):
        dx = np.abs(x - cx); dx = np.minimum(dx, NX2 - dx)
        sdf2 = np.maximum(sdf2, 9.0 - np.sqrt(dx**2 + (y - cy) ** 2 + (z - cz) ** 2))
    sdf2 = np.maximum(sdf2, rt - np.sqrt((y - cy) ** 2 + (z - cz) ** 2))
s = peclet.flow.Solver(NX2, NY2, NZ2)
s.set_rho(1.0); s.set_mu(MU); s.set_dt(100.0)
s.set_body_force(FX, 0.0, 0.0)
s.set_solid(np.asfortranarray(sdf2), cutcell_pressure=True)
for _ in range(60):
    s.step()
u, v, w, p = s.get_uf(), s.get_vf(), s.get_wf(), s.get_p()
ox, oy, oz = s.get_ox(), s.get_oy(), s.get_oz()
net2 = pnm.extract_network_flow(
    np.ascontiguousarray(sdf2.T, dtype=np.float32), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
    np.ascontiguousarray(u.T), np.ascontiguousarray(v.T), np.ascontiguousarray(w.T),
    np.ascontiguousarray(p.T), np.ascontiguousarray(ox.T), np.ascontiguousarray(oy.T),
    np.ascontiguousarray(oz.T), grad_p_zyx=[0.0, 0.0, -FX])
Q2 = np.array(net2["throat_flow"])
dp2 = np.array(net2["throat_dp"])
res2 = np.array(net2["pore_residual"])
F2 = np.array([(ox[i, :, :] * u[i, :, :]).sum() for i in range(NX2)]).mean()
print(f"pores {len(net2['pores'])}, throats {len(Q2)}, F = {F2:.4e}")
for t, (a, b) in enumerate(net2["throats"]):
    print(f"  throat {a}-{b}: Q={Q2[t]: .4e} dp={dp2[t]: .4e} g={Q2[t]/dp2[t]: .4e}")
check("case2 mass residuals", np.abs(res2).max() < 1e-4 * abs(F2),
      f"max |residual|/F = {np.abs(res2).max()/abs(F2):.2e}")
check("case2 g>0 on all throats", bool(np.all(Q2 * dp2 > 0)),
      f"{int((Q2*dp2 > 0).sum())}/{len(Q2)}")
# the four tube columns are independent parallel channels of 3 throats each, every throat in a
# column carrying that column's full flux: sum |Q| = 3 * F_total
check("case2 total flux", abs(np.abs(Q2).sum() / 3.0 - abs(F2)) < 1e-2 * abs(F2),
      f"sum|Q|/3 = {np.abs(Q2).sum()/3.0:.4e} vs F = {F2:.4e}")

print("PASS" if fails == 0 else f"FAIL ({fails})")
del s  # release the solver's device Views BEFORE interpreter exit (flow's finalize contract)
sys.exit(1 if fails else 0)
