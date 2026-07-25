// peclet-pnm — distributed network-flow extraction vs the single-rank oracle.
//
// A sphere-lattice SDF + deterministic smooth MAC fields (u,v,w,p and openness — not a physical
// flow; the bookkeeping only needs consistency) are extracted (a) via extract_network_flow_mpi
// distributed over the ORB blocks and (b) via the single-rank extract_network_flow_k on rank 0.
// The distributed result is global and identical on every rank; it must match the oracle: throat
// list and pore ids exactly; Q/A/dp/pressures/residuals to accumulation-order tolerance (both
// paths sum with atomics); pore positions to the CUDA FMA-contraction tolerance. Two configs:
// with openness fields and fully open. np = 1, 2, 4, any Kokkos backend.
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "pore_extraction.hpp"
#include "pore_extraction_mpi.hpp"

using pnm::Pore;

namespace {

float sphereLatticeSdf(int gx, int gy, int gz, std::array<int, 3> gd) {
  const float R = 0.22f * std::min(gd[0], std::min(gd[1], gd[2]));
  float best = 1e30f;
  for (int cz = 0; cz < 2; ++cz)
    for (int cy = 0; cy < 2; ++cy)
      for (int cx = 0; cx < 2; ++cx) {
        const float ccx = gd[0] * (2 * cx + 1) / 4.0f, ccy = gd[1] * (2 * cy + 1) / 4.0f,
                    ccz = gd[2] * (2 * cz + 1) / 4.0f;
        float dx = std::fabs(gx - ccx), dy = std::fabs(gy - ccy), dz = std::fabs(gz - ccz);
        dx = std::min(dx, gd[0] - dx);
        dy = std::min(dy, gd[1] - dy);
        dz = std::min(dz, gd[2] - dz);
        best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - R);
      }
  return best;
}

double fld(int c, int gx, int gy, int gz, std::array<int, 3> gd) {
  const double x = 2.0 * M_PI * gx / gd[0], y = 2.0 * M_PI * gy / gd[1],
               z = 2.0 * M_PI * gz / gd[2];
  switch (c) {
    case 0: return std::sin(x) * std::cos(y) + 0.3 * std::cos(2 * z);
    case 1: return std::cos(x) * std::sin(z) - 0.2 * std::sin(y);
    case 2: return std::sin(y) * std::sin(z) + 0.1 * std::cos(x);
    case 3: return std::cos(x + y) + 0.5 * std::sin(z - x);          // p
    default: {
      const double s = 0.5 + 0.5 * std::sin(x + 2 * y - z);          // openness in [0,1]
      return s < 0.02 ? 0.0 : s;
    }
  }
}

bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int runConfig(const char* name, bool withOpen, MPI_Comm comm) {
  const std::array<int, 3> gd{36, 30, 24};
  const std::array<float, 3> org{0.5f, -1.0f, 2.0f};
  const std::array<float, 3> spc{0.5f, 1.0f, 1.5f};
  const std::array<double, 3> gp{1.0e-3, -2.0e-3, 0.5e-3};
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  std::array<int, 3> bo{}, bs{};
  pnm::mpi_block_of(gd, comm, bo, bs);
  const std::size_t nl = std::size_t(bs[0]) * bs[1] * bs[2];
  std::vector<float> sdfL(nl);
  std::vector<double> fldL[7];
  for (int c = 0; c < 7; ++c)
    fldL[c].resize(nl);
  for (int z = 0; z < bs[2]; ++z)
    for (int y = 0; y < bs[1]; ++y)
      for (int x = 0; x < bs[0]; ++x) {
        const std::size_t i = (std::size_t(z) * bs[1] + y) * bs[0] + x;
        const int gx = bo[0] + x, gy = bo[1] + y, gz = bo[2] + z;
        sdfL[i] = sphereLatticeSdf(gx, gy, gz, gd);
        for (int c = 0; c < 4; ++c)
          fldL[c][i] = fld(c, gx, gy, gz, gd);
        for (int c = 4; c < 7; ++c)
          fldL[c][i] = fld(4, gx + c, gy, gz, gd);  // three distinct openness fields
      }
  std::vector<double> none;
  auto dist = pnm::extract_network_flow_mpi(
      sdfL, gd, org, spc, fldL[0], fldL[1], fldL[2], fldL[3], withOpen ? fldL[4] : none,
      withOpen ? fldL[5] : none, withOpen ? fldL[6] : none, gp, comm);

  int fail = 0;
  if (rank == 0) {
    const std::size_t n = std::size_t(gd[0]) * gd[1] * gd[2];
    std::vector<float> sdfG(n);
    std::vector<double> fldG[7];
    for (int c = 0; c < 7; ++c)
      fldG[c].resize(n);
    for (int z = 0; z < gd[2]; ++z)
      for (int y = 0; y < gd[1]; ++y)
        for (int x = 0; x < gd[0]; ++x) {
          const std::size_t i = (std::size_t(z) * gd[1] + y) * gd[0] + x;
          sdfG[i] = sphereLatticeSdf(x, y, z, gd);
          for (int c = 0; c < 4; ++c)
            fldG[c][i] = fld(c, x, y, z, gd);
          for (int c = 4; c < 7; ++c)
            fldG[c][i] = fld(4, x + c, y, z, gd);
        }
    auto orc = pnm::extract_network_flow_k(sdfG, gd, org, spc, fldG[0], fldG[1], fldG[2], fldG[3],
                                           withOpen ? fldG[4] : none, withOpen ? fldG[5] : none,
                                           withOpen ? fldG[6] : none, gp);
    double qs = 1e-30;
    for (double q : orc.throat_flow)
      qs = std::max(qs, std::fabs(q));
    if (dist.throats != orc.throats) {
      std::printf("[%s] FAIL throat list: %zu vs %zu\n", name, dist.throats.size(),
                  orc.throats.size());
      fail = 1;
    } else {
      for (std::size_t t = 0; t < orc.throats.size() && !fail; ++t) {
        if (!close(dist.throat_flow[t], orc.throat_flow[t], 1e-10 * qs) ||
            !close(dist.throat_area[t], orc.throat_area[t], 1e-10 * qs) ||
            !close(dist.throat_dp[t], orc.throat_dp[t], 1e-8)) {
          std::printf("[%s] FAIL throat %zu: Q %.15e vs %.15e  A %.15e vs %.15e  dp %g vs %g\n",
                      name, t, dist.throat_flow[t], orc.throat_flow[t], dist.throat_area[t],
                      orc.throat_area[t], dist.throat_dp[t], orc.throat_dp[t]);
          fail = 1;
        }
      }
    }
    if (dist.pores.size() != orc.pores.size()) {
      std::printf("[%s] FAIL pore count %zu vs %zu\n", name, dist.pores.size(), orc.pores.size());
      fail = 1;
    } else {
      for (std::size_t k = 0; k < orc.pores.size() && !fail; ++k) {
        const Pore &a = dist.pores[k], &b = orc.pores[k];
        if (a.radius != b.radius || !close(a.x, b.x, 1e-5 * spc[0]) ||
            !close(a.y, b.y, 1e-5 * spc[1]) || !close(a.z, b.z, 1e-5 * spc[2]) ||
            !close(dist.pore_pressure[k], orc.pore_pressure[k], 1e-9) ||
            !close(dist.pore_residual[k], orc.pore_residual[k], 1e-10 * qs)) {
          std::printf("[%s] FAIL pore %zu (r %.9g vs %.9g, p %.15e vs %.15e)\n", name, k,
                      a.radius, b.radius, dist.pore_pressure[k], orc.pore_pressure[k]);
          fail = 1;
        }
      }
    }
    if (!fail)
      std::printf("[%s] PASS: %zu pores, %zu throats match the single-rank oracle\n", name,
                  orc.pores.size(), orc.throats.size());
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, comm);
  return fail;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int fail = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    fail += runConfig("openness", true, MPI_COMM_WORLD);
    fail += runConfig("fully_open", false, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return fail ? 1 : 0;
}
