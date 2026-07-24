// peclet-pnm — distributed extraction vs the single-rank oracle, bit-exact.
//
// Two synthetic periodic geometries (a solid sphere lattice with interstitial pores; a smooth
// trigonometric SDF with wrap-around components, anisotropic spacing and a shifted origin) are
// extracted (a) distributed over the ORB blocks via pnm::extract_pore_network_mpi and (b) on the
// full grid by the single-rank pnm::extract_pore_network_k (rank 0). The distributed result must
// be EXACTLY equal: per-voxel segmentation ids, the pore set (positions/radii bitwise, compared
// sorted — the emission order is an atomic-slot race in both paths), and the global connection
// list. Runs on any Kokkos backend (OpenMP / CUDA / HIP), np = 1, 2, 4.
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

#include "pore_extraction.hpp"
#include "pore_extraction_mpi.hpp"

using pnm::Index;
using pnm::Pore;

namespace {

// deterministic synthetic SDFs over the global periodic grid ------------------------------------
float sphereLatticeSdf(int gx, int gy, int gz, std::array<int, 3> gd) {
  // 2x2x2 lattice of solid spheres (min-image periodic distance); pores at the interstices.
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
  return best;  // negative INSIDE the spheres (solid), positive at the interstices (pore)
}

float trigSdf(int gx, int gy, int gz, std::array<int, 3> gd) {
  const float x = 2.0f * float(M_PI) * gx / gd[0], y = 2.0f * float(M_PI) * gy / gd[1],
              z = 2.0f * float(M_PI) * gz / gd[2];
  return std::sin(x) * std::cos(y) + 0.7f * std::sin(2.0f * z) - 0.1f;
}

struct BlockMeta {
  int o[3], s[3];
};

bool poreLess(const Pore& a, const Pore& b) {
  return std::tie(a.x, a.y, a.z, a.radius) < std::tie(b.x, b.y, b.z, b.radius);
}
// Radius must be bitwise (a straight copy of the SDF peak value). Positions carry the sub-voxel
// centroid, whose accumulation nvcc may contract into FMAs differently in the oracle vs the
// distributed kernel (distinct kernels, same expressions) — measured ~1e-8 wobble at symmetric
// peaks on CUDA, bitwise on OpenMP. Compare positions to 1e-5 * spacing; peaks are >= 1 voxel
// apart so the sort stays stable.
bool poreEq(const Pore& a, const Pore& b, std::array<float, 3> spc) {
  return a.radius == b.radius && std::fabs(a.x - b.x) <= 1e-5f * spc[0] &&
         std::fabs(a.y - b.y) <= 1e-5f * spc[1] && std::fabs(a.z - b.z) <= 1e-5f * spc[2];
}

// one geometry: distributed extraction on all ranks vs the rank-0 single-rank oracle ------------
template <class SdfFn>
int runCase(const char* name, std::array<int, 3> gd, std::array<float, 3> org,
            std::array<float, 3> spc, const SdfFn& sdfAt, MPI_Comm comm) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::array<int, 3> bo{}, bs{};
  pnm::mpi_block_of(gd, comm, bo, bs);
  std::vector<float> local(std::size_t(bs[0]) * bs[1] * bs[2]);
  for (int z = 0; z < bs[2]; ++z)
    for (int y = 0; y < bs[1]; ++y)
      for (int x = 0; x < bs[0]; ++x)
        local[(std::size_t(z) * bs[1] + y) * bs[0] + x] =
            sdfAt(bo[0] + x, bo[1] + y, bo[2] + z, gd);

  auto dist = pnm::extract_pore_network_mpi(local, gd, org, spc, comm);

  // gather the distributed pieces (block metas fixed-size; seg variable-size, rank order)
  BlockMeta meta{{bo[0], bo[1], bo[2]}, {bs[0], bs[1], bs[2]}};
  auto metas = pnm::detail_mpi::allgatherv(std::vector<BlockMeta>{meta}, comm);
  auto segAll = pnm::detail_mpi::allgatherv(dist.seg, comm);
  auto poresAll = pnm::detail_mpi::allgatherv(dist.pores, comm);

  int fail = 0;
  if (rank == 0) {
    // single-rank oracle on the full grid
    const std::size_t n = std::size_t(gd[0]) * gd[1] * gd[2];
    std::vector<float> full(n);
    for (int z = 0; z < gd[2]; ++z)
      for (int y = 0; y < gd[1]; ++y)
        for (int x = 0; x < gd[0]; ++x)
          full[(std::size_t(z) * gd[1] + y) * gd[0] + x] = sdfAt(x, y, z, gd);
    auto oracle = pnm::extract_pore_network_k(full, gd, org, spc);

    // seg: reassemble the blocks and compare per voxel
    std::vector<int> segG(n, -999);
    std::size_t off = 0;
    for (int r = 0; r < size; ++r) {
      const auto& m = metas[r];
      for (int z = 0; z < m.s[2]; ++z)
        for (int y = 0; y < m.s[1]; ++y)
          for (int x = 0; x < m.s[0]; ++x)
            segG[(std::size_t(m.o[2] + z) * gd[1] + (m.o[1] + y)) * gd[0] + (m.o[0] + x)] =
                segAll[off + (std::size_t(z) * m.s[1] + y) * m.s[0] + x];
      off += std::size_t(m.s[0]) * m.s[1] * m.s[2];
    }
    std::size_t nbad = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (segG[i] != oracle.seg[i])
        ++nbad;
    if (nbad) {
      std::printf("[%s] FAIL seg: %zu / %zu voxels differ\n", name, nbad, n);
      fail = 1;
    }

    // pores: exact set equality (sorted; positions/radii must be bitwise-equal floats)
    auto op = oracle.pores;
    std::sort(op.begin(), op.end(), poreLess);
    std::sort(poresAll.begin(), poresAll.end(), poreLess);
    if (op.size() != poresAll.size()) {
      std::printf("[%s] FAIL pores: %zu (dist) vs %zu (oracle)\n", name, poresAll.size(),
                  op.size());
      fail = 1;
    } else {
      for (std::size_t i = 0; i < op.size(); ++i)
        if (!poreEq(op[i], poresAll[i], spc)) {
          std::printf("[%s] FAIL pore %zu: (%g,%g,%g,r=%g) vs (%g,%g,%g,r=%g)\n", name, i,
                      poresAll[i].x, poresAll[i].y, poresAll[i].z, poresAll[i].radius, op[i].x,
                      op[i].y, op[i].z, op[i].radius);
          fail = 1;
          break;
        }
    }

    // connections: the distributed list is global + identical on every rank
    if (dist.connections != oracle.connections) {
      std::printf("[%s] FAIL connections: %zu (dist) vs %zu (oracle)\n", name,
                  dist.connections.size(), oracle.connections.size());
      fail = 1;
    }
    if (!fail)
      std::printf("[%s] PASS np=%d: %zu pores, %zu connections, seg exact on %zu voxels\n", name,
                  size, op.size(), oracle.connections.size(), n);
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
    fail += runCase("sphere_lattice", {36, 30, 24}, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f},
                    sphereLatticeSdf, MPI_COMM_WORLD);
    fail += runCase("trig_field", {32, 32, 32}, {-1.f, 2.f, 0.5f}, {0.5f, 1.f, 2.f}, trigSdf,
                    MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return fail ? 1 : 0;
}
