// peclet-pnm — the single-rank extraction contract on synthetic SDFs (any Kokkos backend).
//
// Every expected number below was derived by hand from the geometry and then confirmed against
// the implementation; a change in any of them is a change of the extraction's behaviour.
//
//   one spherical pore (16^3, R = 5.5 at voxel (8,8,8))
//       -> 1 pore, radius exactly R, at the integer centre; 1 solid; no throat.
//   two overlapping spherical pores (24x16x16, centres (8,8,8) and (16,8,8), R = 5.5)
//       -> 2 pores, radius R each, at their centres; 1 solid; ONE throat (1,2) at the mid-plane.
//   2x2x2 periodic lattice of solid spheres (36x30x24, the tests/kokkos_mpi geometry)
//       -> 8 pores at the cell body centres x in {0,18}, y in {0,15}, z in {0,12} with radius
//          sqrt(9^2 + 7.5^2 + 6^2) - 0.22*24 = 7.882 (the SDF value at the peak voxel); 12 throats
//          = the face-adjacent body-centre pairs (8 pores x 3 axes / 2); 8 solid spheres each
//          touching all 8 pores (64 pore-solid contacts) => 76 connections; no debris (label 0).
//
// Also: the staged path (extract_pores -> segment_volume -> extract_topology) equals the fused
// extract_pore_network stage for stage; two runs are bitwise equal (seg + connections bitwise,
// pores as a sorted set — the emission order is an atomic-slot race); origin/spacing map the pore
// position; empty input -> empty output.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "../synthetic_sdf.hpp"
#include "pore_extraction.hpp"

using pnm::Pore;
using pnm::PoreNetwork;

namespace {

int failures = 0;

#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

using Pair = std::pair<int, int>;

bool poreLess(const Pore& a, const Pore& b) {
  return std::tie(a.x, a.y, a.z, a.radius) < std::tie(b.x, b.y, b.z, b.radius);
}
std::vector<Pore> sorted(std::vector<Pore> p) {
  std::sort(p.begin(), p.end(), poreLess);
  return p;
}
bool poreBitwise(const Pore& a, const Pore& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.radius == b.radius;
}
bool poresBitwise(const std::vector<Pore>& a, const std::vector<Pore>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), poreBitwise);
}

/// A throat is a connection between two PORE labels (> 0); pore-solid contacts are the rest.
std::vector<Pair> throats(const std::vector<Pair>& conns) {
  std::vector<Pair> t;
  for (const auto& c : conns)
    if (c.first > 0 && c.second > 0)
      t.push_back(c);
  return t;
}
int minLabel(const std::vector<int>& seg) {
  return *std::min_element(seg.begin(), seg.end());
}
int maxLabel(const std::vector<int>& seg) {
  return *std::max_element(seg.begin(), seg.end());
}
std::size_t countLabel(const std::vector<int>& seg, int l) {
  return std::size_t(std::count(seg.begin(), seg.end(), l));
}

/// Position to 1e-5*spacing: the sub-voxel centroid accumulation may be FMA-contracted
/// differently per backend (measured ~1e-8 wobble at symmetric peaks on CUDA, bitwise on OpenMP).
bool poreAt(const Pore& p, std::array<float, 3> xyz, std::array<float, 3> spc) {
  return std::fabs(p.x - xyz[0]) <= 1e-5f * spc[0] && std::fabs(p.y - xyz[1]) <= 1e-5f * spc[1] &&
         std::fabs(p.z - xyz[2]) <= 1e-5f * spc[2];
}

void report(const char* name, const PoreNetwork& net) {
  std::printf("[%s] %zu pores, %zu connections (%zu throats), labels %d..%d\n", name,
              net.pores.size(), net.connections.size(), throats(net.connections).size(),
              minLabel(net.seg), maxLabel(net.seg));
}

// staged == fused, stage for stage; two fused runs bitwise equal ------------------------------
void checkStagedAndDeterministic(const char* name, const std::vector<float>& sdf,
                                 std::array<int, 3> gd, std::array<float, 3> org,
                                 std::array<float, 3> spc, const PoreNetwork& net) {
  const auto pores = pnm::extract_pores_k(sdf, gd, org, spc);
  const auto seg = pnm::segment_volume_k(sdf, gd, spc);
  const auto conns = pnm::extract_topology_k(seg, gd);
  CHECK(poresBitwise(sorted(pores), sorted(net.pores)));
  CHECK(seg == net.seg);
  CHECK(conns == net.connections);

  const auto again = pnm::extract_pore_network_k(sdf, gd, org, spc);
  CHECK(poresBitwise(sorted(again.pores), sorted(net.pores)));
  CHECK(again.seg == net.seg);
  CHECK(again.connections == net.connections);
  std::printf("[%s] staged == fused, two runs bitwise equal\n", name);
}

// 1. one spherical pore --------------------------------------------------------------------------
void testOneSphere() {
  const std::array<int, 3> gd{16, 16, 16}, c{8, 8, 8};
  const float R = 5.5f;
  const std::array<float, 3> org{0.f, 0.f, 0.f}, spc{1.f, 1.f, 1.f};
  const auto sdf = pnm::test::sampleGrid(
      gd, [&](int x, int y, int z) { return pnm::test::spherePoreSdf(x, y, z, c, R); });
  const auto net = pnm::extract_pore_network_k(sdf, gd, org, spc);
  report("one_sphere", net);

  CHECK(net.pores.size() == 1);
  if (net.pores.size() == 1) {
    CHECK(net.pores[0].radius == R);  // a straight copy of the peak SDF value
    CHECK(poreAt(net.pores[0], {8.f, 8.f, 8.f}, spc));
  }
  CHECK(maxLabel(net.seg) == 1);
  CHECK(minLabel(net.seg) == -1);
  CHECK(countLabel(net.seg, 0) == 0);
  CHECK((net.connections == std::vector<Pair>{{-1, 1}}));
  CHECK(throats(net.connections).empty());
  // pore voxels are exactly the SDF > 0 voxels, all labelled 1
  std::size_t nPore = 0;
  for (std::size_t i = 0; i < sdf.size(); ++i) {
    if (sdf[i] > 0.f)
      ++nPore;
    CHECK((sdf[i] > 0.f) == (net.seg[i] == 1));
  }
  CHECK(countLabel(net.seg, 1) == nPore);
  checkStagedAndDeterministic("one_sphere", sdf, gd, org, spc, net);

  // origin/spacing: the pore position is origin + voxel*spacing (anisotropic, shifted origin);
  // the count contract is unchanged.
  const std::array<float, 3> org2{-1.f, 2.f, 0.5f}, spc2{0.5f, 1.f, 2.f};
  const auto net2 = pnm::extract_pore_network_k(sdf, gd, org2, spc2);
  CHECK(net2.pores.size() == 1);
  if (net2.pores.size() == 1) {
    CHECK(net2.pores[0].radius == R);
    CHECK(poreAt(net2.pores[0],
                 {org2[0] + 8 * spc2[0], org2[1] + 8 * spc2[1], org2[2] + 8 * spc2[2]}, spc2));
  }
  CHECK((net2.connections == std::vector<Pair>{{-1, 1}}));
}

// 2. two overlapping spherical pores -------------------------------------------------------------
void testTwoSpheres() {
  const std::array<int, 3> gd{24, 16, 16}, c1{8, 8, 8}, c2{16, 8, 8};
  const float R = 5.5f;
  const std::array<float, 3> org{0.f, 0.f, 0.f}, spc{1.f, 1.f, 1.f};
  const auto sdf = pnm::test::sampleGrid(
      gd, [&](int x, int y, int z) { return pnm::test::twoSpherePoresSdf(x, y, z, c1, c2, R); });
  const auto net = pnm::extract_pore_network_k(sdf, gd, org, spc);
  report("two_spheres", net);

  CHECK(net.pores.size() == 2);
  const auto p = sorted(net.pores);
  if (p.size() == 2) {
    CHECK(p[0].radius == R && p[1].radius == R);
    CHECK(poreAt(p[0], {8.f, 8.f, 8.f}, spc));
    CHECK(poreAt(p[1], {16.f, 8.f, 8.f}, spc));
  }
  CHECK(maxLabel(net.seg) == 2);
  CHECK(minLabel(net.seg) == -1);
  CHECK(countLabel(net.seg, 0) == 0);
  CHECK(net.connections == (std::vector<Pair>{{-1, 1}, {-1, 2}, {1, 2}}));
  CHECK((throats(net.connections) == std::vector<Pair>{{1, 2}}));
  // pore 1 owns the peak at c1 (first appearance in x-fastest order), pore 2 the peak at c2; the
  // mid-plane x = 12 belongs to one of them (tie broken on voxel index) — every pore voxel is
  // labelled and the two basins split the union.
  const auto idx = [&](int x, int y, int z) { return (std::size_t(z) * gd[1] + y) * gd[0] + x; };
  CHECK(net.seg[idx(8, 8, 8)] == 1);
  CHECK(net.seg[idx(16, 8, 8)] == 2);
  CHECK(net.seg[idx(10, 8, 8)] == 1);
  CHECK(net.seg[idx(14, 8, 8)] == 2);
  for (std::size_t i = 0; i < sdf.size(); ++i)
    CHECK((sdf[i] > 0.f) == (net.seg[i] > 0));
  checkStagedAndDeterministic("two_spheres", sdf, gd, org, spc, net);
}

// 3. 2x2x2 periodic lattice of solid spheres ----------------------------------------------------
void testSphereLattice() {
  const std::array<int, 3> gd{36, 30, 24};
  const std::array<float, 3> org{0.f, 0.f, 0.f}, spc{1.f, 1.f, 1.f};
  const auto sdf = pnm::test::sampleGrid(
      gd, [&](int x, int y, int z) { return pnm::test::sphereLatticeSdf(x, y, z, gd); });
  const auto net = pnm::extract_pore_network_k(sdf, gd, org, spc);
  report("sphere_lattice", net);

  // 8 body-centre pores, radius = the peak SDF value = sqrt(9^2+7.5^2+6^2) - 0.22*24.
  CHECK(net.pores.size() == 8);
  const float Rpeak = pnm::test::sphereLatticeSdf(0, 0, 0, gd);
  CHECK(std::fabs(Rpeak - (std::sqrt(9.f * 9.f + 7.5f * 7.5f + 6.f * 6.f) - 0.22f * 24.f)) < 1e-5f);
  const auto p = sorted(net.pores);
  std::size_t k = 0;
  for (float x : {0.f, 18.f})
    for (float y : {0.f, 15.f})
      for (float z : {0.f, 12.f}) {  // sorted by (x, y, z)
        if (k < p.size()) {
          CHECK(p[k].radius == Rpeak);
          CHECK(poreAt(p[k], {x, y, z}, spc));
        }
        ++k;
      }

  // 8 solid spheres (labels -1..-8), no debris, pore ids 1..8.
  CHECK(maxLabel(net.seg) == 8);
  CHECK(minLabel(net.seg) == -8);
  CHECK(countLabel(net.seg, 0) == 0);
  for (std::size_t i = 0; i < sdf.size(); ++i)
    CHECK((sdf[i] > 0.f) == (net.seg[i] > 0));

  // 12 face throats between body centres; ids follow first appearance in x-fastest order:
  // 1:(0,0,0) 2:(18,0,0) 3:(0,15,0) 4:(18,15,0) 5..8: the same at z = 12.
  const std::vector<Pair> expectedThroats{{1, 2}, {1, 3}, {1, 5}, {2, 4}, {2, 6}, {3, 4},
                                          {3, 7}, {4, 8}, {5, 6}, {5, 7}, {6, 8}, {7, 8}};
  CHECK(throats(net.connections) == expectedThroats);
  // every sphere touches every pore: 64 pore-solid contacts, 76 connections in total.
  CHECK(net.connections.size() == 76);
  std::vector<Pair> expected;
  for (int s = -8; s <= -1; ++s)
    for (int q = 1; q <= 8; ++q)
      expected.push_back({s, q});
  expected.insert(expected.end(), expectedThroats.begin(), expectedThroats.end());
  CHECK(net.connections == expected);

  checkStagedAndDeterministic("sphere_lattice", sdf, gd, org, spc, net);
}

// 4. empty input ---------------------------------------------------------------------------------
void testEmpty() {
  const std::array<int, 3> gd{0, 0, 0};
  const std::array<float, 3> z{0.f, 0.f, 0.f}, one{1.f, 1.f, 1.f};
  const std::vector<float> none;
  CHECK(pnm::extract_pores_k(none, gd, z, one).empty());
  CHECK(pnm::segment_volume_k(none, gd, one).empty());
  CHECK(pnm::extract_topology_k(std::vector<int>{}, gd).empty());
  const auto net = pnm::extract_pore_network_k(none, gd, z, one);
  CHECK(net.pores.empty() && net.seg.empty() && net.connections.empty());
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  std::printf("peclet-pnm single-rank tests on %s\n", Kokkos::DefaultExecutionSpace::name());
  testOneSphere();
  testTwoSpheres();
  testSphereLattice();
  testEmpty();
  if (failures)
    std::fprintf(stderr, "%d CHECK(s) failed\n", failures);
  else
    std::printf("all checks passed\n");
  return failures ? 1 : 0;
}
