// peclet-pnm — deterministic synthetic SDFs shared by the single-rank (tests/kokkos) and the
// distributed (tests/kokkos_mpi) tests. Sign convention: NEGATIVE inside solid, positive in the
// pore space (suite-wide). Every generator is a pure function of the integer voxel coordinates so
// a block sampled on one rank equals the corresponding slice of the full grid sampled on another.
#ifndef PECLET_PNM_TESTS_SYNTHETIC_SDF_HPP
#define PECLET_PNM_TESTS_SYNTHETIC_SDF_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pnm::test {

/// 2x2x2 lattice of SOLID spheres in a periodic box (min-image distance); pores at the
/// interstices. Sphere centres at gd*(2c+1)/4 per axis, radius 0.22*min(gd).
inline float sphereLatticeSdf(int gx, int gy, int gz, std::array<int, 3> gd) {
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

/// Smooth trigonometric field with wrap-around components (exercises the periodic images).
inline float trigSdf(int gx, int gy, int gz, std::array<int, 3> gd) {
  const float x = 2.0f * float(M_PI) * gx / gd[0], y = 2.0f * float(M_PI) * gy / gd[1],
              z = 2.0f * float(M_PI) * gz / gd[2];
  return std::sin(x) * std::cos(y) + 0.7f * std::sin(2.0f * z) - 0.1f;
}

/// One spherical PORE (SDF > 0 inside) of radius `radius` centred on the integer voxel `c`; solid
/// everywhere outside. The SDF peak is exactly `radius` at `c` (no distance rounding).
inline float spherePoreSdf(int gx, int gy, int gz, std::array<int, 3> c, float radius) {
  const float dx = float(gx - c[0]), dy = float(gy - c[1]), dz = float(gz - c[2]);
  return radius - std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// Two overlapping spherical pores (the union = max of the two SDFs); a throat at the mid-plane
/// when the centres are closer than 2*radius.
inline float twoSpherePoresSdf(int gx, int gy, int gz, std::array<int, 3> c1, std::array<int, 3> c2,
                               float radius) {
  return std::max(spherePoreSdf(gx, gy, gz, c1, radius), spherePoreSdf(gx, gy, gz, c2, radius));
}

/// Sample `fn(gx, gy, gz)` over the whole `gd` grid in pnm's flat x-fastest layout.
template <class SdfFn>
std::vector<float> sampleGrid(std::array<int, 3> gd, const SdfFn& fn) {
  std::vector<float> v(std::size_t(gd[0]) * gd[1] * gd[2]);
  for (int z = 0; z < gd[2]; ++z)
    for (int y = 0; y < gd[1]; ++y)
      for (int x = 0; x < gd[0]; ++x)
        v[(std::size_t(z) * gd[1] + y) * gd[0] + x] = fn(x, y, z);
  return v;
}

}  // namespace pnm::test

#endif  // PECLET_PNM_TESTS_SYNTHETIC_SDF_HPP
