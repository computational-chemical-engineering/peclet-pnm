// cfd-gpu — portable (Kokkos) pore-network extraction from an SDF.
//
// Kokkos port of pore_extraction.cu (the pnm_backend module): pore detection (local maxima + weighted
// centroid), marker-controlled watershed segmentation of the solid (init markers -> union-find CCL ->
// flood fill), gradient-path pore basins, and boundary-pair topology. Grid-stride __global__ kernels ->
// Kokkos::parallel_for, atomicAdd/atomicMin -> Kokkos::atomic_*, cudaMalloc/Memcpy -> Kokkos::View +
// deep_copy. The thrust includes in the .cu were dead (sort/unique is host std::sort). Host orchestration
// (label renumber via std::map, topology sort/unique) stays on the host. Runs on any Kokkos backend.
#ifndef CFD_PORE_EXTRACTION_HPP
#define CFD_PORE_EXTRACTION_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace pnm {

struct Pore { float x, y, z, radius; };
struct I3 { int x, y, z; };

using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;

KOKKOS_INLINE_FUNCTION int get_idx(int x, int y, int z, I3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

// ---- pore detection (local maxima of the SDF + weight-centroid sub-voxel position) ----
inline std::vector<Pore> extract_pores_k(const std::vector<float>& sdf_h, std::array<int, 3> resolution,
                                         std::array<float, 3> origin, std::array<float, 3> spacing) {
  if (sdf_h.empty()) return {};
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const float ox = origin[0], oy = origin[1], oz = origin[2];
  const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
  const std::size_t n = sdf_h.size();
  const int max_pores = 1000000;

  Kokkos::View<float*, Mem> sdf("sdf", n);
  Kokkos::View<Pore*, Mem> pores("pores", max_pores);
  Kokkos::View<int, Mem> counter("counter");
  { auto hm = Kokkos::create_mirror_view(sdf);
    for (std::size_t i = 0; i < n; ++i) hm(i) = sdf_h[i];
    Kokkos::deep_copy(sdf, hm); }
  Kokkos::deep_copy(counter, 0);

  Exec space;
  using MD = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "pnm::extract_pores", MD(space, {0, 0, 0}, {res.x, res.y, res.z}),
      KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int ci = get_idx(ix, iy, iz, res);
        const float cv = sdf(ci);
        if (cv <= 0.0f) return;
        bool peak = true;
        for (int dz = -1; dz <= 1 && peak; ++dz)
          for (int dy = -1; dy <= 1 && peak; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0) continue;
              const int ni = get_idx(ix + dx, iy + dy, iz + dz, res);
              const float nv = sdf(ni);
              if (nv > cv || (nv == cv && ni > ci)) { peak = false; break; }
            }
        if (!peak) return;
        float sw = 0.0f, px = 0.0f, py = 0.0f, pz = 0.0f;
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const float v = sdf(get_idx(ix + dx, iy + dy, iz + dz, res));
              float w = v > 0.0f ? v : 0.0f; w = w * w;
              sw += w; px += dx * w; py += dy * w; pz += dz * w;
            }
        float fx = 0, fy = 0, fz = 0;
        if (sw > 1e-6f) { fx = px / sw; fy = py / sw; fz = pz / sw; }
        const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
        if (slot < max_pores)
          pores(slot) = Pore{ox + (ix + fx) * sx, oy + (iy + fy) * sy, oz + (iz + fz) * sz, cv};
      });
  space.fence();

  int h_count = 0; { auto hc = Kokkos::create_mirror_view(counter); Kokkos::deep_copy(hc, counter); h_count = hc(); }
  if (h_count > max_pores) h_count = max_pores;
  std::vector<Pore> out(h_count);
  { auto hp = Kokkos::create_mirror_view(pores);
    Kokkos::deep_copy(hp, pores);
    for (int i = 0; i < h_count; ++i) out[i] = hp(i); }
  return out;
}

// ---- marker-controlled watershed segmentation of the solid + gradient-path pore basins ----
inline std::vector<int> segment_volume_k(const std::vector<float>& sdf_h, std::array<int, 3> resolution,
                                         std::array<float, 3> spacing) {
  if (sdf_h.empty()) return {};
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const std::size_t n = sdf_h.size();
  const float min_sp = std::min(spacing[0], std::min(spacing[1], spacing[2]));
  const float thr = -1.5f * min_sp;

  Kokkos::View<float*, Mem> sdf("sdf", n);
  Kokkos::View<int*, Mem> labels("labels", n), roots("roots", n);
  Kokkos::View<int, Mem> changed("changed");
  { auto hm = Kokkos::create_mirror_view(sdf);
    for (std::size_t i = 0; i < n; ++i) hm(i) = sdf_h[i];
    Kokkos::deep_copy(sdf, hm); }

  Exec space;
  using MD = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
  const auto full = MD(space, {0, 0, 0}, {res.x, res.y, res.z});

  // 1. init markers (deep solid -> own index, else -1)
  Kokkos::parallel_for("pnm::init_markers", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
    const int i = get_idx(ix, iy, iz, res);
    labels(i) = (sdf(i) < thr) ? i : -1;
  });
  space.fence();

  // 2. union-find CCL on markers (26-connectivity, 13 forward neighbours) + path compression, to fixpoint
  auto flatten = [&]() {
    Kokkos::parallel_for("pnm::flatten", Kokkos::RangePolicy<Exec>(space, 0, n), KOKKOS_LAMBDA(std::size_t idx) {
      int l = labels(idx);
      if (l != -1) { while (l != labels(l)) l = labels(l); labels(idx) = l; }
    });
    space.fence();
  };
  int h_changed = 1;
  while (h_changed) {
    Kokkos::deep_copy(changed, 0);
    Kokkos::parallel_for("pnm::merge_markers", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
      const int idx = get_idx(ix, iy, iz, res);
      const int my = labels(idx);
      if (my == -1) return;
      const int dz_l[13] = {1,1,1,1,1,1,1,1,1,0,0,0,0};
      const int dy_l[13] = {-1,-1,-1,0,0,0,1,1,1,1,1,1,0};
      const int dx_l[13] = {-1,0,1,-1,0,1,-1,0,1,-1,0,1,1};
      for (int k = 0; k < 13; ++k) {
        const int ni = get_idx(ix + dx_l[k], iy + dy_l[k], iz + dz_l[k], res);
        const int nl = labels(ni);
        if (nl != -1 && my != nl) {
          int rm = my; while (rm != labels(rm)) rm = labels(rm);
          int rn = nl; while (rn != labels(rn)) rn = labels(rn);
          if (rm != rn) {
            const int small = rm < rn ? rm : rn, large = rm < rn ? rn : rm;
            Kokkos::atomic_min(&labels(large), small);
            changed() = 1;
          }
        }
      }
    });
    space.fence();
    flatten();
    auto hc = Kokkos::create_mirror_view(changed); Kokkos::deep_copy(hc, changed); h_changed = hc();
  }

  // 3. flood-fill the remaining (shallow) solid voxels (26-connectivity, smallest neighbour label), to fixpoint
  h_changed = 1;
  while (h_changed) {
    Kokkos::deep_copy(changed, 0);
    Kokkos::parallel_for("pnm::flood", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
      const int idx = get_idx(ix, iy, iz, res);
      if (sdf(idx) >= 0.0f) return;          // pore: ignore
      if (labels(idx) != -1) return;         // already labelled
      int best = -1;
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            const int nl = labels(get_idx(ix + dx, iy + dy, iz + dz, res));
            if (nl != -1 && (best == -1 || nl < best)) best = nl;
          }
      if (best != -1) { labels(idx) = best; changed() = 1; }
    });
    space.fence();
    auto hc = Kokkos::create_mirror_view(changed); Kokkos::deep_copy(hc, changed); h_changed = hc();
  }

  // 4. gradient-path pore basins (ascent for pores, descent for solids; 26-connectivity, tie-break on index)
  Kokkos::parallel_for("pnm::gradient_path", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
    const int ci = get_idx(ix, iy, iz, res);
    const bool ascent = (sdf(ci) > 0.0f);
    int walker = ci; const int MAX_STEPS = 512;
    for (int s = 0; s < MAX_STEPS; ++s) {
      int best = walker; float bv = sdf(walker);
      const int wx = walker % res.x, wy = (walker / res.x) % res.y, wz = walker / (res.x * res.y);
      for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            const int ni = get_idx(wx + dx, wy + dy, wz + dz, res);
            const float nv = sdf(ni);
            if (ascent) { if (nv > bv) { bv = nv; best = ni; } else if (nv == bv && ni > best) best = ni; }
            else        { if (nv < bv) { bv = nv; best = ni; } else if (nv == bv && ni > best) best = ni; }
          }
      if (best == walker) break;
      walker = best;
    }
    roots(ci) = walker;
  });
  space.fence();

  std::vector<int> solid_seg(n), pore_roots(n);
  { auto hl = Kokkos::create_mirror_view(labels); Kokkos::deep_copy(hl, labels);
    auto hr = Kokkos::create_mirror_view(roots); Kokkos::deep_copy(hr, roots);
    for (std::size_t i = 0; i < n; ++i) { solid_seg[i] = hl(i); pore_roots[i] = hr(i); } }

  // 5. combine + renumber on host (pores >0, solids <0, debris 0) -- matches the CUDA host pass
  std::vector<int> seg(n);
  std::map<int, int> lmap; int next_pore = 1, next_solid = -1;
  for (std::size_t i = 0; i < n; ++i) {
    if (sdf_h[i] > 0) {
      const int r = pore_roots[i];
      if (lmap.find(r) == lmap.end()) lmap[r] = next_pore++;
      seg[i] = lmap[r];
    } else {
      const int l = solid_seg[i];
      if (l == -1) seg[i] = 0;
      else { if (lmap.find(l) == lmap.end()) lmap[l] = next_solid--; seg[i] = lmap[l]; }
    }
  }
  return seg;
}

// ---- boundary-pair topology (unique adjacent-label pairs across +x/+y/+z faces) ----
inline std::vector<std::pair<int, int>> extract_topology_k(const std::vector<int>& seg_h,
                                                           std::array<int, 3> resolution) {
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const std::size_t n = seg_h.size();
  const int max_pairs = (int)(n * 3);

  Kokkos::View<int*, Mem> seg("seg", n);
  Kokkos::View<int*, Mem> pairs("pairs", (std::size_t)max_pairs * 2);  // flattened (l1,l2) interleaved
  Kokkos::View<int, Mem> cnt("cnt");
  { auto hm = Kokkos::create_mirror_view(seg);
    for (std::size_t i = 0; i < n; ++i) hm(i) = seg_h[i];
    Kokkos::deep_copy(seg, hm); }
  Kokkos::deep_copy(cnt, 0);

  Exec space;
  using MD = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "pnm::boundary_pairs", MD(space, {0, 0, 0}, {res.x, res.y, res.z}),
      KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int idx = get_idx(ix, iy, iz, res);
        const int my = seg(idx);
        const int dx_l[3] = {1,0,0}, dy_l[3] = {0,1,0}, dz_l[3] = {0,0,1};
        for (int k = 0; k < 3; ++k) {
          const int nl = seg(get_idx(ix + dx_l[k], iy + dy_l[k], iz + dz_l[k], res));
          if (my != nl) {
            const int l1 = my < nl ? my : nl, l2 = my < nl ? nl : my;
            const int slot = Kokkos::atomic_fetch_add(&cnt(), 1);
            if (slot < max_pairs) { pairs(2 * slot) = l1; pairs(2 * slot + 1) = l2; }
          }
        }
      });
  space.fence();

  int h_count = 0; { auto hc = Kokkos::create_mirror_view(cnt); Kokkos::deep_copy(hc, cnt); h_count = hc(); }
  if (h_count > max_pairs) h_count = max_pairs;
  std::vector<std::pair<int, int>> result; result.reserve(h_count);
  { auto hp = Kokkos::create_mirror_view(pairs); Kokkos::deep_copy(hp, pairs);
    for (int i = 0; i < h_count; ++i) result.push_back({hp(2 * i), hp(2 * i + 1)}); }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

}  // namespace pnm

#endif  // CFD_PORE_EXTRACTION_HPP
