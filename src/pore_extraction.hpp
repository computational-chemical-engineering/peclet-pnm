/// @file
/// @brief flow — portable (Kokkos) pore-network extraction from an SDF.
///
/// Kokkos port of pore_extraction.cu (the pnm_backend module): pore detection (local maxima + weighted
/// centroid), marker-controlled watershed segmentation of the solid (init markers -> union-find CCL ->
/// flood fill), gradient-path pore basins, and boundary-pair topology. Grid-stride __global__ kernels ->
/// Kokkos::parallel_for, atomicAdd/atomicMin -> Kokkos::atomic_*, cudaMalloc/Memcpy -> Kokkos::View +
/// deep_copy. The thrust includes in the .cu were dead (sort/unique is host std::sort). Host orchestration
/// (label renumber via std::map, topology sort/unique) stays on the host. Runs on any Kokkos backend.
#ifndef PECLET_FLOW_PORE_EXTRACTION_HPP
#define PECLET_FLOW_PORE_EXTRACTION_HPP

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

/// Bulk host->device upload of a whole std::vector via one deep_copy over an unmanaged host view —
/// replaces the per-element `create_mirror_view` + fill loop (F3). `d` must already be sized to `h`.
template <class T>
inline void uploadVec(const std::vector<T>& h, const Kokkos::View<T*, Mem>& d) {
  if (h.empty()) return;
  Kokkos::deep_copy(
      d, Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h.data(),
                                                                                            h.size()));
}

/// Download the first `count` elements of a device view into a fresh std::vector via one deep_copy —
/// replaces the `create_mirror_view` (whole view) + element loop (S2/G1), and only moves what is used.
template <class V>
inline std::vector<typename V::value_type> downloadN(const V& d, std::size_t count) {
  std::vector<typename V::value_type> out(count);
  if (count)
    Kokkos::deep_copy(
        Kokkos::View<typename V::value_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            out.data(), count),
        Kokkos::subview(d, Kokkos::make_pair(std::size_t(0), count)));
  return out;
}

KOKKOS_INLINE_FUNCTION int get_idx(int x, int y, int z, I3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

// ---- pore detection (local maxima of the SDF + weight-centroid sub-voxel position) ----
// Device core: operates on an already-uploaded device SDF, so a fused pipeline uploads the SDF once.
inline std::vector<Pore> extractPoresView(const Kokkos::View<float*, Mem>& sdf,
                                            std::array<int, 3> resolution, std::array<float, 3> origin,
                                            std::array<float, 3> spacing) {
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const float ox = origin[0], oy = origin[1], oz = origin[2];
  const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
  const std::size_t n = sdf.extent(0);
  const int max_pores = 1000000;

  Kokkos::View<Pore*, Mem> pores("pores", max_pores);
  Kokkos::View<int, Mem> counter("counter");
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
  return downloadN(pores, static_cast<std::size_t>(h_count));
}

// Host wrapper: upload the SDF, then run the device core.
inline std::vector<Pore> extract_pores_k(const std::vector<float>& sdf_h, std::array<int, 3> resolution,
                                         std::array<float, 3> origin, std::array<float, 3> spacing) {
  if (sdf_h.empty()) return {};
  Kokkos::View<float*, Mem> sdf("sdf", sdf_h.size());
  uploadVec(sdf_h, sdf);
  return extractPoresView(sdf, resolution, origin, spacing);
}

// ---- marker-controlled watershed segmentation of the solid + gradient-path pore basins ----
// Device core: takes an uploaded device SDF, returns the (device-resident) segmentation View.
inline Kokkos::View<int*, Mem> segmentVolumeView(const Kokkos::View<float*, Mem>& sdf,
                                                   std::array<int, 3> resolution,
                                                   std::array<float, 3> spacing) {
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const std::size_t n = sdf.extent(0);
  const float min_sp = std::min(spacing[0], std::min(spacing[1], spacing[2]));
  const float thr = -1.5f * min_sp;

  Kokkos::View<int*, Mem> labels("labels", n), roots("roots", n);
  Kokkos::View<int, Mem> changed("changed");

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

  // 5. combine + renumber ON DEVICE (pores >0 ascending, solids <0 descending, debris 0), matching the
  // host first-encounter relabel exactly (F2). A label's id is its rank in voxel-index order of first
  // appearance — which equals the exclusive prefix sum of a "first-occurrence" flag, so it parallelises
  // without the host std::map + the two full-volume D2Hs (labels/roots stay on device). The root/label
  // values ARE voxel indices, so per-label scratch is size-n arrays indexed by that value.
  Kokkos::View<int*, Mem> seg("seg", n);
  Kokkos::View<int*, Mem> minPoreIdx("minPoreIdx", n), minSolidIdx("minSolidIdx", n);
  Kokkos::View<int*, Mem> poreFirst("poreFirst", n), solidFirst("solidFirst", n);
  Kokkos::View<int*, Mem> poreRank("poreRank", n), solidRank("solidRank", n);
  Kokkos::View<int*, Mem> poreId("poreId", n), solidId("solidId", n);
  constexpr int kBig = 0x7fffffff;
  Kokkos::deep_copy(minPoreIdx, kBig);
  Kokkos::deep_copy(minSolidIdx, kBig);
  const std::size_t nn = n;
  using R1 = Kokkos::RangePolicy<Exec>;
  // (a) per-root/label min voxel index of first appearance (pores use `roots`, solids use `labels`).
  Kokkos::parallel_for("pnm::relabel_min", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
    if (sdf(i) > 0.0f)
      Kokkos::atomic_min(&minPoreIdx(roots(i)), static_cast<int>(i));
    else if (labels(i) != -1)
      Kokkos::atomic_min(&minSolidIdx(labels(i)), static_cast<int>(i));
  });
  space.fence();
  // (b) flag the voxel that is the first appearance of its label.
  Kokkos::parallel_for("pnm::relabel_first", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
    poreFirst(i) = (sdf(i) > 0.0f && minPoreIdx(roots(i)) == static_cast<int>(i)) ? 1 : 0;
    solidFirst(i) =
        (sdf(i) <= 0.0f && labels(i) != -1 && minSolidIdx(labels(i)) == static_cast<int>(i)) ? 1 : 0;
  });
  space.fence();
  // (c) exclusive prefix sums ⇒ the 0-based rank (= first-encounter order) of each first voxel.
  Kokkos::parallel_scan("pnm::relabel_porescan", R1(space, 0, nn),
                        KOKKOS_LAMBDA(std::size_t i, int& upd, const bool fin) {
                          const int v = poreFirst(i);
                          if (fin) poreRank(i) = upd;
                          upd += v;
                        });
  Kokkos::parallel_scan("pnm::relabel_solidscan", R1(space, 0, nn),
                        KOKKOS_LAMBDA(std::size_t i, int& upd, const bool fin) {
                          const int v = solidFirst(i);
                          if (fin) solidRank(i) = upd;
                          upd += v;
                        });
  space.fence();
  // (d) assign each label its signed id at its first voxel (pores 1,2,…; solids −1,−2,…).
  Kokkos::parallel_for("pnm::relabel_assign", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
    if (poreFirst(i)) poreId(roots(i)) = poreRank(i) + 1;
    if (solidFirst(i)) solidId(labels(i)) = -(solidRank(i) + 1);
  });
  space.fence();
  // (e) scatter ids to every voxel: pore→poreId, labelled solid→solidId, unlabelled solid (debris)→0.
  Kokkos::parallel_for("pnm::relabel_seg", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
    seg(i) = (sdf(i) > 0.0f) ? poreId(roots(i)) : (labels(i) == -1 ? 0 : solidId(labels(i)));
  });
  space.fence();
  return seg;  // device-resident; the fused pipeline feeds it straight to the topology stage
}

// Host wrapper: upload the SDF, segment on device, download the segmentation.
inline std::vector<int> segment_volume_k(const std::vector<float>& sdf_h, std::array<int, 3> resolution,
                                         std::array<float, 3> spacing) {
  if (sdf_h.empty()) return {};
  Kokkos::View<float*, Mem> sdf("sdf", sdf_h.size());
  uploadVec(sdf_h, sdf);
  return downloadN(segmentVolumeView(sdf, resolution, spacing), sdf_h.size());
}

// ---- boundary-pair topology (unique adjacent-label pairs across +x/+y/+z faces) ----
// Device core: takes the (device-resident) segmentation View directly — no re-upload.
inline std::vector<std::pair<int, int>> extractTopologyView(const Kokkos::View<int*, Mem>& seg,
                                                              std::array<int, 3> resolution) {
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const std::size_t n = seg.extent(0);
  const int max_pairs = (int)(n * 3);

  Kokkos::View<int*, Mem> pairs("pairs", (std::size_t)max_pairs * 2);  // flattened (l1,l2) interleaved
  Kokkos::View<int, Mem> cnt("cnt");
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
  std::vector<int> flat = downloadN(pairs, static_cast<std::size_t>(2 * h_count));  // only the used slots
  std::vector<std::pair<int, int>> result; result.reserve(h_count);
  for (int i = 0; i < h_count; ++i) result.push_back({flat[2 * i], flat[2 * i + 1]});
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

// Host wrapper: upload the segmentation, then run the device core.
inline std::vector<std::pair<int, int>> extract_topology_k(const std::vector<int>& seg_h,
                                                           std::array<int, 3> resolution) {
  if (seg_h.empty()) return {};
  Kokkos::View<int*, Mem> seg("seg", seg_h.size());
  uploadVec(seg_h, seg);
  return extractTopologyView(seg, resolution);
}

/// The full pore network from one extraction: pores, the per-voxel segmentation (flat), and the
/// label-adjacency connections.
struct PoreNetwork {
  std::vector<Pore> pores;
  std::vector<int> seg;
  std::vector<std::pair<int, int>> connections;
};

// ---- fused pipeline (F1): upload the SDF ONCE, keep it + the segmentation device-resident across all
// three stages (extract_pores → segment_volume → extract_topology), so neither the SDF nor seg is
// re-uploaded or round-tripped between stages. Only the final results cross back to the host. Each
// stage's result is identical to calling the three functions separately. ----
inline PoreNetwork extract_pore_network_k(const std::vector<float>& sdf_h,
                                          std::array<int, 3> resolution, std::array<float, 3> origin,
                                          std::array<float, 3> spacing) {
  PoreNetwork out;
  if (sdf_h.empty()) return out;
  Kokkos::View<float*, Mem> sdf("sdf", sdf_h.size());
  uploadVec(sdf_h, sdf);
  out.pores = extractPoresView(sdf, resolution, origin, spacing);
  Kokkos::View<int*, Mem> seg = segmentVolumeView(sdf, resolution, spacing);  // stays on device
  out.connections = extractTopologyView(seg, resolution);
  out.seg = downloadN(seg, sdf_h.size());
  return out;
}

}  // namespace pnm

#endif  // PECLET_FLOW_PORE_EXTRACTION_HPP
