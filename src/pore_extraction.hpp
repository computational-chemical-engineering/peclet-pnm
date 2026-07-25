/// @file
/// @brief flow — portable (Kokkos) pore-network extraction from an SDF.
///
/// Kokkos port of pore_extraction.cu (the pnm_backend module): pore detection (local maxima +
/// weighted centroid), marker-controlled watershed segmentation of the solid (init markers ->
/// union-find CCL -> flood fill), gradient-path pore basins, and boundary-pair topology.
/// Grid-stride __global__ kernels -> Kokkos::parallel_for, atomicAdd/atomicMin -> Kokkos::atomic_*,
/// cudaMalloc/Memcpy -> Kokkos::View + deep_copy. The thrust includes in the .cu were dead
/// (sort/unique is host std::sort). Host orchestration (label renumber via std::map, topology
/// sort/unique) stays on the host. Runs on any Kokkos backend.
#ifndef PECLET_FLOW_PORE_EXTRACTION_HPP
#define PECLET_FLOW_PORE_EXTRACTION_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <map>
#include <utility>
#include <vector>

namespace pnm {

struct Pore {
  float x, y, z, radius;
};
struct I3 {
  int x, y, z;
};

using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;

/// Bulk host->device upload of a whole std::vector via one deep_copy over an unmanaged host view —
/// replaces the per-element `create_mirror_view` + fill loop (F3). `d` must already be sized to
/// `h`.
template <class T>
inline void uploadVec(const std::vector<T>& h, const Kokkos::View<T*, Mem>& d) {
  if (h.empty())
    return;
  Kokkos::deep_copy(
      d, Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
             h.data(), h.size()));
}

/// Download the first `count` elements of a device view into a fresh std::vector via one deep_copy
/// — replaces the `create_mirror_view` (whole view) + element loop (S2/G1), and only moves what is
/// used.
template <class V>
inline std::vector<typename V::value_type> downloadN(const V& d, std::size_t count) {
  std::vector<typename V::value_type> out(count);
  if (count)
    Kokkos::deep_copy(Kokkos::View<typename V::value_type*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(out.data(), count),
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
// Device core: operates on an already-uploaded device SDF, so a fused pipeline uploads the SDF
// once.
inline std::vector<Pore> extractPoresView(const Kokkos::View<float*, Mem>& sdf,
                                          std::array<int, 3> resolution,
                                          std::array<float, 3> origin,
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
        if (cv <= 0.0f)
          return;
        bool peak = true;
        for (int dz = -1; dz <= 1 && peak; ++dz)
          for (int dy = -1; dy <= 1 && peak; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0)
                continue;
              const int ni = get_idx(ix + dx, iy + dy, iz + dz, res);
              const float nv = sdf(ni);
              if (nv > cv || (nv == cv && ni > ci)) {
                peak = false;
                break;
              }
            }
        if (!peak)
          return;
        float sw = 0.0f, px = 0.0f, py = 0.0f, pz = 0.0f;
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const float v = sdf(get_idx(ix + dx, iy + dy, iz + dz, res));
              float w = v > 0.0f ? v : 0.0f;
              w = w * w;
              sw += w;
              px += dx * w;
              py += dy * w;
              pz += dz * w;
            }
        float fx = 0, fy = 0, fz = 0;
        if (sw > 1e-6f) {
          fx = px / sw;
          fy = py / sw;
          fz = pz / sw;
        }
        const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
        if (slot < max_pores)
          pores(slot) = Pore{ox + (ix + fx) * sx, oy + (iy + fy) * sy, oz + (iz + fz) * sz, cv};
      });
  space.fence();

  int h_count = 0;
  {
    auto hc = Kokkos::create_mirror_view(counter);
    Kokkos::deep_copy(hc, counter);
    h_count = hc();
  }
  if (h_count > max_pores)
    h_count = max_pores;
  return downloadN(pores, static_cast<std::size_t>(h_count));
}

// Host wrapper: upload the SDF, then run the device core.
inline std::vector<Pore> extract_pores_k(const std::vector<float>& sdf_h,
                                         std::array<int, 3> resolution, std::array<float, 3> origin,
                                         std::array<float, 3> spacing) {
  if (sdf_h.empty())
    return {};
  Kokkos::View<float*, Mem> sdf("sdf", sdf_h.size());
  uploadVec(sdf_h, sdf);
  return extractPoresView(sdf, resolution, origin, spacing);
}

// ---- marker-controlled watershed segmentation of the solid + gradient-path pore basins ----
// Device core: takes an uploaded device SDF, returns the (device-resident) segmentation View.
// `rootsOut` (optional) receives the gradient-path pore roots (roots(i)==i at the pore peaks) —
// the network-flow extraction uses them to locate pore centers per label.
inline Kokkos::View<int*, Mem> segmentVolumeView(const Kokkos::View<float*, Mem>& sdf,
                                                 std::array<int, 3> resolution,
                                                 std::array<float, 3> spacing,
                                                 Kokkos::View<int*, Mem>* rootsOut = nullptr) {
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
  Kokkos::parallel_for(
      "pnm::init_markers", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int i = get_idx(ix, iy, iz, res);
        labels(i) = (sdf(i) < thr) ? i : -1;
      });
  space.fence();

  // 2. union-find CCL on markers (26-connectivity, 13 forward neighbours) + path compression, to
  // fixpoint
  auto flatten = [&]() {
    Kokkos::parallel_for(
        "pnm::flatten", Kokkos::RangePolicy<Exec>(space, 0, n), KOKKOS_LAMBDA(std::size_t idx) {
          int l = labels(idx);
          if (l != -1) {
            while (l != labels(l))
              l = labels(l);
            labels(idx) = l;
          }
        });
    space.fence();
  };
  int h_changed = 1;
  while (h_changed) {
    Kokkos::deep_copy(changed, 0);
    Kokkos::parallel_for(
        "pnm::merge_markers", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int idx = get_idx(ix, iy, iz, res);
          const int my = labels(idx);
          if (my == -1)
            return;
          const int dz_l[13] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
          const int dy_l[13] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0};
          const int dx_l[13] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1};
          for (int k = 0; k < 13; ++k) {
            const int ni = get_idx(ix + dx_l[k], iy + dy_l[k], iz + dz_l[k], res);
            const int nl = labels(ni);
            if (nl != -1 && my != nl) {
              int rm = my;
              while (rm != labels(rm))
                rm = labels(rm);
              int rn = nl;
              while (rn != labels(rn))
                rn = labels(rn);
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
    auto hc = Kokkos::create_mirror_view(changed);
    Kokkos::deep_copy(hc, changed);
    h_changed = hc();
  }

  // 3. flood-fill the remaining (shallow) solid voxels (26-connectivity, smallest neighbour label),
  // to fixpoint. Jacobi (double-buffered): each sweep reads only the previous sweep's labels, so
  // the result is DETERMINISTIC — the old in-place sweep could observe same-sweep writes (a device
  // race) — and sweep-for-sweep identical to the distributed flood (pore_extraction_mpi.hpp),
  // which is what makes the multi-rank segmentation bit-exact to this single-rank path.
  Kokkos::View<int*, Mem> labelsN("labelsN", n);
  h_changed = 1;
  while (h_changed) {
    Kokkos::deep_copy(changed, 0);
    Kokkos::deep_copy(labelsN, labels);
    Kokkos::parallel_for(
        "pnm::flood", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int idx = get_idx(ix, iy, iz, res);
          if (sdf(idx) >= 0.0f)
            return;  // pore: ignore
          if (labels(idx) != -1)
            return;  // already labelled
          int best = -1;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const int nl = labels(get_idx(ix + dx, iy + dy, iz + dz, res));
                if (nl != -1 && (best == -1 || nl < best))
                  best = nl;
              }
          if (best != -1) {
            labelsN(idx) = best;
            changed() = 1;
          }
        });
    space.fence();
    std::swap(labels, labelsN);
    auto hc = Kokkos::create_mirror_view(changed);
    Kokkos::deep_copy(hc, changed);
    h_changed = hc();
  }

  // 4. gradient-path pore basins (ascent for pores, descent for solids; 26-connectivity, tie-break
  // on index)
  Kokkos::parallel_for(
      "pnm::gradient_path", full, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int ci = get_idx(ix, iy, iz, res);
        const bool ascent = (sdf(ci) > 0.0f);
        int walker = ci;
        const int MAX_STEPS = 512;
        for (int s = 0; s < MAX_STEPS; ++s) {
          int best = walker;
          float bv = sdf(walker);
          const int wx = walker % res.x, wy = (walker / res.x) % res.y,
                    wz = walker / (res.x * res.y);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const int ni = get_idx(wx + dx, wy + dy, wz + dz, res);
                const float nv = sdf(ni);
                if (ascent) {
                  if (nv > bv) {
                    bv = nv;
                    best = ni;
                  } else if (nv == bv && ni > best)
                    best = ni;
                } else {
                  if (nv < bv) {
                    bv = nv;
                    best = ni;
                  } else if (nv == bv && ni > best)
                    best = ni;
                }
              }
          if (best == walker)
            break;
          walker = best;
        }
        roots(ci) = walker;
      });
  space.fence();

  // 5. combine + renumber ON DEVICE (pores >0 ascending, solids <0 descending, debris 0), matching
  // the host first-encounter relabel exactly (F2). A label's id is its rank in voxel-index order of
  // first appearance — which equals the exclusive prefix sum of a "first-occurrence" flag, so it
  // parallelises without the host std::map + the two full-volume D2Hs (labels/roots stay on
  // device). The root/label values ARE voxel indices, so per-label scratch is size-n arrays indexed
  // by that value.
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
  // (a) per-root/label min voxel index of first appearance (pores use `roots`, solids use
  // `labels`).
  Kokkos::parallel_for(
      "pnm::relabel_min", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
        if (sdf(i) > 0.0f)
          Kokkos::atomic_min(&minPoreIdx(roots(i)), static_cast<int>(i));
        else if (labels(i) != -1)
          Kokkos::atomic_min(&minSolidIdx(labels(i)), static_cast<int>(i));
      });
  space.fence();
  // (b) flag the voxel that is the first appearance of its label.
  Kokkos::parallel_for(
      "pnm::relabel_first", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
        poreFirst(i) = (sdf(i) > 0.0f && minPoreIdx(roots(i)) == static_cast<int>(i)) ? 1 : 0;
        solidFirst(i) =
            (sdf(i) <= 0.0f && labels(i) != -1 && minSolidIdx(labels(i)) == static_cast<int>(i))
                ? 1
                : 0;
      });
  space.fence();
  // (c) exclusive prefix sums ⇒ the 0-based rank (= first-encounter order) of each first voxel.
  Kokkos::parallel_scan(
      "pnm::relabel_porescan", R1(space, 0, nn),
      KOKKOS_LAMBDA(std::size_t i, int& upd, const bool fin) {
        const int v = poreFirst(i);
        if (fin)
          poreRank(i) = upd;
        upd += v;
      });
  Kokkos::parallel_scan(
      "pnm::relabel_solidscan", R1(space, 0, nn),
      KOKKOS_LAMBDA(std::size_t i, int& upd, const bool fin) {
        const int v = solidFirst(i);
        if (fin)
          solidRank(i) = upd;
        upd += v;
      });
  space.fence();
  // (d) assign each label its signed id at its first voxel (pores 1,2,…; solids −1,−2,…).
  Kokkos::parallel_for(
      "pnm::relabel_assign", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
        if (poreFirst(i))
          poreId(roots(i)) = poreRank(i) + 1;
        if (solidFirst(i))
          solidId(labels(i)) = -(solidRank(i) + 1);
      });
  space.fence();
  // (e) scatter ids to every voxel: pore→poreId, labelled solid→solidId, unlabelled solid
  // (debris)→0.
  Kokkos::parallel_for(
      "pnm::relabel_seg", R1(space, 0, nn), KOKKOS_LAMBDA(std::size_t i) {
        seg(i) = (sdf(i) > 0.0f) ? poreId(roots(i)) : (labels(i) == -1 ? 0 : solidId(labels(i)));
      });
  space.fence();
  if (rootsOut)
    *rootsOut = roots;
  return seg;  // device-resident; the fused pipeline feeds it straight to the topology stage
}

// Host wrapper: upload the SDF, segment on device, download the segmentation.
inline std::vector<int> segment_volume_k(const std::vector<float>& sdf_h,
                                         std::array<int, 3> resolution,
                                         std::array<float, 3> spacing) {
  if (sdf_h.empty())
    return {};
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

  Kokkos::View<int*, Mem> pairs("pairs",
                                (std::size_t)max_pairs * 2);  // flattened (l1,l2) interleaved
  Kokkos::View<int, Mem> cnt("cnt");
  Kokkos::deep_copy(cnt, 0);

  Exec space;
  using MD = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "pnm::boundary_pairs", MD(space, {0, 0, 0}, {res.x, res.y, res.z}),
      KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int idx = get_idx(ix, iy, iz, res);
        const int my = seg(idx);
        const int dx_l[3] = {1, 0, 0}, dy_l[3] = {0, 1, 0}, dz_l[3] = {0, 0, 1};
        for (int k = 0; k < 3; ++k) {
          const int nl = seg(get_idx(ix + dx_l[k], iy + dy_l[k], iz + dz_l[k], res));
          if (my != nl) {
            const int l1 = my < nl ? my : nl, l2 = my < nl ? nl : my;
            const int slot = Kokkos::atomic_fetch_add(&cnt(), 1);
            if (slot < max_pairs) {
              pairs(2 * slot) = l1;
              pairs(2 * slot + 1) = l2;
            }
          }
        }
      });
  space.fence();

  int h_count = 0;
  {
    auto hc = Kokkos::create_mirror_view(cnt);
    Kokkos::deep_copy(hc, cnt);
    h_count = hc();
  }
  if (h_count > max_pairs)
    h_count = max_pairs;
  std::vector<int> flat =
      downloadN(pairs, static_cast<std::size_t>(2 * h_count));  // only the used slots
  std::vector<std::pair<int, int>> result;
  result.reserve(h_count);
  for (int i = 0; i < h_count; ++i)
    result.push_back({flat[2 * i], flat[2 * i + 1]});
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

// Host wrapper: upload the segmentation, then run the device core.
inline std::vector<std::pair<int, int>> extract_topology_k(const std::vector<int>& seg_h,
                                                           std::array<int, 3> resolution) {
  if (seg_h.empty())
    return {};
  Kokkos::View<int*, Mem> seg("seg", seg_h.size());
  uploadVec(seg_h, seg);
  return extractTopologyView(seg, resolution);
}

// ---- network flow: throat flow rates + pore-center pressures from a MAC flow field ----
// The method transferred from the Voronoi PNM (pnm_voronoi/pnm_voro): there, the throat flow was
// the integral of u·n over the Voronoi facet (facet sliced against the grid, MAC velocities
// interpolated) and the pore pressure a trilinear interpolation of the cell-centered periodic p
// at the pore center (Voronoi vertex), with the macroscopic gradient added along the min-image
// pore-to-pore vector. On the segmentation network both transfer EXACTLY and more simply:
//   * a throat (label pair) is a set of grid-aligned voxel faces, and on flow's staggered MAC
//     grid the openness-weighted face velocity IS the discrete flux carrier (div = sum of
//     o·u·A over the cell faces) — so Q_ij = sum over interface faces of o·u·A, no slicing or
//     interpolation, and the per-pore signed boundary sum reproduces the solver's divergence
//     (~0 to solve tolerance) — reported as `pore_residual`, the built-in correctness check.
//   * the pore center is the basin's SDF peak (the gradient-path root) with the same sub-voxel
//     centroid refinement as extract_pores; p is trilinearly interpolated there (periodic).
// Conventions (flow's MAC layout): u(i,j,k) lives on the -x face of cell (i,j,k); ox is that
// face's openness (fluid area fraction); cell centers sit at origin + i*spacing (pnm convention).
// CAVEAT: throats are keyed by label PAIR — two disjoint interfaces between the same two pores
// (e.g. once directly and once through the periodic wrap) merge into one entry and their fluxes
// add (possibly cancelling). The Voronoi network kept such parallel throats separate.
struct NetworkFlow {
  std::vector<Pore> pores;                   // ordered by pore label id (pores[k] <-> label k+1)
  std::vector<double> pore_pressure;         // periodic-part p at the pore center (trilinear)
  std::vector<double> pore_residual;         // net signed outflow over the pore's WHOLE boundary
  std::vector<std::pair<int, int>> throats;  // pore-pore label pairs (li < lj)
  std::vector<double> throat_flow;           // Q through the interface, positive li -> lj
  std::vector<double> throat_area;           // open area of the interface (sum o·A over faces)
  std::vector<double> throat_dp;             // total-pressure drop P_i - P_j =
                                             //   (p_i - p_j) - grad_p · minimage(x_j - x_i)
};

inline NetworkFlow extract_network_flow_k(
    const std::vector<float>& sdf_h, std::array<int, 3> resolution, std::array<float, 3> origin,
    std::array<float, 3> spacing, const std::vector<double>& u_h, const std::vector<double>& v_h,
    const std::vector<double>& w_h, const std::vector<double>& p_h,
    const std::vector<double>& ox_h, const std::vector<double>& oy_h,
    const std::vector<double>& oz_h,  // pass empty vectors for a fully-open (non-cut-cell) grid
    std::array<double, 3> grad_p) {
  NetworkFlow out;
  if (sdf_h.empty())
    return out;
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const std::size_t n = sdf_h.size();
  Exec space;
  using R1 = Kokkos::RangePolicy<Exec>;

  Kokkos::View<float*, Mem> sdf("nf::sdf", n);
  uploadVec(sdf_h, sdf);
  Kokkos::View<int*, Mem> roots;
  Kokkos::View<int*, Mem> seg = segmentVolumeView(sdf, resolution, spacing, &roots);

  // number of pore labels
  int np = 0;
  Kokkos::parallel_reduce(
      "nf::np", R1(space, 0, n),
      KOKKOS_LAMBDA(std::size_t i, int& m) {
        if (seg(i) > m)
          m = seg(i);
      },
      Kokkos::Max<int>(np));
  space.fence();
  if (np == 0)
    return out;

  // pore centers: the basin peak (gradient-path root) per label + centroid refinement
  Kokkos::View<int*, Mem> peak("nf::peak", np);
  Kokkos::parallel_for(
      "nf::peaks", R1(space, 0, n), KOKKOS_LAMBDA(std::size_t i) {
        if (sdf(i) > 0.0f && roots(i) == static_cast<int>(i))
          peak(seg(i) - 1) = static_cast<int>(i);
      });
  space.fence();
  Kokkos::View<Pore*, Mem> poresD("nf::pores", np);
  {
    const float oxo = origin[0], oyo = origin[1], ozo = origin[2];
    const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
    const I3 r = res;
    Kokkos::parallel_for(
        "nf::centers", R1(space, 0, np), KOKKOS_LAMBDA(int k) {
          const int ci = peak(k);
          const int ix = ci % r.x, iy = (ci / r.x) % r.y, iz = ci / (r.x * r.y);
          float sw = 0.0f, px = 0.0f, py = 0.0f, pz = 0.0f;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const float v = sdf(get_idx(ix + dx, iy + dy, iz + dz, r));
                float w = v > 0.0f ? v : 0.0f;
                w = w * w;
                sw += w;
                px += dx * w;
                py += dy * w;
                pz += dz * w;
              }
          float fx = 0, fy = 0, fz = 0;
          if (sw > 1e-6f) {
            fx = px / sw;
            fy = py / sw;
            fz = pz / sw;
          }
          poresD(k) = Pore{oxo + (ix + fx) * sx, oyo + (iy + fy) * sy, ozo + (iz + fz) * sz,
                           sdf(ci)};
        });
    space.fence();
  }
  out.pores = downloadN(poresD, static_cast<std::size_t>(np));

  // fields on device
  auto upl = [&](const std::vector<double>& h, const char* nm, bool required) {
    Kokkos::View<double*, Mem> d(nm, required || !h.empty() ? n : 0);
    if (!h.empty())
      uploadVec(h, d);
    return d;
  };
  Kokkos::View<double*, Mem> u = upl(u_h, "nf::u", true), v = upl(v_h, "nf::v", true),
                             w = upl(w_h, "nf::w", true), p = upl(p_h, "nf::p", true);
  const bool hasOpen = !ox_h.empty();
  Kokkos::View<double*, Mem> ox = upl(ox_h, "nf::ox", false), oy = upl(oy_h, "nf::oy", false),
                             oz = upl(oz_h, "nf::oz", false);

  // pore-center pressures: trilinear interpolation of the cell-centered periodic p
  Kokkos::View<double*, Mem> ppres("nf::ppres", np);
  {
    const float oxo = origin[0], oyo = origin[1], ozo = origin[2];
    const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
    const I3 r = res;
    Kokkos::parallel_for(
        "nf::pore_pressure", R1(space, 0, np), KOKKOS_LAMBDA(int k) {
          const Pore po = poresD(k);
          const double g[3] = {(po.x - oxo) / sx, (po.y - oyo) / sy, (po.z - ozo) / sz};
          int b[3];
          double f[3];
          for (int a = 0; a < 3; ++a) {
            const double fl = Kokkos::floor(g[a]);
            b[a] = static_cast<int>(fl);
            f[a] = g[a] - fl;
          }
          double acc = 0.0;
          for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                const double wt = (dx ? f[0] : 1.0 - f[0]) * (dy ? f[1] : 1.0 - f[1]) *
                                  (dz ? f[2] : 1.0 - f[2]);
                acc += wt * p(get_idx(b[0] + dx, b[1] + dy, b[2] + dz, r));
              }
          ppres(k) = acc;
        });
    space.fence();
  }
  {
    auto hv = downloadN(ppres, static_cast<std::size_t>(np));
    out.pore_pressure.assign(hv.begin(), hv.end());
  }

  // FLOW-basin labels for every cell: gradient ascent from ALL cells (not just sdf>0). Cut cells
  // whose CENTER is inside the solid still carry real openness-weighted flux (the near-wall
  // staircase); keyed on seg alone that flux bypasses the pore-pore interface (measured: 6% of
  // the tube flux through the wall annulus, exactly the pore residual). Ascent assigns each such
  // cell to the pore basin whose flow it carries; deep solid ends at a solid peak (negative
  // label) and carries no flux.
  Kokkos::View<int*, Mem> flowLab("nf::flowLab", n);
  {
    const I3 r = res;
    Kokkos::parallel_for(
        "nf::flow_basins", R1(space, 0, n), KOKKOS_LAMBDA(std::size_t i0) {
          int walker = static_cast<int>(i0);
          const int MAX_STEPS = 512;
          for (int s0 = 0; s0 < MAX_STEPS; ++s0) {
            int best = walker;
            float bv = sdf(walker);
            const int wx = walker % r.x, wy = (walker / r.x) % r.y, wz = walker / (r.x * r.y);
            for (int dz = -1; dz <= 1; ++dz)
              for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                  if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                  const int ni = get_idx(wx + dx, wy + dy, wz + dz, r);
                  const float nv = sdf(ni);
                  if (nv > bv) {
                    bv = nv;
                    best = ni;
                  } else if (nv == bv && ni > best)
                    best = ni;
                }
            if (best == walker)
              break;
            walker = best;
          }
          flowLab(i0) = seg(walker);  // seg at a pore peak == its pore label id
        });
    space.fence();
  }

  // throat list: adjacent flow-basin pairs with OPEN interface faces (o > 0)
  {
    const I3 r = res;
    const bool ho = hasOpen;
    const std::int64_t maxp = static_cast<std::int64_t>(n) * 3;
    Kokkos::View<std::int64_t*, Mem> praw("nf::praw", static_cast<std::size_t>(maxp));
    Kokkos::View<int, Mem> pcnt("nf::pcnt");
    Kokkos::deep_copy(pcnt, 0);
    using MD = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "nf::throat_pairs", MD(space, {0, 0, 0}, {r.x, r.y, r.z}),
        KOKKOS_LAMBDA(int ixx, int iyy, int izz) {
          const int c = get_idx(ixx, iyy, izz, r);
          const int a = flowLab(c);
          for (int d = 0; d < 3; ++d) {
            const int nb = get_idx(ixx + (d == 0), iyy + (d == 1), izz + (d == 2), r);
            const int b = flowLab(nb);
            if (a == b || a <= 0 || b <= 0)
              continue;
            const double opn = ho ? (d == 0 ? ox(nb) : (d == 1 ? oy(nb) : oz(nb))) : 1.0;
            if (opn <= 0.0)
              continue;
            const int lo = a < b ? a : b, hi = a < b ? b : a;
            const int s0 = Kokkos::atomic_fetch_add(&pcnt(), 1);
            if (s0 < maxp)
              praw(s0) = (static_cast<std::int64_t>(lo) << 32) | hi;
          }
        });
    space.fence();
    auto hc = Kokkos::create_mirror_view(pcnt);
    Kokkos::deep_copy(hc, pcnt);
    auto keys0 = downloadN(praw, static_cast<std::size_t>(std::min<std::int64_t>(hc(), maxp)));
    std::sort(keys0.begin(), keys0.end());
    keys0.erase(std::unique(keys0.begin(), keys0.end()), keys0.end());
    for (auto k : keys0)
      out.throats.push_back({static_cast<int>(k >> 32), static_cast<int>(k & 0x7fffffff)});
  }
  const std::size_t nt = out.throats.size();
  std::vector<std::int64_t> keys(nt);
  for (std::size_t t = 0; t < nt; ++t)
    keys[t] = (static_cast<std::int64_t>(out.throats[t].first) << 32) | out.throats[t].second;
  Kokkos::View<std::int64_t*, Mem> keyD("nf::keys", nt);
  uploadVec(keys, keyD);

  // accumulate: openness-weighted MAC face fluxes over every flow-basin boundary face, plus the
  // area-weighted throat centroid (min-imaged relative to the lower pore's center).
  Kokkos::View<double*, Mem> Q("nf::Q", nt), A("nf::A", nt), resid("nf::resid", np);
  Kokkos::View<double*, Mem> Cx("nf::Cx", nt), Cy("nf::Cy", nt), Cz("nf::Cz", nt);
  {
    const double Ax = double(spacing[1]) * spacing[2], Ay = double(spacing[0]) * spacing[2],
                 Az = double(spacing[0]) * spacing[1];
    const I3 r = res;
    const std::int64_t ntl = static_cast<std::int64_t>(nt);
    const bool ho = hasOpen;
    const float oxo = origin[0], oyo = origin[1], ozo = origin[2];
    const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
    const double Lx = double(r.x) * sx, Ly = double(r.y) * sy, Lz = double(r.z) * sz;
    // Min-image anchor per throat: the LOWER pore's PEAK VOXEL center — integer coordinates, so
    // the periodic-image branch is deterministic (the refined float centroid wobbles ~1e-8 under
    // CUDA FMA contraction, which flips the image of faces exactly half a period away —
    // measured on a symmetric sphere lattice).
    auto peakl = peak;
    using MD = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "nf::throat_flux", MD(space, {0, 0, 0}, {r.x, r.y, r.z}),
        KOKKOS_LAMBDA(int ixx, int iyy, int izz) {
          const int c = get_idx(ixx, iyy, izz, r);
          const int a = flowLab(c);
          for (int d = 0; d < 3; ++d) {
            const int nb = get_idx(ixx + (d == 0), iyy + (d == 1), izz + (d == 2), r);
            const int b = flowLab(nb);
            if (a == b)
              continue;
            // the shared face is the -d face of the +d neighbour: velocity/openness live there
            const double vel = d == 0 ? u(nb) : (d == 1 ? v(nb) : w(nb));
            const double opn = ho ? (d == 0 ? ox(nb) : (d == 1 ? oy(nb) : oz(nb))) : 1.0;
            const double area = d == 0 ? Ax : (d == 1 ? Ay : Az);
            const double q = opn * vel * area;  // positive = flow from c to the +d neighbour
            if (a > 0)
              Kokkos::atomic_add(&resid(a - 1), q);
            if (b > 0)
              Kokkos::atomic_add(&resid(b - 1), -q);
            if (a > 0 && b > 0 && opn > 0.0) {
              const int lo = a < b ? a : b, hi = a < b ? b : a;
              const std::int64_t key = (static_cast<std::int64_t>(lo) << 32) | hi;
              // binary search the sorted throat keys
              std::int64_t l = 0, hgh = ntl - 1, slot = -1;
              while (l <= hgh) {
                const std::int64_t mid = l + (hgh - l) / 2;
                if (keyD(mid) == key) {
                  slot = mid;
                  break;
                }
                if (keyD(mid) < key)
                  l = mid + 1;
                else
                  hgh = mid - 1;
              }
              if (slot >= 0) {
                const double w0 = opn * area;
                Kokkos::atomic_add(&Q(slot), a < b ? q : -q);
                Kokkos::atomic_add(&A(slot), w0);
                // face center, min-imaged relative to the lower pore's peak voxel center
                const int pk = peakl(lo - 1);
                const int pkx = pk % r.x, pky = (pk / r.x) % r.y, pkz = pk / (r.x * r.y);
                double fp[3] = {oxo + (ixx + (d == 0 ? 0.5 : 0.0)) * double(sx),
                                oyo + (iyy + (d == 1 ? 0.5 : 0.0)) * double(sy),
                                ozo + (izz + (d == 2 ? 0.5 : 0.0)) * double(sz)};
                const double pc[3] = {oxo + pkx * double(sx), oyo + pky * double(sy),
                                      ozo + pkz * double(sz)};
                const double Lw[3] = {Lx, Ly, Lz};
                for (int a2 = 0; a2 < 3; ++a2) {
                  double dv = fp[a2] - pc[a2];
                  dv -= Lw[a2] * Kokkos::round(dv / Lw[a2]);
                  fp[a2] = dv;
                }
                Kokkos::atomic_add(&Cx(slot), w0 * fp[0]);
                Kokkos::atomic_add(&Cy(slot), w0 * fp[1]);
                Kokkos::atomic_add(&Cz(slot), w0 * fp[2]);
              }
            }
          }
        });
    space.fence();
  }
  std::vector<double> hcx, hcy, hcz;
  {
    auto hq = downloadN(Q, nt);
    auto ha = downloadN(A, nt);
    auto hr = downloadN(resid, static_cast<std::size_t>(np));
    hcx = downloadN(Cx, nt);
    hcy = downloadN(Cy, nt);
    hcz = downloadN(Cz, nt);
    out.throat_flow.assign(hq.begin(), hq.end());
    out.throat_area.assign(ha.begin(), ha.end());
    out.pore_residual.assign(hr.begin(), hr.end());
  }

  // total-pressure drop per throat: periodic parts + the macroscopic gradient along the throat-
  // anchored two-leg min-image path i -> throat centroid -> j. (The Voronoi original used a
  // single min-image between the pore centers; anchoring at the throat disambiguates pores at
  // exactly half the period and follows the physical path of THIS interface.)
  out.throat_dp.resize(nt);
  const double L[3] = {double(res.x) * spacing[0], double(res.y) * spacing[1],
                       double(res.z) * spacing[2]};
  const auto peakH = downloadN(peak, static_cast<std::size_t>(np));
  for (std::size_t t = 0; t < nt; ++t) {
    const int li = out.throats[t].first, lj = out.throats[t].second;
    const Pore& pi = out.pores[li - 1];
    const Pore& pj = out.pores[lj - 1];
    const double aw = out.throat_area[t];
    const int pk = peakH[li - 1];
    const double anc[3] = {origin[0] + (pk % res.x) * double(spacing[0]),
                           origin[1] + ((pk / res.x) % res.y) * double(spacing[1]),
                           origin[2] + (pk / (res.x * res.y)) * double(spacing[2])};
    const int pkj = peakH[lj - 1];
    const double ancj[3] = {origin[0] + (pkj % res.x) * double(spacing[0]),
                            origin[1] + ((pkj / res.x) % res.y) * double(spacing[1]),
                            origin[2] + (pkj / (res.x * res.y)) * double(spacing[2])};
    double macro = 0.0;
    const double pip[3] = {double(pi.x), double(pi.y), double(pi.z)};
    const double pjp[3] = {double(pj.x), double(pj.y), double(pj.z)};
    const int N[3] = {res.x, res.y, res.z};
    for (int a = 0; a < 3; ++a) {
      // Two-leg path i -> throat -> j; every position term cancels except the integer periodic
      // image count k, which is decided in SNAPPED integer-cell arithmetic (peak-voxel anchors +
      // the bias-rounded throat centroid) so float wobble (refined centroids ~1e-8, atomic sum
      // order ~1e-13) can never flip the image branch.
      const double ct = (hcx[t] * (a == 0) + hcy[t] * (a == 1) + hcz[t] * (a == 2)) / aw;
      const long long Dc = llround((ancj[a] - anc[a]) / double(spacing[a])) -
                           llround(ct / double(spacing[a]) + 1e-6);
      const long long k = llround(double(Dc) / N[a]);
      macro += grad_p[a] * ((pjp[a] - pip[a]) - L[a] * double(k));
    }
    out.throat_dp[t] = (out.pore_pressure[li - 1] - out.pore_pressure[lj - 1]) - macro;
  }
  return out;
}

/// The full pore network from one extraction: pores, the per-voxel segmentation (flat), and the
/// label-adjacency connections.
struct PoreNetwork {
  std::vector<Pore> pores;
  std::vector<int> seg;
  std::vector<std::pair<int, int>> connections;
};

// ---- fused pipeline (F1): upload the SDF ONCE, keep it + the segmentation device-resident across
// all three stages (extract_pores → segment_volume → extract_topology), so neither the SDF nor seg
// is re-uploaded or round-tripped between stages. Only the final results cross back to the host.
// Each stage's result is identical to calling the three functions separately. ----
inline PoreNetwork extract_pore_network_k(const std::vector<float>& sdf_h,
                                          std::array<int, 3> resolution,
                                          std::array<float, 3> origin,
                                          std::array<float, 3> spacing) {
  PoreNetwork out;
  if (sdf_h.empty())
    return out;
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
