/// @file
/// @brief peclet.pnm — the ONE set of stage kernels shared by the single-rank
/// (pore_extraction.hpp) and the distributed (pore_extraction_mpi.hpp) pipelines.
///
/// Every stage kernel is templated on a small GEOMETRY POLICY that maps the owned box a rank
/// iterates over onto the field it reads (the whole periodic grid for a single rank, a
/// ghost-extended block for an MPI rank) and onto GLOBAL voxel ids (the tie-break identity that
/// makes every fixpoint decomposition-independent):
///
///   nx, ny, nz              owned box (the cells this rank iterates over and writes)
///   g                       offset of owned (0,0,0) in field coordinates (0 single-rank, the
///                           ghost width for a block)
///   cell(lx,ly,lz)          field index of FIELD coords (periodic wrap single-rank; plain for a
///                           block — ghosts hold the wrapped neighbours)
///   gid(lx,ly,lz)           global voxel id of field coords (== cell single-rank)
///   gcoord(lx,ly,lz,...)    wrapped global integer coords (the float positions come from these)
///   inOwned(ix,iy,iz)       owned coords inside the owned box? (always single-rank: a periodic
///                           neighbour IS owned; a block routes cross-block adjacency through its
///                           boundary graph instead)
///   owned(ix,iy,iz)         dense owned-box index (== cell single-rank)
///   ownedCoords(o,...)      the inverse of owned()
///   localOf(gid)            field index of a global id or -1 when it has no image in the field
///   cellAtGlobal(gx,gy,gz)  field index of (wrapped) global coords
///
/// `GridGeo` is the single-rank policy (trivial: everything is the grid itself); `BlockGeo` the
/// extended-block policy of the MPI pipeline. The MPI file adds only the halo exchanges, the
/// ownership/merge orchestration and the distributed reductions around these kernels.
///
/// Arithmetic is IDENTICAL between the two instantiations by construction (same expression, same
/// evaluation order) — that is what keeps the multi-rank results bit-exact to single-rank.
#ifndef PECLET_PNM_PORE_KERNELS_HPP
#define PECLET_PNM_PORE_KERNELS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <Kokkos_Core.hpp>
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
/// Global voxel / face id (== peclet::core::Index; asserted where core is included).
using Index = std::int64_t;

/// Bulk host->device upload of a whole std::vector via one deep_copy over an unmanaged host view.
/// `d` must already be sized to `h`.
template <class T>
inline void uploadVec(const std::vector<T>& h, const Kokkos::View<T*, Mem>& d) {
  if (h.empty())
    return;
  Kokkos::deep_copy(
      d, Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
             h.data(), h.size()));
}

/// Download the first `count` elements of a device view into a fresh std::vector via one
/// deep_copy — only moves what is used.
template <class V>
inline std::vector<typename V::value_type> downloadN(const V& d, std::size_t count) {
  std::vector<typename V::value_type> out(count);
  if (count)
    Kokkos::deep_copy(Kokkos::View<typename V::value_type*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(out.data(), count),
                      Kokkos::subview(d, Kokkos::make_pair(std::size_t(0), count)));
  return out;
}

/// Read a device scalar (the `changed` / counter flags) back to the host.
template <class V>
inline typename V::value_type readScalar(const V& d) {
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);
  return h();
}

KOKKOS_INLINE_FUNCTION int get_idx(int x, int y, int z, I3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

KOKKOS_INLINE_FUNCTION int wrapc(int a, int n) {
  const int r = a % n;
  return r < 0 ? r + n : r;
}

/// Single-rank geometry policy: the field IS the whole periodic grid, owned == field coords,
/// gid == index. All members plain ints so the POD captures by value into lambdas.
struct GridGeo {
  int nx, ny, nz;
  int g;  // 0

  static GridGeo of(I3 res) { return GridGeo{res.x, res.y, res.z, 0}; }
  KOKKOS_INLINE_FUNCTION Index cell(int lx, int ly, int lz) const {
    return (Index(wrapc(lz, nz)) * ny + wrapc(ly, ny)) * nx + wrapc(lx, nx);
  }
  KOKKOS_INLINE_FUNCTION Index gid(int lx, int ly, int lz) const { return cell(lx, ly, lz); }
  KOKKOS_INLINE_FUNCTION void gcoord(int lx, int ly, int lz, int& gx, int& gy, int& gz) const {
    gx = wrapc(lx, nx);
    gy = wrapc(ly, ny);
    gz = wrapc(lz, nz);
  }
  KOKKOS_INLINE_FUNCTION bool inOwned(int, int, int) const { return true; }
  KOKKOS_INLINE_FUNCTION Index owned(int ix, int iy, int iz) const { return cell(ix, iy, iz); }
  KOKKOS_INLINE_FUNCTION void ownedCoords(Index o, int& ix, int& iy, int& iz) const {
    ix = int(o % nx);
    iy = int((o / nx) % ny);
    iz = int(o / (Index(nx) * ny));
  }
  KOKKOS_INLINE_FUNCTION Index localOf(Index gid) const { return gid; }
  KOKKOS_INLINE_FUNCTION Index cellAtGlobal(int gx, int gy, int gz) const {
    return cell(gx, gy, gz);
  }
  std::size_t numOwned() const { return std::size_t(nx) * ny * nz; }
};

/// Extended-block geometry policy (the MPI pipeline): ghost-inclusive local box <-> global
/// periodic grid.
struct BlockGeo {
  int ex, ey, ez;     // extended sizes (inner + 2g)
  int ox, oy, oz;     // global origin of the extended box (originInclGhost; may be negative)
  int nx, ny, nz;     // inner (owned) sizes
  int gnx, gny, gnz;  // global dims
  int g;              // ghost width

  static KOKKOS_INLINE_FUNCTION int wrapc(int a, int n) { return pnm::wrapc(a, n); }
  KOKKOS_INLINE_FUNCTION Index lidx(int lx, int ly, int lz) const {
    return (Index(lz) * ey + ly) * ex + lx;
  }
  KOKKOS_INLINE_FUNCTION Index cell(int lx, int ly, int lz) const { return lidx(lx, ly, lz); }
  /// gid of the (periodically wrapped) global cell under extended-local coords.
  KOKKOS_INLINE_FUNCTION Index gidAt(int lx, int ly, int lz) const {
    const int gx = wrapc(ox + lx, gnx), gy = wrapc(oy + ly, gny), gz = wrapc(oz + lz, gnz);
    return (Index(gz) * gny + gy) * gnx + gx;
  }
  KOKKOS_INLINE_FUNCTION Index gid(int lx, int ly, int lz) const { return gidAt(lx, ly, lz); }
  KOKKOS_INLINE_FUNCTION void gcoord(int lx, int ly, int lz, int& gx, int& gy, int& gz) const {
    gx = wrapc(ox + lx, gnx);
    gy = wrapc(oy + ly, gny);
    gz = wrapc(oz + lz, gnz);
  }
  KOKKOS_INLINE_FUNCTION bool inOwned(int ix, int iy, int iz) const {
    return ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz;
  }
  KOKKOS_INLINE_FUNCTION Index owned(int ix, int iy, int iz) const {
    return (Index(iz) * ny + iy) * nx + ix;
  }
  KOKKOS_INLINE_FUNCTION void ownedCoords(Index o, int& ix, int& iy, int& iz) const {
    ix = int(o % nx);
    iy = int((o / nx) % ny);
    iz = int(o / (Index(nx) * ny));
  }
  /// gid -> extended-local linear index if the cell has an image in this extended box, else -1.
  /// (All images of a cell carry identical values after an exchange — selfCopy covers periodic
  /// self-images — so any image is good for reading.)
  KOKKOS_INLINE_FUNCTION Index localOf(Index gid) const {
    const int gx = int(gid % gnx);
    const Index t = gid / gnx;
    const int gy = int(t % gny), gz = int(t / gny);
    const int lx = wrapc(gx - ox, gnx);
    if (lx >= ex)
      return Index(-1);
    const int ly = wrapc(gy - oy, gny);
    if (ly >= ey)
      return Index(-1);
    const int lz = wrapc(gz - oz, gnz);
    if (lz >= ez)
      return Index(-1);
    return lidx(lx, ly, lz);
  }
  KOKKOS_INLINE_FUNCTION Index cellAtGlobal(int gx, int gy, int gz) const {
    return localOf((Index(wrapc(gz, gnz)) * gny + wrapc(gy, gny)) * gnx + wrapc(gx, gnx));
  }
  /// Is the owned cell within one cell of a block face (the only cells that can see a ghost)?
  KOKKOS_INLINE_FUNCTION bool onSurface(int ix, int iy, int iz) const {
    return !(ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1 && iz > 0 && iz < nz - 1);
  }
  std::size_t numOwned() const { return std::size_t(nx) * ny * nz; }
  std::size_t numExt() const { return std::size_t(ex) * ey * ez; }
};

namespace kernels {

using MD3 = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
using R1 = Kokkos::RangePolicy<Exec>;

/// The owned box of a geometry as an MDRange policy.
template <class Geo>
inline MD3 ownedRange(const Exec& space, const Geo& geo) {
  return MD3(space, {0, 0, 0}, {geo.nx, geo.ny, geo.nz});
}

/// Film-face sentinel of the per-patch throat labelling: "interface face awaiting attachment".
constexpr Index kSent = 0x7ffffffffffffffeLL;

// ---------------------------------------------------------------------------------------------
// device stencil primitives
// ---------------------------------------------------------------------------------------------

/// Local SDF maximum under the global (sdf, gid) lexicographic order (26-neighbourhood).
template <class Geo, class SdfView>
KOKKOS_INLINE_FUNCTION bool isPeak(const Geo& g, const SdfView& sdf, int lx, int ly, int lz,
                                   float cv, Index cgid) {
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        const float nv = sdf(g.cell(lx + dx, ly + dy, lz + dz));
        const Index ngid = g.gid(lx + dx, ly + dy, lz + dz);
        if (nv > cv || (nv == cv && ngid > cgid))
          return false;
      }
  return true;
}

/// The pore record at a peak: sub-voxel position by the SDF^2-weighted centroid of the 3^3
/// neighbourhood (positive part only), in global (wrapped) integer coords, radius = sdf there.
template <class Geo, class SdfView>
KOKKOS_INLINE_FUNCTION Pore poreAt(const Geo& g, const SdfView& sdf, int lx, int ly, int lz,
                                   float ox, float oy, float oz, float sx, float sy, float sz) {
  float sw = 0.0f, px = 0.0f, py = 0.0f, pz = 0.0f;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const float v = sdf(g.cell(lx + dx, ly + dy, lz + dz));
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
  int gx, gy, gz;
  g.gcoord(lx, ly, lz, gx, gy, gz);
  return Pore{ox + (gx + fx) * sx, oy + (gy + fy) * sy, oz + (gz + fz) * sz,
              sdf(g.cell(lx, ly, lz))};
}

/// Steepest 26-neighbour under the (sdf, gid) lexicographic order: ascent (pores) or descent
/// (solids); returns the neighbour's gid, or `cgid` itself at a peak. Ties on value go to the
/// larger gid.
template <class Geo, class SdfView>
KOKKOS_INLINE_FUNCTION Index steepest(const Geo& g, const SdfView& sdf, int lx, int ly, int lz,
                                      Index cgid, bool ascent) {
  Index best = cgid;
  float bv = sdf(g.cell(lx, ly, lz));
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        const Index ni = g.gid(lx + dx, ly + dy, lz + dz);
        const float nv = sdf(g.cell(lx + dx, ly + dy, lz + dz));
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
  return best;
}

/// Trilinear interpolation of a cell-centered periodic field at grid position gp (in cells).
template <class Geo, class FieldView>
KOKKOS_INLINE_FUNCTION double trilinear(const Geo& g, const FieldView& p, const double gp[3]) {
  int b[3];
  double f[3];
  for (int a = 0; a < 3; ++a) {
    const double fl = Kokkos::floor(gp[a]);
    b[a] = static_cast<int>(fl);
    f[a] = gp[a] - fl;
  }
  double acc = 0.0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const double wt =
            (dx ? f[0] : 1.0 - f[0]) * (dy ? f[1] : 1.0 - f[1]) * (dz ? f[2] : 1.0 - f[2]);
        acc += wt * p(g.cellAtGlobal(b[0] + dx, b[1] + dy, b[2] + dz));
      }
  return acc;
}

/// Face adjacency of the per-patch throat CCL: two faces (cell offset (dx,dy,dz), directions
/// d and d2) connect when their doubled center offset satisfies |2*dc + e_d2 - e_d|^2 <= 8 — a
/// generous edge/corner adjacency in pure integer arithmetic (periodic-safe, decomposition-
/// independent).
KOKKOS_INLINE_FUNCTION bool facesAdjacent(int dx, int dy, int dz, int d, int d2) {
  const int Dx = 2 * dx + (d2 == 0) - (d == 0);
  const int Dy = 2 * dy + (d2 == 1) - (d == 1);
  const int Dz = 2 * dz + (d2 == 2) - (d == 2);
  return Dx * Dx + Dy * Dy + Dz * Dz <= 8;
}

/// Device binary search over a sorted key view; returns the position or -1.
template <class KeyView>
KOKKOS_INLINE_FUNCTION Index bsearchKey(const KeyView& keys, Index nkeys,
                                        typename KeyView::non_const_value_type key) {
  Index lo = 0, hi = nkeys - 1;
  while (lo <= hi) {
    const Index mid = lo + (hi - lo) / 2;
    const auto k = keys(mid);
    if (k == key)
      return mid;
    if (k < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return Index(-1);
}

/// Per-face field accessors for the film-attachment sweep: label / pair key of face (cell e, d).
/// Flat: one 3*n array indexed 3*e+d (single-rank); Split: one field per direction (the MPI
/// pipeline exchanges each direction as its own extended field).
template <class V>
struct FlatFaces {
  V v;
  KOKKOS_INLINE_FUNCTION Index operator()(Index e, int d) const { return v(3 * e + d); }
};
template <class V>
struct SplitFaces {
  V v0, v1, v2;
  KOKKOS_INLINE_FUNCTION Index operator()(Index e, int d) const {
    return d == 0 ? v0(e) : (d == 1 ? v1(e) : v2(e));
  }
};

// ---------------------------------------------------------------------------------------------
// pore detection
// ---------------------------------------------------------------------------------------------

/// Emit the owned peaks (local SDF maxima, sdf > 0) as Pore records into `pores` (atomic slot
/// counter `counter`, capacity `maxPores`).
template <class Geo, class SdfView, class PoreView, class CountView>
inline void detectPores(const Exec& space, const Geo& geo, const SdfView& sdf,
                        std::array<float, 3> origin, std::array<float, 3> spacing,
                        const PoreView& pores, const CountView& counter, int maxPores) {
  const Geo g = geo;
  const float ox = origin[0], oy = origin[1], oz = origin[2];
  const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
  Kokkos::parallel_for(
      "pnm::extract_pores", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int lx = ix + g.g, ly = iy + g.g, lz = iz + g.g;
        const float cv = sdf(g.cell(lx, ly, lz));
        if (cv <= 0.0f)
          return;
        if (!isPeak(g, sdf, lx, ly, lz, cv, g.gid(lx, ly, lz)))
          return;
        const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
        if (slot < maxPores)
          pores(slot) = poreAt(g, sdf, lx, ly, lz, ox, oy, oz, sx, sy, sz);
      });
  space.fence();
}

// ---------------------------------------------------------------------------------------------
// marker-controlled watershed: markers -> union-find CCL -> flood
// ---------------------------------------------------------------------------------------------

/// Deep-solid markers: parent(o) = o where sdf < thr, else -1 (owned-dense).
template <class Geo, class SdfView, class ParentView>
inline void initMarkers(const Exec& space, const Geo& geo, const SdfView& sdf, float thr,
                        const ParentView& parent) {
  using L = typename ParentView::non_const_value_type;
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::init_markers", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const Index o = g.owned(ix, iy, iz);
        parent(o) = (sdf(g.cell(ix + g.g, iy + g.g, iz + g.g)) < thr) ? L(o) : L(-1);
      });
  space.fence();
}

/// Path compression: every labelled entry points at its root.
template <class ParentView>
inline void cclFlatten(const Exec& space, const ParentView& parent, std::size_t n) {
  using L = typename ParentView::non_const_value_type;
  Kokkos::parallel_for(
      "pnm::flatten", R1(space, 0, n), KOKKOS_LAMBDA(std::size_t idx) {
        L l = parent(idx);
        if (l >= 0) {
          while (l != parent(l))
            l = parent(l);
          parent(idx) = l;
        }
      });
  space.fence();
}

/// One union sweep of the marker CCL (26-connectivity, 13 forward neighbours, union-by-min via
/// atomic_min on the larger root). Owned-owned adjacency only: a periodic neighbour is owned
/// single-rank; a block's cross-block pairs go through its boundary graph.
template <class Geo, class ParentView, class FlagView>
inline void cclMergeMarkers(const Exec& space, const Geo& geo, const ParentView& parent,
                            const FlagView& changed) {
  using L = typename ParentView::non_const_value_type;
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::merge_markers", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const Index o = g.owned(ix, iy, iz);
        const L my = parent(o);
        if (my == -1)
          return;
        const int dz_l[13] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
        const int dy_l[13] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0};
        const int dx_l[13] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1};
        for (int k = 0; k < 13; ++k) {
          const int jx = ix + dx_l[k], jy = iy + dy_l[k], jz = iz + dz_l[k];
          if (!g.inOwned(jx, jy, jz))
            continue;
          const L nl = parent(g.owned(jx, jy, jz));
          if (nl != -1 && my != nl) {
            L rm = my;
            while (rm != parent(rm))
              rm = parent(rm);
            L rn = nl;
            while (rn != parent(rn))
              rn = parent(rn);
            if (rm != rn) {
              const L small = rm < rn ? rm : rn, large = rm < rn ? rn : rm;
              Kokkos::atomic_min(&parent(large), small);
              changed() = 1;
            }
          }
        }
      });
  space.fence();
}

/// Union-find to fixpoint: `merge()` launches one union sweep (setting `changed`), then a
/// flatten; repeat until a sweep changes nothing.
template <class ParentView, class FlagView, class MergeFn>
inline void cclFixpoint(const Exec& space, const ParentView& parent, std::size_t n,
                        const FlagView& changed, const MergeFn& merge) {
  int h_changed = 1;
  while (h_changed) {
    Kokkos::deep_copy(changed, 0);
    merge();
    cclFlatten(space, parent, n);
    h_changed = readScalar(changed);
  }
}

/// One Jacobi flood sweep: every unlabelled solid voxel takes the smallest labelled 26-neighbour
/// (reads `labels`, writes `labelsN`; the caller swaps). Labels are voxel ids in the field's
/// label type.
template <class Geo, class SdfView, class LabelView, class FlagView>
inline void floodSweep(const Exec& space, const Geo& geo, const SdfView& sdf,
                       const LabelView& labels, const LabelView& labelsN, const FlagView& changed) {
  using L = typename LabelView::non_const_value_type;
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::flood", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int lx = ix + g.g, ly = iy + g.g, lz = iz + g.g;
        const Index e = g.cell(lx, ly, lz);
        if (sdf(e) >= 0.0f)
          return;  // pore: ignore
        if (labels(e) != -1)
          return;  // already labelled
        L best = -1;
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0)
                continue;
              const L nl = labels(g.cell(lx + dx, ly + dy, lz + dz));
              if (nl != -1 && (best == -1 || nl < best))
                best = nl;
            }
        if (best != -1) {
          labelsN(e) = best;
          changed() = 1;
        }
      });
  space.fence();
}

/// Jacobi flood to fixpoint. Double-buffered: each sweep reads only the previous sweep's labels,
/// so the result is DETERMINISTIC (an in-place sweep could observe same-sweep writes — a device
/// race) and sweep-for-sweep identical between the single-rank and the distributed pipeline.
/// `sync(localChanged) -> globalChanged` lets a block Allreduce the flag and exchange its ghosts
/// between sweeps (identity single-rank). `labels` holds the result on return (the views are
/// swapped per sweep).
template <class Geo, class SdfView, class LabelView, class FlagView, class SyncFn>
inline void floodFixpoint(const Exec& space, const Geo& geo, const SdfView& sdf, LabelView& labels,
                          LabelView& labelsN, const FlagView& changed, const SyncFn& sync) {
  int h_changed = 1;
  while (h_changed) {
    Kokkos::deep_copy(changed, 0);
    Kokkos::deep_copy(labelsN, labels);
    floodSweep(space, geo, sdf, labels, labelsN, changed);
    std::swap(labels, labelsN);
    h_changed = sync(readScalar(changed));
  }
}

// ---------------------------------------------------------------------------------------------
// gradient-path basins
// ---------------------------------------------------------------------------------------------

/// Single-rank gradient walk: from every owned cell follow the steepest neighbour (ascent for
/// pores, descent for solids; `ascentAll` = ascent from every cell, the flow-basin variant) for
/// at most 512 steps; `emit(o, walker)` receives the end point (a gid == field index here).
/// GridGeo only — the walk itself does not distribute (its RESULT is the forest root that
/// forestInit + resolveHoldAtGhost compute on a block).
template <class SdfView, class EmitFn>
inline void gradientWalk(const Exec& space, const GridGeo& geo, const SdfView& sdf, bool ascentAll,
                         const EmitFn& emit) {
  const GridGeo g = geo;
  Kokkos::parallel_for(
      "pnm::gradient_path", R1(space, 0, g.numOwned()), KOKKOS_LAMBDA(std::size_t i0) {
        const Index ci = Index(i0);
        const bool ascent = ascentAll || (sdf(ci) > 0.0f);
        Index walker = ci;
        const int MAX_STEPS = 512;
        for (int s = 0; s < MAX_STEPS; ++s) {
          int wx, wy, wz;
          g.ownedCoords(walker, wx, wy, wz);
          const Index best = steepest(g, sdf, wx, wy, wz, walker, ascent);
          if (best == walker)
            break;
          walker = best;
        }
        emit(ci, walker);
      });
  space.fence();
}

/// Steepest-neighbour forest for the hold-at-ghost root resolution. `target(e)` per owned cell:
/// cells with `chase(e)` false finalize immediately as `finalOf(e, gid)`; chasing cells store the
/// gid of their steepest ascent neighbour, or `peakOf(e, gid)` when they are a peak. (Negative
/// values are the finalized encoding; a non-negative value is a pending gid.)
template <class Geo, class SdfView, class TargetView, class ChaseFn, class FinalFn, class PeakFn>
inline void forestInit(const Exec& space, const Geo& geo, const SdfView& sdf,
                       const TargetView& target, const ChaseFn& chase, const FinalFn& finalOf,
                       const PeakFn& peakOf) {
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::grad_step", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int lx = ix + g.g, ly = iy + g.g, lz = iz + g.g;
        const Index ci = g.cell(lx, ly, lz);
        const Index cgid = g.gid(lx, ly, lz);
        if (!chase(ci)) {
          target(ci) = finalOf(ci, cgid);
          return;
        }
        const Index best = steepest(g, sdf, lx, ly, lz, cgid, true);
        target(ci) = (best == cgid) ? peakOf(ci, cgid) : best;
      });
  space.fence();
}

/// One round of pointer jumping with the hold-at-ghost invariant: chase within the field, adopt a
/// finalized (negative) value, or HOLD at the last in-field cell when the next hop leaves the
/// field (never store a gid the field cannot reach — that strands the chase outside the ghost
/// ring). Returns the number of owned cells still pending.
template <class Geo, class TargetView>
inline int resolveHoldAtGhost(const Exec& space, const Geo& geo, const TargetView& target) {
  const Geo g = geo;
  int pending = 0;
  Kokkos::parallel_reduce(
      "pnm::resolve_roots", R1(space, 0, g.numOwned()),
      KOKKOS_LAMBDA(std::size_t o, int& pend) {
        int ix, iy, iz;
        g.ownedCoords(Index(o), ix, iy, iz);
        const Index ci = g.cell(ix + g.g, iy + g.g, iz + g.g);
        Index t = target(ci);
        if (t < 0)
          return;  // finalized
        for (int s = 0; s < 64; ++s) {
          const Index tl = g.localOf(t);  // in-field by invariant
          if (tl < 0)
            break;
          const Index tv = target(tl);
          if (tv < 0) {  // that cell knows its root: adopt, finalized
            t = tv;
            break;
          }
          const Index nl = g.localOf(tv);
          if (nl < 0)
            break;  // next hop leaves the field: hold at tl, wait for its owner
          t = tv;   // advance within the field
        }
        if (t != target(ci))
          target(ci) = t;
        if (t >= 0)
          pend += 1;
      },
      pending);
  space.fence();
  return pending;
}

// ---------------------------------------------------------------------------------------------
// topology
// ---------------------------------------------------------------------------------------------

/// Boundary pairs: for every owned cell, the (min,max) label pairs across its +x/+y/+z faces
/// where the label changes, appended to the interleaved `pairs` (atomic `cnt`, capacity
/// `maxPairs`).
template <class Geo, class SegView, class PairView, class CountView>
inline void boundaryPairs(const Exec& space, const Geo& geo, const SegView& seg,
                          const PairView& pairs, const CountView& cnt, Index maxPairs) {
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::boundary_pairs", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int lx = ix + g.g, ly = iy + g.g, lz = iz + g.g;
        const int my = seg(g.cell(lx, ly, lz));
        const int dx_l[3] = {1, 0, 0}, dy_l[3] = {0, 1, 0}, dz_l[3] = {0, 0, 1};
        for (int k = 0; k < 3; ++k) {
          const int nl = seg(g.cell(lx + dx_l[k], ly + dy_l[k], lz + dz_l[k]));
          if (my != nl) {
            const int slot = Kokkos::atomic_fetch_add(&cnt(), 1);
            if (slot < maxPairs) {
              pairs(2 * slot) = my < nl ? my : nl;
              pairs(2 * slot + 1) = my < nl ? nl : my;
            }
          }
        }
      });
  space.fence();
}

/// Interleaved device pairs -> host (l1,l2) list, sorted and unique.
template <class PairView>
inline std::vector<std::pair<int, int>> uniquePairs(const PairView& pairs, std::size_t count) {
  std::vector<int> flat = downloadN(pairs, 2 * count);
  std::vector<std::pair<int, int>> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = {flat[2 * i], flat[2 * i + 1]};
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

/// The largest label over the owned box (= the number of pore labels).
template <class Geo, class SegView>
inline int maxLabel(const Exec& space, const Geo& geo, const SegView& seg) {
  const Geo g = geo;
  int np = 0;
  Kokkos::parallel_reduce(
      "pnm::max_label", R1(space, 0, g.numOwned()),
      KOKKOS_LAMBDA(std::size_t o, int& m) {
        int ix, iy, iz;
        g.ownedCoords(Index(o), ix, iy, iz);
        const int sv = seg(g.cell(ix + g.g, iy + g.g, iz + g.g));
        if (sv > m)
          m = sv;
      },
      Kokkos::Max<int>(np));
  space.fence();
  return np;
}

// ---------------------------------------------------------------------------------------------
// network flow: per-patch throats over the interface FACES (fid = 3*cell + d, the +d face)
// ---------------------------------------------------------------------------------------------

/// Interface predicate + CORE tier per owned face: an interface face separates two different
/// positive flow-basin labels with positive openness; CORE = both cells fluid-centered (sdf > 0),
/// which defines the patches by CCL; the rest are FILM faces that attach later. Writes (owned-
/// dense 3*o+d) `core`, `parent` (= f for core faces, -1 otherwise) and `fpair`
/// ((lo<<32|hi) pair key, -1 for a non-interface face).
template <class Geo, class LabView, class SdfView, class OpenView, class CoreView, class ParentView,
          class PairView>
inline void faceInit(const Exec& space, const Geo& geo, const LabView& flowLab, const SdfView& sdf,
                     const OpenView& ox, const OpenView& oy, const OpenView& oz, bool hasOpen,
                     const CoreView& core, const ParentView& parent, const PairView& fpair) {
  using L = typename ParentView::non_const_value_type;
  const Geo g = geo;
  const bool ho = hasOpen;
  Kokkos::parallel_for(
      "pnm::face_init", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int lx = ix + g.g, ly = iy + g.g, lz = iz + g.g;
        const Index o = g.owned(ix, iy, iz);
        const Index ce = g.cell(lx, ly, lz);
        const int a = flowLab(ce);
        for (int d = 0; d < 3; ++d) {
          const Index f = 3 * o + d;
          const Index nb = g.cell(lx + (d == 0), ly + (d == 1), lz + (d == 2));
          const int b = flowLab(nb);
          const double opn = ho ? (d == 0 ? ox(nb) : (d == 1 ? oy(nb) : oz(nb))) : 1.0;
          const bool itf = (a != b && a > 0 && b > 0 && opn > 0.0);
          core(f) = (itf && sdf(ce) > 0.0f && sdf(nb) > 0.0f) ? 1 : 0;
          parent(f) = core(f) ? L(f) : L(-1);
          fpair(f) = itf ? ((Index(a < b ? a : b) << 32) | (a < b ? b : a)) : Index(-1);
        }
      });
  space.fence();
}

/// One union sweep of the face CCL over the faces with parent >= 0 (same pair key + adjacency;
/// owned-owned adjacency only, see cclMergeMarkers).
template <class Geo, class ParentView, class PairView, class FlagView>
inline void faceMerge(const Exec& space, const Geo& geo, const ParentView& parent,
                      const PairView& fpair, const FlagView& changed) {
  using L = typename ParentView::non_const_value_type;
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::face_merge", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const Index o = g.owned(ix, iy, iz);
        for (int d = 0; d < 3; ++d) {
          const Index f = 3 * o + d;
          if (parent(f) < 0)
            continue;
          const Index pk = fpair(f);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const int jx = ix + dx, jy = iy + dy, jz = iz + dz;
                if (!g.inOwned(jx, jy, jz))
                  continue;
                const Index o2 = g.owned(jx, jy, jz);
                for (int d2 = 0; d2 < 3; ++d2) {
                  const Index f2 = 3 * o2 + d2;
                  if (f2 == f || parent(f2) < 0 || fpair(f2) != pk)
                    continue;
                  if (!facesAdjacent(dx, dy, dz, d, d2))
                    continue;
                  L rm = L(f);
                  while (rm != parent(rm))
                    rm = parent(rm);
                  L rn = L(f2);
                  while (rn != parent(rn))
                    rn = parent(rn);
                  if (rm != rn) {
                    const L sml = rm < rn ? rm : rn, lrg = rm < rn ? rn : rm;
                    Kokkos::atomic_min(&parent(lrg), sml);
                    changed() = 1;
                  }
                }
              }
        }
      });
  space.fence();
}

/// Per-component minimum GLOBAL face id (3*gid+d) of the local face CCL, keyed by the component
/// root: the decomposition-independent patch label. `mg` must be pre-filled with a large value.
template <class Geo, class ParentView, class MinView>
inline void faceMinGid(const Exec& space, const Geo& geo, const ParentView& parent,
                       const MinView& mg) {
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::face_mingid", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const Index o = g.owned(ix, iy, iz);
        const Index gid = g.gid(ix + g.g, iy + g.g, iz + g.g);
        for (int d = 0; d < 3; ++d)
          if (parent(3 * o + d) >= 0)
            Kokkos::atomic_min(&mg(parent(3 * o + d)), 3 * gid + d);
      });
  space.fence();
}

/// Core-tier labels: faceLab = rootOf(parent) for the CCL'd faces, kSent for the remaining
/// interface (film) faces, -1 for non-interface faces. `rootOf(parentValue)` maps a component
/// root to its patch label (identity single-rank; the min global fid of faceMinGid on a block).
template <class ParentView, class PairView, class LabView, class RootFn>
inline void faceLabelCore(const Exec& space, std::size_t nf, const ParentView& parent,
                          const PairView& fpair, const LabView& faceLab, const RootFn& rootOf) {
  Kokkos::parallel_for(
      "pnm::face_label", R1(space, 0, nf), KOKKOS_LAMBDA(std::size_t f) {
        faceLab(f) = parent(f) >= 0 ? rootOf(parent(f)) : (fpair(f) >= 0 ? kSent : Index(-1));
      });
  space.fence();
}

/// Leftover-tier labels: the still-kSent films (their own CCL tier) get rootOf(parent).
template <class ParentView, class LabView, class RootFn>
inline void faceLabelLeftover(const Exec& space, std::size_t nf, const ParentView& parent,
                              const LabView& faceLab, const RootFn& rootOf) {
  Kokkos::parallel_for(
      "pnm::leftover_label", R1(space, 0, nf), KOKKOS_LAMBDA(std::size_t f) {
        if (faceLab(f) == kSent)
          faceLab(f) = rootOf(parent(f));
      });
  space.fence();
}

/// Leftover tier init: parent = f for the films no core patch reached, -1 otherwise.
template <class ParentView, class LabView>
inline void leftoverInit(const Exec& space, std::size_t nf, const LabView& faceLab,
                         const ParentView& parent) {
  using L = typename ParentView::non_const_value_type;
  Kokkos::parallel_for(
      "pnm::leftover_init", R1(space, 0, nf),
      KOKKOS_LAMBDA(std::size_t f) { parent(f) = (faceLab(f) == kSent) ? L(f) : L(-1); });
  space.fence();
}

/// One Jacobi min-propagation sweep of the film attachment: every film face (interface, not
/// core) takes the minimum attached label among its adjacent same-pair faces (`nbLab(e, d)` /
/// `nbPair(e, d)` read the neighbour faces — flat single-rank, per-direction exchanged fields on
/// a block). A film can never bridge two core patches: it only ever adopts, never unions.
template <class Geo, class LabView, class PairView, class CoreView, class NbLab, class NbPair,
          class FlagView>
inline void filmAttachSweep(const Exec& space, const Geo& geo, const LabView& faceLab,
                            const PairView& fpair, const CoreView& core, const NbLab& nbLab,
                            const NbPair& nbPair, const FlagView& changed) {
  const Geo g = geo;
  Kokkos::parallel_for(
      "pnm::film_attach", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const Index o = g.owned(ix, iy, iz);
        for (int d = 0; d < 3; ++d) {
          const Index f = 3 * o + d;
          if (core(f) || faceLab(f) < 0)
            continue;
          Index best = faceLab(f);
          const Index pk = fpair(f);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const Index e2 = g.cell(ix + g.g + dx, iy + g.g + dy, iz + g.g + dz);
                for (int d2 = 0; d2 < 3; ++d2) {
                  const Index l2 = nbLab(e2, d2);
                  if (l2 < 0 || l2 >= kSent || l2 >= best)
                    continue;
                  if (nbPair(e2, d2) != pk)
                    continue;
                  if (!facesAdjacent(dx, dy, dz, d, d2))
                    continue;
                  best = l2;
                }
              }
          if (best < faceLab(f)) {
            faceLab(f) = best;
            changed() = 1;
          }
        }
      });
  space.fence();
}

/// Global unique (patch root, pair key) list -> the throat list ordered by (pair, root fid), the
/// sorted root keys and the (key position -> throat slot) map for the device lookup.
struct ThroatSlots {
  std::vector<std::pair<int, int>> throats;  // (li, lj) per slot
  std::vector<Index> keySorted;              // patch roots, ascending
  std::vector<int> slotOf;                   // keySorted position -> throat slot
};
inline ThroatSlots throatSlots(std::vector<std::pair<Index, Index>> rootPair) {
  std::sort(rootPair.begin(), rootPair.end(),
            [](const std::pair<Index, Index>& x, const std::pair<Index, Index>& y) {
              return x.second != y.second ? x.second < y.second : x.first < y.first;
            });
  ThroatSlots ts;
  std::vector<Index> rootF;
  for (const auto& rp : rootPair) {
    ts.throats.push_back({int(rp.second >> 32), int(rp.second & 0x7fffffff)});
    rootF.push_back(rp.first);
  }
  ts.keySorted = rootF;
  std::sort(ts.keySorted.begin(), ts.keySorted.end());
  ts.slotOf.resize(rootF.size());
  for (std::size_t t = 0; t < rootF.size(); ++t) {
    const auto it = std::lower_bound(ts.keySorted.begin(), ts.keySorted.end(), rootF[t]);
    ts.slotOf[std::size_t(it - ts.keySorted.begin())] = int(t);
  }
  return ts;
}

/// Accumulate over every owned +face where the flow-basin label changes: the openness-weighted
/// MAC flux o·u·A into the per-pore residual (signed) and, for interface faces of a throat patch,
/// into the throat's flow (positive lower -> higher label), open area and area-weighted centroid
/// (min-imaged relative to the LOWER pore's peak-voxel anchor `anc{X,Y,Z}` — integer-derived, so
/// the periodic-image branch is deterministic under FMA wobble).
template <class Geo, class LabView, class FieldView, class OpenView, class FaceLabView,
          class KeyView, class SlotView, class AncView, class AccView>
inline void throatFlux(const Exec& space, const Geo& geo, const LabView& flowLab,
                       const FieldView& u, const FieldView& v, const FieldView& w,
                       const OpenView& ox, const OpenView& oy, const OpenView& oz, bool hasOpen,
                       const FaceLabView& faceLab, const KeyView& keyD, const SlotView& slotD,
                       std::size_t nt, const AncView& ancX, const AncView& ancY,
                       const AncView& ancZ, std::array<float, 3> origin,
                       std::array<float, 3> spacing, std::array<int, 3> gdims, const AccView& Q,
                       const AccView& A, const AccView& Cx, const AccView& Cy, const AccView& Cz,
                       const AccView& resid) {
  const Geo g = geo;
  const double Ax = double(spacing[1]) * spacing[2], Ay = double(spacing[0]) * spacing[2],
               Az = double(spacing[0]) * spacing[1];
  const std::int64_t ntl = static_cast<std::int64_t>(nt);
  const bool ho = hasOpen;
  const float oxo = origin[0], oyo = origin[1], ozo = origin[2];
  const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
  const double Lx = double(gdims[0]) * sx, Ly = double(gdims[1]) * sy, Lz = double(gdims[2]) * sz;
  Kokkos::parallel_for(
      "pnm::throat_flux", ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const int lx = ix + g.g, ly = iy + g.g, lz = iz + g.g;
        const Index o = g.owned(ix, iy, iz);
        const int a = flowLab(g.cell(lx, ly, lz));
        for (int d = 0; d < 3; ++d) {
          const Index nb = g.cell(lx + (d == 0), ly + (d == 1), lz + (d == 2));
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
          const Index rt = faceLab(3 * o + d);  // patch root, -1 = no throat face
          if (rt >= 0) {
            const Index k = bsearchKey(keyD, ntl, rt);
            const int slot = k >= 0 ? slotD(k) : -1;
            const int lo = a < b ? a : b;
            if (slot >= 0) {
              const double w0 = opn * area;
              Kokkos::atomic_add(&Q(slot), a < b ? q : -q);
              Kokkos::atomic_add(&A(slot), w0);
              // face center, min-imaged relative to the lower pore's peak voxel center
              int gx, gy, gz;
              g.gcoord(lx, ly, lz, gx, gy, gz);
              double fp[3] = {oxo + (gx + (d == 0 ? 0.5 : 0.0)) * double(sx),
                              oyo + (gy + (d == 1 ? 0.5 : 0.0)) * double(sy),
                              ozo + (gz + (d == 2 ? 0.5 : 0.0)) * double(sz)};
              const double pc[3] = {ancX(lo - 1), ancY(lo - 1), ancZ(lo - 1)};
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

/// Total-pressure drop per throat: periodic parts + the macroscopic gradient along the throat-
/// anchored two-leg min-image path i -> throat centroid -> j. Every position term cancels except
/// the integer periodic image count k, decided in SNAPPED integer-cell arithmetic (peak-voxel
/// anchors + the bias-rounded throat centroid) so float wobble can never flip the image branch.
/// (A single min-image between the pore centers is ambiguous at exactly half the period.)
inline std::vector<double> throatPressureDrops(
    const std::vector<std::pair<int, int>>& throats, const std::vector<Pore>& pores,
    const std::vector<double>& porePressure, const std::vector<double>& throatArea,
    const std::vector<double>& Cx, const std::vector<double>& Cy, const std::vector<double>& Cz,
    const std::vector<double>& ancX, const std::vector<double>& ancY,
    const std::vector<double>& ancZ, std::array<float, 3> spacing, std::array<int, 3> gdims,
    std::array<double, 3> grad_p) {
  const std::size_t nt = throats.size();
  std::vector<double> dp(nt);
  const double L[3] = {double(gdims[0]) * spacing[0], double(gdims[1]) * spacing[1],
                       double(gdims[2]) * spacing[2]};
  for (std::size_t t = 0; t < nt; ++t) {
    const int li = throats[t].first, lj = throats[t].second;
    const Pore& pi = pores[li - 1];
    const Pore& pj = pores[lj - 1];
    const double aw = throatArea[t];
    const double anc[3] = {ancX[li - 1], ancY[li - 1], ancZ[li - 1]};
    const double ancj[3] = {ancX[lj - 1], ancY[lj - 1], ancZ[lj - 1]};
    double macro = 0.0;
    const double pip[3] = {double(pi.x), double(pi.y), double(pi.z)};
    const double pjp[3] = {double(pj.x), double(pj.y), double(pj.z)};
    for (int a = 0; a < 3; ++a) {
      const double ct = (Cx[t] * (a == 0) + Cy[t] * (a == 1) + Cz[t] * (a == 2)) / aw;
      const long long Dc = llround((ancj[a] - anc[a]) / double(spacing[a])) -
                           llround(ct / double(spacing[a]) + 1e-6);
      const long long k = llround(double(Dc) / gdims[a]);
      macro += grad_p[a] * ((pjp[a] - pip[a]) - L[a] * double(k));
    }
    dp[t] = (porePressure[li - 1] - porePressure[lj - 1]) - macro;
  }
  return dp;
}

}  // namespace kernels
}  // namespace pnm

#endif  // PECLET_PNM_PORE_KERNELS_HPP
