/// @file
/// @brief peclet.pnm — distributed (MPI) pore-network extraction on the core block decomposition.
///
/// The SDF is decomposed over ranks by the shared ORB (peclet::core::decomp::BlockDecomposer, the
/// same deterministic partition flow/dem use) and every stage of the single-rank pipeline
/// (pore_extraction.hpp) runs per-rank on a g=1 extended block with core's GridHalo ghost
/// exchange. Labels are GLOBAL voxel ids (Index/int64), which makes every fixpoint
/// decomposition-independent, so the multi-rank result is BIT-EXACT to the single-rank pipeline:
///
///   * pore detection      — 3^3 stencil on the exchanged SDF; a rank emits the peaks it owns.
///   * marker CCL          — local union-find over owned cells (verbatim single-rank kernels on
///                           the owned box), then per-component min-gid, then ONE global merge:
///                           the cross-block adjacency graph on boundary labels (surface data) is
///                           allgathered and union-found on the host — no iteration to
///                           convergence. Fixpoint = min gid of the global component, the same
///                           value the single-rank atomic_min CCL converges to.
///   * flood fill          — Jacobi (double-buffered) min-label sweeps with a halo exchange +
///                           changed-flag Allreduce per sweep; sweep-for-sweep identical to the
///                           single-rank Jacobi flood.
///   * gradient-path roots — the 512-step walk does not distribute, but its RESULT is a forest
///                           (each pore voxel points at its steepest 26-neighbour under the
///                           (sdf, gid) lexicographic order; peaks are roots). Roots are resolved
///                           by pointer jumping: chase within the block, stop at a ghost, exchange
///                           the target field, repeat. The forest is fixed, so the fixpoint is the
///                           unique root regardless of order. (No 512-step cap — a >512-step
///                           monotone path would be the one divergence from the single-rank walk.)
///   * renumbering         — solid labels ARE their component min-gid, so solid ids are just the
///                           ascending sort of the global label set; pore ids need the
///                           first-appearance (min gid) per root, reduced per-rank with a device
///                           UnorderedMap then allgathered. Matches the single-rank
///                           first-encounter order exactly.
///   * topology            — local (+x/+y/+z) face pairs against exchanged seg ghosts, then a
///                           global sort/unique; every rank returns the identical list.
///
/// Include only from MPI-enabled TUs (PECLET_PNM_MPI bindings, tests/kokkos_mpi).
#ifndef PECLET_PNM_PORE_EXTRACTION_MPI_HPP
#define PECLET_PNM_PORE_EXTRACTION_MPI_HPP

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <map>
#include <utility>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "pore_extraction.hpp"

namespace pnm {

using peclet::core::Index;
using peclet::core::IVec;

/// Block geometry for device kernels: extended (ghost-inclusive) local box <-> global periodic
/// grid. All members plain ints so the POD captures by value into lambdas.
struct BlockGeo {
  int ex, ey, ez;     // extended sizes (inner + 2g)
  int ox, oy, oz;     // global origin of the extended box (originInclGhost; may be negative)
  int nx, ny, nz;     // inner sizes
  int gnx, gny, gnz;  // global dims
  int g;              // ghost width

  static KOKKOS_INLINE_FUNCTION int wrapc(int a, int n) {
    int r = a % n;
    return r < 0 ? r + n : r;
  }
  KOKKOS_INLINE_FUNCTION Index lidx(int lx, int ly, int lz) const {
    return (Index(lz) * ey + ly) * ex + lx;
  }
  /// gid of the (periodically wrapped) global cell under extended-local coords.
  KOKKOS_INLINE_FUNCTION Index gidAt(int lx, int ly, int lz) const {
    const int gx = wrapc(ox + lx, gnx), gy = wrapc(oy + ly, gny), gz = wrapc(oz + lz, gnz);
    return (Index(gz) * gny + gy) * gnx + gx;
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
};

namespace detail_mpi {

inline int allreduceMaxInt(int v, MPI_Comm comm) {
  int g = 0;
  MPI_Allreduce(&v, &g, 1, MPI_INT, MPI_MAX, comm);
  return g;
}

/// Allgatherv a local vector of T (POD) — every rank returns the concatenation.
template <class T>
inline std::vector<T> allgatherv(const std::vector<T>& local, MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  const int nloc = static_cast<int>(local.size() * sizeof(T));
  std::vector<int> counts(size, 0), displs(size + 1, 0);
  MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  for (int r = 0; r < size; ++r)
    displs[r + 1] = displs[r] + counts[r];
  std::vector<T> all(displs[size] / sizeof(T));
  MPI_Allgatherv(local.data(), nloc, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE,
                 comm);
  return all;
}

/// Device binary search over a sorted key array; returns the payload index or -1.
KOKKOS_INLINE_FUNCTION Index bsearchKey(const peclet::core::View<Index>& keys, Index nkeys,
                                        Index key) {
  Index lo = 0, hi = nkeys - 1;
  while (lo <= hi) {
    const Index mid = lo + (hi - lo) / 2;
    const Index k = keys(mid);
    if (k == key)
      return mid;
    if (k < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return Index(-1);
}

/// Per-rank (key -> min value) reduction on device via UnorderedMap, with capacity-retry.
/// `emit(i, key, val)` semantics are provided by the caller lambda `KeyVal(i, k, v) -> bool`
/// (returns false to skip element i). Returns host (key, minValue) pairs.
template <class KeyValFn>
inline std::vector<std::pair<Index, Index>> minByKey(std::size_t n, const KeyValFn& kv) {
  using Map = Kokkos::UnorderedMap<Index, Index, Exec>;
  std::size_t cap = 4096;
  for (;;) {
    Map map(cap);
    int failed = 0;
    Kokkos::parallel_reduce(
        "pnm::mpi::min_by_key", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(std::size_t i, int& f) {
          Index key, val;
          if (!kv(i, key, val))
            return;
          auto r = map.insert(key, val);
          if (r.failed()) {
            f += 1;
            return;
          }
          Kokkos::atomic_min(&map.value_at(r.index()), val);
        },
        failed);
    Kokkos::fence();
    if (failed == 0) {
      // Harvest valid entries.
      const std::size_t mcap = map.capacity();
      Kokkos::View<Index*, Mem> keys("pnm::mpi::keys", map.size()),
          vals("pnm::mpi::vals", map.size());
      Kokkos::View<int, Mem> slot("pnm::mpi::slot");
      Kokkos::deep_copy(slot, 0);
      Kokkos::parallel_for(
          "pnm::mpi::harvest", Kokkos::RangePolicy<Exec>(0, mcap), KOKKOS_LAMBDA(std::size_t k) {
            if (map.valid_at(k)) {
              const int s = Kokkos::atomic_fetch_add(&slot(), 1);
              keys(s) = map.key_at(k);
              vals(s) = map.value_at(k);
            }
          });
      Kokkos::fence();
      auto hk = downloadN(keys, map.size());
      auto hv = downloadN(vals, map.size());
      std::vector<std::pair<Index, Index>> out(hk.size());
      for (std::size_t i = 0; i < hk.size(); ++i)
        out[i] = {hk[i], hv[i]};
      return out;
    }
    cap *= 4;
  }
}

}  // namespace detail_mpi

/// One owned pore-peak record of the distributed network-flow extraction (namespace-scope: nvcc
/// forbids function-local types in extended-lambda captures).
struct PoreRec {
  int id;
  int pgx, pgy, pgz;  // peak voxel global coords: the deterministic min-image anchor
  Pore po;
  double press;
};

/// Result of one distributed extraction. `pores` and `seg` are RANK-LOCAL (the pores whose peak
/// voxel this rank owns; the dense labels of this rank's inner block, x-fastest);
/// `connections` is the GLOBAL unique pair list, identical on every rank.
struct MpiPoreNetwork {
  std::vector<Pore> pores;
  std::vector<int> seg;
  std::vector<std::pair<int, int>> connections;
  std::array<int, 3> block_origin{0, 0, 0};  // xyz, this rank's inner block
  std::array<int, 3> block_size{0, 0, 0};
};

/// This rank's ORB block of the global grid (deterministic; same partition as flow/dem).
inline void mpi_block_of(std::array<int, 3> gdims, MPI_Comm comm, std::array<int, 3>& origin,
                         std::array<int, 3>& size) {
  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(nranks),
                                               IVec<3>{gdims[0], gdims[1], gdims[2]});
  for (int a = 0; a < 3; ++a) {
    origin[a] = static_cast<int>(dec.origins()[rank][a]);
    size[a] = static_cast<int>(dec.sizes()[rank][a]);
  }
}

/// Distributed fused extraction. `sdf_local` is this rank's INNER block (x-fastest, no ghosts),
/// sized to the ORB block from mpi_block_of. Bit-exact to the single-rank
/// extract_pore_network_k on the gathered grid (pores as a set; seg per voxel; connections).
inline MpiPoreNetwork extract_pore_network_mpi(const std::vector<float>& sdf_local,
                                               std::array<int, 3> gdims,
                                               std::array<float, 3> origin,
                                               std::array<float, 3> spacing, MPI_Comm comm) {
  namespace dm = detail_mpi;
  using peclet::core::View;
  using peclet::core::halo::GridHalo;
  using peclet::core::halo::GridHaloTopology;

  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);

  // ---- decomposition + g=1 halo (full 26-neighbour ghost ring, periodic all axes) ----
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(nranks),
                                               IVec<3>{gdims[0], gdims[1], gdims[2]});
  constexpr int G = 1;
  GridHaloTopology<3> topo;
  topo.buildTopology(dec, rank, G, {true, true, true}, comm);
  GridHalo<float> haloF;
  haloF.init(topo);
  GridHalo<Index> haloI;
  haloI.init(topo);
  GridHalo<int> haloS;
  haloS.init(topo);

  const auto& idxr = topo.indexer();
  BlockGeo geo{};
  geo.ex = (int)idxr.sizeInclGhost()[0];
  geo.ey = (int)idxr.sizeInclGhost()[1];
  geo.ez = (int)idxr.sizeInclGhost()[2];
  geo.ox = (int)idxr.originInclGhost()[0];
  geo.oy = (int)idxr.originInclGhost()[1];
  geo.oz = (int)idxr.originInclGhost()[2];
  geo.nx = (int)idxr.sizeInner()[0];
  geo.ny = (int)idxr.sizeInner()[1];
  geo.nz = (int)idxr.sizeInner()[2];
  geo.gnx = gdims[0];
  geo.gny = gdims[1];
  geo.gnz = gdims[2];
  geo.g = G;

  MpiPoreNetwork out;
  out.block_origin = {geo.ox + G, geo.oy + G, geo.oz + G};
  out.block_size = {geo.nx, geo.ny, geo.nz};
  const std::size_t nInner = std::size_t(geo.nx) * geo.ny * geo.nz;
  const std::size_t nExt = std::size_t(geo.ex) * geo.ey * geo.ez;
  if (sdf_local.size() != nInner) {
    std::fprintf(stderr, "[pnm::mpi] rank %d: sdf_local size %zu != block %dx%dx%d\n", rank,
                 sdf_local.size(), geo.nx, geo.ny, geo.nz);
    MPI_Abort(comm, 1);
  }

  Exec space;
  using MD3 = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
  const auto inner = MD3(space, {0, 0, 0}, {geo.nx, geo.ny, geo.nz});
  const auto ext = MD3(space, {0, 0, 0}, {geo.ex, geo.ey, geo.ez});

  // ---- SDF onto the extended block + one exchange ----
  View<float> sdfE("pnm::mpi::sdfE", nExt);
  {
    View<float> sdfI("pnm::mpi::sdfI", nInner);
    uploadVec(sdf_local, sdfI);
    const BlockGeo g = geo;
    Kokkos::parallel_for(
        "pnm::mpi::sdf_stage", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          sdfE(g.lidx(ix + 1, iy + 1, iz + 1)) = sdfI((Index(iz) * g.ny + iy) * g.nx + ix);
        });
    space.fence();
  }
  haloF.exchange(sdfE);

  // ---- pore detection (owned peaks under the global (sdf, gid) tie-break) ----
  {
    const BlockGeo g = geo;
    const float ox = origin[0], oy = origin[1], oz = origin[2];
    const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
    const int max_pores = 1000000;
    View<Pore> pores("pnm::mpi::pores", max_pores);
    Kokkos::View<int, Mem> counter("pnm::mpi::pore_count");
    Kokkos::deep_copy(counter, 0);
    Kokkos::parallel_for(
        "pnm::mpi::extract_pores", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const Index ci = g.lidx(lx, ly, lz);
          const float cv = sdfE(ci);
          if (cv <= 0.0f)
            return;
          const Index cgid = g.gidAt(lx, ly, lz);
          bool peak = true;
          for (int dz = -1; dz <= 1 && peak; ++dz)
            for (int dy = -1; dy <= 1 && peak; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const float nv = sdfE(g.lidx(lx + dx, ly + dy, lz + dz));
                const Index ngid = g.gidAt(lx + dx, ly + dy, lz + dz);
                if (nv > cv || (nv == cv && ngid > cgid)) {
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
                const float v = sdfE(g.lidx(lx + dx, ly + dy, lz + dz));
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
          // Global (wrapped) integer coords -> identical float positions to the single rank.
          const int gx = BlockGeo::wrapc(g.ox + lx, g.gnx), gy = BlockGeo::wrapc(g.oy + ly, g.gny),
                    gz = BlockGeo::wrapc(g.oz + lz, g.gnz);
          const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
          if (slot < max_pores)
            pores(slot) = Pore{ox + (gx + fx) * sx, oy + (gy + fy) * sy, oz + (gz + fz) * sz, cv};
        });
    space.fence();
    auto hc = Kokkos::create_mirror_view(counter);
    Kokkos::deep_copy(hc, counter);
    out.pores = downloadN(pores, std::min<std::size_t>(hc(), max_pores));
  }

  // ---- markers + LOCAL union-find CCL over the owned box (single-rank kernels, owned indices) --
  const float min_sp = std::min(spacing[0], std::min(spacing[1], spacing[2]));
  const float thr = -1.5f * min_sp;

  View<Index> labelE("pnm::mpi::labelE", nExt);  // gid labels, -1 = unlabelled
  {
    Kokkos::View<int*, Mem> parent("pnm::mpi::parent", nInner);
    Kokkos::View<int, Mem> changed("pnm::mpi::changed");
    const BlockGeo g = geo;
    Kokkos::parallel_for(
        "pnm::mpi::init_markers", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const Index o = (Index(iz) * g.ny + iy) * g.nx + ix;
          parent(o) = (sdfE(g.lidx(ix + 1, iy + 1, iz + 1)) < thr) ? int(o) : -1;
        });
    space.fence();
    auto flatten = [&]() {
      Kokkos::parallel_for(
          "pnm::mpi::flatten", Kokkos::RangePolicy<Exec>(space, 0, nInner),
          KOKKOS_LAMBDA(std::size_t o) {
            int l = parent(o);
            if (l != -1) {
              while (l != parent(l))
                l = parent(l);
              parent(o) = l;
            }
          });
      space.fence();
    };
    int h_changed = 1;
    while (h_changed) {
      Kokkos::deep_copy(changed, 0);
      Kokkos::parallel_for(
          "pnm::mpi::merge_markers", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
            const Index o = (Index(iz) * g.ny + iy) * g.nx + ix;
            const int my = parent(o);
            if (my == -1)
              return;
            const int dz_l[13] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
            const int dy_l[13] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0};
            const int dx_l[13] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1};
            for (int k = 0; k < 13; ++k) {
              const int jx = ix + dx_l[k], jy = iy + dy_l[k], jz = iz + dz_l[k];
              if (jx < 0 || jx >= g.nx || jy < 0 || jy >= g.ny || jz < 0 || jz >= g.nz)
                continue;  // cross-block adjacency goes through the boundary graph
              const int nl = parent((Index(jz) * g.ny + jy) * g.nx + jx);
              if (nl != -1 && my != nl) {
                int rm = my;
                while (rm != parent(rm))
                  rm = parent(rm);
                int rn = nl;
                while (rn != parent(rn))
                  rn = parent(rn);
                if (rm != rn) {
                  const int small = rm < rn ? rm : rn, large = rm < rn ? rn : rm;
                  Kokkos::atomic_min(&parent(large), small);
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
    // Per-component min gid -> the component label (what the single-rank atomic_min CCL yields
    // when parents are voxel ids), scattered onto the extended label field.
    View<Index> mg("pnm::mpi::mg", nInner);
    Kokkos::deep_copy(mg, Index(0x7fffffffffffffffLL));
    Kokkos::parallel_for(
        "pnm::mpi::ccl_mingid", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const Index o = (Index(iz) * g.ny + iy) * g.nx + ix;
          if (parent(o) != -1)
            Kokkos::atomic_min(&mg(parent(o)), g.gidAt(ix + 1, iy + 1, iz + 1));
        });
    space.fence();
    Kokkos::parallel_for(
        "pnm::mpi::ccl_scatter", ext, KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const Index e = g.lidx(lx, ly, lz);
          const int ix = lx - 1, iy = ly - 1, iz = lz - 1;
          if (ix >= 0 && ix < g.nx && iy >= 0 && iy < g.ny && iz >= 0 && iz < g.nz) {
            const int p = parent((Index(iz) * g.ny + iy) * g.nx + ix);
            labelE(e) = (p == -1) ? Index(-1) : mg(p);
          } else {
            labelE(e) = -1;
          }
        });
    space.fence();
  }
  haloI.exchange(labelE);

  // ---- global merge: boundary equivalence graph, allgathered, union-found on the host ----
  {
    const BlockGeo g = geo;
    const Index maxPairs = Index(nInner) * 4 + 1024;
    View<Index> bpairs("pnm::mpi::bpairs", std::size_t(2 * maxPairs));
    Kokkos::View<int, Mem> bcnt("pnm::mpi::bcnt");
    Kokkos::deep_copy(bcnt, 0);
    Kokkos::parallel_for(
        "pnm::mpi::boundary_pairs", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          // Only owned cells within one cell of a block face can see a ghost.
          if (ix > 0 && ix < g.nx - 1 && iy > 0 && iy < g.ny - 1 && iz > 0 && iz < g.nz - 1)
            return;
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const Index my = labelE(g.lidx(lx, ly, lz));
          if (my == -1)
            return;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const int jx = ix + dx, jy = iy + dy, jz = iz + dz;
                const bool ghost = (jx < 0 || jx >= g.nx || jy < 0 || jy >= g.ny || jz < 0 ||
                                    jz >= g.nz);  // neighbour outside the owned box
                if (!ghost)
                  continue;
                const Index nl = labelE(g.lidx(lx + dx, ly + dy, lz + dz));
                if (nl == -1 || nl == my)
                  continue;
                const int s = Kokkos::atomic_fetch_add(&bcnt(), 1);
                if (s < maxPairs) {
                  bpairs(2 * s) = my < nl ? my : nl;
                  bpairs(2 * s + 1) = my < nl ? nl : my;
                }
              }
        });
    space.fence();
    auto hc = Kokkos::create_mirror_view(bcnt);
    Kokkos::deep_copy(hc, bcnt);
    std::vector<Index> flat = downloadN(bpairs, std::size_t(2 * std::min<Index>(hc(), maxPairs)));
    // Local dedupe before the allgather.
    std::vector<std::pair<Index, Index>> loc(flat.size() / 2);
    for (std::size_t i = 0; i < loc.size(); ++i)
      loc[i] = {flat[2 * i], flat[2 * i + 1]};
    std::sort(loc.begin(), loc.end());
    loc.erase(std::unique(loc.begin(), loc.end()), loc.end());
    auto all = dm::allgatherv(loc, comm);
    // Host union-find, union-by-min-gid: the class root is the min gid, matching the single-rank
    // fixpoint. Deterministic on every rank (identical input).
    std::map<Index, Index> par;
    auto find = [&par](Index x) {
      while (true) {
        auto it = par.find(x);
        if (it == par.end() || it->second == x)
          return x;
        x = it->second;
      }
    };
    for (const auto& pr : all) {
      const Index ra = find(pr.first), rb = find(pr.second);
      if (ra == rb)
        continue;
      const Index lo = ra < rb ? ra : rb, hi = ra < rb ? rb : ra;
      par[hi] = lo;
      par.emplace(lo, lo);
    }
    std::vector<Index> rkeys, rvals;
    for (const auto& kv : par) {
      const Index root = find(kv.first);
      if (root != kv.first) {
        rkeys.push_back(kv.first);
        rvals.push_back(root);
      }
    }
    if (!rkeys.empty()) {
      View<Index> dk("pnm::mpi::rk", rkeys.size()), dv("pnm::mpi::rv", rvals.size());
      uploadVec(rkeys, dk);
      uploadVec(rvals, dv);
      const Index nk = Index(rkeys.size());
      Kokkos::parallel_for(
          "pnm::mpi::apply_merge", ext, KOKKOS_LAMBDA(int lx, int ly, int lz) {
            const Index e = g.lidx(lx, ly, lz);
            const Index l = labelE(e);
            if (l == -1)
              return;
            const Index k = dm::bsearchKey(dk, nk, l);
            if (k >= 0)
              labelE(e) = dv(k);
          });
      space.fence();
    }
  }
  haloI.exchange(labelE);

  // ---- flood fill of the shallow solid: Jacobi sweeps, exchange + Allreduce per sweep ----
  {
    const BlockGeo g = geo;
    View<Index> labelN("pnm::mpi::labelN", nExt);
    Kokkos::View<int, Mem> changed("pnm::mpi::fchanged");
    int h_changed = 1;
    while (h_changed) {
      Kokkos::deep_copy(changed, 0);
      Kokkos::deep_copy(labelN, labelE);
      Kokkos::parallel_for(
          "pnm::mpi::flood", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
            const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
            const Index e = g.lidx(lx, ly, lz);
            if (sdfE(e) >= 0.0f)
              return;  // pore: ignore
            if (labelE(e) != -1)
              return;  // already labelled
            Index best = -1;
            for (int dz = -1; dz <= 1; ++dz)
              for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                  if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                  const Index nl = labelE(g.lidx(lx + dx, ly + dy, lz + dz));
                  if (nl != -1 && (best == -1 || nl < best))
                    best = nl;
                }
            if (best != -1) {
              labelN(e) = best;
              changed() = 1;
            }
          });
      space.fence();
      std::swap(labelE, labelN);
      auto hc = Kokkos::create_mirror_view(changed);
      Kokkos::deep_copy(hc, changed);
      h_changed = dm::allreduceMaxInt(hc(), comm);
      if (h_changed)
        haloI.exchange(labelE);
    }
  }

  // ---- gradient-path pore roots: steepest-neighbour forest + cross-rank root resolution ----
  // The chase state per cell is either FINALIZED (a negative value ~root once the tree root is
  // known) or a HOLD position: the gid of a cell inside THIS extended block whose own state we are
  // waiting on. A cell never stores a gid it cannot reach — the naive "adopt whatever the ghost
  // points at" strands the chase on a remote mid-chain gid outside the ghost ring (measured: basin
  // fragmentation on plateau-heavy fields). Roots finalize at init; each round the exchange
  // imports the neighbours' newly finalized boundary values, so every pending chain finalizes one
  // cross-block hop per round — guaranteed progress, loop until no cell is pending.
  View<Index> targetE("pnm::mpi::targetE", nExt);
  {
    const BlockGeo g = geo;
    Kokkos::deep_copy(targetE, Index(-1));
    Kokkos::parallel_for(
        "pnm::mpi::grad_step", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const Index ci = g.lidx(lx, ly, lz);
          const Index cgid = g.gidAt(lx, ly, lz);
          if (sdfE(ci) <= 0.0f) {  // solids: self-root (their basins are unused, as single-rank)
            targetE(ci) = ~cgid;
            return;
          }
          Index best = cgid;
          float bv = sdfE(ci);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const float nv = sdfE(g.lidx(lx + dx, ly + dy, lz + dz));
                const Index ngid = g.gidAt(lx + dx, ly + dy, lz + dz);
                if (nv > bv) {
                  bv = nv;
                  best = ngid;
                } else if (nv == bv && ngid > best)
                  best = ngid;
              }
          targetE(ci) = (best == cgid) ? ~cgid : best;  // peak: finalized self-root
        });
    space.fence();
    int rounds = 0;
    for (;;) {
      haloI.exchange(targetE);
      int pending = 0;
      Kokkos::parallel_reduce(
          "pnm::mpi::resolve_roots",
          Kokkos::RangePolicy<Exec>(space, 0, nInner),
          KOKKOS_LAMBDA(std::size_t o, int& pend) {
            const int ix = int(o % g.nx), iy = int((o / g.nx) % g.ny),
                      iz = int(o / (Index(g.nx) * g.ny));
            const Index ci = g.lidx(ix + 1, iy + 1, iz + 1);
            Index t = targetE(ci);
            if (t < 0)
              return;  // finalized
            for (int s = 0; s < 64; ++s) {
              const Index tl = g.localOf(t);  // in-block by invariant
              if (tl < 0)
                break;
              const Index tv = targetE(tl);
              if (tv < 0) {  // that cell knows its root: adopt, finalized
                t = tv;
                break;
              }
              const Index nl = g.localOf(tv);
              if (nl < 0)
                break;  // next hop leaves the block: hold at tl, wait for its owner
              t = tv;    // advance within the block
            }
            if (t != targetE(ci))
              targetE(ci) = t;
            if (t >= 0)
              pend += 1;
          },
          pending);
      space.fence();
      int globalPending = 0;
      MPI_Allreduce(&pending, &globalPending, 1, MPI_INT, MPI_SUM, comm);
      if (!globalPending)
        break;
      if (++rounds > 100000) {
        std::fprintf(stderr, "[pnm::mpi] gradient-root resolution did not converge\n");
        MPI_Abort(comm, 2);
      }
    }
  }

  // ---- global renumbering (single-rank first-encounter order == ascending min-appearance gid) --
  View<int> segE("pnm::mpi::segE", nExt);
  {
    const BlockGeo g = geo;
    // Pore roots: per-rank (root -> min appearance gid), then a global min-reduce.
    auto locPores = dm::minByKey(nInner, KOKKOS_LAMBDA(std::size_t o, Index& k, Index& v) {
      const int ix = int(o % g.nx), iy = int((o / g.nx) % g.ny), iz = int(o / (Index(g.nx) * g.ny));
      const Index e = g.lidx(ix + 1, iy + 1, iz + 1);
      if (sdfE(e) <= 0.0f)
        return false;
      k = ~targetE(e);  // finalized state stores ~root
      v = g.gidAt(ix + 1, iy + 1, iz + 1);
      return true;
    });
    // Solid labels: min appearance gid over ALL labelled voxels (a flood-filled shallow voxel can
    // precede the component's min marker in gid order, so the label value alone is NOT the first
    // appearance).
    auto locSolids = dm::minByKey(nInner, KOKKOS_LAMBDA(std::size_t o, Index& k, Index& v) {
      const int ix = int(o % g.nx), iy = int((o / g.nx) % g.ny), iz = int(o / (Index(g.nx) * g.ny));
      const Index e = g.lidx(ix + 1, iy + 1, iz + 1);
      if (sdfE(e) > 0.0f || labelE(e) == -1)
        return false;
      k = labelE(e);
      v = g.gidAt(ix + 1, iy + 1, iz + 1);
      return true;
    });
    auto allPores = dm::allgatherv(locPores, comm);
    auto allSolids = dm::allgatherv(locSolids, comm);
    // min-reduce per key, then order by min appearance (== single-rank first-encounter order).
    auto reduceSort = [](std::vector<std::pair<Index, Index>>& v) {
      std::map<Index, Index> m;
      for (const auto& kv : v) {
        auto it = m.find(kv.first);
        if (it == m.end())
          m.emplace(kv.first, kv.second);
        else if (kv.second < it->second)
          it->second = kv.second;
      }
      std::vector<std::pair<Index, Index>> order;  // (minApp, key)
      order.reserve(m.size());
      for (const auto& kv : m)
        order.push_back({kv.second, kv.first});
      std::sort(order.begin(), order.end());
      return order;
    };
    auto poreOrder = reduceSort(allPores);
    auto solidOrder = reduceSort(allSolids);
    // key -> dense id maps (pores 1,2,...; solids -1,-2,...), sorted by key for device bsearch.
    std::vector<std::pair<Index, int>> pmap(poreOrder.size()), smap(solidOrder.size());
    for (std::size_t i = 0; i < poreOrder.size(); ++i)
      pmap[i] = {poreOrder[i].second, int(i) + 1};
    for (std::size_t i = 0; i < solidOrder.size(); ++i)
      smap[i] = {solidOrder[i].second, -(int(i) + 1)};
    std::sort(pmap.begin(), pmap.end());
    std::sort(smap.begin(), smap.end());
    std::vector<Index> pk(pmap.size()), sk(smap.size());
    std::vector<int> pv(pmap.size()), sv(smap.size());
    for (std::size_t i = 0; i < pmap.size(); ++i) {
      pk[i] = pmap[i].first;
      pv[i] = pmap[i].second;
    }
    for (std::size_t i = 0; i < smap.size(); ++i) {
      sk[i] = smap[i].first;
      sv[i] = smap[i].second;
    }
    View<Index> dpk("pnm::mpi::dpk", pk.size()), dsk("pnm::mpi::dsk", sk.size());
    Kokkos::View<int*, Mem> dpv("pnm::mpi::dpv", pv.size()), dsv("pnm::mpi::dsv", sv.size());
    uploadVec(pk, dpk);
    uploadVec(sk, dsk);
    uploadVec(pv, dpv);
    uploadVec(sv, dsv);
    const Index npk = Index(pk.size()), nsk = Index(sk.size());
    Kokkos::deep_copy(segE, 0);
    Kokkos::parallel_for(
        "pnm::mpi::seg_assign", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const Index e = g.lidx(ix + 1, iy + 1, iz + 1);
          int id = 0;
          if (sdfE(e) > 0.0f) {
            const Index k = dm::bsearchKey(dpk, npk, ~targetE(e));
            id = (k >= 0) ? dpv(k) : 0;
          } else if (labelE(e) != -1) {
            const Index k = dm::bsearchKey(dsk, nsk, labelE(e));
            id = (k >= 0) ? dsv(k) : 0;
          }
          segE(e) = id;
        });
    space.fence();
  }
  haloS.exchange(segE);

  // ---- topology: local (+x/+y/+z) pairs against exchanged seg ghosts, global sort/unique ----
  {
    const BlockGeo g = geo;
    const Index maxPairs = Index(nInner) * 3;
    Kokkos::View<int*, Mem> pairs("pnm::mpi::pairs", std::size_t(2 * maxPairs));
    Kokkos::View<int, Mem> cnt("pnm::mpi::tcnt");
    Kokkos::deep_copy(cnt, 0);
    Kokkos::parallel_for(
        "pnm::mpi::topology", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const int my = segE(g.lidx(lx, ly, lz));
          const int dx_l[3] = {1, 0, 0}, dy_l[3] = {0, 1, 0}, dz_l[3] = {0, 0, 1};
          for (int k = 0; k < 3; ++k) {
            const int nl = segE(g.lidx(lx + dx_l[k], ly + dy_l[k], lz + dz_l[k]));
            if (my != nl) {
              const int s = Kokkos::atomic_fetch_add(&cnt(), 1);
              if (s < maxPairs) {
                pairs(2 * s) = my < nl ? my : nl;
                pairs(2 * s + 1) = my < nl ? nl : my;
              }
            }
          }
        });
    space.fence();
    auto hc = Kokkos::create_mirror_view(cnt);
    Kokkos::deep_copy(hc, cnt);
    std::vector<int> flat = downloadN(pairs, std::size_t(2 * std::min<Index>(hc(), maxPairs)));
    std::vector<std::pair<int, int>> loc(flat.size() / 2);
    for (std::size_t i = 0; i < loc.size(); ++i)
      loc[i] = {flat[2 * i], flat[2 * i + 1]};
    std::sort(loc.begin(), loc.end());
    loc.erase(std::unique(loc.begin(), loc.end()), loc.end());
    auto all = dm::allgatherv(loc, comm);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    out.connections = std::move(all);
  }

  // ---- download this rank's inner seg block ----
  {
    const BlockGeo g = geo;
    Kokkos::View<int*, Mem> segI("pnm::mpi::segI", nInner);
    Kokkos::parallel_for(
        "pnm::mpi::seg_pack", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          segI((Index(iz) * g.ny + iy) * g.nx + ix) = segE(g.lidx(ix + 1, iy + 1, iz + 1));
        });
    space.fence();
    out.seg = downloadN(segI, nInner);
  }
  return out;
}

// ---- distributed network flow: throat flow rates + pore pressures from a MAC field ------------
// The distributed counterpart of extract_network_flow_k (see pore_extraction.hpp for the method).
// Composes with the validated distributed segmentation: runs extract_pore_network_mpi for the
// labels, then re-stages the fields on a g=1 extended block and resolves the FLOW-BASIN label of
// every cell by propagating the label itself with the same hold-at-ghost finalization scheme as
// the pore roots (finalized = -label stored, pending = an in-block gid): no remote root-to-label
// lookup ever needed. Fluxes/areas/centroids/residuals accumulate rank-locally over owned +faces
// and are Allreduce-summed; the pore list (with trilinearly interpolated pressures, needing a
// g=2 halo of p around each owned peak) is allgathered. EVERY rank returns the identical global
// network. Bit-exact to the single-rank extract_network_flow_k up to float-sum ordering
// (atomics) and the CUDA FMA-contraction wobble on pore centroids.
inline NetworkFlow extract_network_flow_mpi(
    const std::vector<float>& sdf_local, std::array<int, 3> gdims, std::array<float, 3> origin,
    std::array<float, 3> spacing, const std::vector<double>& u_h, const std::vector<double>& v_h,
    const std::vector<double>& w_h, const std::vector<double>& p_h,
    const std::vector<double>& ox_h, const std::vector<double>& oy_h,
    const std::vector<double>& oz_h, std::array<double, 3> grad_p, MPI_Comm comm) {
  namespace dm = detail_mpi;
  using peclet::core::View;
  using peclet::core::halo::GridHalo;
  using peclet::core::halo::GridHaloTopology;
  NetworkFlow out;

  // 1. distributed segmentation (validated pipeline) -> this rank's dense labels
  MpiPoreNetwork base = extract_pore_network_mpi(sdf_local, gdims, origin, spacing, comm);

  int rank = 0, nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(nranks),
                                               IVec<3>{gdims[0], gdims[1], gdims[2]});
  constexpr int G = 1;
  GridHaloTopology<3> topo;
  topo.buildTopology(dec, rank, G, {true, true, true}, comm);
  GridHalo<float> haloF;
  haloF.init(topo);
  GridHalo<Index> haloI;
  haloI.init(topo);
  GridHalo<int> haloS;
  haloS.init(topo);
  GridHalo<double> haloD;
  haloD.init(topo);

  const auto& idxr = topo.indexer();
  BlockGeo geo{};
  geo.ex = (int)idxr.sizeInclGhost()[0];
  geo.ey = (int)idxr.sizeInclGhost()[1];
  geo.ez = (int)idxr.sizeInclGhost()[2];
  geo.ox = (int)idxr.originInclGhost()[0];
  geo.oy = (int)idxr.originInclGhost()[1];
  geo.oz = (int)idxr.originInclGhost()[2];
  geo.nx = (int)idxr.sizeInner()[0];
  geo.ny = (int)idxr.sizeInner()[1];
  geo.nz = (int)idxr.sizeInner()[2];
  geo.gnx = gdims[0];
  geo.gny = gdims[1];
  geo.gnz = gdims[2];
  geo.g = G;
  const std::size_t nInner = std::size_t(geo.nx) * geo.ny * geo.nz;
  const std::size_t nExt = std::size_t(geo.ex) * geo.ey * geo.ez;
  Exec space;
  using MD3 = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>;
  using R1 = Kokkos::RangePolicy<Exec>;
  const auto inner = MD3(space, {0, 0, 0}, {geo.nx, geo.ny, geo.nz});
  const BlockGeo g = geo;

  // 2. stage sdf / seg / fields onto the extended block + exchange
  auto stageF = [&](const std::vector<float>& h) {
    View<float> e("nf::mpi::f", nExt), i0("nf::mpi::fi", nInner);
    uploadVec(h, i0);
    Kokkos::parallel_for(
        "pnm::nfmpi::stageF", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          e(g.lidx(ix + 1, iy + 1, iz + 1)) = i0((Index(iz) * g.ny + iy) * g.nx + ix);
        });
    space.fence();
    return e;
  };
  auto stageI = [&](const std::vector<int>& h) {
    View<int> e("nf::mpi::s", nExt);
    Kokkos::View<int*, Mem> i0("nf::mpi::si", nInner);
    uploadVec(h, i0);
    Kokkos::parallel_for(
        "pnm::nfmpi::stageI", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          e(g.lidx(ix + 1, iy + 1, iz + 1)) = i0((Index(iz) * g.ny + iy) * g.nx + ix);
        });
    space.fence();
    return e;
  };
  auto stageD = [&](const std::vector<double>& h, bool required) {
    View<double> e("nf::mpi::d", (required || !h.empty()) ? nExt : 0);
    if (h.empty())
      return e;
    View<double> i0("nf::mpi::di", nInner);
    uploadVec(h, i0);
    Kokkos::parallel_for(
        "pnm::nfmpi::stageD", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          e(g.lidx(ix + 1, iy + 1, iz + 1)) = i0((Index(iz) * g.ny + iy) * g.nx + ix);
        });
    space.fence();
    return e;
  };
  View<float> sdfE = stageF(sdf_local);
  haloF.exchange(sdfE);
  View<int> segE = stageI(base.seg);
  haloS.exchange(segE);
  View<double> uE = stageD(u_h, true), vE = stageD(v_h, true), wE = stageD(w_h, true);
  haloD.exchange(uE);
  haloD.exchange(vE);
  haloD.exchange(wE);
  const bool hasOpen = !ox_h.empty();
  View<double> oxE = stageD(ox_h, false), oyE = stageD(oy_h, false), ozE = stageD(oz_h, false);
  if (hasOpen) {
    haloD.exchange(oxE);
    haloD.exchange(oyE);
    haloD.exchange(ozE);
  }

  // 3. flow-basin labels for EVERY cell: propagate the LABEL along the steepest-ascent forest.
  // State per cell (Index field): finalized = -(label+1) (label 0 = enclosed solid basin, no
  // pore); pending = the in-block gid currently held. Pore cells finalize immediately with their
  // own seg label; solid maxima finalize as 0; everything else chases, adopting only finalized
  // ghost values (never storing an out-of-block gid).
  View<Index> labWork("pnm::nfmpi::labWork", nExt);
  {
    Kokkos::deep_copy(labWork, Index(-1));  // ghosts: "finalized, label 0" until exchanged
    Kokkos::parallel_for(
        "pnm::nfmpi::basin_init", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const Index ci = g.lidx(lx, ly, lz);
          if (sdfE(ci) > 0.0f) {  // pore cell: basin label is its own segmentation label
            labWork(ci) = -(Index(segE(ci)) + 1);
            return;
          }
          const Index cgid = g.gidAt(lx, ly, lz);
          Index best = cgid;
          float bv = sdfE(ci);
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const float nv = sdfE(g.lidx(lx + dx, ly + dy, lz + dz));
                const Index ngid = g.gidAt(lx + dx, ly + dy, lz + dz);
                if (nv > bv) {
                  bv = nv;
                  best = ngid;
                } else if (nv == bv && ngid > best)
                  best = ngid;
              }
          labWork(ci) = (best == cgid) ? Index(-1) : best;  // solid max: label 0 (excluded)
        });
    space.fence();
    int rounds = 0;
    for (;;) {
      haloI.exchange(labWork);
      int pending = 0;
      Kokkos::parallel_reduce(
          "pnm::nfmpi::basin_resolve", R1(space, 0, nInner),
          KOKKOS_LAMBDA(std::size_t o, int& pend) {
            const int ix = int(o % g.nx), iy = int((o / g.nx) % g.ny),
                      iz = int(o / (Index(g.nx) * g.ny));
            const Index ci = g.lidx(ix + 1, iy + 1, iz + 1);
            Index t = labWork(ci);
            if (t < 0)
              return;  // finalized
            for (int s0 = 0; s0 < 64; ++s0) {
              const Index tl = g.localOf(t);
              if (tl < 0)
                break;
              const Index tv = labWork(tl);
              if (tv < 0) {  // that cell knows its basin label: adopt, finalized
                t = tv;
                break;
              }
              const Index nl = g.localOf(tv);
              if (nl < 0)
                break;  // next hop leaves the block: hold, wait for its owner
              t = tv;
            }
            if (t != labWork(ci))
              labWork(ci) = t;
            if (t >= 0)
              pend += 1;
          },
          pending);
      space.fence();
      int globalPending = 0;
      MPI_Allreduce(&pending, &globalPending, 1, MPI_INT, MPI_SUM, comm);
      if (!globalPending)
        break;
      if (++rounds > 100000) {
        std::fprintf(stderr, "[pnm::mpi] flow-basin resolution did not converge\n");
        MPI_Abort(comm, 3);
      }
    }
  }
  View<int> flowLab("pnm::nfmpi::flowLab", nExt);
  Kokkos::parallel_for(
      "pnm::nfmpi::basin_label", R1(space, 0, nExt),
      KOKKOS_LAMBDA(std::size_t e) { flowLab(e) = int(-labWork(e) - 1); });
  space.fence();
  haloS.exchange(flowLab);

  // 4. global pore list: each owned basin peak emits (id, Pore, trilinear pressure). The 2x2x2
  // interpolation cube around a peak can reach 2 cells out, so p gets its own g=2 halo.
  int np = 0;
  {
    int npLoc = 0;
    Kokkos::parallel_reduce(
        "pnm::nfmpi::np", R1(space, 0, nInner),
        KOKKOS_LAMBDA(std::size_t o, int& m) {
          const int ix = int(o % g.nx), iy = int((o / g.nx) % g.ny),
                    iz = int(o / (Index(g.nx) * g.ny));
          const int sv = segE(g.lidx(ix + 1, iy + 1, iz + 1));
          if (sv > m)
            m = sv;
        },
        Kokkos::Max<int>(npLoc));
    space.fence();
    MPI_Allreduce(&npLoc, &np, 1, MPI_INT, MPI_MAX, comm);
  }
  if (np == 0)
    return out;

  GridHaloTopology<3> topo2;
  topo2.buildTopology(dec, rank, 2, {true, true, true}, comm);
  GridHalo<double> haloP2;
  haloP2.init(topo2);
  const auto& idxr2 = topo2.indexer();
  BlockGeo g2{};
  g2.ex = (int)idxr2.sizeInclGhost()[0];
  g2.ey = (int)idxr2.sizeInclGhost()[1];
  g2.ez = (int)idxr2.sizeInclGhost()[2];
  g2.ox = (int)idxr2.originInclGhost()[0];
  g2.oy = (int)idxr2.originInclGhost()[1];
  g2.oz = (int)idxr2.originInclGhost()[2];
  g2.nx = geo.nx;
  g2.ny = geo.ny;
  g2.nz = geo.nz;
  g2.gnx = gdims[0];
  g2.gny = gdims[1];
  g2.gnz = gdims[2];
  g2.g = 2;
  View<double> pE2("pnm::nfmpi::pE2", std::size_t(g2.ex) * g2.ey * g2.ez);
  {
    View<double> i0("pnm::nfmpi::pi", nInner);
    uploadVec(p_h, i0);
    const BlockGeo gg = g2;
    Kokkos::parallel_for(
        "pnm::nfmpi::stageP2", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          pE2(gg.lidx(ix + 2, iy + 2, iz + 2)) = i0((Index(iz) * g.ny + iy) * g.nx + ix);
        });
    space.fence();
  }
  haloP2.exchange(pE2);

  std::vector<PoreRec> recs;
  {
    Kokkos::View<PoreRec*, Mem> buf("pnm::nfmpi::precs", nInner ? nInner : 1);
    Kokkos::View<int, Mem> cnt("pnm::nfmpi::pcnt");
    Kokkos::deep_copy(cnt, 0);
    const float oxo = origin[0], oyo = origin[1], ozo = origin[2];
    const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
    const BlockGeo gg = g2;
    Kokkos::parallel_for(
        "pnm::nfmpi::pores", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const Index ci = g.lidx(lx, ly, lz);
          const float cv = sdfE(ci);
          if (cv <= 0.0f)
            return;
          const Index cgid = g.gidAt(lx, ly, lz);
          bool peak = true;
          for (int dz = -1; dz <= 1 && peak; ++dz)
            for (int dy = -1; dy <= 1 && peak; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const float nv = sdfE(g.lidx(lx + dx, ly + dy, lz + dz));
                const Index ngid = g.gidAt(lx + dx, ly + dy, lz + dz);
                if (nv > cv || (nv == cv && ngid > cgid)) {
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
                const float v0 = sdfE(g.lidx(lx + dx, ly + dy, lz + dz));
                float w0 = v0 > 0.0f ? v0 : 0.0f;
                w0 = w0 * w0;
                sw += w0;
                px += dx * w0;
                py += dy * w0;
                pz += dz * w0;
              }
          float fx = 0, fy = 0, fz = 0;
          if (sw > 1e-6f) {
            fx = px / sw;
            fy = py / sw;
            fz = pz / sw;
          }
          const int gx = BlockGeo::wrapc(g.ox + lx, g.gnx), gy = BlockGeo::wrapc(g.oy + ly, g.gny),
                    gz = BlockGeo::wrapc(g.oz + lz, g.gnz);
          PoreRec rec;
          rec.id = segE(ci);
          rec.pgx = gx;
          rec.pgy = gy;
          rec.pgz = gz;
          rec.po = Pore{oxo + (gx + fx) * sx, oyo + (gy + fy) * sy, ozo + (gz + fz) * sz, cv};
          // trilinear p at the refined position: base cells are within peak +- 2 (g2 ring)
          const double gp3[3] = {(rec.po.x - oxo) / sx, (rec.po.y - oyo) / sy,
                                 (rec.po.z - ozo) / sz};
          int b[3];
          double f[3];
          for (int a = 0; a < 3; ++a) {
            const double fl = Kokkos::floor(gp3[a]);
            b[a] = int(fl);
            f[a] = gp3[a] - fl;
          }
          double acc = 0.0;
          for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx) {
                const double wt = (dx ? f[0] : 1.0 - f[0]) * (dy ? f[1] : 1.0 - f[1]) *
                                  (dz ? f[2] : 1.0 - f[2]);
                const Index e2 = gg.localOf(
                    (Index(BlockGeo::wrapc(b[2] + dz, gg.gnz)) * gg.gny +
                     BlockGeo::wrapc(b[1] + dy, gg.gny)) *
                        gg.gnx +
                    BlockGeo::wrapc(b[0] + dx, gg.gnx));
                acc += wt * pE2(e2);
              }
          rec.press = acc;
          const int s0 = Kokkos::atomic_fetch_add(&cnt(), 1);
          buf(s0) = rec;
        });
    space.fence();
    auto hc = Kokkos::create_mirror_view(cnt);
    Kokkos::deep_copy(hc, cnt);
    recs = downloadN(buf, std::size_t(hc()));
  }
  auto allRecs = dm::allgatherv(recs, comm);
  out.pores.resize(np);
  out.pore_pressure.assign(np, 0.0);
  std::vector<double> ancX(np), ancY(np), ancZ(np);  // peak-voxel min-image anchors
  for (const auto& r0 : allRecs) {
    out.pores[r0.id - 1] = r0.po;
    out.pore_pressure[r0.id - 1] = r0.press;
    ancX[r0.id - 1] = origin[0] + r0.pgx * double(spacing[0]);
    ancY[r0.id - 1] = origin[1] + r0.pgy * double(spacing[1]);
    ancZ[r0.id - 1] = origin[2] + r0.pgz * double(spacing[2]);
  }
  View<double> ancXD("pnm::nfmpi::ancX", np), ancYD("pnm::nfmpi::ancY", np),
      ancZD("pnm::nfmpi::ancZ", np);
  uploadVec(ancX, ancXD);
  uploadVec(ancY, ancYD);
  uploadVec(ancZ, ancZD);

  // 5. throat keys from owned +faces of the flow-basin field (open faces only), global unique
  {
    const bool ho = hasOpen;
    const std::int64_t maxp = std::int64_t(nInner) * 3;
    Kokkos::View<std::int64_t*, Mem> praw("pnm::nfmpi::praw", std::size_t(maxp));
    Kokkos::View<int, Mem> pcnt("pnm::nfmpi::prawcnt");
    Kokkos::deep_copy(pcnt, 0);
    Kokkos::parallel_for(
        "pnm::nfmpi::throat_pairs", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const int a = flowLab(g.lidx(lx, ly, lz));
          for (int d = 0; d < 3; ++d) {
            const Index nb = g.lidx(lx + (d == 0), ly + (d == 1), lz + (d == 2));
            const int b = flowLab(nb);
            if (a == b || a <= 0 || b <= 0)
              continue;
            const double opn = ho ? (d == 0 ? oxE(nb) : (d == 1 ? oyE(nb) : ozE(nb))) : 1.0;
            if (opn <= 0.0)
              continue;
            const int lo = a < b ? a : b, hi = a < b ? b : a;
            const int s0 = Kokkos::atomic_fetch_add(&pcnt(), 1);
            if (s0 < maxp)
              praw(s0) = (std::int64_t(lo) << 32) | hi;
          }
        });
    space.fence();
    auto hc = Kokkos::create_mirror_view(pcnt);
    Kokkos::deep_copy(hc, pcnt);
    auto loc = downloadN(praw, std::size_t(std::min<std::int64_t>(hc(), maxp)));
    std::sort(loc.begin(), loc.end());
    loc.erase(std::unique(loc.begin(), loc.end()), loc.end());
    auto all = dm::allgatherv(loc, comm);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    for (auto k : all)
      out.throats.push_back({int(k >> 32), int(k & 0x7fffffff)});
  }
  const std::size_t nt = out.throats.size();
  std::vector<std::int64_t> keys(nt);
  for (std::size_t t = 0; t < nt; ++t)
    keys[t] = (std::int64_t(out.throats[t].first) << 32) | out.throats[t].second;
  Kokkos::View<std::int64_t*, Mem> keyD("pnm::nfmpi::keys", nt);
  uploadVec(keys, keyD);

  // 6. rank-local accumulation over owned +faces, then a global sum-reduce
  std::vector<double> Qh(nt, 0.0), Ah(nt, 0.0), Cxh(nt, 0.0), Cyh(nt, 0.0), Czh(nt, 0.0),
      Rh(np, 0.0);
  {
    View<double> Q("pnm::nfmpi::Q", nt), A("pnm::nfmpi::A", nt), Cx("pnm::nfmpi::Cx", nt),
        Cy("pnm::nfmpi::Cy", nt), Cz("pnm::nfmpi::Cz", nt), resid("pnm::nfmpi::res", np);
    const double Axf = double(spacing[1]) * spacing[2], Ayf = double(spacing[0]) * spacing[2],
                 Azf = double(spacing[0]) * spacing[1];
    const std::int64_t ntl = std::int64_t(nt);
    const bool ho = hasOpen;
    const float oxo = origin[0], oyo = origin[1], ozo = origin[2];
    const float sx = spacing[0], sy = spacing[1], sz = spacing[2];
    const double Lx = double(g.gnx) * sx, Ly = double(g.gny) * sy, Lz = double(g.gnz) * sz;
    Kokkos::parallel_for(
        "pnm::nfmpi::throat_flux", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const int a = flowLab(g.lidx(lx, ly, lz));
          for (int d = 0; d < 3; ++d) {
            const Index nb = g.lidx(lx + (d == 0), ly + (d == 1), lz + (d == 2));
            const int b = flowLab(nb);
            if (a == b)
              continue;
            const double vel = d == 0 ? uE(nb) : (d == 1 ? vE(nb) : wE(nb));
            const double opn = ho ? (d == 0 ? oxE(nb) : (d == 1 ? oyE(nb) : ozE(nb))) : 1.0;
            const double area = d == 0 ? Axf : (d == 1 ? Ayf : Azf);
            const double q = opn * vel * area;
            if (a > 0)
              Kokkos::atomic_add(&resid(a - 1), q);
            if (b > 0)
              Kokkos::atomic_add(&resid(b - 1), -q);
            if (a > 0 && b > 0 && opn > 0.0) {
              const int lo = a < b ? a : b, hi = a < b ? b : a;
              const std::int64_t key = (std::int64_t(lo) << 32) | hi;
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
                const int gx = BlockGeo::wrapc(g.ox + lx, g.gnx),
                          gy = BlockGeo::wrapc(g.oy + ly, g.gny),
                          gz = BlockGeo::wrapc(g.oz + lz, g.gnz);
                double fp[3] = {oxo + (gx + (d == 0 ? 0.5 : 0.0)) * double(sx),
                                oyo + (gy + (d == 1 ? 0.5 : 0.0)) * double(sy),
                                ozo + (gz + (d == 2 ? 0.5 : 0.0)) * double(sz)};
                const double pc[3] = {ancXD(lo - 1), ancYD(lo - 1), ancZD(lo - 1)};
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
    auto red = [&](const View<double>& d, std::vector<double>& h) {
      auto loc = downloadN(d, h.size());
      MPI_Allreduce(loc.data(), h.data(), int(h.size()), MPI_DOUBLE, MPI_SUM, comm);
    };
    red(Q, Qh);
    red(A, Ah);
    red(Cx, Cxh);
    red(Cy, Cyh);
    red(Cz, Czh);
    red(resid, Rh);
  }
  out.throat_flow = Qh;
  out.throat_area = Ah;
  out.pore_residual = Rh;

  // 7. dp: identical on every rank (throat-anchored two-leg min-image, as single-rank)
  out.throat_dp.resize(nt);
  const double L[3] = {double(gdims[0]) * spacing[0], double(gdims[1]) * spacing[1],
                       double(gdims[2]) * spacing[2]};
  for (std::size_t t = 0; t < nt; ++t) {
    const int li = out.throats[t].first, lj = out.throats[t].second;
    const Pore& pi = out.pores[li - 1];
    const Pore& pj = out.pores[lj - 1];
    const double aw = out.throat_area[t];
    const double anc[3] = {ancX[li - 1], ancY[li - 1], ancZ[li - 1]};
    const double ancj[3] = {ancX[lj - 1], ancY[lj - 1], ancZ[lj - 1]};
    double macro = 0.0;
    const double pip[3] = {double(pi.x), double(pi.y), double(pi.z)};
    const double pjp[3] = {double(pj.x), double(pj.y), double(pj.z)};
    const int N[3] = {gdims[0], gdims[1], gdims[2]};
    for (int a = 0; a < 3; ++a) {
      // identical snapped-integer image decision as the single-rank path (see pore_extraction.hpp)
      const double ct = (Cxh[t] * (a == 0) + Cyh[t] * (a == 1) + Czh[t] * (a == 2)) / aw;
      const long long Dc = llround((ancj[a] - anc[a]) / double(spacing[a])) -
                           llround(ct / double(spacing[a]) + 1e-6);
      const long long k = llround(double(Dc) / N[a]);
      macro += grad_p[a] * ((pjp[a] - pip[a]) - L[a] * double(k));
    }
    out.throat_dp[t] = (out.pore_pressure[li - 1] - out.pore_pressure[lj - 1]) - macro;
  }
  return out;
}

}  // namespace pnm

#endif  // PECLET_PNM_PORE_EXTRACTION_MPI_HPP
