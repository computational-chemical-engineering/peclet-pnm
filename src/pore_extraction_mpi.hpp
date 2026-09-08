/// @file
/// @brief peclet.pnm — distributed (MPI) pore-network extraction on the core block decomposition.
///
/// The SDF is decomposed over ranks by the shared ORB (peclet::core::decomp::BlockDecomposer, the
/// same deterministic partition flow/dem use) and every stage of the single-rank pipeline
/// (pore_extraction.hpp) runs per-rank on a g=1 extended block with core's GridHalo ghost
/// exchange — the SAME stage kernels (pore_kernels.hpp), instantiated on the `BlockGeo` geometry
/// instead of the single-rank `GridGeo`. Labels are GLOBAL voxel ids (Index/int64), which makes
/// every fixpoint decomposition-independent, so the multi-rank result is BIT-EXACT to the
/// single-rank pipeline. This file holds only what is distributed: the halo exchanges, the
/// ownership/merge orchestration and the reductions:
///
///   * pore detection      — 3^3 stencil on the exchanged SDF; a rank emits the peaks it owns.
///   * marker CCL          — local union-find over owned cells (the single-rank kernels on the
///                           owned box), then per-component min-gid, then ONE global merge:
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
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "pore_extraction.hpp"

namespace pnm {

static_assert(std::is_same_v<Index, peclet::core::Index>,
              "pnm::Index must be core's Index (the halo exchanges carry it)");
using peclet::core::IVec;

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

/// Sort + unique in place.
template <class T>
inline void sortUnique(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// The extended-block geometry of a halo topology (ghost width G) on the global grid `gdims`.
template <class Topo>
inline BlockGeo blockGeoOf(const Topo& topo, std::array<int, 3> gdims, int G) {
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
  return geo;
}

/// Stage a host inner block (x-fastest, no ghosts) onto a fresh extended device field (ghosts
/// zero-initialised); the caller exchanges.
template <class T>
inline peclet::core::View<T> stageInner(const Exec& space, const BlockGeo& geo,
                                        const std::vector<T>& h, const char* name) {
  const BlockGeo g = geo;
  peclet::core::View<T> e(name, g.numExt());
  Kokkos::View<T*, Mem> i0("pnm::mpi::stage_in", g.numOwned());
  uploadVec(h, i0);
  Kokkos::parallel_for(
      "pnm::mpi::stage", kernels::ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        e(g.cell(ix + g.g, iy + g.g, iz + g.g)) = i0(g.owned(ix, iy, iz));
      });
  space.fence();
  return e;
}

/// Pack the owned box of an extended device field into a host vector.
template <class T>
inline std::vector<T> packInner(const Exec& space, const BlockGeo& geo,
                                const peclet::core::View<T>& e) {
  const BlockGeo g = geo;
  Kokkos::View<T*, Mem> i0("pnm::mpi::pack", g.numOwned());
  Kokkos::parallel_for(
      "pnm::mpi::seg_pack", kernels::ownedRange(space, g), KOKKOS_LAMBDA(int ix, int iy, int iz) {
        i0(g.owned(ix, iy, iz)) = e(g.cell(ix + g.g, iy + g.g, iz + g.g));
      });
  space.fence();
  return downloadN(i0, g.numOwned());
}

/// Interleaved device (a,b) Index pairs -> host list, locally sorted/unique.
template <class PairView>
inline std::vector<std::pair<Index, Index>> localPairs(const PairView& d, std::size_t count) {
  std::vector<Index> flat = downloadN(d, 2 * count);
  std::vector<std::pair<Index, Index>> loc(count);
  for (std::size_t i = 0; i < count; ++i)
    loc[i] = {flat[2 * i], flat[2 * i + 1]};
  sortUnique(loc);
  return loc;
}

/// ONE global merge of a cross-block equivalence graph: the local (a,b) pairs are allgathered
/// and union-found on the host (union-by-min-gid: the class root is the min gid, matching the
/// single-rank fixpoint; deterministic on every rank — identical input). Returns the
/// (key -> root) remap for every key that is not its own root, as sorted device key/value views
/// for kernels::bsearchKey (`nk` = 0 when nothing merges).
struct MergeRemap {
  peclet::core::View<Index> keys, roots;
  Index nk = 0;
};
inline MergeRemap globalMerge(const std::vector<std::pair<Index, Index>>& loc, MPI_Comm comm) {
  auto all = allgatherv(loc, comm);
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
      rkeys.push_back(kv.first);  // std::map order: ascending keys, as bsearchKey needs
      rvals.push_back(root);
    }
  }
  MergeRemap m;
  m.nk = Index(rkeys.size());
  m.keys = peclet::core::View<Index>("pnm::mpi::rk", rkeys.size());
  m.roots = peclet::core::View<Index>("pnm::mpi::rv", rvals.size());
  uploadVec(rkeys, m.keys);
  uploadVec(rvals, m.roots);
  return m;
}

/// Apply a MergeRemap to the entries of `field` selected by `pred(i)`: label -> its class root.
template <class FieldView, class PredFn>
inline void applyRemap(const Exec& space, const MergeRemap& m, const FieldView& field,
                       std::size_t n, const PredFn& pred) {
  if (m.nk == 0)
    return;
  const auto dk = m.keys;
  const auto dv = m.roots;
  const Index nk = m.nk;
  Kokkos::parallel_for(
      "pnm::mpi::apply_merge", kernels::R1(space, 0, n), KOKKOS_LAMBDA(std::size_t i) {
        if (!pred(i))
          return;
        const Index k = kernels::bsearchKey(dk, nk, field(i));
        if (k >= 0)
          field(i) = dv(k);
      });
  space.fence();
}

/// Cross-rank fixpoint of the hold-at-ghost pointer jumping (kernels::resolveHoldAtGhost):
/// exchange the state field, resolve one round on every rank, Allreduce the pending count, until
/// no cell is pending. Progress is guaranteed (every pending chain finalizes one cross-block hop
/// per round); the round cap is a collective guard.
template <class Halo, class TargetView>
inline void resolveToFixpoint(const Exec& space, const BlockGeo& geo, Halo& halo,
                              const TargetView& target, MPI_Comm comm, const char* what) {
  int rounds = 0;
  for (;;) {
    halo.exchange(target);
    const int pending = kernels::resolveHoldAtGhost(space, geo, target);
    int globalPending = 0;
    MPI_Allreduce(&pending, &globalPending, 1, MPI_INT, MPI_SUM, comm);
    if (!globalPending)
      break;
    if (++rounds > 100000) {
      // collective: the round count and the Allreduce'd stop are identical on every rank
      throw std::runtime_error(std::string("[pnm::mpi] ") + what + " did not converge");
    }
  }
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
  namespace kn = kernels;
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

  const BlockGeo geo = dm::blockGeoOf(topo, gdims, G);
  const BlockGeo g = geo;
  MpiPoreNetwork out;
  out.block_origin = {geo.ox + G, geo.oy + G, geo.oz + G};
  out.block_size = {geo.nx, geo.ny, geo.nz};
  const std::size_t nInner = geo.numOwned();
  const std::size_t nExt = geo.numExt();
  // A caller-side size error is rank-local, but everything below is collective: agree on it
  // first so every rank throws together instead of the good ranks hanging in the next exchange.
  {
    int bad = sdf_local.size() != nInner ? 1 : 0, anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad)
      throw std::runtime_error(bad ? "[pnm::mpi] rank " + std::to_string(rank) +
                                         ": sdf_local size " + std::to_string(sdf_local.size()) +
                                         " != block " + std::to_string(geo.nx) + "x" +
                                         std::to_string(geo.ny) + "x" + std::to_string(geo.nz)
                                   : "[pnm::mpi] rank " + std::to_string(rank) +
                                         ": sdf_local size mismatch on another rank");
  }

  Exec space;
  const auto inner = kn::ownedRange(space, geo);
  const auto ext = kn::MD3(space, {0, 0, 0}, {geo.ex, geo.ey, geo.ez});

  // ---- SDF onto the extended block + one exchange ----
  View<float> sdfE = dm::stageInner(space, geo, sdf_local, "pnm::mpi::sdfE");
  haloF.exchange(sdfE);

  // ---- pore detection (owned peaks under the global (sdf, gid) tie-break) ----
  {
    const int max_pores = 1000000;
    View<Pore> pores("pnm::mpi::pores", max_pores);
    Kokkos::View<int, Mem> counter("pnm::mpi::pore_count");
    Kokkos::deep_copy(counter, 0);
    kn::detectPores(space, geo, sdfE, origin, spacing, pores, counter, max_pores);
    out.pores = downloadN(pores, std::min<std::size_t>(readScalar(counter), max_pores));
  }

  // ---- markers + LOCAL union-find CCL over the owned box (single-rank kernels, owned indices) --
  const float min_sp = std::min(spacing[0], std::min(spacing[1], spacing[2]));
  const float thr = -1.5f * min_sp;

  View<Index> labelE("pnm::mpi::labelE", nExt);  // gid labels, -1 = unlabelled
  {
    Kokkos::View<int*, Mem> parent("pnm::mpi::parent", nInner);
    Kokkos::View<int, Mem> changed("pnm::mpi::changed");
    kn::initMarkers(space, geo, sdfE, thr, parent);
    kn::cclFixpoint(space, parent, nInner, changed,
                    [&]() { kn::cclMergeMarkers(space, geo, parent, changed); });
    // Per-component min gid -> the component label (what the single-rank atomic_min CCL yields
    // when parents are voxel ids), scattered onto the extended label field.
    View<Index> mg("pnm::mpi::mg", nInner);
    Kokkos::deep_copy(mg, Index(0x7fffffffffffffffLL));
    Kokkos::parallel_for(
        "pnm::mpi::ccl_mingid", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          const Index o = g.owned(ix, iy, iz);
          if (parent(o) != -1)
            Kokkos::atomic_min(&mg(parent(o)), g.gid(ix + 1, iy + 1, iz + 1));
        });
    space.fence();
    Kokkos::parallel_for(
        "pnm::mpi::ccl_scatter", ext, KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const Index e = g.cell(lx, ly, lz);
          const int ix = lx - 1, iy = ly - 1, iz = lz - 1;
          if (g.inOwned(ix, iy, iz)) {
            const int p = parent(g.owned(ix, iy, iz));
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
    const Index maxPairs = Index(nInner) * 4 + 1024;
    View<Index> bpairs("pnm::mpi::bpairs", std::size_t(2 * maxPairs));
    Kokkos::View<int, Mem> bcnt("pnm::mpi::bcnt");
    Kokkos::deep_copy(bcnt, 0);
    Kokkos::parallel_for(
        "pnm::mpi::boundary_pairs", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
          // Only owned cells within one cell of a block face can see a ghost.
          if (!g.onSurface(ix, iy, iz))
            return;
          const int lx = ix + 1, ly = iy + 1, lz = iz + 1;
          const Index my = labelE(g.cell(lx, ly, lz));
          if (my == -1)
            return;
          for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                if (g.inOwned(ix + dx, iy + dy, iz + dz))
                  continue;  // owned-owned adjacency was the local CCL's job
                const Index nl = labelE(g.cell(lx + dx, ly + dy, lz + dz));
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
    const auto loc =
        dm::localPairs(bpairs, std::size_t(std::min<Index>(readScalar(bcnt), maxPairs)));
    const dm::MergeRemap merge = dm::globalMerge(loc, comm);
    dm::applyRemap(
        space, merge, labelE, nExt, KOKKOS_LAMBDA(std::size_t e) { return labelE(e) != -1; });
  }
  haloI.exchange(labelE);

  // ---- flood fill of the shallow solid: Jacobi sweeps, exchange + Allreduce per sweep ----
  {
    View<Index> labelN("pnm::mpi::labelN", nExt);
    Kokkos::View<int, Mem> changed("pnm::mpi::fchanged");
    kn::floodFixpoint(space, geo, sdfE, labelE, labelN, changed, [&](int c) {
      const int gc = dm::allreduceMaxInt(c, comm);
      if (gc)
        haloI.exchange(labelE);
      return gc;
    });
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
    Kokkos::deep_copy(targetE, Index(-1));
    kn::forestInit(
        space, geo, sdfE, targetE, KOKKOS_LAMBDA(Index ci) { return sdfE(ci) > 0.0f; },
        KOKKOS_LAMBDA(Index, Index cgid) { return ~cgid; },   // solids: self-root (unused)
        KOKKOS_LAMBDA(Index, Index cgid) { return ~cgid; });  // peak: finalized self-root
    dm::resolveToFixpoint(space, geo, haloI, targetE, comm, "gradient-root resolution");
  }

  // ---- global renumbering (single-rank first-encounter order == ascending min-appearance gid) --
  View<int> segE("pnm::mpi::segE", nExt);
  {
    // Pore roots: per-rank (root -> min appearance gid), then a global min-reduce.
    auto locPores = dm::minByKey(
        nInner, KOKKOS_LAMBDA(std::size_t o, Index & k, Index & v) {
          int ix, iy, iz;
          g.ownedCoords(Index(o), ix, iy, iz);
          const Index e = g.cell(ix + 1, iy + 1, iz + 1);
          if (sdfE(e) <= 0.0f)
            return false;
          k = ~targetE(e);  // finalized state stores ~root
          v = g.gid(ix + 1, iy + 1, iz + 1);
          return true;
        });
    // Solid labels: min appearance gid over ALL labelled voxels (a flood-filled shallow voxel can
    // precede the component's min marker in gid order, so the label value alone is NOT the first
    // appearance).
    auto locSolids = dm::minByKey(
        nInner, KOKKOS_LAMBDA(std::size_t o, Index & k, Index & v) {
          int ix, iy, iz;
          g.ownedCoords(Index(o), ix, iy, iz);
          const Index e = g.cell(ix + 1, iy + 1, iz + 1);
          if (sdfE(e) > 0.0f || labelE(e) == -1)
            return false;
          k = labelE(e);
          v = g.gid(ix + 1, iy + 1, iz + 1);
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
          const Index e = g.cell(ix + 1, iy + 1, iz + 1);
          int id = 0;
          if (sdfE(e) > 0.0f) {
            const Index k = kn::bsearchKey(dpk, npk, ~targetE(e));
            id = (k >= 0) ? dpv(k) : 0;
          } else if (labelE(e) != -1) {
            const Index k = kn::bsearchKey(dsk, nsk, labelE(e));
            id = (k >= 0) ? dsv(k) : 0;
          }
          segE(e) = id;
        });
    space.fence();
  }
  haloS.exchange(segE);

  // ---- topology: local (+x/+y/+z) pairs against exchanged seg ghosts, global sort/unique ----
  {
    const Index maxPairs = Index(nInner) * 3;
    Kokkos::View<int*, Mem> pairs("pnm::mpi::pairs", std::size_t(2 * maxPairs));
    Kokkos::View<int, Mem> cnt("pnm::mpi::tcnt");
    Kokkos::deep_copy(cnt, 0);
    kn::boundaryPairs(space, geo, segE, pairs, cnt, maxPairs);
    auto loc = kn::uniquePairs(pairs, std::size_t(std::min<Index>(readScalar(cnt), maxPairs)));
    auto all = dm::allgatherv(loc, comm);
    dm::sortUnique(all);
    out.connections = std::move(all);
  }

  // ---- download this rank's inner seg block ----
  out.seg = dm::packInner(space, geo, segE);
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
    const std::vector<double>& w_h, const std::vector<double>& p_h, const std::vector<double>& ox_h,
    const std::vector<double>& oy_h, const std::vector<double>& oz_h, std::array<double, 3> grad_p,
    MPI_Comm comm) {
  namespace dm = detail_mpi;
  namespace kn = kernels;
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

  const BlockGeo geo = dm::blockGeoOf(topo, gdims, G);
  const BlockGeo g = geo;
  const std::size_t nInner = geo.numOwned();
  const std::size_t nExt = geo.numExt();
  Exec space;
  using R1 = kn::R1;
  const auto inner = kn::ownedRange(space, geo);

  // 2. stage sdf / seg / fields onto the extended block + exchange
  View<float> sdfE = dm::stageInner(space, geo, sdf_local, "nf::mpi::f");
  haloF.exchange(sdfE);
  View<int> segE = dm::stageInner(space, geo, base.seg, "nf::mpi::s");
  haloS.exchange(segE);
  auto stageD = [&](const std::vector<double>& h, bool required) {
    if (h.empty())
      return View<double>("nf::mpi::d", required ? nExt : 0);
    View<double> e = dm::stageInner(space, geo, h, "nf::mpi::d");
    haloD.exchange(e);
    return e;
  };
  View<double> uE = stageD(u_h, true), vE = stageD(v_h, true), wE = stageD(w_h, true);
  const bool hasOpen = !ox_h.empty();
  View<double> oxE = stageD(ox_h, false), oyE = stageD(oy_h, false), ozE = stageD(oz_h, false);

  // 3. flow-basin labels for EVERY cell: propagate the LABEL along the steepest-ascent forest.
  // State per cell (Index field): finalized = -(label+1) (label 0 = enclosed solid basin, no
  // pore); pending = the in-block gid currently held. Pore cells finalize immediately with their
  // own seg label; solid maxima finalize as 0; everything else chases, adopting only finalized
  // ghost values (never storing an out-of-block gid).
  View<Index> labWork("pnm::nfmpi::labWork", nExt);
  {
    Kokkos::deep_copy(labWork, Index(-1));  // ghosts: "finalized, label 0" until exchanged
    kn::forestInit(
        space, geo, sdfE, labWork, KOKKOS_LAMBDA(Index ci) { return sdfE(ci) <= 0.0f; },
        KOKKOS_LAMBDA(Index ci, Index) { return -(Index(segE(ci)) + 1); },  // pore: own label
        KOKKOS_LAMBDA(Index, Index) { return Index(-1); });  // solid max: label 0 (excluded)
    dm::resolveToFixpoint(space, geo, haloI, labWork, comm, "flow-basin resolution");
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
    const int npLoc = kn::maxLabel(space, geo, segE);
    MPI_Allreduce(&npLoc, &np, 1, MPI_INT, MPI_MAX, comm);
  }
  if (np == 0)
    return out;

  GridHaloTopology<3> topo2;
  topo2.buildTopology(dec, rank, 2, {true, true, true}, comm);
  GridHalo<double> haloP2;
  haloP2.init(topo2);
  const BlockGeo g2 = dm::blockGeoOf(topo2, gdims, 2);
  View<double> pE2 = dm::stageInner(space, g2, p_h, "pnm::nfmpi::pE2");
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
          const Index ci = g.cell(lx, ly, lz);
          const float cv = sdfE(ci);
          if (cv <= 0.0f)
            return;
          if (!kn::isPeak(g, sdfE, lx, ly, lz, cv, g.gid(lx, ly, lz)))
            return;
          PoreRec rec;
          rec.id = segE(ci);
          g.gcoord(lx, ly, lz, rec.pgx, rec.pgy, rec.pgz);
          rec.po = kn::poreAt(g, sdfE, lx, ly, lz, oxo, oyo, ozo, sx, sy, sz);
          // trilinear p at the refined position: base cells are within peak +- 2 (g2 ring)
          const double gp3[3] = {(rec.po.x - oxo) / sx, (rec.po.y - oyo) / sy,
                                 (rec.po.z - ozo) / sz};
          rec.press = kn::trilinear(gg, pE2, gp3);
          const int s0 = Kokkos::atomic_fetch_add(&cnt(), 1);
          buf(s0) = rec;
        });
    space.fence();
    recs = downloadN(buf, std::size_t(readScalar(cnt)));
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

  // 5. per-patch throats: distributed CCL over the interface faces. Local union-find on owned
  // faces (fid = 3*gid(cell)+d, same identity/adjacency as single-rank), per-component min fid,
  // then ONE global merge of the cross-block face-adjacency graph — the marker-CCL recipe. The
  // patch root (min global fid) is decomposition-independent, so the throat list matches the
  // single-rank order exactly.
  const std::size_t nfL = 3 * nInner;
  Kokkos::View<Index*, Mem> faceLab("pnm::nfmpi::faceLab", nfL);  // final: patch label (min fid)
  kn::ThroatSlots slots;
  constexpr Index kSent = kn::kSent;  // film awaiting attachment
  {
    Kokkos::View<int*, Mem> parent("pnm::nfmpi::fparent", nfL);
    Kokkos::View<Index*, Mem> fpairL("pnm::nfmpi::fpairL", nfL);
    Kokkos::View<char*, Mem> fcoreL("pnm::nfmpi::fcoreL", nfL);
    Kokkos::View<int, Mem> changed("pnm::nfmpi::fch");
    Kokkos::View<Index*, Mem> mg("pnm::nfmpi::fmg", nfL);  // component root -> min global fid
    auto rootOf = KOKKOS_LAMBDA(int p) {
      return mg(p);
    };
    // per-direction extended fields of the face labels / pair keys: exchanged so a boundary face
    // can see its ghost-side neighbours (the marker CCL exchanges one cell field; a face field is
    // three of them)
    View<Index> flabE[3], fpairE[3];
    for (int d = 0; d < 3; ++d) {
      flabE[d] = View<Index>("pnm::nfmpi::flabE", nExt);
      fpairE[d] = View<Index>("pnm::nfmpi::fpairE", nExt);
      Kokkos::deep_copy(flabE[d], Index(-1));
      Kokkos::deep_copy(fpairE[d], Index(-1));
    }
    const kn::SplitFaces<View<Index>> nbLab{flabE[0], flabE[1], flabE[2]};
    const kn::SplitFaces<View<Index>> nbPair{fpairE[0], fpairE[1], fpairE[2]};
    // scatter the owned face labels of direction d (all attached labels, or only the faces
    // parent >= 0 — the tier currently being merged) and exchange
    auto scatterLabels = [&](bool byParent) {
      for (int d = 0; d < 3; ++d) {
        auto fl = flabE[d];
        const int dd0 = d;
        Kokkos::parallel_for(
            "pnm::nfmpi::face_scatter", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
              const Index f = 3 * g.owned(ix, iy, iz) + dd0;
              const Index lbl = faceLab(f);
              const bool take = byParent ? (parent(f) >= 0) : (lbl >= 0 && lbl < kSent);
              fl(g.cell(ix + 1, iy + 1, iz + 1)) = take ? lbl : Index(-1);
            });
        space.fence();
        haloI.exchange(flabE[d]);
      }
    };
    // cross-block adjacency pairs (owned face with parent >= 0 vs ghost-cell face of the same
    // pair), then ONE global merge applied to the owned labels of that tier
    const std::int64_t maxbp = std::int64_t(nInner) * 8 + 1024;
    Kokkos::View<Index*, Mem> bpairs("pnm::nfmpi::fbp", std::size_t(2 * maxbp));
    Kokkos::View<int, Mem> bcnt("pnm::nfmpi::fbpcnt");
    auto boundaryMerge = [&]() {
      Kokkos::deep_copy(bcnt, 0);
      Kokkos::parallel_for(
          "pnm::nfmpi::face_bpairs", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
            if (!g.onSurface(ix, iy, iz))
              return;  // only faces within one cell of a block face can touch a ghost face
            const Index o = g.owned(ix, iy, iz);
            for (int d = 0; d < 3; ++d) {
              if (parent(3 * o + d) < 0)
                continue;  // this tier's faces only
              const Index myl = faceLab(3 * o + d);
              const Index pk = fpairL(3 * o + d);
              for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                  for (int dx = -1; dx <= 1; ++dx) {
                    if (g.inOwned(ix + dx, iy + dy, iz + dz))
                      continue;
                    const Index e2 = g.cell(ix + 1 + dx, iy + 1 + dy, iz + 1 + dz);
                    for (int d2 = 0; d2 < 3; ++d2) {
                      const Index ol = nbLab(e2, d2);
                      if (ol < 0 || ol == myl)
                        continue;
                      if (nbPair(e2, d2) != pk)
                        continue;
                      if (!kn::facesAdjacent(dx, dy, dz, d, d2))
                        continue;
                      const int s0 = Kokkos::atomic_fetch_add(&bcnt(), 1);
                      if (s0 < maxbp) {
                        bpairs(2 * s0) = myl < ol ? myl : ol;
                        bpairs(2 * s0 + 1) = myl < ol ? ol : myl;
                      }
                    }
                  }
            }
          });
      space.fence();
      const auto loc =
          dm::localPairs(bpairs, std::size_t(std::min<std::int64_t>(readScalar(bcnt), maxbp)));
      const dm::MergeRemap merge = dm::globalMerge(loc, comm);
      dm::applyRemap(
          space, merge, faceLab, nfL, KOKKOS_LAMBDA(std::size_t f) { return parent(f) >= 0; });
    };
    // one local CCL tier: fixpoint over the faces with parent >= 0, then their min global fid
    auto cclTier = [&]() {
      kn::cclFixpoint(space, parent, nfL, changed,
                      [&]() { kn::faceMerge(space, geo, parent, fpairL, changed); });
      Kokkos::deep_copy(mg, Index(0x7fffffffffffffffLL));
      kn::faceMinGid(space, geo, parent, mg);
    };

    // interface predicate + CORE tier (both cells fluid-centered — see single-rank rationale)
    kn::faceInit(space, geo, flowLab, sdfE, oxE, oyE, ozE, hasOpen, fcoreL, parent, fpairL);
    cclTier();
    kn::faceLabelCore(space, nfL, parent, fpairL, faceLab, rootOf);
    // pair keys onto the extended fields (once) + the core labels, exchange, boundary merge
    for (int d = 0; d < 3; ++d) {
      auto fq = fpairE[d];
      const int dd0 = d;
      Kokkos::parallel_for(
          "pnm::nfmpi::pair_scatter", inner, KOKKOS_LAMBDA(int ix, int iy, int iz) {
            fq(g.cell(ix + 1, iy + 1, iz + 1)) = fpairL(3 * g.owned(ix, iy, iz) + dd0);
          });
      space.fence();
      haloI.exchange(fpairE[d]);
    }
    scatterLabels(false);
    boundaryMerge();
    // film attachment: min reachable core-patch label, propagated through films across ranks
    // (scatter+exchange the current labels each sweep; Jacobi min => deterministic fixpoint)
    {
      int rounds2 = 0;
      for (;;) {
        scatterLabels(false);
        Kokkos::deep_copy(changed, 0);
        kn::filmAttachSweep(space, geo, faceLab, fpairL, fcoreL, nbLab, nbPair, changed);
        if (!dm::allreduceMaxInt(readScalar(changed), comm))
          break;
        if (++rounds2 > 100000) {
          // collective: the round count and the Allreduce'd stop are identical on every rank
          throw std::runtime_error("[pnm::mpi] film attachment did not converge");
        }
      }
    }
    // leftover films (no core patch reachable anywhere): own patches — local CCL + one boundary
    // merge, exactly like the core tier but restricted to still-kSent faces
    kn::leftoverInit(space, nfL, faceLab, parent);
    cclTier();
    kn::faceLabelLeftover(space, nfL, parent, faceLab, rootOf);
    scatterLabels(true);
    boundaryMerge();
    // global unique (patch root, pair) -> throat list, ordered by (pair, root fid)
    {
      auto flabH = downloadN(faceLab, nfL);
      auto fpairH = downloadN(fpairL, nfL);
      std::vector<std::pair<Index, Index>> locR;  // (root, pair)
      for (std::size_t f = 0; f < nfL; ++f)
        if (flabH[f] >= 0)
          locR.push_back({flabH[f], fpairH[f]});
      dm::sortUnique(locR);
      auto allR = dm::allgatherv(locR, comm);
      dm::sortUnique(allR);
      slots = kn::throatSlots(std::move(allR));
    }
  }
  out.throats = slots.throats;
  const std::size_t nt = out.throats.size();
  Kokkos::View<Index*, Mem> keyD("pnm::nfmpi::keys", nt);
  Kokkos::View<int*, Mem> slotD("pnm::nfmpi::slots", nt);
  uploadVec(slots.keySorted, keyD);
  uploadVec(slots.slotOf, slotD);

  // 6. rank-local accumulation over owned +faces, then a global sum-reduce
  std::vector<double> Qh(nt, 0.0), Ah(nt, 0.0), Cxh(nt, 0.0), Cyh(nt, 0.0), Czh(nt, 0.0),
      Rh(np, 0.0);
  {
    View<double> Q("pnm::nfmpi::Q", nt), A("pnm::nfmpi::A", nt), Cx("pnm::nfmpi::Cx", nt),
        Cy("pnm::nfmpi::Cy", nt), Cz("pnm::nfmpi::Cz", nt), resid("pnm::nfmpi::res", np);
    kn::throatFlux(space, geo, flowLab, uE, vE, wE, oxE, oyE, ozE, hasOpen, faceLab, keyD, slotD,
                   nt, ancXD, ancYD, ancZD, origin, spacing, gdims, Q, A, Cx, Cy, Cz, resid);
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
  out.throat_dp =
      kn::throatPressureDrops(out.throats, out.pores, out.pore_pressure, out.throat_area, Cxh, Cyh,
                              Czh, ancX, ancY, ancZ, spacing, gdims, grad_p);
  return out;
}

}  // namespace pnm

#endif  // PECLET_PNM_PORE_EXTRACTION_MPI_HPP
