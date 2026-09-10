/// @file
/// @brief peclet.pnm — portable (Kokkos) single-rank pore-network extraction from an SDF.
///
/// Pore detection (local maxima of the SDF + weighted centroid), marker-controlled watershed
/// segmentation of the solid (init markers -> union-find CCL -> flood fill), gradient-path pore
/// basins, boundary-pair throat topology, and the network-flow extraction from a peclet.flow MAC
/// field. The stage kernels live in pore_kernels.hpp (shared with the distributed pipeline,
/// instantiated here on the trivial `GridGeo` geometry — the field is the whole periodic grid);
/// this file is the single-rank orchestration: device-resident buffers, the fixpoint loops, the
/// device renumbering and the host topology sort/unique. The distributed pipeline
/// (pore_extraction_mpi.hpp) runs the SAME kernels on the core block decomposition, bit-exact.
#ifndef PECLET_PNM_PORE_EXTRACTION_HPP
#define PECLET_PNM_PORE_EXTRACTION_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <utility>
#include <vector>

#include "pore_kernels.hpp"

namespace pnm {

// ---- pore detection (local maxima of the SDF + weight-centroid sub-voxel position) ----
// Device core: operates on an already-uploaded device SDF, so a fused pipeline uploads the SDF
// once.
inline std::vector<Pore> extractPoresView(const Kokkos::View<float*, Mem>& sdf,
                                          std::array<int, 3> resolution,
                                          std::array<float, 3> origin,
                                          std::array<float, 3> spacing) {
  const GridGeo geo = GridGeo::of(I3{resolution[0], resolution[1], resolution[2]});
  const int max_pores = 1000000;
  Kokkos::View<Pore*, Mem> pores("pores", max_pores);
  Kokkos::View<int, Mem> counter("counter");
  Kokkos::deep_copy(counter, 0);
  Exec space;
  kernels::detectPores(space, geo, sdf, origin, spacing, pores, counter, max_pores);
  const int h_count = std::min(readScalar(counter), max_pores);
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
  const GridGeo geo = GridGeo::of(I3{resolution[0], resolution[1], resolution[2]});
  const std::size_t n = sdf.extent(0);
  const float min_sp = std::min(spacing[0], std::min(spacing[1], spacing[2]));
  const float thr = -1.5f * min_sp;

  Kokkos::View<int*, Mem> labels("labels", n), roots("roots", n);
  Kokkos::View<int, Mem> changed("changed");
  Exec space;

  // 1. init markers (deep solid -> own index, else -1)
  kernels::initMarkers(space, geo, sdf, thr, labels);

  // 2. union-find CCL on markers (26-connectivity, 13 forward neighbours) + path compression, to
  // fixpoint. Labels are voxel ids, so the fixpoint label of a component is its min voxel id.
  kernels::cclFixpoint(space, labels, n, changed,
                       [&]() { kernels::cclMergeMarkers(space, geo, labels, changed); });

  // 3. flood-fill the remaining (shallow) solid voxels (26-connectivity, smallest neighbour label),
  // to fixpoint. Jacobi (double-buffered) — deterministic, and sweep-for-sweep identical to the
  // distributed flood, which is what makes the multi-rank segmentation bit-exact to this path.
  {
    Kokkos::View<int*, Mem> labelsN("labelsN", n);
    kernels::floodFixpoint(space, geo, sdf, labels, labelsN, changed, [](int c) { return c; });
  }

  // 4. gradient-path pore basins (ascent for pores, descent for solids; 26-connectivity, tie-break
  // on index)
  kernels::gradientWalk(
      space, geo, sdf, false, KOKKOS_LAMBDA(Index i, Index walker) { roots(i) = int(walker); });

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
  using R1 = kernels::R1;
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
  const GridGeo geo = GridGeo::of(I3{resolution[0], resolution[1], resolution[2]});
  const std::size_t n = seg.extent(0);
  const Index max_pairs = Index(n) * 3;
  Kokkos::View<int*, Mem> pairs("pairs",
                                (std::size_t)max_pairs * 2);  // flattened (l1,l2) interleaved
  Kokkos::View<int, Mem> cnt("cnt");
  Kokkos::deep_copy(cnt, 0);
  Exec space;
  kernels::boundaryPairs(space, geo, seg, pairs, cnt, max_pairs);
  const Index h_count = std::min<Index>(readScalar(cnt), max_pairs);
  return kernels::uniquePairs(pairs, static_cast<std::size_t>(h_count));  // only the used slots
}

// Host wrapper: upload the segmentation from a caller-owned buffer (the binding hands over the
// NumPy array's memory directly), then run the device core.
inline std::vector<std::pair<int, int>> extract_topology_k(const int* seg_h, std::size_t n,
                                                           std::array<int, 3> resolution) {
  if (n == 0)
    return {};
  Kokkos::View<int*, Mem> seg("seg", n);
  Kokkos::deep_copy(
      seg, Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
               seg_h, n));
  return extractTopologyView(seg, resolution);
}

inline std::vector<std::pair<int, int>> extract_topology_k(const std::vector<int>& seg_h,
                                                           std::array<int, 3> resolution) {
  return extract_topology_k(seg_h.data(), seg_h.size(), resolution);
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
// Throats are PER-PATCH: a throat is a CONNECTED patch of interface faces, so two disjoint
// interfaces between the same two pores (e.g. once directly and once through the periodic wrap)
// are separate parallel throats — `throats` can repeat the same label pair. Like the Voronoi
// network, where each throat was its own facet object.
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
    const std::vector<double>& w_h, const std::vector<double>& p_h, const std::vector<double>& ox_h,
    const std::vector<double>& oy_h,
    const std::vector<double>& oz_h,  // pass empty vectors for a fully-open (non-cut-cell) grid
    std::array<double, 3> grad_p) {
  namespace kn = kernels;
  NetworkFlow out;
  if (sdf_h.empty())
    return out;
  const I3 res{resolution[0], resolution[1], resolution[2]};
  const GridGeo geo = GridGeo::of(res);
  const std::size_t n = sdf_h.size();
  Exec space;
  using R1 = kn::R1;

  Kokkos::View<float*, Mem> sdf("nf::sdf", n);
  uploadVec(sdf_h, sdf);
  Kokkos::View<int*, Mem> roots;
  Kokkos::View<int*, Mem> seg = segmentVolumeView(sdf, resolution, spacing, &roots);

  // number of pore labels
  const int np = kn::maxLabel(space, geo, seg);
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
    const GridGeo g = geo;
    Kokkos::parallel_for(
        "nf::centers", R1(space, 0, np), KOKKOS_LAMBDA(int k) {
          int ix, iy, iz;
          g.ownedCoords(peak(k), ix, iy, iz);
          poresD(k) = kn::poreAt(g, sdf, ix, iy, iz, oxo, oyo, ozo, sx, sy, sz);
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
    const GridGeo g = geo;
    Kokkos::parallel_for(
        "nf::pore_pressure", R1(space, 0, np), KOKKOS_LAMBDA(int k) {
          const Pore po = poresD(k);
          const double gp[3] = {(po.x - oxo) / sx, (po.y - oyo) / sy, (po.z - ozo) / sz};
          ppres(k) = kn::trilinear(g, p, gp);
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
  kn::gradientWalk(
      space, geo, sdf, true, KOKKOS_LAMBDA(Index i, Index walker) {
        flowLab(i) = seg(walker);  // seg at a pore peak == its pore label id
      });

  // ---- per-patch throats: CCL over the interface FACES -------------------------------------
  // A throat is a CONNECTED patch of open interface faces, not the whole label pair: two pores
  // touching at two disjoint places (e.g. once directly and once through the periodic wrap) are
  // two parallel throats — pair-keyed, their fluxes would merge and can cancel. Faces are
  // identified by fid = 3*cell + d (the +d face of `cell`); adjacency is kernels::facesAdjacent.
  // Each patch is keyed by its minimum fid, so the throat list is grid-deterministic.
  // Two-tier patching: CORE faces — both cells FLUID-centered (sdf > 0) — define the patches by
  // CCL; FILM faces (at least one solid-centered cell; they exist only because the flow basins
  // extend into the cut-cell staircase) then ATTACH to the minimum reachable core patch by label
  // propagation, so a film can never bridge two core patches into one. The fluid-centered
  // criterion is geometric and parameter-free: a wall-hugging film cannot contain a core face by
  // construction (measured: an openness threshold could not separate two capsule throats bridged
  // by a wall film whose staircase faces reach openness ~0.7). Films reaching no core patch
  // become their own patches. Every interface face lands in exactly one patch — exact bookkeeping.
  const std::size_t nf = 3 * n;
  Kokkos::View<Index*, Mem> fpar("nf::fpar", nf);    // final patch label per face, -1 = none
  Kokkos::View<Index*, Mem> fpair("nf::fpair", nf);  // (lo<<32|hi) pair key per face
  {
    Kokkos::View<Index*, Mem> par("nf::fccl", nf);  // CCL parents (core, then leftover)
    Kokkos::View<char*, Mem> core("nf::fcore", nf);
    Kokkos::View<int, Mem> changed("nf::fchanged");
    kn::faceInit(space, geo, flowLab, sdf, ox, oy, oz, hasOpen, core, par, fpair);
    // one CCL fixpoint over the faces with par >= 0 (used twice: core tier, then leftover films)
    auto cclPass = [&]() {
      kn::cclFixpoint(space, par, nf, changed,
                      [&]() { kn::faceMerge(space, geo, par, fpair, changed); });
    };
    // single-rank: the CCL root IS the min fid of the patch (parents are fids, union-by-min)
    auto rootOf = KOKKOS_LAMBDA(Index parent) {
      return parent;
    };
    cclPass();  // core tier
    kn::faceLabelCore(space, nf, par, fpair, fpar, rootOf);
    // film attachment: min reachable core-patch label, propagated through films (Jacobi min)
    const kn::FlatFaces<Kokkos::View<Index*, Mem>> nbLab{fpar}, nbPair{fpair};
    int h_changed = 1;
    while (h_changed) {
      Kokkos::deep_copy(changed, 0);
      kn::filmAttachSweep(space, geo, fpar, fpair, core, nbLab, nbPair, changed);
      h_changed = readScalar(changed);
    }
    // leftover films (no core patch reachable): their own patches by a second CCL tier
    kn::leftoverInit(space, nf, fpar, par);
    cclPass();
    kn::faceLabelLeftover(space, nf, par, fpar, rootOf);
  }
  // unique patch roots -> throat slots, ordered by (pair, root fid)
  kn::ThroatSlots slots;
  {
    Kokkos::View<Index*, Mem> rbuf("nf::rbuf", nf ? nf : 1), pbuf("nf::pbuf", nf ? nf : 1);
    Kokkos::View<int, Mem> rcnt("nf::rcnt");
    Kokkos::deep_copy(rcnt, 0);
    Kokkos::parallel_for(
        "nf::face_roots", R1(space, 0, nf), KOKKOS_LAMBDA(std::size_t f) {
          if (fpar(f) == Index(f)) {
            const int s0 = Kokkos::atomic_fetch_add(&rcnt(), 1);
            rbuf(s0) = f;
            pbuf(s0) = fpair(f);
          }
        });
    space.fence();
    const std::size_t nr = std::size_t(readScalar(rcnt));
    auto hr = downloadN(rbuf, nr);
    auto hp = downloadN(pbuf, nr);
    std::vector<std::pair<Index, Index>> rootPair(nr);
    for (std::size_t i = 0; i < nr; ++i)
      rootPair[i] = {hr[i], hp[i]};
    slots = kn::throatSlots(std::move(rootPair));
  }
  out.throats = slots.throats;
  const std::size_t nt = out.throats.size();
  Kokkos::View<Index*, Mem> keyD("nf::keys", nt);
  Kokkos::View<int*, Mem> slotD("nf::slots", nt);
  uploadVec(slots.keySorted, keyD);
  uploadVec(slots.slotOf, slotD);

  // Min-image anchor per pore: its PEAK VOXEL center — integer coordinates, so the periodic-image
  // branch is deterministic (the refined float centroid wobbles ~1e-8 under CUDA FMA
  // contraction, which flips the image of faces exactly half a period away — measured on a
  // symmetric sphere lattice).
  const auto peakH = downloadN(peak, static_cast<std::size_t>(np));
  std::vector<double> ancX(np), ancY(np), ancZ(np);
  for (int k = 0; k < np; ++k) {
    const int pk = peakH[k];
    ancX[k] = origin[0] + (pk % res.x) * double(spacing[0]);
    ancY[k] = origin[1] + ((pk / res.x) % res.y) * double(spacing[1]);
    ancZ[k] = origin[2] + (pk / (res.x * res.y)) * double(spacing[2]);
  }
  Kokkos::View<double*, Mem> ancXD("nf::ancX", np), ancYD("nf::ancY", np), ancZD("nf::ancZ", np);
  uploadVec(ancX, ancXD);
  uploadVec(ancY, ancYD);
  uploadVec(ancZ, ancZD);

  // accumulate: openness-weighted MAC face fluxes over every flow-basin boundary face, plus the
  // area-weighted throat centroid (min-imaged relative to the lower pore's anchor).
  Kokkos::View<double*, Mem> Q("nf::Q", nt), A("nf::A", nt), resid("nf::resid", np);
  Kokkos::View<double*, Mem> Cx("nf::Cx", nt), Cy("nf::Cy", nt), Cz("nf::Cz", nt);
  kn::throatFlux(space, geo, flowLab, u, v, w, ox, oy, oz, hasOpen, fpar, keyD, slotD, nt, ancXD,
                 ancYD, ancZD, origin, spacing, resolution, Q, A, Cx, Cy, Cz, resid);
  std::vector<double> hcx = downloadN(Cx, nt), hcy = downloadN(Cy, nt), hcz = downloadN(Cz, nt);
  {
    auto hq = downloadN(Q, nt);
    auto ha = downloadN(A, nt);
    auto hr = downloadN(resid, static_cast<std::size_t>(np));
    out.throat_flow.assign(hq.begin(), hq.end());
    out.throat_area.assign(ha.begin(), ha.end());
    out.pore_residual.assign(hr.begin(), hr.end());
  }

  // total-pressure drop per throat (throat-anchored two-leg min-image path)
  out.throat_dp =
      kn::throatPressureDrops(out.throats, out.pores, out.pore_pressure, out.throat_area, hcx, hcy,
                              hcz, ancX, ancY, ancZ, spacing, resolution, grad_p);
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

#endif  // PECLET_PNM_PORE_EXTRACTION_HPP
