/// @file
/// @brief nanobind module `pnm` — Kokkos pore-network extraction from SDF geometry.
///
/// Matches the numpy convention: SDF is (Nz,Ny,Nx) C-order, and every triple that describes it
/// (`origin_zyx`, `spacing_zyx`, `shape_zyx`, `grad_p_zyx`) is stated in that order and carries the
/// `_zyx` suffix (NAMING.md §1.7). VTI reading (SDFReader) is pure C++ (sdf_reader.cpp,
/// backend-free); the pore / segmentation / topology / network-flow compute is Kokkos (any
/// backend). Exposes `SDFReader`, `Pore`, `extract_pores`, `segment_volume`, `extract_topology`,
/// `extract_pore_network`, `extract_network_flow` (+ `mpi_block` and the `*_mpi` collectives when
/// built with PECLET_PNM_MPI). A C-order (Nz,Ny,Nx) buffer is contiguous x-fastest, so it maps
/// onto the flat layout directly via the shared bridge (peclet::core::python, core).
///
/// Arrays in, arrays out: a segmentation is returned as an int32 (Nz,Ny,Nx) C-order ndarray
/// (the flat x-fastest label vector re-shaped in place — no copy, no reshape on the caller's
/// side), a connection / throat list as an (M,2) int32 ndarray, and the network-flow scalars as
/// float64 ndarrays; the host vectors are moved into the capsule that backs the array
/// (vector_to_ndarray). `extract_topology` takes that segmentation back without a copy. Only the
/// pores stay a Python list of `Pore` objects (small, and each carries four named scalars).
///
/// Precision policy: the SDF is float32 (the VTI's storage type; the kernels compute in float),
/// `origin_zyx` / `spacing_zyx` are taken as Python floats (double) and NARROWED to float32 before
/// the kernels, so pore centres and radii are float32 in the input unit system; the MAC fields of
/// the network-flow extraction (u, v, w, p, openness) are float64 and the flow sums are double.
///
/// Kokkos teardown follows the suite-wide pattern of peclet/core/python/kokkos_teardown.hpp: Kokkos
/// is initialized at import and the module's single atexit hook (also `pnm.finalize()`) releases
/// every registered View owner and THEN calls Kokkos::finalize. pnm keeps no Kokkos state between
/// calls (its registry stays empty), but the finalize is still REQUIRED on CUDA (else
/// cudaErrorCudartUnloading at exit).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>  // optional openness arrays, optional shape_zyx
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>  // list[Pore] and the z-y-x triples

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"
#include "pore_extraction.hpp"
#include "sdf_reader.h"

#ifdef PECLET_PNM_MPI
#include <mpi.h>

#include "pore_extraction_mpi.hpp"
#endif

namespace nb = nanobind;
using pnm::Pore;

#ifdef PECLET_PNM_MPI
// Ensure MPI_Init has been called (mirrors the flow/dem idiom); safe to call repeatedly. If WE
// initialized MPI (no mpi4py in the driver), also finalize it at exit so mpirun sees a clean
// shutdown; when mpi4py did the init, its own atexit hook finalizes.
static void ensure_mpi_init() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    std::atexit([]() {
      int fin = 0;
      MPI_Finalized(&fin);
      if (!fin)
        MPI_Finalize();
    });
  }
}
#endif

// C-order (Nz,Ny,Nx) float SDF -> flat x-fastest vector + res = (Nx,Ny,Nz).
static std::vector<float> to_sdf(nb::ndarray<float, nb::c_contig> a, std::array<int, 3>& res) {
  if (a.ndim() != 3)
    throw std::runtime_error("SDF array must be 3D (Nz,Ny,Nx)");
  res = {(int)a.shape(2), (int)a.shape(1), (int)a.shape(0)};
  return peclet::core::python::ndarray_to_vector<float>(nb::ndarray<>(a));
}

// C-order (Nz,Ny,Nx) double field -> flat x-fastest vector, shape-checked against the SDF's res.
static std::vector<double> to_field(nb::ndarray<double, nb::c_contig> a,
                                    const std::array<int, 3>& res, const char* name) {
  if (a.ndim() != 3 || (int)a.shape(2) != res[0] || (int)a.shape(1) != res[1] ||
      (int)a.shape(0) != res[2])
    throw std::runtime_error(std::string(name) + " must be (Nz,Ny,Nx) matching the SDF");
  return peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(a));
}

// ---- outputs: host vectors moved into capsule-owned NumPy arrays (no element-wise boxing) ----

// Flat x-fastest labels of an (Nx,Ny,Nz) grid -> int32 (Nz,Ny,Nx) C-order array over the SAME
// memory (an (Nz,Ny,Nx) C-order buffer is x-fastest, so this is a re-shape, not a transpose).
static nb::ndarray<nb::numpy, int> labels_to_ndarray(std::vector<int>&& seg,
                                                     const std::array<int, 3>& res) {
  const std::size_t nx = res[0], ny = res[1], nz = res[2];
  if (seg.size() != nx * ny * nz)
    throw std::runtime_error("internal: segmentation size does not match the grid");
  return peclet::core::python::vector_to_ndarray<int>(std::move(seg), {nz, ny, nx},
                                                      {std::int64_t(ny * nx), std::int64_t(nx), 1});
}

// Label pairs -> (M,2) int32 array over the pair vector's own storage.
static nb::ndarray<nb::numpy, int> pairs_to_ndarray(std::vector<std::pair<int, int>>&& v) {
  using P = std::pair<int, int>;
  static_assert(sizeof(P) == 2 * sizeof(int) && std::is_standard_layout_v<P>,
                "std::pair<int,int> must be two adjacent ints to alias it as an (M,2) array");
  const std::size_t m = v.size();
  auto* held = new std::vector<P>(std::move(v));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<P>*>(p); });
  return nb::ndarray<nb::numpy, int>(reinterpret_cast<int*>(held->data()), {m, 2}, owner, {2, 1},
                                     nb::dtype<int>(), nb::device::cpu::value, 0);
}

// Per-pore / per-throat scalars -> 1-D float64 array.
static nb::ndarray<nb::numpy, double> doubles_to_ndarray(std::vector<double>&& v) {
  const std::size_t n = v.size();
  return peclet::core::python::vector_to_ndarray<double>(std::move(v), {n}, {1});
}

// The network-flow result as the documented dict (pores: list[Pore]; everything else ndarray).
static nb::dict network_flow_dict(pnm::NetworkFlow&& net) {
  nb::dict d;
  d["pores"] = net.pores;
  d["pore_pressure"] = doubles_to_ndarray(std::move(net.pore_pressure));
  d["pore_residual"] = doubles_to_ndarray(std::move(net.pore_residual));
  d["throats"] = pairs_to_ndarray(std::move(net.throats));
  d["throat_flow"] = doubles_to_ndarray(std::move(net.throat_flow));
  d["throat_area"] = doubles_to_ndarray(std::move(net.throat_area));
  d["throat_dp"] = doubles_to_ndarray(std::move(net.throat_dp));
  return d;
}

// Common docstring tail of extract_network_flow / extract_network_flow_mpi (nanobind keeps the
// doc POINTER, so the assembled strings below are function-local statics).
static constexpr const char* kNetworkFlowDictDoc =
    "\n\nReturns a dict: 'pores' (list[Pore], ordered by label: pores[k] is label k+1), "
    "'pore_pressure' and 'pore_residual' (float64 (N,)), 'throats' ((M,2) int32 label pairs "
    "a < b, per PATCH), 'throat_flow', 'throat_area' and 'throat_dp' (float64 (M,)).\n\n"
    "PRECONDITION (cross-repo, not checkable here): the openness arrays must be the face "
    "openness the velocity field was projected with. For peclet.flow's cut-cell IBM that means "
    "the solid was set with set_solid(..., cutcell_pressure=True) — without it every openness "
    "flow reports is 0, so every throat_flow and throat_area comes back 0; for the ghost-cell "
    "IBM (set_ghost_projection) pass get_ox_proj()/get_oy_proj()/get_oz_proj() instead.";

NB_MODULE(_pnm, m) {
  m.attr("__doc__") =
      "pnm — pore-network extraction from SDF geometry (Kokkos).\n\n"
      "Arrays are (Nz,Ny,Nx) C-order and every describing triple is z-y-x with a `_zyx` suffix. "
      "Precision: the SDF is float32 and the geometry kernels compute in float32; origin_zyx / "
      "spacing_zyx are doubles narrowed to float32 (pore centres and radii are float32 in the "
      "input "
      "unit system); the network-flow MAC fields (u, v, w, p, ox, oy, oz) are float64 and their "
      "sums are double. SDF sign: negative inside the solid, positive in the pore space.";
  // Kokkos init + the release-then-finalize atexit hook + finalize() + execution_space: the
  // suite-wide teardown pattern (peclet/core/python/kokkos_teardown.hpp). pnm's functions hold no
  // Kokkos state between calls and return host-vector-backed arrays, so its registry stays empty;
  // the atexit Kokkos::finalize is still REQUIRED on CUDA (else cudaErrorCudartUnloading at exit).
  peclet::core::python::install(m);

  // VTI reader (pure C++; sdf_reader.cpp). Returns (sdf_3d[nz,ny,nx], origin_zyx, spacing_zyx).
  nb::class_<SDFReader>(m, "SDFReader")
      .def_static(
          "read_vti",
          [](const std::string& filename) {
            auto* data = new SDFData(SDFReader::read_vti(filename));
            std::size_t shape[3]{(std::size_t)data->resolution[2], (std::size_t)data->resolution[1],
                                 (std::size_t)data->resolution[0]};
            std::vector<double> org{data->origin[2], data->origin[1], data->origin[0]};
            std::vector<double> spc{data->spacing[2], data->spacing[1], data->spacing[0]};
            // C-contiguous (nz,ny,nx) float array referencing the reader's buffer; the capsule
            // keeps the SDFData alive for exactly as long as Python holds the array.
            nb::capsule owner(data, [](void* p) noexcept { delete static_cast<SDFData*>(p); });
            nb::ndarray<nb::numpy, float> sdf_3d(data->sdf_values.data(), 3, shape, owner);
            return nb::make_tuple(sdf_3d, org, spc);
          },
          "Reads a VTI (VTK ImageData) SDF volume. Returns the tuple (sdf, origin_zyx, "
          "spacing_zyx), in that order: sdf a float32 (Nz,Ny,Nx) C-order array over the "
          "reader's buffer (no copy), origin_zyx and spacing_zyx the z-y-x lists of the grid "
          "origin and cell size (float64).");

  nb::class_<Pore>(m, "Pore",
                   "One detected pore: a strict local maximum of the SDF over its 26-neighbourhood "
                   "(periodic in all three directions). The centre is the peak voxel's position "
                   "plus a squared-SDF-weighted sub-voxel offset over the 3x3x3 stencil, in the "
                   "unit system of origin_zyx / spacing_zyx; the radius is the SDF value at the "
                   "peak, i.e. the physical inscribed-sphere radius. All fields are float32. "
                   "The centre is exposed as three self-named scalars x, y, z (not a triple), so "
                   "no axis order is implied and no _zyx suffix applies — the documented "
                   "exception to NAMING.md 1.7.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](Pore* p, float x, float y, float z, float radius) { new (p) Pore{x, y, z, radius}; },
          nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("radius"))
      .def_rw("x", &Pore::x, "Centre x coordinate (physical units of origin_zyx / spacing_zyx).")
      .def_rw("y", &Pore::y, "Centre y coordinate.")
      .def_rw("z", &Pore::z, "Centre z coordinate.")
      .def_rw("radius", &Pore::radius,
              "Inscribed-sphere radius = the SDF value at the peak voxel (physical units).")
      .def("__repr__", [](const Pore& p) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "Pore(x=%g, y=%g, z=%g, radius=%g)", (double)p.x,
                      (double)p.y, (double)p.z, (double)p.radius);
        return std::string(buf);
      });

  m.def(
      "extract_pores",
      [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> origin_zyx,
         std::vector<double> spacing_zyx) {
        std::array<int, 3> res;
        auto v = to_sdf(sdf, res);
        std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        return pnm::extract_pores_k(v, res, org, spc);
      },
      nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"),
      "Pore detection: every voxel with sdf > 0 that is a strict local maximum of the SDF over its "
      "26 neighbours (periodic wrap in x, y, z; ties broken towards the higher flat index) becomes "
      "a Pore. sdf is a float32 (Nz,Ny,Nx) C-order array; origin_zyx / spacing_zyx are the grid's "
      "z-y-x origin and cell size (narrowed to float32). Returns the list of Pore(x, y, z, radius) "
      "in the input unit system, in device-completion order (NOT sorted; sort by (z, y, x) for a "
      "reproducible order). Capped at 1e6 pores.");

  m.def(
      "segment_volume",
      [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> spacing_zyx) {
        std::array<int, 3> res;
        auto v = to_sdf(sdf, res);
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        return labels_to_ndarray(pnm::segment_volume_k(v, res, spc), res);
      },
      nb::arg("sdf"), nb::arg("spacing_zyx"),
      "Marker-controlled watershed segmentation of the SDF grid. Returns an int32 (Nz,Ny,Nx) "
      "C-order label array (sdf.shape; the same memory as the kernels' flat x-fastest label "
      "vector — no copy, no reshape needed): pore voxels (sdf > 0) carry the "
      "id of the pore basin they belong to, 1, 2, ... in first-encounter (flat-index) order of the "
      "basin peaks — the same peaks extract_pores finds — assigned by a gradient-ascent path to "
      "the local SDF maximum; solid voxels (sdf <= 0) carry the NEGATIVE id -1, -2, ... of their "
      "connected solid grain (26-connected components of the deep solid, sdf < -1.5 * min "
      "spacing, flooded outwards by a Jacobi min-label sweep), and 0 marks solid debris no grain "
      "reached. Periodic in all three directions. spacing_zyx only sets the deep-solid marker "
      "threshold.");

  m.def(
      "extract_topology",
      [](nb::ndarray<const int, nb::c_contig> segmentation,
         std::optional<std::vector<int>> shape_zyx) {
        std::array<int, 3> res;
        if (segmentation.ndim() == 3) {
          res = {(int)segmentation.shape(2), (int)segmentation.shape(1),
                 (int)segmentation.shape(0)};
          if (shape_zyx && (shape_zyx->size() != 3 || (*shape_zyx)[0] != res[2] ||
                            (*shape_zyx)[1] != res[1] || (*shape_zyx)[2] != res[0]))
            throw std::runtime_error("shape_zyx does not match the (Nz,Ny,Nx) segmentation");
        } else if (segmentation.ndim() == 1) {
          if (!shape_zyx)
            throw std::runtime_error("a flat segmentation needs shape_zyx = (Nz, Ny, Nx)");
          if (shape_zyx->size() != 3)
            throw std::runtime_error("shape_zyx must be (Nz, Ny, Nx)");
          res = {(*shape_zyx)[2], (*shape_zyx)[1], (*shape_zyx)[0]};
          if (segmentation.size() != std::size_t(res[0]) * res[1] * res[2])
            throw std::runtime_error("segmentation length does not match shape_zyx");
        } else {
          throw std::runtime_error("segmentation must be (Nz,Ny,Nx) or flat");
        }
        return pairs_to_ndarray(
            pnm::extract_topology_k(segmentation.data(), segmentation.size(), res));
      },
      nb::arg("segmentation"), nb::arg("shape_zyx") = nb::none(),
      "Label adjacency of a segment_volume result: the sorted, unique (a, b) pairs with a < b of "
      "labels that share a voxel face (+x, +y, +z, periodic wrap), as an (M,2) int32 array. "
      "segmentation is the int32 (Nz,Ny,Nx) array segment_volume returned (read in place, no "
      "copy); a flat x-fastest label vector is accepted too, then shape_zyx = (Nz, Ny, Nx) of "
      "the grid it was made on (= sdf.shape) is required. Rows with a label <= 0 are pore-solid "
      "(or grain-grain / debris) contacts; the pore-to-pore throats are the rows with both "
      "labels > 0. One row per label pair (a per-PATCH throat list, which can repeat a pair, is "
      "what extract_network_flow returns).");

  // Fused pipeline (F1): SDF uploaded once, segmentation device-resident across all three stages.
  m.def(
      "extract_pore_network",
      [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> origin_zyx,
         std::vector<double> spacing_zyx) {
        std::array<int, 3> res;
        auto v = to_sdf(sdf, res);
        std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        pnm::PoreNetwork net = pnm::extract_pore_network_k(v, res, org, spc);
        return nb::make_tuple(net.pores, labels_to_ndarray(std::move(net.seg), res),
                              pairs_to_ndarray(std::move(net.connections)));
      },
      nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"),
      "Fused extraction (SDF uploaded once, segmentation device-resident across stages): returns "
      "the tuple (pores, segmentation, connections), in that order, each element exactly what "
      "extract_pores, segment_volume and extract_topology return (list[Pore]; int32 (Nz,Ny,Nx) "
      "array; (M,2) int32 array).\n\n"
      "UNITS: everything is in the system `origin_zyx` / `spacing_zyx` and the SDF are stated in "
      "(the VTI's own, straight from SDFReader.read_vti) — pore centres are physical coordinates "
      "and a pore radius is the physical inscribed radius, not a voxel count. Pass "
      "spacing_zyx = [1,1,1] and origin_zyx = [0,0,0] to work in voxels. Measured on "
      "flow/data/packing_ring.vti: doubling the spacing, the origin and the SDF together leaves "
      "the pore count (7199) and the throat-connection count (53020) unchanged and doubles every "
      "radius BITWISE. One sub-voxel caveat: the centre's intra-cell offset is guarded by an "
      "absolute `sw > 1e-6` on a squared-distance weight sum, so a near-degenerate peak's centre "
      "can shift by a fraction of a cell (measured up to 0.37 cells) under a change of unit "
      "system. Radii, counts and topology are unaffected.");

  // Network flow: throat flow rates + pore-center pressures from a peclet.flow MAC field (the
  // method transferred from the Voronoi PNM). Pass flow's fields transposed to (Nz,Ny,Nx):
  // u = s.get_uf().T etc. (zero-copy view of the [x,y,z] Fortran array).
  static const std::string kNetworkFlowDoc =
      std::string(
          "Pore-network FLOW data from a MAC field on the same grid as the SDF: segments the SDF, "
          "then returns per-pore-label centers/pressures (trilinear p at the basin peak, periodic) "
          "and per-throat flow rates (sum of openness-weighted MAC face fluxes o*u*A over the "
          "label-interface faces; positive from the lower to the higher label). All arrays are "
          "(Nz,Ny,Nx) C-order on the SDF grid: pass flow's fields as get_uf().T, get_p().T, "
          "get_ox().T, ... u(i,j,k) is the -x face velocity of cell (i,j,k) (flow's MAC layout); "
          "omit ox/oy/oz for a fully open grid. grad_p_zyx adds the macroscopic gradient along "
          "the min-image pore-to-pore vector to throat_dp (= P_i - P_j, drives flow i->j when "
          "positive). pore_residual is the signed flux sum over each pore's whole boundary — "
          "~solver tolerance when u is flow's projected divergence-free field. Throats are "
          "PER-PATCH (a connected patch of interface faces): two disjoint interfaces between the "
          "same two pores are separate parallel throats, so the throat list can repeat a label "
          "pair.") +
      kNetworkFlowDictDoc;
  m.def(
      "extract_network_flow",
      [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> origin_zyx,
         std::vector<double> spacing_zyx, nb::ndarray<double, nb::c_contig> u,
         nb::ndarray<double, nb::c_contig> v, nb::ndarray<double, nb::c_contig> w,
         nb::ndarray<double, nb::c_contig> p, std::optional<nb::ndarray<double, nb::c_contig>> ox,
         std::optional<nb::ndarray<double, nb::c_contig>> oy,
         std::optional<nb::ndarray<double, nb::c_contig>> oz, std::vector<double> grad_p_zyx) {
        std::array<int, 3> res;
        auto sv = to_sdf(sdf, res);
        if ((bool)ox != (bool)oy || (bool)ox != (bool)oz)
          throw std::runtime_error("pass all three openness arrays (ox,oy,oz) or none");
        std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        std::array<double, 3> gp{grad_p_zyx[2], grad_p_zyx[1], grad_p_zyx[0]};
        auto net = pnm::extract_network_flow_k(
            sv, res, org, spc, to_field(u, res, "u"), to_field(v, res, "v"), to_field(w, res, "w"),
            to_field(p, res, "p"), ox ? to_field(*ox, res, "ox") : std::vector<double>{},
            oy ? to_field(*oy, res, "oy") : std::vector<double>{},
            oz ? to_field(*oz, res, "oz") : std::vector<double>{}, gp);
        return network_flow_dict(std::move(net));
      },
      nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"), nb::arg("u"), nb::arg("v"),
      nb::arg("w"), nb::arg("p"), nb::arg("ox") = nb::none(), nb::arg("oy") = nb::none(),
      nb::arg("oz") = nb::none(), nb::arg("grad_p_zyx") = std::vector<double>{0.0, 0.0, 0.0},
      kNetworkFlowDoc.c_str());

#ifdef PECLET_PNM_MPI
  // Distributed path (built with -DPECLET_PNM_MPI=ON): the SDF is decomposed over ranks by the
  // shared core ORB (same deterministic partition as flow/dem). Query this rank's block with
  // mpi_block, pass the LOCAL block's SDF to extract_pore_network_mpi. Bit-exact to single-rank.
  // Rank / size queries are mpi4py's job (MPI.COMM_WORLD.rank / .size), not this module's.
  m.def(
      "mpi_block",
      [](std::vector<int> global_shape_zyx) {
        ensure_mpi_init();
        if (global_shape_zyx.size() != 3)
          throw std::runtime_error("global_shape_zyx must be (Nz, Ny, Nx)");
        std::array<int, 3> gd{global_shape_zyx[2], global_shape_zyx[1], global_shape_zyx[0]};
        std::array<int, 3> o{}, s{};
        pnm::mpi_block_of(gd, MPI_COMM_WORLD, o, s);
        return nb::make_tuple(nb::make_tuple(o[2], o[1], o[0]), nb::make_tuple(s[2], s[1], s[0]));
      },
      nb::arg("global_shape_zyx"),
      "This rank's ORB block of the global (Nz,Ny,Nx) grid (collective; MPI_Init is called if "
      "needed): returns the tuple (offset_zyx, shape_zyx), in that order — two int 3-tuples, the "
      "block's first VOXEL index per axis and its voxel count per axis. offset_zyx is an integer "
      "grid offset, NOT the physical origin_zyx the extraction functions take (that stays the "
      "global grid's). Slice the global SDF as sdf[oz:oz+sz, oy:oy+sy, ox:ox+sx] and pass the "
      "block to extract_pore_network_mpi / extract_network_flow_mpi.");
  m.def(
      "extract_pore_network_mpi",
      [](nb::ndarray<float, nb::c_contig> sdf_local, std::vector<int> global_shape_zyx,
         std::vector<double> origin_zyx, std::vector<double> spacing_zyx) {
        ensure_mpi_init();
        std::array<int, 3> res;
        auto v = to_sdf(sdf_local, res);
        std::array<int, 3> gd{global_shape_zyx[2], global_shape_zyx[1], global_shape_zyx[0]};
        std::array<int, 3> bo{}, bs{};
        pnm::mpi_block_of(gd, MPI_COMM_WORLD, bo, bs);
        if (res != bs)
          throw std::runtime_error("sdf_local shape does not match this rank's mpi_block");
        std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        auto net = pnm::extract_pore_network_mpi(v, gd, org, spc, MPI_COMM_WORLD);
        return nb::make_tuple(net.pores, labels_to_ndarray(std::move(net.seg), res),
                              pairs_to_ndarray(std::move(net.connections)));
      },
      nb::arg("sdf_local"), nb::arg("global_shape_zyx"), nb::arg("origin_zyx"),
      nb::arg("spacing_zyx"),
      "Distributed fused extraction (collective over MPI_COMM_WORLD). sdf_local is this rank's "
      "ORB block (Nz,Ny,Nx C-order, from mpi_block); origin_zyx / spacing_zyx are the GLOBAL "
      "grid's. Returns the tuple (pores, segmentation, connections), in that order: the pores "
      "whose peak this rank owns (list[Pore]), this rank's block of the segmentation (int32 "
      "array of sdf_local.shape, global label ids), and the GLOBAL connection list ((M,2) int32, "
      "identical on every rank). Bit-exact to the single-rank extract_pore_network on the "
      "gathered grid.");
  static const std::string kNetworkFlowMpiDoc =
      std::string(
          "Distributed network-flow extraction (collective over MPI_COMM_WORLD): all arrays are "
          "this rank's ORB block (Nz,Ny,Nx C-order, from mpi_block; MAC fields from a distributed "
          "peclet.flow run on the SAME BlockDecomposer). Returns the same dict as "
          "extract_network_flow, GLOBAL and identical on every rank.") +
      kNetworkFlowDictDoc;
  m.def(
      "extract_network_flow_mpi",
      [](nb::ndarray<float, nb::c_contig> sdf_local, std::vector<int> global_shape_zyx,
         std::vector<double> origin_zyx, std::vector<double> spacing_zyx,
         nb::ndarray<double, nb::c_contig> u, nb::ndarray<double, nb::c_contig> v,
         nb::ndarray<double, nb::c_contig> w, nb::ndarray<double, nb::c_contig> p,
         std::optional<nb::ndarray<double, nb::c_contig>> ox,
         std::optional<nb::ndarray<double, nb::c_contig>> oy,
         std::optional<nb::ndarray<double, nb::c_contig>> oz, std::vector<double> grad_p_zyx) {
        ensure_mpi_init();
        std::array<int, 3> res;
        auto sv = to_sdf(sdf_local, res);
        std::array<int, 3> gd{global_shape_zyx[2], global_shape_zyx[1], global_shape_zyx[0]};
        std::array<int, 3> bo{}, bs{};
        pnm::mpi_block_of(gd, MPI_COMM_WORLD, bo, bs);
        if (res != bs)
          throw std::runtime_error("sdf_local shape does not match this rank's mpi_block");
        if ((bool)ox != (bool)oy || (bool)ox != (bool)oz)
          throw std::runtime_error("pass all three openness arrays (ox,oy,oz) or none");
        std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        std::array<double, 3> gp{grad_p_zyx[2], grad_p_zyx[1], grad_p_zyx[0]};
        auto net = pnm::extract_network_flow_mpi(
            sv, gd, org, spc, to_field(u, res, "u"), to_field(v, res, "v"), to_field(w, res, "w"),
            to_field(p, res, "p"), ox ? to_field(*ox, res, "ox") : std::vector<double>{},
            oy ? to_field(*oy, res, "oy") : std::vector<double>{},
            oz ? to_field(*oz, res, "oz") : std::vector<double>{}, gp, MPI_COMM_WORLD);
        return network_flow_dict(std::move(net));
      },
      nb::arg("sdf_local"), nb::arg("global_shape_zyx"), nb::arg("origin_zyx"),
      nb::arg("spacing_zyx"), nb::arg("u"), nb::arg("v"), nb::arg("w"), nb::arg("p"),
      nb::arg("ox") = nb::none(), nb::arg("oy") = nb::none(), nb::arg("oz") = nb::none(),
      nb::arg("grad_p_zyx") = std::vector<double>{0.0, 0.0, 0.0}, kNetworkFlowMpiDoc.c_str());
#endif
}
