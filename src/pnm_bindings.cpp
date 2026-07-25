/// @file
/// @brief nanobind module `pnm` — Kokkos pore-network extraction from SDF geometry.
///
/// Matches the numpy convention: SDF is (Nz,Ny,Nx) C-order, origin/spacing are zyx. VTI reading
/// (SDFReader) is pure C++ (sdf_reader.cpp, backend-free); the pore/segmentation/topology compute
/// is the Kokkos GPU port. Exposes `SDFReader`, `extract_pores`, `segment_volume`,
/// `extract_topology_gpu`. A C-order (Nz,Ny,Nx) buffer is contiguous x-fastest, so it maps onto the
/// solver's flat layout directly via the shared bridge (peclet::core::python, core).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>  // optional openness arrays (extract_network_flow)
#include <nanobind/stl/pair.h>      // std::pair conversion (topology connections)
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

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

NB_MODULE(_pnm, m) {
  m.attr("__doc__") = "pnm — pore-network extraction from SDF geometry (Kokkos)";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // atexit Kokkos::finalize is REQUIRED on CUDA (else cudaErrorCudartUnloading at exit when
  // Kokkos's device state outlives the CUDA runtime). pnm returns host-vector-backed arrays, so
  // finalize is always clean here. See peclet-flow's flow_bindings.cpp.
  nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  }));
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());

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
          "Reads VTI; returns (sdf_3d[nz,ny,nx], origin_zyx, spacing_zyx)");

  nb::class_<Pore>(m, "Pore")
      .def_rw("x", &Pore::x)
      .def_rw("y", &Pore::y)
      .def_rw("z", &Pore::z)
      .def_rw("radius", &Pore::radius);

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
      nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"));

  m.def(
      "segment_volume",
      [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> spacing_zyx) {
        std::array<int, 3> res;
        auto v = to_sdf(sdf, res);
        std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1],
                                 (float)spacing_zyx[0]};
        return pnm::segment_volume_k(v, res, spc);
      },
      nb::arg("sdf"), nb::arg("spacing_zyx"));

  m.def(
      "extract_topology_gpu",
      [](std::vector<int> segmentation, std::vector<int> shape_zyx) {
        std::array<int, 3> res{shape_zyx[2], shape_zyx[1], shape_zyx[0]};
        return pnm::extract_topology_k(segmentation, res);
      },
      nb::arg("segmentation"), nb::arg("shape"));

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
        return nb::make_tuple(net.pores, net.seg, net.connections);
      },
      nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"),
      "Fused extraction (SDF uploaded once, segmentation device-resident across stages): returns "
      "(pores, segmentation_flat, connections).");

  // Network flow: throat flow rates + pore-center pressures from a peclet.flow MAC field (the
  // method transferred from the Voronoi PNM). Pass flow's fields transposed to (Nz,Ny,Nx):
  // u = s.get_uf().T etc. (zero-copy view of the [x,y,z] Fortran array).
  m.def(
      "extract_network_flow",
      [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> origin_zyx,
         std::vector<double> spacing_zyx, nb::ndarray<double, nb::c_contig> u,
         nb::ndarray<double, nb::c_contig> v, nb::ndarray<double, nb::c_contig> w,
         nb::ndarray<double, nb::c_contig> p,
         std::optional<nb::ndarray<double, nb::c_contig>> ox,
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
            sv, res, org, spc, to_field(u, res, "u"), to_field(v, res, "v"),
            to_field(w, res, "w"), to_field(p, res, "p"),
            ox ? to_field(*ox, res, "ox") : std::vector<double>{},
            oy ? to_field(*oy, res, "oy") : std::vector<double>{},
            oz ? to_field(*oz, res, "oz") : std::vector<double>{}, gp);
        nb::dict d;
        d["pores"] = net.pores;
        d["pore_pressure"] = net.pore_pressure;
        d["pore_residual"] = net.pore_residual;
        d["throats"] = net.throats;
        d["throat_flow"] = net.throat_flow;
        d["throat_area"] = net.throat_area;
        d["throat_dp"] = net.throat_dp;
        return d;
      },
      nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"), nb::arg("u"), nb::arg("v"),
      nb::arg("w"), nb::arg("p"), nb::arg("ox") = nb::none(), nb::arg("oy") = nb::none(),
      nb::arg("oz") = nb::none(), nb::arg("grad_p_zyx") = std::vector<double>{0.0, 0.0, 0.0},
      "Pore-network FLOW data from a MAC field on the same grid as the SDF: segments the SDF, "
      "then returns per-pore-label centers/pressures (trilinear p at the basin peak, periodic) "
      "and per-throat flow rates (sum of openness-weighted MAC face fluxes o*u*A over the "
      "label-interface faces; positive from the lower to the higher label). All arrays are "
      "(Nz,Ny,Nx) C-order on the SDF grid: pass flow's fields as get_uf().T, get_p().T, "
      "get_ox().T, ... u(i,j,k) is the -x face velocity of cell (i,j,k) (flow's MAC layout); "
      "omit ox/oy/oz for a fully open grid. grad_p_zyx adds the macroscopic gradient along the "
      "min-image pore-to-pore vector to throat_dp (= P_i - P_j, drives flow i->j when positive). "
      "pore_residual is the signed flux sum over each pore's whole boundary — ~solver tolerance "
      "when u is flow's projected divergence-free field. Throats are PER-PATCH (a connected patch "
      "of interface faces): two disjoint interfaces between the same two pores are separate "
      "parallel throats, so the throat list can repeat a label pair.");

#ifdef PECLET_PNM_MPI
  // Distributed path (built with -DPECLET_PNM_MPI=ON): the SDF is decomposed over ranks by the
  // shared core ORB (same deterministic partition as flow/dem). Query this rank's block with
  // mpi_block, pass the LOCAL block's SDF to extract_pore_network_mpi. Bit-exact to single-rank.
  m.def(
      "mpi_rank",
      []() {
        ensure_mpi_init();
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
      },
      "This rank's index in MPI_COMM_WORLD (MPI_Init is called if needed).");
  m.def(
      "mpi_size",
      []() {
        ensure_mpi_init();
        int s = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
      },
      "Number of ranks in MPI_COMM_WORLD (MPI_Init is called if needed).");
  m.def(
      "mpi_block",
      [](std::vector<int> global_shape_zyx) {
        ensure_mpi_init();
        std::array<int, 3> gd{global_shape_zyx[2], global_shape_zyx[1], global_shape_zyx[0]};
        std::array<int, 3> o{}, s{};
        pnm::mpi_block_of(gd, MPI_COMM_WORLD, o, s);
        return nb::make_tuple(std::vector<int>{o[2], o[1], o[0]},
                              std::vector<int>{s[2], s[1], s[0]});
      },
      nb::arg("global_shape_zyx"),
      "This rank's ORB block of the global grid: (origin_zyx, shape_zyx). Slice the global SDF "
      "with these and pass the local block to extract_pore_network_mpi.");
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
        return nb::make_tuple(net.pores, net.seg, net.connections);
      },
      nb::arg("sdf_local"), nb::arg("global_shape_zyx"), nb::arg("origin_zyx"),
      nb::arg("spacing_zyx"),
      "Distributed fused extraction (collective over MPI_COMM_WORLD). sdf_local is this rank's "
      "ORB block (Nz,Ny,Nx C-order, from mpi_block). Returns (pores_owned_by_this_rank, "
      "segmentation_flat_local_block, connections_global). Bit-exact to the single-rank "
      "extract_pore_network on the gathered grid.");
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
            sv, gd, org, spc, to_field(u, res, "u"), to_field(v, res, "v"),
            to_field(w, res, "w"), to_field(p, res, "p"),
            ox ? to_field(*ox, res, "ox") : std::vector<double>{},
            oy ? to_field(*oy, res, "oy") : std::vector<double>{},
            oz ? to_field(*oz, res, "oz") : std::vector<double>{}, gp, MPI_COMM_WORLD);
        nb::dict d;
        d["pores"] = net.pores;
        d["pore_pressure"] = net.pore_pressure;
        d["pore_residual"] = net.pore_residual;
        d["throats"] = net.throats;
        d["throat_flow"] = net.throat_flow;
        d["throat_area"] = net.throat_area;
        d["throat_dp"] = net.throat_dp;
        return d;
      },
      nb::arg("sdf_local"), nb::arg("global_shape_zyx"), nb::arg("origin_zyx"),
      nb::arg("spacing_zyx"), nb::arg("u"), nb::arg("v"), nb::arg("w"), nb::arg("p"),
      nb::arg("ox") = nb::none(), nb::arg("oy") = nb::none(), nb::arg("oz") = nb::none(),
      nb::arg("grad_p_zyx") = std::vector<double>{0.0, 0.0, 0.0},
      "Distributed network-flow extraction (collective over MPI_COMM_WORLD): all arrays are this "
      "rank's ORB block (Nz,Ny,Nx C-order, from mpi_block; MAC fields from a distributed "
      "peclet.flow run on the SAME BlockDecomposer). Returns the same dict as "
      "extract_network_flow, GLOBAL and identical on every rank.");
#endif
}
