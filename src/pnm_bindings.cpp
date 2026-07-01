/// @file
/// @brief nanobind module `pnm` — Kokkos pore-network extraction from SDF geometry.
///
/// Matches the numpy convention: SDF is (Nz,Ny,Nx) C-order, origin/spacing are zyx. VTI reading
/// (SDFReader) is pure C++ (sdf_reader.cpp, backend-free); the pore/segmentation/topology compute is
/// the Kokkos GPU port. Exposes `SDFReader`, `extract_pores`, `segment_volume`, `extract_topology_gpu`.
/// A C-order (Nz,Ny,Nx) buffer is contiguous x-fastest, so it maps onto the solver's flat layout
/// directly via the shared bridge (peclet::core::python, core).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>  // std::pair conversion (topology connections)
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <vector>

#include "pore_extraction.hpp"
#include "sdf_reader.h"
#include "peclet/core/python/ndarray_interop.hpp"

namespace nb = nanobind;
using pnm::Pore;

// C-order (Nz,Ny,Nx) float SDF -> flat x-fastest vector + res = (Nx,Ny,Nz).
static std::vector<float> to_sdf(nb::ndarray<float, nb::c_contig> a, std::array<int, 3>& res) {
  if (a.ndim() != 3) throw std::runtime_error("SDF array must be 3D (Nz,Ny,Nx)");
  res = {(int)a.shape(2), (int)a.shape(1), (int)a.shape(0)};
  return peclet::core::python::ndarray_to_vector<float>(nb::ndarray<>(a));
}

NB_MODULE(_pnm, m) {
  m.attr("__doc__") = "pnm — pore-network extraction from SDF geometry (Kokkos)";
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  // atexit Kokkos::finalize is REQUIRED on CUDA (else cudaErrorCudartUnloading at exit when Kokkos's
  // device state outlives the CUDA runtime). pnm returns host-vector-backed arrays, so finalize is
  // always clean here. See flow_bindings.cpp.
  nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
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
            // C-contiguous (nz,ny,nx) float array referencing the reader's buffer; the capsule keeps
            // the SDFData alive for exactly as long as Python holds the array.
            nb::capsule owner(data, [](void* p) noexcept { delete static_cast<SDFData*>(p); });
            nb::ndarray<nb::numpy, float> sdf_3d(data->sdf_values.data(), 3, shape, owner);
            return nb::make_tuple(sdf_3d, org, spc);
          },
          "Reads VTI; returns (sdf_3d[nz,ny,nx], origin_zyx, spacing_zyx)");

  nb::class_<Pore>(m, "Pore")
      .def_rw("x", &Pore::x).def_rw("y", &Pore::y)
      .def_rw("z", &Pore::z).def_rw("radius", &Pore::radius);

  m.def("extract_pores",
        [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> origin_zyx,
           std::vector<double> spacing_zyx) {
          std::array<int, 3> res; auto v = to_sdf(sdf, res);
          std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
          std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1], (float)spacing_zyx[0]};
          return pnm::extract_pores_k(v, res, org, spc);
        }, nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"));

  m.def("segment_volume",
        [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> spacing_zyx) {
          std::array<int, 3> res; auto v = to_sdf(sdf, res);
          std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1], (float)spacing_zyx[0]};
          return pnm::segment_volume_k(v, res, spc);
        }, nb::arg("sdf"), nb::arg("spacing_zyx"));

  m.def("extract_topology_gpu",
        [](std::vector<int> segmentation, std::vector<int> shape_zyx) {
          std::array<int, 3> res{shape_zyx[2], shape_zyx[1], shape_zyx[0]};
          return pnm::extract_topology_k(segmentation, res);
        }, nb::arg("segmentation"), nb::arg("shape"));

  // Fused pipeline (F1): SDF uploaded once, segmentation device-resident across all three stages.
  m.def("extract_pore_network",
        [](nb::ndarray<float, nb::c_contig> sdf, std::vector<double> origin_zyx,
           std::vector<double> spacing_zyx) {
          std::array<int, 3> res; auto v = to_sdf(sdf, res);
          std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
          std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1], (float)spacing_zyx[0]};
          pnm::PoreNetwork net = pnm::extract_pore_network_k(v, res, org, spc);
          return nb::make_tuple(net.pores, net.seg, net.connections);
        }, nb::arg("sdf"), nb::arg("origin_zyx"), nb::arg("spacing_zyx"),
        "Fused extraction (SDF uploaded once, segmentation device-resident across stages): returns "
        "(pores, segmentation_flat, connections).");
}
