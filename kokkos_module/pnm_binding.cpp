// cfd-gpu — pybind11 module for the Kokkos pore-network extraction (the canonical pnm_backend). Matches the
// numpy convention: SDF is (Nz,Ny,Nx) C-order, origin/spacing are zyx. VTI reading (SDFReader) is pure C++
// (sdf_reader.cpp, CUDA-free); the pore/segmentation/topology compute is the Kokkos GPU port.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <vector>

#include "pore_extraction_kokkos.hpp"
#include "sdf_reader.h"

namespace py = pybind11;
using pnmk::Pore;

PYBIND11_MODULE(pnm_kokkos, m) {
  m.doc() = "cfd-gpu pore-network extraction (Kokkos)";
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  py::module_::import("atexit").attr("register")(py::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }));
  m.attr("execution_space") = py::str(Kokkos::DefaultExecutionSpace::name());

  // VTI reader (pure C++; sdf_reader.cpp). Returns (sdf_3d[nz,ny,nx], origin_zyx, spacing_zyx).
  py::class_<SDFReader>(m, "SDFReader")
      .def_static(
          "read_vti",
          [](const std::string& filename) {
            auto* data = new SDFData(SDFReader::read_vti(filename));
            std::vector<py::ssize_t> shape{(py::ssize_t)data->resolution[2],
                                           (py::ssize_t)data->resolution[1],
                                           (py::ssize_t)data->resolution[0]};
            std::vector<double> org{data->origin[2], data->origin[1], data->origin[0]};
            std::vector<double> spc{data->spacing[2], data->spacing[1], data->spacing[0]};
            py::capsule free_when_done(data, [](void* f) { delete reinterpret_cast<SDFData*>(f); });
            auto sdf_3d = py::array_t<float>(shape, data->sdf_values.data(), free_when_done);
            return py::make_tuple(sdf_3d, org, spc);
          },
          "Reads VTI; returns (sdf_3d[nz,ny,nx], origin_zyx, spacing_zyx)");

  py::class_<Pore>(m, "Pore")
      .def_readwrite("x", &Pore::x).def_readwrite("y", &Pore::y)
      .def_readwrite("z", &Pore::z).def_readwrite("radius", &Pore::radius);

  auto to_sdf = [](py::array_t<float, py::array::c_style | py::array::forcecast> a,
                   std::array<int, 3>& res) {
    py::buffer_info b = a.request();
    if (b.ndim != 3) throw std::runtime_error("SDF array must be 3D (Nz,Ny,Nx)");
    res = {(int)b.shape[2], (int)b.shape[1], (int)b.shape[0]};  // (Nx,Ny,Nz)
    const float* p = static_cast<float*>(b.ptr);
    return std::vector<float>(p, p + (b.shape[0] * b.shape[1] * b.shape[2]));
  };

  m.def("extract_pores",
        [to_sdf](py::array_t<float, py::array::c_style | py::array::forcecast> sdf,
                 std::vector<double> origin_zyx, std::vector<double> spacing_zyx) {
          std::array<int, 3> res; auto v = to_sdf(sdf, res);
          std::array<float, 3> org{(float)origin_zyx[2], (float)origin_zyx[1], (float)origin_zyx[0]};
          std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1], (float)spacing_zyx[0]};
          return pnmk::extract_pores_k(v, res, org, spc);
        }, py::arg("sdf"), py::arg("origin_zyx"), py::arg("spacing_zyx"));

  m.def("segment_volume",
        [to_sdf](py::array_t<float, py::array::c_style | py::array::forcecast> sdf,
                 std::vector<double> spacing_zyx) {
          std::array<int, 3> res; auto v = to_sdf(sdf, res);
          std::array<float, 3> spc{(float)spacing_zyx[2], (float)spacing_zyx[1], (float)spacing_zyx[0]};
          return pnmk::segment_volume_k(v, res, spc);
        }, py::arg("sdf"), py::arg("spacing_zyx"));

  m.def("extract_topology_gpu",
        [](std::vector<int> segmentation, std::vector<int> shape_zyx) {
          std::array<int, 3> res{shape_zyx[2], shape_zyx[1], shape_zyx[0]};
          return pnmk::extract_topology_k(segmentation, res);
        }, py::arg("segmentation"), py::arg("shape"));
}
