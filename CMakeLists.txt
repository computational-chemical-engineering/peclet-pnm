cmake_minimum_required(VERSION 3.24)
project(peclet_pnm LANGUAGES CXX)

# peclet.pnm — pore-network extraction from SDF geometry (Kokkos GPU compute + the pure-C++ SDFReader
# VTI reader), as an importable Python module. Split out of peclet-flow (2026-07): the CFD solve lives
# in peclet.flow; this package is the "pnm_from_sdf" extraction feature on its own.
#
# The Kokkos backend (CUDA / HIP / OpenMP) is selected by the install prefix
# (extern/install/<backend>, built by ../tools/bootstrap_deps.sh), exactly like flow/dem.
#
# Build (nanobind is found via the active interpreter):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
#   cmake --build build -j        ->  build/peclet/pnm/_pnm.*.so   (import as peclet.pnm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Redistributable single-GPU CUDA wheel: libcudart is provided by the `nvidia-cuda-runtime` PyPI
# dependency (installed to site-packages/nvidia/cu13/lib), not the system. Mirrors peclet-flow.
option(PECLET_CUDA_RUNTIME_WHEEL "RPATH the module to the nvidia-cuda-runtime wheel's libcudart" OFF)

# Dependencies via the vendored PecletDeps helper: an installed Kokkos prefix + sibling checkout for
# the dev/suite build, or FetchContent-built Kokkos + fetched core headers for a self-contained
# sdist/wheel (cibuildwheel). See cmake/PecletDeps.cmake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PecletDeps)
peclet_require_kokkos()
peclet_require_nanobind()
peclet_sibling_include(peclet-core "${PECLET_TPX_TAG}" "../core" TPX_INC)

# pnm — pore-network extraction (Kokkos compute + pure-C++ SDFReader VTI reader).
# NB_STATIC: bundle nanobind's runtime into the module. NOMINSIZE: nanobind's -Os is rejected by nvcc
# since the Kokkos device sources compile as CXX through the launch compiler. The extension is
# assembled into the PEP-420 peclet namespace as `peclet.pnm` (private `_pnm` re-exported by
# peclet/pnm/__init__.py, kept as a plain file under packaging/ and staged into <build>/peclet/... so
# `PYTHONPATH=<build> python …` works in the dev loop too).
nanobind_add_module(pnm NB_STATIC NOMINSIZE src/pnm_bindings.cpp src/sdf_reader.cpp)
set_target_properties(pnm PROPERTIES OUTPUT_NAME _pnm        # -> peclet.pnm._pnm (NB_MODULE(_pnm))
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/pnm")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/pnm_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/pnm/__init__.py COPYONLY)
target_include_directories(pnm PRIVATE src "${TPX_INC}")
target_link_libraries(pnm PRIVATE Kokkos::kokkos)
# HIP/lld strictness: nanobind's --gc-sections can drop Kokkos SharedAllocationRecord<HIPSpace>
# vtables it only later finds referenced. Keep sections on HIP (nvcc/ld is lenient).
if(Kokkos_ENABLE_HIP)
  target_link_options(pnm PRIVATE -Wl,--no-gc-sections)
endif()

# Redistributable CUDA wheel: point the module at the nvidia-cuda-runtime wheel's libcudart via a
# relative $ORIGIN RPATH (peclet/pnm -> site-packages -> nvidia/cu13/lib).
if(PECLET_CUDA_RUNTIME_WHEEL)
  set_target_properties(pnm PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

# --- pip / scikit-build-core install rule -------------------------------------------------------
if(DEFINED SKBUILD)
  install(TARGETS pnm LIBRARY DESTINATION peclet/pnm COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/pnm_init.py
          DESTINATION peclet/pnm RENAME __init__.py COMPONENT python)
endif()
