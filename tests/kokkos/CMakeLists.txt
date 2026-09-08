# peclet-pnm — single-rank extraction tests (any Kokkos backend, no MPI).
#
# Normally built from the root CMake under -DPECLET_PNM_BUILD_TESTS=ON (one tree per backend),
# reusing the module's resolved Kokkos target + core include path. Also a standalone project:
#   cmake -S tests/kokkos -B build_kokkos -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>"
#   cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(pnm_kokkos_tests LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  find_package(Kokkos CONFIG REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
  include(PecletDeps)
  peclet_sibling_include(peclet-core "${PECLET_CORE_TAG}" "../../../core" PECLET_CORE_INCLUDE)
  enable_testing()
endif()

add_executable(test_pnm test_pnm.cpp)
target_include_directories(test_pnm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           "${PECLET_CORE_INCLUDE}")
target_link_libraries(test_pnm PRIVATE Kokkos::kokkos)
add_test(NAME pnm_single_rank COMMAND test_pnm)
