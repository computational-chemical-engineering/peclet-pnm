# peclet-pnm — distributed (MPI) extraction tests: the multi-rank pipeline vs the single-rank
# oracle, bit-exact (seg per voxel, pores as a sorted set, connections list), np = 1, 2, 4.
#
# Standalone find_package project (like flow's tests/kokkos_mpi): Kokkos + MPI + the header-only
# core + the pnm headers. Build:
#   cmake -S tests/kokkos_mpi -B build_kmpi \
#         -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
#         -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
#   cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(pnm_kokkos_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(MPI REQUIRED)

set(TPX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core" CACHE PATH "core repo")
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
endif()

enable_testing()
foreach(t pnm_mpi pnm_flow_mpi)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${TPX_DIR}/include)
  target_link_libraries(test_${t} PRIVATE Kokkos::kokkos MPI::MPI_CXX)
  foreach(np 1 2 4)
    add_test(NAME ${t}_np${np} COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_${t}>)
  endforeach()
endforeach()
