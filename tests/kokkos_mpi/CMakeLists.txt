# peclet-pnm — distributed (MPI) extraction tests: the multi-rank pipeline vs the single-rank
# oracle, bit-exact (seg per voxel, pores as a sorted set, connections list), np = 1, 2, 4.
#
# Normally built from the root CMake under -DPECLET_PNM_MPI=ON -DPECLET_PNM_BUILD_TESTS=ON (one
# tree per backend), reusing the module's resolved Kokkos + MPI targets and core include path.
# Also a standalone project:
#   cmake -S tests/kokkos_mpi -B build_kmpi \
#         -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
#         -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun     # FORCE mpirun: ParaView's mpiexec runs singletons
#   cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
# Extra launcher flags (e.g. --oversubscribe on a 2-core runner) go in -DMPIEXEC_PREFLAGS=...
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(pnm_kokkos_mpi LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  find_package(Kokkos CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
  include(PecletDeps)
  peclet_sibling_include(peclet-core "${PECLET_CORE_TAG}" "../../../core" PECLET_CORE_INCLUDE)
  enable_testing()
endif()

if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
endif()
if(NOT MPIEXEC_NUMPROC_FLAG)
  set(MPIEXEC_NUMPROC_FLAG -np)
endif()
separate_arguments(_pnm_mpiexec_preflags NATIVE_COMMAND "${MPIEXEC_PREFLAGS}")

foreach(t pnm_mpi pnm_flow_mpi)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                             "${PECLET_CORE_INCLUDE}")
  target_link_libraries(test_${t} PRIVATE Kokkos::kokkos MPI::MPI_CXX)
  foreach(np 1 2 4)
    add_test(NAME ${t}_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_pnm_mpiexec_preflags}
                     $<TARGET_FILE:test_${t}>)
    set_tests_properties(${t}_np${np} PROPERTIES PROCESSORS ${np})
  endforeach()
endforeach()
