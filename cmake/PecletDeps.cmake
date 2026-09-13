# PecletDeps.cmake — self-contained dependency provisioning for a peclet compute package.
#
# A peclet compute wheel must build both ways:
#   * DEV / suite build — Kokkos (+ArborX) come from an installed prefix on CMAKE_PREFIX_PATH
#     (../extern/install/<backend> from tools/bootstrap_deps.sh) and the sibling headers
#     (core, morton) from ../<sibling>/include. Fast; the developer workflow.
#   * SELF-CONTAINED sdist/wheel (cibuildwheel) — the umbrella and siblings are ABSENT (the build runs on
#     an isolated copy of this one repo), so everything is FetchContent-built at the suite-pinned
#     versions. This is what makes `pip install peclet-flow` produce a working OpenMP CPU wheel.
#
# Selection is automatic (prefix/sibling present -> use it; else fetch), and can be forced off/on with
# -DPECLET_VENDOR_DEPS=ON. Vendored Kokkos is OpenMP+Serial only (portable CPU wheel); GPU/MPI builds use
# the prefix path. Keep PECLET_*_TAG in lockstep with tools/bootstrap_deps.sh.
include_guard(GLOBAL)
include(FetchContent)

set(PECLET_KOKKOS_TAG "5.1.1" CACHE STRING "Vendored Kokkos git tag")
set(PECLET_ARBORX_TAG "v2.1"  CACHE STRING "Vendored ArborX git tag")
set(PECLET_CORE_TAG    "v1.0.0"  CACHE STRING "Vendored core git tag (headers)")
set(PECLET_MORTON_TAG "v1.0.0"  CACHE STRING "Vendored morton git tag (headers)")
option(PECLET_VENDOR_DEPS "Force FetchContent-build of Kokkos/ArborX/siblings (self-contained wheel)" OFF)

# Extra -D flags for the VENDORED Kokkos configure (wheel builds only; ignored when Kokkos comes
# from a prefix). The nested configure below is a separate CMake run, so nothing set on THIS command
# line reaches it -- and a platform whose OpenMP needs coaxing has no other way in. Measured needs
# (2026-09-13 wheel probe): Windows wants -DOpenMP_RUNTIME_MSVC=llvm, because MSVC's default
# /openmp is OpenMP 2.0 and Kokkos requires 3.0; a platform with no usable OpenMP at all takes
# -DKokkos_ENABLE_OPENMP=OFF and gets a Serial wheel (the later -D wins over the default below).
set(PECLET_VENDOR_KOKKOS_ARGS "" CACHE STRING "Extra -D flags for the vendored Kokkos configure")
# ...and from the ENVIRONMENT, which is how a wheel build actually reaches in: cibuildwheel sets
# environment variables, and nesting one -D inside another (CMAKE_ARGS="-DA=ON -DB=-DC=x") did not
# survive the trip on Windows -- the whole string arrived as a single argument, so B was never set
# and only A got a (nonsense, truthy) value. One variable, one job.
if(NOT PECLET_VENDOR_KOKKOS_ARGS AND DEFINED ENV{PECLET_VENDOR_KOKKOS_ARGS})
  set(PECLET_VENDOR_KOKKOS_ARGS "$ENV{PECLET_VENDOR_KOKKOS_ARGS}")
endif()
separate_arguments(PECLET_VENDOR_KOKKOS_ARGS NATIVE_COMMAND "${PECLET_VENDOR_KOKKOS_ARGS}")

# nanobind — found via the active interpreter (scikit-build-core supplies it as a build requirement),
# identical to the umbrella SuiteNanobind helper but vendored so an isolated sdist build needs no ../cmake.
# MUST be a macro: find_package(Python) sets variables nanobind reads at module-creation time in the
# caller's scope.
macro(peclet_require_nanobind)
  if(NOT COMMAND nanobind_add_module)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    if(NOT nanobind_DIR)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _peclet_nb OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _peclet_nb_rc)
      if(_peclet_nb_rc EQUAL 0 AND EXISTS "${_peclet_nb}")
        set(nanobind_DIR "${_peclet_nb}")
      endif()
    endif()
    find_package(nanobind CONFIG REQUIRED)
    message(STATUS "[peclet] nanobind from ${nanobind_DIR}")
  endif()
endmacro()

# Vendored Kokkos/ArborX are *installed* to a staging prefix (not added as bare build-tree subprojects):
# ArborX REQUIREs an installed Kokkos package config (`find_package(Kokkos ... CONFIG)`), which a
# FetchContent subproject target does not provide — so a subproject Kokkos makes ArborX's own
# find_package(Kokkos) fail. Instead we build+install each into a shared staging prefix exactly like
# tools/bootstrap_deps.sh, then find_package() both (satisfying our code AND ArborX). Built once (marker).
set(PECLET_STAGE_PREFIX "${CMAKE_BINARY_DIR}/_peclet_deps" CACHE PATH "Vendored-deps staging install prefix")

function(_peclet_stage_build name url tag)  # extra -D configure args via ARGN
  FetchContent_Declare(${name} GIT_REPOSITORY "${url}" GIT_TAG "${tag}" GIT_SHALLOW TRUE)
  FetchContent_GetProperties(${name})
  if(NOT ${name}_POPULATED)
    FetchContent_Populate(${name})
  endif()
  if(EXISTS "${PECLET_STAGE_PREFIX}/.peclet_${name}_installed")
    return()
  endif()
  message(STATUS "[peclet] building+installing ${name} ${tag} -> ${PECLET_STAGE_PREFIX}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${${name}_SOURCE_DIR}" -B "${${name}_BINARY_DIR}"
            -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${PECLET_STAGE_PREFIX}"
            "-DCMAKE_PREFIX_PATH=${PECLET_STAGE_PREFIX}" -DCMAKE_CXX_STANDARD=20
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON ${ARGN}
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "[peclet] ${name} configure failed (${_rc})")
  endif()
  # --config Release matters on a MULTI-CONFIG generator (Visual Studio), where CMAKE_BUILD_TYPE
  # above is ignored and `--build --target install` otherwise installs the DEBUG build. Linking that
  # into a Release extension is LNK2038, "RuntimeLibrary MDd_DynamicDebug doesn't match
  # MD_DynamicRelease" -- measured on the 2026-09-13 Windows wheel probe. Single-config generators
  # ignore the flag.
  execute_process(COMMAND ${CMAKE_COMMAND} --build "${${name}_BINARY_DIR}" --config Release
                          --target install --parallel
                  RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "[peclet] ${name} build/install failed (${_rc})")
  endif()
  file(WRITE "${PECLET_STAGE_PREFIX}/.peclet_${name}_installed" "")
endfunction()

# Kokkos — prefix if present (unless forced), else vendored OpenMP+Serial installed to the staging prefix.
macro(peclet_require_kokkos)
  if(NOT PECLET_VENDOR_DEPS)
    find_package(Kokkos CONFIG QUIET)
  endif()
  if(Kokkos_FOUND)
    message(STATUS "[peclet] Kokkos ${Kokkos_VERSION} from prefix (${Kokkos_DEVICES})")
  else()
    # Host-parallel backend: OpenMP where the toolchain HAS one, the C++ std::thread backend where
    # it does not -- never a hard -DKokkos_ENABLE_OPENMP=ON, which simply fails the build off Linux.
    # Kokkos requires OpenMP >= 3.0 and two mainstream toolchains cannot give it (measured,
    # 2026-09-13 wheel probe): AppleClang ships no OpenMP at all, and MSVC reports 2.0 whatever
    # runtime is selected, because it defines _OPENMP as 200203 even under /openmp:llvm -- the gate
    # is a macro version, not a capability.
    #
    # Kokkos::Threads needs no OpenMP runtime, so it costs no vendored libomp (Homebrew publishes
    # only a tahoe bottle, which would drag the macOS wheel floor from 11 to 26) and no asserted
    # version the build system cannot check. The suite is indifferent to which one it gets: exactly
    # one line in eight repositories names Kokkos::OpenMP, and it is #ifdef-guarded.
    #
    # Both are multi-threaded. `execution_space` on the installed module reports which you have, and
    # peclet::core::cpuBudget sizes the pool the same way for either.
    find_package(OpenMP 3.0 QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
      set(_peclet_kokkos_omp ON)
      set(_peclet_kokkos_thr OFF)
    else()
      set(_peclet_kokkos_omp OFF)
      set(_peclet_kokkos_thr ON)
      message(STATUS "[peclet] no OpenMP >= 3.0 for ${CMAKE_CXX_COMPILER_ID}; "
                     "vendoring Kokkos with the C++ threads backend instead")
    endif()
    message(STATUS "[peclet] vendored Kokkos: OPENMP=${_peclet_kokkos_omp} THREADS=${_peclet_kokkos_thr} "
                   "SERIAL=ON ${PECLET_VENDOR_KOKKOS_ARGS}")
    _peclet_stage_build(kokkos "https://github.com/kokkos/kokkos.git" "${PECLET_KOKKOS_TAG}"
                        -DKokkos_ENABLE_OPENMP=${_peclet_kokkos_omp}
                        -DKokkos_ENABLE_THREADS=${_peclet_kokkos_thr} -DKokkos_ENABLE_SERIAL=ON
                        ${PECLET_VENDOR_KOKKOS_ARGS})
    list(APPEND CMAKE_PREFIX_PATH "${PECLET_STAGE_PREFIX}")
    find_package(Kokkos CONFIG REQUIRED)
    message(STATUS "[peclet] vendored Kokkos ${Kokkos_VERSION} @ ${PECLET_STAGE_PREFIX}")
  endif()
endmacro()

# ArborX — header-only but REQUIREs an installed Kokkos config; built against + installed to the same prefix.
macro(peclet_require_arborx)
  if(NOT PECLET_VENDOR_DEPS)
    find_package(ArborX CONFIG QUIET)
  endif()
  if(NOT ArborX_FOUND)
    _peclet_stage_build(arborx "https://github.com/arborx/ArborX.git" "${PECLET_ARBORX_TAG}")
    list(APPEND CMAKE_PREFIX_PATH "${PECLET_STAGE_PREFIX}")
    find_package(ArborX CONFIG REQUIRED)
    message(STATUS "[peclet] vendored ArborX ${ArborX_VERSION} @ ${PECLET_STAGE_PREFIX}")
  endif()
endmacro()

# Sibling header include dir (core / morton). Returns the sibling checkout if present, else a
# FetchContent-fetched source tree's include/ (header-only — declared but not built).
function(peclet_sibling_include repo tag sibling_reldir outvar)
  set(_local "${CMAKE_CURRENT_SOURCE_DIR}/${sibling_reldir}/include")
  if(EXISTS "${_local}" AND NOT PECLET_VENDOR_DEPS)
    set(${outvar} "${_local}" PARENT_SCOPE)
    return()
  endif()
  string(TOLOWER "peclet_sib_${repo}" _name)
  FetchContent_Declare(${_name}
    GIT_REPOSITORY "https://github.com/computational-chemical-engineering/${repo}.git"
    GIT_TAG ${tag} GIT_SHALLOW TRUE)
  FetchContent_GetProperties(${_name})
  if(NOT ${_name}_POPULATED)
    FetchContent_Populate(${_name})
  endif()
  set(${outvar} "${${_name}_SOURCE_DIR}/include" PARENT_SCOPE)
  message(STATUS "[peclet] vendored ${repo} headers -> ${${_name}_SOURCE_DIR}/include")
endfunction()
