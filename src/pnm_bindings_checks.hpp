/// @file
/// @brief Input validation of the `peclet.pnm` Python binding (src/pnm_bindings.cpp only).
///
/// The extraction kernels (pore_extraction*.hpp) trust their caller; the binding refuses a z-y-x
/// triple of the wrong length, a NaN/Inf, a non-positive cell size or a negative count as
/// ValueError (std::invalid_argument) before it reaches them. A valid value takes exactly the path
/// it took before. The message names the function, the parameter, the admissible range with its
/// dimension (pnm works in the caller's consistent length units, those of the SDF) and the
/// offending value. Checks that predate these (array rank, shape matching) keep their RuntimeError.
///
/// origin_zyx and spacing_zyx are narrowed to float32 before the kernels, so they are checked as
/// float32: 1e39 is not finite there and 1e-50 is not > 0. Binding-only: no kernel header includes
/// this file.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pore_kernels.hpp"  // pnm::Pore

namespace pnm::binding_checks {

inline std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
inline void require(bool ok, const char* who, const std::string& name, const std::string& rule,
                    double v) {
  if (!ok) {
    throw std::invalid_argument(std::string(who) + ": " + name + " must be " + rule + ", got " +
                                num(v));
  }
}
inline std::string component(const char* name, std::size_t i) {
  return std::string(name) + "[" + std::to_string(i) + "]";
}

/// A z-y-x triple: exactly three values.
inline void triple(const char* who, const char* name, std::size_t size) {
  if (size != 3) {
    throw std::invalid_argument(std::string(who) + ": " + name +
                                " must be 3 values (z, y, x), got " + std::to_string(size));
  }
}
/// Each component finite once narrowed to float32 (a coordinate: any sign).
inline void finiteEach(const char* who, const char* name, const std::vector<double>& a,
                       const char* dim) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(std::isfinite(static_cast<float>(a[i])), who, component(name, i),
            std::string("finite as float32 (") + dim + ")", a[i]);
  }
}
/// Each component finite and > 0 once narrowed to float32 (a cell size).
inline void positiveEach(const char* who, const char* name, const std::vector<double>& a,
                         const char* dim) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float f = static_cast<float>(a[i]);
    require(std::isfinite(f) && f > 0.0f, who, component(name, i),
            std::string("finite and > 0 as float32 (") + dim + ")", a[i]);
  }
}
/// Each component an integer >= `lo` (a voxel count).
inline void countEach(const char* who, const char* name, const std::vector<int>& a, int lo) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(a[i] >= lo, who, component(name, i),
            "an integer >= " + std::to_string(lo) + " (voxels)", a[i]);
  }
}

// --- the checks of one binding entry point ------------------------------------------------

/// spacing_zyx: the grid's cell size per axis, three values > 0 (length units).
inline void spacingZyx(const char* who, const std::vector<double>& spacing) {
  triple(who, "spacing_zyx", spacing.size());
  positiveEach(who, "spacing_zyx", spacing, "length units");
}
/// origin_zyx + spacing_zyx of extract_pores / extract_pore_network(_mpi).
inline void grid(const char* who, const std::vector<double>& origin,
                 const std::vector<double>& spacing) {
  triple(who, "origin_zyx", origin.size());
  finiteEach(who, "origin_zyx", origin, "length units");
  spacingZyx(who, spacing);
}
/// The global grid of the ORB decomposition: at least one voxel per axis. `sized` = the length
/// was already checked (mpi_block's own RuntimeError predates this round).
inline void globalShapeZyx(const char* who, const std::vector<int>& shape, bool sized) {
  if (!sized) {
    triple(who, "global_shape_zyx", shape.size());
  }
  countEach(who, "global_shape_zyx", shape, 1);
}
/// extract_topology's shape_zyx, when given as three values: voxel counts >= 0 whose product
/// fits in std::size_t (a wrapped product could match the flat array's length). A wrong length
/// is left to the binding's own RuntimeError.
inline void topologyShape(const std::optional<std::vector<int>>& shape) {
  if (!shape || shape->size() != 3) {
    return;
  }
  countEach("extract_topology", "shape_zyx", *shape, 0);
  std::size_t n = 1;
  for (int s : *shape) {
    require(!__builtin_mul_overflow(n, static_cast<std::size_t>(s), &n), "extract_topology",
            "shape_zyx", "a grid whose voxel count fits in 64 bits", static_cast<double>(s));
  }
}

/// Pore(x, y, z, radius): a finite centre and a finite radius >= 0 (Pore() is all zeros).
inline void initPore(Pore* p, float x, float y, float z, float radius) {
  require(std::isfinite(x), "Pore", "x", "finite (length units)", x);
  require(std::isfinite(y), "Pore", "y", "finite (length units)", y);
  require(std::isfinite(z), "Pore", "z", "finite (length units)", z);
  require(std::isfinite(radius) && radius >= 0.0f, "Pore", "radius",
          "finite and >= 0 (length units)", radius);
  new (p) Pore{x, y, z, radius};
}

}  // namespace pnm::binding_checks
