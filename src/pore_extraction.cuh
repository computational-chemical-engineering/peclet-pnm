/// @file
/// @brief Pore-network extraction from an SDF (interface).
#pragma once

#include "sdf_reader.h"
#include <tuple>
#include <vector>

#include <array>

struct Pore {
  float x, y, z;
  float radius;
};

// Extracts simply pore centers (local maxima)
std::vector<Pore> extract_pores_gpu(const SDFData &sdf);

// Performs full segmentation
// Returns:
// 1. std::vector<int> segmentation_grid (flattened, same size as SDF)
//    - values > 0 are Pore IDs
//    - values < 0 are Solid IDs
//    - values == 0 ? Boundary? (Usually we partition everything)
std::vector<int> segment_volume_gpu(const SDFData &sdf);

// Extracts topology from segmentation
// Returns list of unique connections (ID_A, ID_B)
std::vector<std::pair<int, int>>
extract_topology_gpu(const std::vector<int> &segmentation,
                     const std::array<int, 3> &resolution);
