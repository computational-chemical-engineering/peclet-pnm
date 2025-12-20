#pragma once

#include <vector>
#include <string>
#include <array>
#include <iostream>

struct SDFData {
    std::vector<float> sdf_values; // Flattened SDF grid
    std::array<int, 3> resolution;
    std::array<float, 3> origin;
    std::array<float, 3> spacing;
    
    // Helper to get total number of elements
    size_t size() const {
        return resolution[0] * resolution[1] * resolution[2];
    }
};

class SDFReader {
public:
    static SDFData read_vti(const std::string& filename);
};
