/// @file
/// @brief Pore-network extraction from an SDF (implementation).
#include "pore_extraction.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      throw std::runtime_error("CUDA Error");                                  \
    }                                                                          \
  }

// --------------------------------------------------------
// Device Helper: 3D Indexing with Periodic Wrapping
// --------------------------------------------------------
__device__ int get_idx(int x, int y, int z, int3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

__device__ int3 get_coords(int idx, int3 res) {
  int x = idx % res.x;
  int y = (idx / res.x) % res.y;
  int z = idx / (res.x * res.y);
  return make_int3(x, y, z);
}

// --------------------------------------------------------
// Kernel: Extract Pores (Local Maxima)
// --------------------------------------------------------
__global__ void extract_pores_kernel(const float *__restrict__ sdf, int3 res,
                                     float3 origin, float3 spacing,
                                     Pore *__restrict__ pores_out,
                                     int *__restrict__ counter, int max_pores) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int current_idx = get_idx(idx_x, idx_y, idx_z, res);
  float current_val = sdf[current_idx];

  if (current_val <= 0.0f)
    return;

  bool is_peak = true;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        int neighbor_idx = get_idx(idx_x + dx, idx_y + dy, idx_z + dz, res);
        float neighbor_val = sdf[neighbor_idx];

        if (neighbor_val > current_val) {
          is_peak = false;
          break;
        }
        if (neighbor_val == current_val && neighbor_idx > current_idx) {
          is_peak = false;
          break;
        }
      }
      if (!is_peak)
        break;
    }
    if (!is_peak)
      break;
  }

  if (!is_peak)
    return;

  float sum_weight = 0.0f;
  float3 sum_w_pos = make_float3(0.0f, 0.0f, 0.0f);

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = idx_x + dx;
        int ny = idx_y + dy;
        int nz = idx_z + dz;

        int n_idx = get_idx(nx, ny, nz, res);
        float val = sdf[n_idx];

        float w = fmaxf(0.0f, val);
        w = w * w;

        sum_weight += w;
        sum_w_pos.x += dx * w;
        sum_w_pos.y += dy * w;
        sum_w_pos.z += dz * w;
      }
    }
  }

  float3 offset = make_float3(0.0f, 0.0f, 0.0f);
  if (sum_weight > 1e-6f) {
    offset.x = sum_w_pos.x / sum_weight;
    offset.y = sum_w_pos.y / sum_weight;
    offset.z = sum_w_pos.z / sum_weight;
  }

  float wx = origin.x + (idx_x + offset.x) * spacing.x;
  float wy = origin.y + (idx_y + offset.y) * spacing.y;
  float wz = origin.z + (idx_z + offset.z) * spacing.z;

  int old = atomicAdd(counter, 1);
  if (old < max_pores) {
    pores_out[old] = {wx, wy, wz, current_val};
  }
}

std::vector<Pore> extract_pores_gpu(const SDFData &sdf) {
  if (sdf.sdf_values.empty())
    return {};

  int3 res = {sdf.resolution[0], sdf.resolution[1], sdf.resolution[2]};
  float3 origin = {sdf.origin[0], sdf.origin[1], sdf.origin[2]};
  float3 spacing = {sdf.spacing[0], sdf.spacing[1], sdf.spacing[2]};

  size_t num_elements = sdf.size();

  float *d_sdf;
  CHECK_CUDA(cudaMalloc(&d_sdf, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_sdf, sdf.sdf_values.data(),
                        num_elements * sizeof(float), cudaMemcpyHostToDevice));

  int *d_counter;
  CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

  int max_pores = 1000000;
  Pore *d_pores;
  CHECK_CUDA(cudaMalloc(&d_pores, max_pores * sizeof(Pore)));

  dim3 threads(8, 8, 8);
  dim3 blocks((res.x + threads.x - 1) / threads.x,
              (res.y + threads.y - 1) / threads.y,
              (res.z + threads.z - 1) / threads.z);

  extract_pores_kernel<<<blocks, threads>>>(d_sdf, res, origin, spacing,
                                            d_pores, d_counter, max_pores);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  int h_count = 0;
  CHECK_CUDA(
      cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

  if (h_count > max_pores)
    h_count = max_pores;

  std::vector<Pore> pores(h_count);
  CHECK_CUDA(cudaMemcpy(pores.data(), d_pores, h_count * sizeof(Pore),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_sdf);
  cudaFree(d_counter);
  cudaFree(d_pores);
  return pores;
}

// --------------------------------------------------------
// SEGMENTATION KERNELS
// --------------------------------------------------------

// Step 1: Gradient Path Following
// Stores the index of the root/peak found for each voxel
__global__ void gradient_path_kernel(const float *__restrict__ sdf, int3 res,
                                     int *__restrict__ roots_out) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int current_idx = get_idx(idx_x, idx_y, idx_z, res);
  float start_val = sdf[current_idx];

  // Pores (> 0): Ascent. Solids (< 0): Descent.
  bool ascent = (start_val > 0.0f);

  // Simple iterative path following
  // Limit steps to avoid infinite loops (though loops shouldn't happen in
  // gradient field typically, but plateaus can cause them) Optimization: Check
  // if current is local extremal first.

  int walker = current_idx;
  int steps = 0;
  const int MAX_STEPS = 512; // Heuristic limit

  while (steps < MAX_STEPS) {
    int best_neighbor = walker;
    float best_val = sdf[walker];

    int3 w_coords = get_coords(walker, res);

    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0 && dz == 0)
            continue;

          int n_idx =
              get_idx(w_coords.x + dx, w_coords.y + dy, w_coords.z + dz, res);
          float n_val = sdf[n_idx];

          if (ascent) {
            if (n_val > best_val) {
              best_val = n_val;
              best_neighbor = n_idx;
            } else if (n_val == best_val && n_idx > best_neighbor) {
              // Tie breaking needed to strictly converge to one peak on a
              // plateau
              best_neighbor = n_idx;
            }
          } else { // Descent
            if (n_val < best_val) {
              best_val = n_val;
              best_neighbor = n_idx;
            } else if (n_val == best_val && n_idx > best_neighbor) {
              best_neighbor = n_idx;
            }
          }
        }
      }
    }

    if (best_neighbor == walker) {
      // Reached extremum
      roots_out[current_idx] = walker;
      return;
    }

    walker = best_neighbor;
    steps++;
  }

  // Fallback if max steps reached (should be rare)
  roots_out[current_idx] = walker;
}

// Step 2: Merge Solid Roots (Basin Merging with Threshold)
// Iterates over all voxels. If two adjacent voxels are both "deeply" solid (SDF
// < threshold), their corresponding roots are merged.
// --------------------------------------------------------
// MARKER-CONTROLLED WATERSHED KERNELS
// --------------------------------------------------------

// Step 1: Initialize Markers (Cores)
// If SDF < threshold, it's a marker (Label = Index). Else Label = -1.
__global__ void init_markers_kernel(const float *__restrict__ sdf,
                                    int *__restrict__ labels, int3 res,
                                    float threshold) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Markers are deep solid regions
  if (sdf[idx] < threshold) {
    labels[idx] = idx;
  } else {
    // -1: Unassigned Solid
    // But what about pores? SDF > 0.
    // We will mask pores later or ignore them?
    // Let's assume input to this pipeline is purely solid processing.
    labels[idx] = -1;
  }
}

// Step 2: Connected Component Labeling on Markers
// Merge connected marker voxels.
// Step 2: Connected Component Labeling on Markers
// Merge connected marker voxels. Uses 26-Connectivity (13 forward neighbors).
__global__ void merge_markers_kernel(int *__restrict__ labels, int3 res,
                                     bool *__restrict__ changed) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  int my_label = labels[idx];
  if (my_label == -1)
    return; // Not a marker

  // 13 Forward Neighbors for 26-connectivity
  int dz_list[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
  int dy_list[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0};
  int dx_list[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1};

  for (int i = 0; i < 13; ++i) {
    int n_idx = get_idx(idx_x + dx_list[i], idx_y + dy_list[i],
                        idx_z + dz_list[i], res);
    int n_label = labels[n_idx];

    if (n_label != -1 && my_label != n_label) {
      // Basic CCL merge
      int root_my = my_label;
      while (root_my != labels[root_my])
        root_my = labels[root_my];

      int root_n = n_label;
      while (root_n != labels[root_n])
        root_n = labels[root_n];

      if (root_my != root_n) {
        int r_small = min(root_my, root_n);
        int r_large = max(root_my, root_n);
        atomicMin(&labels[r_large], r_small);
        *changed = true;
      }
    }
  }
}

// Helper to flatten labels: Label[i] = Label[Label[i]]
__global__ void flatten_labels_kernel(int *__restrict__ labels, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int l = labels[idx];
    if (l != -1) {
      while (l != labels[l]) {
        l = labels[l];
      }
      labels[idx] = l;
    }
  }
}

// Step 3: Flood Fill Propagation
// Propagate labels to solid voxels (SDF < 0) that are not yet markers.
// Uses Full 26-Connectivity.
__global__ void propagate_flood_kernel(const float *__restrict__ sdf,
                                       int *__restrict__ labels, int3 res,
                                       bool *__restrict__ changed) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  float val = sdf[idx];
  if (val >= 0.0f)
    return; // Pore, ignore.

  int my_label = labels[idx];
  if (my_label != -1)
    return; // Already assigned/marker.

  int best_label = -1;

  // Check all 26 neighbors
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        int n_idx = get_idx(idx_x + dx, idx_y + dy, idx_z + dz, res);
        int n_label = labels[n_idx];

        if (n_label != -1) {
          // Pick smallest label
          if (best_label == -1 || n_label < best_label) {
            best_label = n_label;
          }
        }
      }
    }
  }

  if (best_label != -1) {
    labels[idx] = best_label;
    *changed = true;
  }
}

// Step 3: Final Assign (Placeholder)
__global__ void assign_final_labels_kernel(const int *__restrict__ roots,
                                           const int *__restrict__ root_labels,
                                           int *__restrict__ final_seg, int N) {
  // Unused
}

// Topology Extraction
// Pairs (ID_A, ID_B) where A and B are neighbors.
__global__ void
extract_boundary_pairs_kernel(const int *__restrict__ segmentation, int3 res,
                              int2 *__restrict__ pairs_out,
                              int *__restrict__ pair_count, int max_pairs) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  int my_label = segmentation[idx];

  int dx_list[] = {1, 0, 0};
  int dy_list[] = {0, 1, 0};
  int dz_list[] = {0, 0, 1};

  for (int i = 0; i < 3; ++i) {
    int n_idx = get_idx(idx_x + dx_list[i], idx_y + dy_list[i],
                        idx_z + dz_list[i], res);
    int n_label = segmentation[n_idx];

    if (my_label != n_label) {
      int l1 = min(my_label, n_label);
      int l2 = max(my_label, n_label);

      int old = atomicAdd(pair_count, 1);
      if (old < max_pairs) {
        pairs_out[old] = make_int2(l1, l2);
      }
    }
  }
}

std::vector<int> segment_volume_gpu(const SDFData &sdf) {
  if (sdf.sdf_values.empty())
    return {};

  int3 res = {sdf.resolution[0], sdf.resolution[1], sdf.resolution[2]};
  float3 spacing = {sdf.spacing[0], sdf.spacing[1], sdf.spacing[2]};
  size_t num_elements = sdf.size();

  // Marker Threshold: Deep inside the object.
  // Try -1.5 * min_spacing (approx 1-2 voxels deep). (Heuristic)
  float min_spacing = min(spacing.x, min(spacing.y, spacing.z));
  float marker_threshold = -1.5f * min_spacing;

  float *d_sdf;
  CHECK_CUDA(cudaMalloc(&d_sdf, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_sdf, sdf.sdf_values.data(),
                        num_elements * sizeof(float), cudaMemcpyHostToDevice));

  // Labels array. Will store Marker IDs and propagated labels.
  int *d_labels;
  CHECK_CUDA(cudaMalloc(&d_labels, num_elements * sizeof(int)));

  dim3 threads(8, 8, 8);
  dim3 blocks((res.x + threads.x - 1) / threads.x,
              (res.y + threads.y - 1) / threads.y,
              (res.z + threads.z - 1) / threads.z);

  // 1. Initialize Markers
  init_markers_kernel<<<blocks, threads>>>(d_sdf, d_labels, res,
                                           marker_threshold);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 2. CCL on Markers
  bool *d_changed;
  CHECK_CUDA(cudaMalloc(&d_changed, sizeof(bool)));
  bool h_changed = true;

  while (h_changed) {
    h_changed = false;
    CHECK_CUDA(cudaMemcpy(d_changed, &h_changed, sizeof(bool),
                          cudaMemcpyHostToDevice));

    merge_markers_kernel<<<blocks, threads>>>(d_labels, res, d_changed);
    CHECK_CUDA(cudaDeviceSynchronize());

    flatten_labels_kernel<<<(num_elements + 255) / 256, 256>>>(d_labels,
                                                               num_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_changed, d_changed, sizeof(bool),
                          cudaMemcpyDeviceToHost));
  }

  // 3. Propagate (Flood Fill) to rest of solids
  h_changed = true;
  while (h_changed) {
    h_changed = false;
    CHECK_CUDA(cudaMemcpy(d_changed, &h_changed, sizeof(bool),
                          cudaMemcpyHostToDevice));

    propagate_flood_kernel<<<blocks, threads>>>(d_sdf, d_labels, res,
                                                d_changed);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_changed, d_changed, sizeof(bool),
                          cudaMemcpyDeviceToHost));
  }

  std::vector<int> solid_seg(num_elements);
  CHECK_CUDA(cudaMemcpy(solid_seg.data(), d_labels, num_elements * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // 4. Extract Pores (Gradient Ascent)
  int *d_pore_roots;
  CHECK_CUDA(cudaMalloc(&d_pore_roots, num_elements * sizeof(int)));
  gradient_path_kernel<<<blocks, threads>>>(d_sdf, res, d_pore_roots);

  std::vector<int> pore_roots(num_elements);
  CHECK_CUDA(cudaMemcpy(pore_roots.data(), d_pore_roots,
                        num_elements * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(d_sdf);
  cudaFree(d_labels);
  cudaFree(d_pore_roots);
  cudaFree(d_changed);

  // 5. Combine and Renumber on Host
  std::vector<int> final_segmentation(num_elements);
  std::map<int, int> label_map;
  int next_pore = 1;
  int next_solid = -1; // Solids are negative

  for (size_t i = 0; i < num_elements; ++i) {
    float val = sdf.sdf_values[i];

    if (val > 0) {
      // Pore
      int r = pore_roots[i];
      if (label_map.find(r) == label_map.end())
        label_map[r] = next_pore++;
      final_segmentation[i] = label_map[r];
    } else {
      // Solid
      int l = solid_seg[i];
      if (l == -1) {
        // Debris / Boundary
        final_segmentation[i] = 0;
      } else {
        if (label_map.find(l) == label_map.end())
          label_map[l] = next_solid--;
        final_segmentation[i] = label_map[l];
      }
    }
  }

  return final_segmentation;
}

std::vector<std::pair<int, int>>
extract_topology_gpu(const std::vector<int> &segmentation, // Host vector
                     const std::array<int, 3> &resolution) {
  size_t num_elements = segmentation.size();

  int *d_seg;
  CHECK_CUDA(cudaMalloc(&d_seg, num_elements * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_seg, segmentation.data(), num_elements * sizeof(int),
                        cudaMemcpyHostToDevice));

  int max_pairs = num_elements * 3; // Theoretical max edges
  int2 *d_pairs;
  CHECK_CUDA(cudaMalloc(&d_pairs, max_pairs * sizeof(int2)));

  int *d_count;
  CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

  dim3 threads(8, 8, 8);
  dim3 blocks((resolution[0] + threads.x - 1) / threads.x,
              (resolution[1] + threads.y - 1) / threads.y,
              (resolution[2] + threads.z - 1) / threads.z);

  int3 res_dim = make_int3(resolution[0], resolution[1], resolution[2]);
  extract_boundary_pairs_kernel<<<blocks, threads>>>(d_seg, res_dim, d_pairs,
                                                     d_count, max_pairs);
  CHECK_CUDA(cudaDeviceSynchronize());

  int h_count = 0;
  CHECK_CUDA(
      cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

  if (h_count > max_pairs)
    h_count = max_pairs;

  // Sort and Unique to remove duplicates
  // Standard approach: Transfer to host due to complexity of tuple sort device
  // side without definition
  std::vector<int2> h_pairs(h_count);
  CHECK_CUDA(cudaMemcpy(h_pairs.data(), d_pairs, h_count * sizeof(int2),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_seg);
  cudaFree(d_pairs);
  cudaFree(d_count);

  std::vector<std::pair<int, int>> result;
  result.reserve(h_count);
  for (auto p : h_pairs) {
    result.push_back({p.x, p.y});
  }

  // Sort and Unique on Host
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());

  return result;
}
