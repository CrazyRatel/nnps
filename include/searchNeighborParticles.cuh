#ifndef NNPS_SEARCH_NEIGHBOR_PARTICLES_CUH
#define NNPS_SEARCH_NEIGHBOR_PARTICLES_CUH

#include "MultiMesh.cuh"
#include "Points.cuh"
#include "gpu_config.h"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

__global__ void searchNeighborParticlesKernel(
    const float* x_coord,
    const float* y_coord,
    const float* z_coord,
    const float* support_h,
    const int* particle_levels,
    const int* candidate_particle_ids,
    const int* candidate_levels,
    const int candidate_count,
    const int particle_count,
    unsigned* neighbor_count) {
    const int particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const float x_i = x_coord[particle_index];
    const float y_i = y_coord[particle_index];
    const float z_i = z_coord[particle_index];
    const float radius_sq = support_h[particle_index] * support_h[particle_index] * 1.5f * 1.5f;
    const int level_i = particle_levels[particle_index];

    unsigned count = 0;
    for (int k = 0; k < candidate_count; ++k) {
        const int candidate_particle = candidate_particle_ids[k];
        if (candidate_particle < 0 || candidate_particle == particle_index ||
            candidate_particle >= particle_count) {
            continue;
        }

        const int candidate_level = candidate_levels[k];
        if (std::abs(level_i - candidate_level) > 1) {
            continue;
        }

        const float dx = x_i - x_coord[candidate_particle];
        const float dy = y_i - y_coord[candidate_particle];
        const float dz = z_i - z_coord[candidate_particle];
        const float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq <= radius_sq) {
            ++count;
        }
    }

    neighbor_count[particle_index] = count;
}

inline void searchNeighborParticles(MultiMesh& mesh) {
    if (mesh.particles.particle_num == 0) {
        return;
    }

    const std::size_t particle_count = mesh.particles.particle_num;
    mesh.particles.host_neighbor_num.resize(particle_count, 0);
    mesh.particles.device_neighbor_num.resize(particle_count, 0);

    std::vector<std::pair<int, int>> candidate_pairs;
    candidate_pairs.reserve(static_cast<std::size_t>(particle_count) * 4);

    for (int level = 0; level <= mesh.L_fine; ++level) {
        if (level < 0 || static_cast<std::size_t>(level) >= mesh.cindexs.size()) {
            continue;
        }

        const auto c_host = thrust::host_vector<int>(mesh.cindexs[level]);
        for (std::size_t i = 0; i < c_host.size(); ++i) {
            const int particle_index = c_host[i];
            if (particle_index >= 0) {
                candidate_pairs.emplace_back(particle_index, level);
            }
        }

        const auto f_host = thrust::host_vector<int>(mesh.findexs[level]);
        for (std::size_t i = 0; i < f_host.size(); ++i) {
            const int particle_index = f_host[i];
            if (particle_index >= 0) {
                candidate_pairs.emplace_back(particle_index, level);
            }
        }
    }

    std::sort(candidate_pairs.begin(), candidate_pairs.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });

    candidate_pairs.erase(
        std::unique(candidate_pairs.begin(), candidate_pairs.end()),
        candidate_pairs.end());

    std::vector<int> candidate_particle_ids;
    std::vector<int> candidate_levels;
    candidate_particle_ids.reserve(candidate_pairs.size());
    candidate_levels.reserve(candidate_pairs.size());

    for (const auto& pair : candidate_pairs) {
        candidate_particle_ids.push_back(pair.first);
        candidate_levels.push_back(pair.second);
    }

    thrust::device_vector<int> d_candidate_particle_ids(candidate_particle_ids.begin(), candidate_particle_ids.end());
    thrust::device_vector<int> d_candidate_levels(candidate_levels.begin(), candidate_levels.end());
    thrust::device_vector<unsigned> d_neighbor_count(particle_count, 0);

    const unsigned block_size = BLOCKSIZE;
    const unsigned grid_size = static_cast<unsigned>((particle_count + block_size - 1) / block_size);

    searchNeighborParticlesKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(mesh.particles.device_x_coord.data()),
        thrust::raw_pointer_cast(mesh.particles.device_y_coord.data()),
        thrust::raw_pointer_cast(mesh.particles.device_z_coord.data()),
        thrust::raw_pointer_cast(mesh.particles.device_h.data()),
        thrust::raw_pointer_cast(mesh.particles.device_level.data()),
        thrust::raw_pointer_cast(d_candidate_particle_ids.data()),
        thrust::raw_pointer_cast(d_candidate_levels.data()),
        static_cast<int>(candidate_particle_ids.size()),
        static_cast<int>(particle_count),
        thrust::raw_pointer_cast(d_neighbor_count.data()));

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Neighbor search kernel launch failed: ") + cudaGetErrorString(status));
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Neighbor search kernel execution failed: ") + cudaGetErrorString(status));
    }

    mesh.particles.device_neighbor_num = d_neighbor_count;
    mesh.particles.host_neighbor_num.resize(particle_count, 0);
    thrust::copy(d_neighbor_count.begin(), d_neighbor_count.end(), mesh.particles.host_neighbor_num.begin());
}

#endif
