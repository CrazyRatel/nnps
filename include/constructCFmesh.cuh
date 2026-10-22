#ifndef NNPS_CONSTRUCT_CFINDEXS_CUH
#define NNPS_CONSTRUCT_CFINDEXS_CUH

#include "MultiMesh.cuh"
#include "constant_memory.cuh"
#include "gpu_config.h"
#include "utilities.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

inline std::runtime_error cfIndexError(const char* operation, cudaError_t status) {
    return std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__device__ __forceinline__ int clampLevelDevice(int value, int lower, int upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

__global__ void computeLevelsKernel(
    const float* support_radius,
    const int particle_count,
    const float max_support,
    const int max_level,
    int* levels) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned>(particle_count)) {
        return;
    }

    if (support_radius[index] <= 0.0f) {
        levels[index] = 0;
        return;
    }

    const float ratio = max_support / support_radius[index];
    const float raw_level = log2f(ratio);
    const int level = static_cast<int>(floorf(raw_level));
    levels[index] = clampLevelDevice(level, 0, max_level);
}

__global__ void fillCFIndexKernel(
    const int* levels,
    const int particle_count,
    int* c_begin,
    int* c_storage,
    int* f_begin,
    int* f_storage,
    int level_upper_bound) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned>(particle_count)) {
        return;
    }

    const int level = levels[index];
    const int c_slot = atomicAdd(&c_begin[level], 1);
    c_storage[static_cast<std::size_t>(level) * particle_count + c_slot] = index;

    for (int fine_level = 0; fine_level < level; ++fine_level) {
        const int f_slot = atomicAdd(&f_begin[fine_level], 1);
        f_storage[static_cast<std::size_t>(fine_level) * particle_count + f_slot] = index;
    }

    (void)level_upper_bound;
}

__global__ void computeHashIndexKernel(
    const float* x_coord,
    const float* y_coord,
    const float* z_coord,
    const int* index_list,
    const int item_count,
    const int hash_table_size,
    int64_t* hash_values) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned>(item_count)) {
        return;
    }

    const int particle_index = index_list[index];
    if (particle_index < 0) {
        hash_values[index] = -1;
        return;
    }

    const float hx = x_coord[particle_index];
    const float hy = y_coord[particle_index];
    const float hz = z_coord[particle_index];

    const int64_t h = static_cast<int64_t>(floorf(hx * 1000.0f)) * P_1 +
                      static_cast<int64_t>(floorf(hy * 1000.0f)) * P_2 +
                      static_cast<int64_t>(floorf(hz * 1000.0f)) * P_3;

    hash_values[index] = llabs(h) % hash_table_size;
}

inline void createCFindexs(MultiMesh& multi_mesh) {
    if (multi_mesh.particles.particle_num == 0) {
        return;
    }
    if (multi_mesh.particles.particle_num > std::numeric_limits<unsigned>::max()) {
        throw std::length_error("C/F index: particle count exceeds unsigned range");
    }

    const unsigned particle_count = static_cast<unsigned>(multi_mesh.particles.particle_num);
    const std::size_t level_count = multi_mesh.cindexs.size();
    if (level_count == 0 || multi_mesh.findexs.size() != level_count) {
        throw std::logic_error("C/F index storage has invalid level count");
    }

    for (std::size_t level = 0; level < level_count; ++level) {
        multi_mesh.cindexs[level].assign(particle_count, -1);
        multi_mesh.findexs[level].assign(particle_count, -1);
    }

    float host_cmax = 1.0f;
    int host_lfine = std::max(0, multi_mesh.L_fine);
    cudaError_t cmax_status = cudaMemcpyFromSymbol(&host_cmax, cmax, sizeof(host_cmax));
    if (cmax_status != cudaSuccess) {
        throw cfIndexError("Read cmax from constant memory failed", cmax_status);
    }

    thrust::device_vector<int> levels(particle_count, 0);
    const unsigned block_size = BLOCKSIZE;
    const unsigned grid_size = (particle_count + block_size - 1) / block_size;

    computeLevelsKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(multi_mesh.particles.device_h.data()),
        static_cast<int>(particle_count),
        host_cmax,
        host_lfine,
        thrust::raw_pointer_cast(levels.data()));

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw cfIndexError("CF level kernel launch failed", status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw cfIndexError("CF level kernel execution failed", status);
    }

    thrust::device_vector<int> c_begin(multi_mesh.L_fine + 1, 0);
    thrust::device_vector<int> f_begin(multi_mesh.L_fine + 1, 0);
    thrust::device_vector<int> c_storage(
        static_cast<std::size_t>(multi_mesh.L_fine + 1) * particle_count, -1);
    thrust::device_vector<int> f_storage(
        static_cast<std::size_t>(multi_mesh.L_fine + 1) * particle_count, -1);

    fillCFIndexKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(levels.data()),
        static_cast<int>(particle_count),
        thrust::raw_pointer_cast(c_begin.data()),
        thrust::raw_pointer_cast(c_storage.data()),
        thrust::raw_pointer_cast(f_begin.data()),
        thrust::raw_pointer_cast(f_storage.data()),
        multi_mesh.L_fine);

    status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw cfIndexError("CF index fill kernel launch failed", status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw cfIndexError("CF index fill kernel execution failed", status);
    }

    std::vector<int> c_count_host(multi_mesh.L_fine + 1);
    std::vector<int> f_count_host(multi_mesh.L_fine + 1);
    thrust::copy(c_begin.begin(), c_begin.end(), c_count_host.begin());
    thrust::copy(f_begin.begin(), f_begin.end(), f_count_host.begin());

    for (int level = 0; level <= multi_mesh.L_fine; ++level) {
        const int c_valid_count = c_count_host[level];
        if (c_valid_count <= 0) {
            multi_mesh.cindexs[level].clear();
            multi_mesh.findexs[level].clear();
            continue;
        }

        multi_mesh.cindexs[level].resize(static_cast<std::size_t>(c_valid_count));
        thrust::copy_n(
            c_storage.begin() + static_cast<std::size_t>(level) * particle_count,
            c_valid_count,
            multi_mesh.cindexs[level].begin());

        const int f_valid_count = f_count_host[level];
        multi_mesh.findexs[level].resize(static_cast<std::size_t>(f_valid_count));
        thrust::copy_n(
            f_storage.begin() + static_cast<std::size_t>(level) * particle_count,
            f_valid_count,
            multi_mesh.findexs[level].begin());
    }
}

inline void calcCCompactMulti(MultiMesh& mesh, int level) {
    if (level < 0 || static_cast<std::size_t>(level) >= mesh.cindexs.size()) {
        return;
    }

    auto& c_values = mesh.cindexs[level];

    auto host_values = thrust::host_vector<int>(c_values);
    std::vector<int> valid_indices;
    valid_indices.reserve(host_values.size());
    for (std::size_t i = 0; i < host_values.size(); ++i) {
        if (host_values[i] >= 0) {
            valid_indices.push_back(static_cast<int>(host_values[i]));
        }
    }

    mesh.c_compact_begin_c[level].assign(valid_indices.begin(), valid_indices.end());
    mesh.c_compact_length_c[level].assign(valid_indices.size(), 1);
}

inline void calcHashIndexMulti(MultiMesh& mesh, int level) {
    if (level < 0 || static_cast<std::size_t>(level) >= mesh.cindexs.size()) {
        return;
    }

    const auto& item_count = mesh.c_compact_begin_c[level].size();
    mesh.hash_indexs_c[level].resize(item_count);
    mesh.hash_indexs_f[level].resize(item_count);

    if (item_count == 0) {
        return;
    }

    thrust::device_vector<int> compact_list(mesh.c_compact_begin_c[level]);
    thrust::device_vector<int64_t> hash_result(item_count, 0);

    computeHashIndexKernel<<<(item_count + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(
        thrust::raw_pointer_cast(mesh.particles.device_x_coord.data()),
        thrust::raw_pointer_cast(mesh.particles.device_y_coord.data()),
        thrust::raw_pointer_cast(mesh.particles.device_z_coord.data()),
        thrust::raw_pointer_cast(compact_list.data()),
        static_cast<int>(item_count),
        static_cast<int>(mesh.hash_table_size),
        thrust::raw_pointer_cast(hash_result.data()));

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw cfIndexError("Hash index kernel launch failed", status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw cfIndexError("Hash index execution failed", status);
    }

    mesh.hash_indexs_c[level] = hash_result;
    mesh.hash_indexs_f[level] = hash_result;
}

inline void resortCcompactByHashMulti(MultiMesh& mesh, int level) {
    if (level < 0 || static_cast<std::size_t>(level) >= mesh.cindexs.size()) {
        return;
    }

    const std::size_t item_count = mesh.hash_indexs_c[level].size();
    if (item_count == 0) {
        return;
    }

    thrust::device_vector<int> compact_list(mesh.c_compact_begin_c[level]);

    thrust::sort_by_key(
        mesh.hash_indexs_c[level].begin(),
        mesh.hash_indexs_c[level].end(),
        compact_list.begin());

        mesh.c_compact_begin_c[level].swap(compact_list);
}

inline void createHtableMulti(MultiMesh& mesh, int level) {
    if (level < 0 || static_cast<std::size_t>(level) >= mesh.h_begin_c.size()) {
        return;
    }

    const std::size_t hash_size = static_cast<std::size_t>(mesh.hash_table_size);
    thrust::fill(mesh.h_begin_c[level].begin(), mesh.h_begin_c[level].end(), -1);
    thrust::fill(mesh.h_length_c[level].begin(), mesh.h_length_c[level].end(), 0);
    thrust::fill(mesh.h_begin_f[level].begin(), mesh.h_begin_f[level].end(), -1);
    thrust::fill(mesh.h_length_f[level].begin(), mesh.h_length_f[level].end(), 0);

    if (mesh.hash_indexs_c[level].empty()) {
        return;
    }

    auto hash_values = thrust::host_vector<int64_t>(mesh.hash_indexs_c[level]);
    auto compact_values = thrust::host_vector<int>(mesh.c_compact_begin_c[level]);
    std::vector<int> h_begin(hash_size, -1);
    std::vector<int> h_length(hash_size, 0);
    for (std::size_t i = 0; i < hash_values.size(); ++i) {
        const int64_t bucket = hash_values[i] % static_cast<int64_t>(hash_size);
        if (bucket < 0) {
            continue;
        }
        const std::size_t bucket_index = static_cast<std::size_t>(bucket);
        if (h_begin[bucket_index] < 0) {
            h_begin[bucket_index] = compact_values[i];
        }
        h_length[bucket_index] += 1;
    }

    mesh.h_begin_c[level] = h_begin;
    mesh.h_length_c[level] = h_length;
}

#endif
