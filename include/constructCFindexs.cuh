#ifndef NNPS_CONSTRUCT_CFINDEXS_CUH
#define NNPS_CONSTRUCT_CFINDEXS_CUH

#include "MultiMesh.cuh"
#include "gpu_config.h"
#include "utilities.cuh"

#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

inline std::runtime_error cfIndexError(const char* operation, cudaError_t status)
{
    return std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void createCFIndexKernel(
    int* levels,
    const float* support_radii,
    int** c_indices,
    int** f_indices,
    unsigned particle_count) {

    const unsigned particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    const int particle_level = min(max(static_cast<int>(floorf(log2f(cmax / support_radii[particle_index]))), 0), Lfine);
    levels[particle_index] = particle_level;
    c_indices[particle_level][particle_index] = particle_index;

    for (int level = 0; level < particle_level; ++level) {
        f_indices[level][particle_index] = particle_index;
    }
}

inline void compactCFIndexVector(thrust::device_vector<int>& indices,
                                 int* selected_count_device,
                                 thrust::device_vector<int>& temporary) {
    temporary = indices;

    void* temporary_storage = nullptr;
    std::size_t temporary_storage_bytes = 0;
    int* input = thrust::raw_pointer_cast(temporary.data());
    int* output = thrust::raw_pointer_cast(indices.data());

    cudaError_t status = cub::DeviceSelect::If(
        temporary_storage, temporary_storage_bytes, input, output,
        selected_count_device, indices.size(), NotMarker(-1));
    if (status != cudaSuccess) {
        throw cfIndexError("C/F index selection size query failed", status);
    }

    status = cudaMalloc(&temporary_storage, temporary_storage_bytes);
    if (status != cudaSuccess) {
        throw cfIndexError("C/F index temporary storage allocation failed", status);
    }

    status = cub::DeviceSelect::If(
        temporary_storage, temporary_storage_bytes, input, output,
        selected_count_device, indices.size(), NotMarker(-1));
    const cudaError_t free_status = cudaFree(temporary_storage);
    if (status != cudaSuccess) {
        throw cfIndexError("C/F index selection failed", status);
    }
    if (free_status != cudaSuccess) {
        throw cfIndexError("C/F index temporary storage release failed", free_status);
    }

    int selected_count = 0;
    status = cudaMemcpy(&selected_count, selected_count_device, sizeof(selected_count),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw cfIndexError("C/F index count copy failed", status);
    }
    indices.resize(static_cast<std::size_t>(selected_count));
}

inline void createCFindexs(MultiMesh& multi_mesh) {
    if (multi_mesh.particles.particle_num == 0) {
        return;
    }
    if (multi_mesh.particles.particle_num > std::numeric_limits<unsigned>::max()) {
        throw std::length_error("C/F index particle count is too large");
    }

    const unsigned particle_count = static_cast<unsigned>(
        multi_mesh.particles.particle_num);
    const std::size_t level_count = multi_mesh.cindexs.size();
    if (level_count == 0 || multi_mesh.findexs.size() != level_count) {
        throw std::logic_error("C/F index storage has an invalid level count");
    }

    std::vector<int*> c_indices_host(level_count);
    std::vector<int*> f_indices_host(level_count);
    for (std::size_t level = 0; level < level_count; ++level) {
        c_indices_host[level] = thrust::raw_pointer_cast(
            multi_mesh.cindexs[level].data());
        f_indices_host[level] = thrust::raw_pointer_cast(
            multi_mesh.findexs[level].data());
    }

    int** c_indices_device = nullptr;
    int** f_indices_device = nullptr;
    const std::size_t pointer_bytes = level_count * sizeof(int*);
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&c_indices_device), pointer_bytes);
    if (status != cudaSuccess) {
        throw cfIndexError("C index pointer allocation failed", status);
    }
    status = cudaMalloc(
        reinterpret_cast<void**>(&f_indices_device), pointer_bytes);
    if (status != cudaSuccess) {
        cudaFree(c_indices_device);
        throw cfIndexError("F index pointer allocation failed", status);
    }

    status = cudaMemcpy(c_indices_device, c_indices_host.data(), pointer_bytes,
                        cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(f_indices_device, f_indices_host.data(), pointer_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        cudaFree(c_indices_device);
        cudaFree(f_indices_device);
        throw cfIndexError("C/F index pointer copy failed", status);
    }

    const unsigned block_size = BLOCKSIZE;
    const unsigned grid_size =
        (particle_count + block_size - 1) / block_size;
    createCFIndexKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(multi_mesh.particles.device_level.data()),
        thrust::raw_pointer_cast(multi_mesh.particles.device_h.data()),
        c_indices_device, f_indices_device, particle_count);

    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    const cudaError_t c_free_status = cudaFree(c_indices_device);
    const cudaError_t f_free_status = cudaFree(f_indices_device);
    if (status != cudaSuccess) {
        throw cfIndexError("C/F index kernel failed", status);
    }
    if (c_free_status != cudaSuccess) {
        throw cfIndexError("C index pointer release failed", c_free_status);
    }
    if (f_free_status != cudaSuccess) {
        throw cfIndexError("F index pointer release failed", f_free_status);
    }

    thrust::device_vector<int> selected_count_device(1);
    thrust::device_vector<int> temporary;
    for (std::size_t level = 0; level < level_count; ++level) {
        compactCFIndexVector(multi_mesh.cindexs[level],
                             thrust::raw_pointer_cast(selected_count_device.data()),
                             temporary);
        compactCFIndexVector(multi_mesh.findexs[level],
                             thrust::raw_pointer_cast(selected_count_device.data()),
                             temporary);
    }
}

#endif
