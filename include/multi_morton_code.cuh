#ifndef NNPS_MULTI_MORTON_CODE_CUH
#define NNPS_MULTI_MORTON_CODE_CUH

#include "MultiMesh.cuh"
#include "constant_memory.cuh"
#include "gpu_config.h"
#include "utilities.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

template <typename MortonType>
__global__ void calculateMultiMortonCodeKernel(
    MortonType* morton_codes,
    const float* x_coordinates,
    const float* y_coordinates,
    const float* z_coordinates,
    std::size_t particle_count,
    int level) {
    const std::size_t particle_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (particle_index >= particle_count) {
        return;
    }

    int position[DIM_NUM];
    getPosByIndex(position, x_coordinates, y_coordinates, z_coordinates, particle_index, level);
    morton_codes[particle_index] = Z_magicbit<MortonType>(position);
}

template <typename MortonType = MORTON_TYPE>
inline void calculateMultiMortonCode(MultiMesh& multi_mesh) {
    const std::size_t particle_count = multi_mesh.particles.particle_num;
    // Check if there are no particles to process
    if (particle_count == 0) {
        throw std::runtime_error("[calculateMultiMortonCode] No particles to process !!!");
        return;
    }

    const unsigned block_size = BLOCKSIZE;
    const unsigned grid_size = static_cast<unsigned>((particle_count + block_size - 1) / block_size);

    MortonType* morton_codes = thrust::raw_pointer_cast(multi_mesh.z_L.data());
    const float* x_coordinates = thrust::raw_pointer_cast(multi_mesh.particles.device_x_coord.data());
    const float* y_coordinates = thrust::raw_pointer_cast(multi_mesh.particles.device_y_coord.data());
    const float* z_coordinates = thrust::raw_pointer_cast(multi_mesh.particles.device_z_coord.data());

    calculateMultiMortonCodeKernel<MortonType><<<grid_size, block_size>>>(
        morton_codes, x_coordinates, y_coordinates, z_coordinates,
        particle_count, multi_mesh.L_max);

    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw std::runtime_error(
            "Morton code kernel launch failed: " +
            std::string(cudaGetErrorString(launch_status)));
    }

    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        throw std::runtime_error(
            "Morton code kernel execution failed: " +
            std::string(cudaGetErrorString(sync_status)));
    }
}

#endif