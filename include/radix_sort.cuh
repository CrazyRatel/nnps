#ifndef NNPS_RADIX_SORT_CUH
#define NNPS_RADIX_SORT_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <cub/device/device_radix_sort.cuh>

#include "Points.cuh"
#include "gpu_config.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdint>

inline std::runtime_error radixSortError(const char* operation, cudaError_t status)
{
    return std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

inline thrust::device_vector<unsigned> getOrderByKey(thrust::device_vector<uint64_t> &key)
{
    if (key.empty()) {
        return {};
    }
    if (key.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("getOrderByKey input is too large for CUB");
    }

    const int num_items = static_cast<int>(key.size());
    thrust::device_vector<unsigned> value(num_items);
    thrust::sequence(thrust::device, value.begin(), value.end());

    thrust::device_vector<unsigned long long> key_alt(key);
    thrust::device_vector<unsigned> value_alt(value);

    unsigned long long *d_key_buf = thrust::raw_pointer_cast(key.data());                    // e.g., [8, 6, 7, 5, 3, 0, 9]
    unsigned long long *d_key_alt_buf = thrust::raw_pointer_cast(key_alt.data());            // e.g., [        ...        ]
    unsigned *d_value_buf = thrust::raw_pointer_cast(value.data());         // e.g., [0, 1, 2, 3, 4, 5, 6]
    unsigned *d_value_alt_buf = thrust::raw_pointer_cast(value_alt.data()); // e.g., [        ...        ]

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<unsigned long long> d_keys(d_key_buf, d_key_alt_buf);
    cub::DoubleBuffer<unsigned> d_values(d_value_buf, d_value_alt_buf);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cudaError_t status = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
    if (status != cudaSuccess) {
        throw radixSortError("CUB radix sort size query failed", status);
    }

    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (status != cudaSuccess) {
        throw radixSortError("Radix sort temporary storage allocation failed", status);
    }

    status = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
    const cudaError_t free_status = cudaFree(d_temp_storage);
    if (status != cudaSuccess) {
        throw radixSortError("CUB radix sort failed", status);
    }
    if (free_status != cudaSuccess) {
        throw radixSortError("Radix sort temporary storage release failed", free_status);
    }

    return d_values.selector == 0 ? value : value_alt;
}

template <typename T>
__global__ void kerSortParticles(unsigned *map,
                                T *x_temp, T *y_temp, T *z_temp, T *h_temp,
                                T *x, T *y, T *z, T *h, unsigned size)
{
    unsigned p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < size)
    {
        h[p] = h_temp[map[p]];
        x[p] = x_temp[map[p]];
        y[p] = y_temp[map[p]];
        z[p] = z_temp[map[p]];
    }
}

inline void sortParticles(thrust::device_vector<unsigned> map, Points& particles)
{
    //临时存储粒子信息
    if (particles.particle_num == 0) {
        return;
    }
    if (map.size() != particles.particle_num) {
        throw std::invalid_argument("sortParticles map size does not match particle count");
    }
    thrust::device_vector<float> x_temp = particles.device_x_coord;
    thrust::device_vector<float> y_temp = particles.device_y_coord;
    thrust::device_vector<float> z_temp = particles.device_z_coord;
    thrust::device_vector<float> h_temp = particles.device_h;

    unsigned *map_ptr = thrust::raw_pointer_cast(map.data());
    float *x_temp_ptr = thrust::raw_pointer_cast(x_temp.data());
    float *y_temp_ptr = thrust::raw_pointer_cast(y_temp.data());
    float *z_temp_ptr = thrust::raw_pointer_cast(z_temp.data());
    float *h_temp_ptr = thrust::raw_pointer_cast(h_temp.data());

    float *x_ptr = thrust::raw_pointer_cast(particles.device_x_coord.data());
    float *y_ptr = thrust::raw_pointer_cast(particles.device_y_coord.data());
    float *z_ptr = thrust::raw_pointer_cast(particles.device_z_coord.data());
    float *h_ptr = thrust::raw_pointer_cast(particles.device_h.data());

    if (particles.particle_num > std::numeric_limits<unsigned>::max()) {
        throw std::length_error("sortParticles particle count is too large");
    }
    const unsigned N = static_cast<unsigned>(particles.particle_num);

    //并行的线程数
    unsigned Block = BLOCKSIZE;
    unsigned Grid = unsigned(N + Block - 1) / Block;

    kerSortParticles<<<Grid, Block>>>(map_ptr,
                                    x_temp_ptr, y_temp_ptr, z_temp_ptr, h_temp_ptr,
                                    x_ptr, y_ptr, z_ptr, h_ptr, N);
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        throw radixSortError("Particle reorder kernel launch failed", launch_status);
    }

    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        throw radixSortError("Particle reorder kernel execution failed", sync_status);
    }
}

#endif