#ifndef NNPS_UTILITIES_H
#define NNPS_UTILITIES_H

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <cub/device/device_reduce.cuh>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

inline void printDeviceInformation() {
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);

    if (countStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed: "
                  << cudaGetErrorString(countStatus) << '\n';
        return;
    }

    std::cout << "CUDA devices available: " << deviceCount << '\n';

    if (deviceCount == 0) {
        return;
    }

    int currentDevice = 0;
    const cudaError_t deviceStatus = cudaGetDevice(&currentDevice);
    if (deviceStatus != cudaSuccess) {
        std::cerr << "Unable to get the current CUDA device: "
                  << cudaGetErrorString(deviceStatus) << '\n';
        return;
    }

    cudaDeviceProp deviceProperties{};
    const cudaError_t propertyStatus = cudaGetDeviceProperties(
        &deviceProperties, currentDevice);
    if (propertyStatus != cudaSuccess) {
        std::cerr << "Unable to query CUDA device properties: "
                  << cudaGetErrorString(propertyStatus) << '\n';
        return;
    }

    std::cout << "Current CUDA device: " << currentDevice << '\n'
              << "Device name: " << deviceProperties.name << '\n'
              << "Compute capability: " << deviceProperties.major << '.'
              << deviceProperties.minor << '\n';
}

inline void setDevice(int deviceId) {
    const cudaError_t status = cudaSetDevice(deviceId);
    if (status != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << deviceId << ": "
                  << cudaGetErrorString(status) << '\n';
    }
}

template <typename Z_T>
__host__ __device__
uint64_t Z_max_range()
{
    /* 有符号去掉一个高位
    * short 至少 16bit，去掉高位 15bit
    * int 至少 16bit, 去掉高位 15bit
    * long 至少 32bit，去掉高位 31bit
    * longlong 至少 64bit，去掉高位 63bit
    * int16_t 15bit
    * int32_t 31bit
    * int64_t 63bit
    * 无符号用全部bit位
    * unsigned short 16
    * unsigned int 16
    * unsigned long 32
    * uint16_t 16
    * uint32_t 32
    * uint64_t 64
    * 默认用最小的，以免出现溢出问题
    */
    uint64_t max_range = 0;
    uint64_t bit_num = 0;
    Z_T x = 1;
    while (x > 0)
    {
        ++bit_num;
        x = x << 1;
    }
    uint64_t e = int(bit_num / 3);
    max_range = pow(2, e - 1) + (pow(2, e - 1) - 1);
    return max_range;
}

template <typename T>
T cubMax(const thrust::device_vector<T>& list)
{
    if (list.empty()) {
        throw std::invalid_argument("cubMax cannot reduce an empty device vector");
    }
    if (list.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("cubMax input is too large for CUB");
    }

    const int num_items = static_cast<int>(list.size());
    const T* d_in = thrust::raw_pointer_cast(list.data());
    T* d_out = nullptr;
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error("cubMax cudaMalloc output failed: " +
                                 std::string(cudaGetErrorString(status)));
    }

    status = cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    if (status != cudaSuccess) {
        cudaFree(d_out);
        throw std::runtime_error("cubMax reduction size query failed: " + std::string(cudaGetErrorString(status)));
    }

    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (status != cudaSuccess) {
        cudaFree(d_out);
        throw std::runtime_error("cubMax cudaMalloc temporary storage failed: " + std::string(cudaGetErrorString(status)));
    }

    status = cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    if (status != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_out);
        throw std::runtime_error("cubMax reduction failed: " + std::string(cudaGetErrorString(status)));
    }

    T result{};
    status = cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_out);
    if (status != cudaSuccess) {
        throw std::runtime_error("cubMax cudaMemcpy failed: " + std::string(cudaGetErrorString(status)));
    }
    return result;
}

template <typename T>
T cubMin(const thrust::device_vector<T>& list)
{
    if (list.empty()) {
        throw std::invalid_argument("cubMin cannot reduce an empty device vector");
    }
    if (list.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("cubMin input is too large for CUB");
    }

    const int num_items = static_cast<int>(list.size());
    const T* d_in = thrust::raw_pointer_cast(list.data());
    T* d_out = nullptr;
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error("cubMin cudaMalloc output failed: " + std::string(cudaGetErrorString(status)));
    }

    status = cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    if (status != cudaSuccess) {
        cudaFree(d_out);
        throw std::runtime_error("cubMin reduction size query failed: " + std::string(cudaGetErrorString(status)));
    }

    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (status != cudaSuccess) {
        cudaFree(d_out);
        throw std::runtime_error("cubMin cudaMalloc temporary storage failed: " + std::string(cudaGetErrorString(status)));
    }

    status = cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    if (status != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_out);
        throw std::runtime_error("cubMin reduction failed: " + std::string(cudaGetErrorString(status)));
    }

    T result{};
    status = cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_out);
    if (status != cudaSuccess) {
        throw std::runtime_error("cubMin cudaMemcpy failed: " + std::string(cudaGetErrorString(status)));
    }
    return result;
}

#endif