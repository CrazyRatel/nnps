#ifndef NNPS_UTILITIES_H
#define NNPS_UTILITIES_H

#include <cuda_runtime.h>

#include <iostream>

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

#endif