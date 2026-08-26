cmake_minimum_required(VERSION 3.31)
project(nnps CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(nnps main.cu)

set_target_properties(nnps PROPERTIES
        CUDA_ARCHITECTURES 120
        CUDA_SEPARABLE_COMPILATION ON)
