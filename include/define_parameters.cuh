#ifndef NNPS_DEFINE_PARAMETERS_CUH
#define NNPS_DEFINE_PARAMETERS_CUH

#include "constant_memory.cuh"
#include "Points.cuh"
#include "utilities.cuh"

#include <thrust/extrema.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

inline int calculateLfineAndSetConstantMemory(Points& points) {
    if (points.particle_num == 0) {
        std::cout<<"Warning: No particles found. Setting Lfine to 0 and No value for cmax, dmin, dmax, m_d."<<std::endl;
        const int empty_level = 0;
        cudaMemcpyToSymbol(Lfine, &empty_level, sizeof(empty_level));
        return empty_level;
    }



    const float maximum_h = cubMax(points.device_h);

    const float minimum_x = cubMin(points.device_x_coord);
    const float minimum_y = cubMin(points.device_y_coord);
    const float minimum_z = cubMin(points.device_z_coord);

    const float maximum_x = cubMax(points.device_x_coord);
    const float maximum_y = cubMax(points.device_y_coord);
    const float maximum_z = cubMax(points.device_z_coord);

    const int grid[DIM_NUM] = {
        static_cast<int>(std::ceil((maximum_x - minimum_x) / maximum_h)),
        static_cast<int>(std::ceil((maximum_y - minimum_y) / maximum_h)),
        static_cast<int>(std::ceil((maximum_z - minimum_z) / maximum_h))
    };
    const int grid_size = std::max({grid[0], grid[1], grid[2], 1});


    const std::uint64_t morton_range = Z_max_range<MORTON_TYPE>();
    const int finest_level = static_cast<int>(std::log2(static_cast<double>(morton_range))) - static_cast<int>(std::ceil(std::log2(static_cast<double>(grid_size))));

    const DATA_TYPE minimum[DIM_NUM] = {minimum_x, minimum_y, minimum_z};
    const DATA_TYPE maximum[DIM_NUM] = {maximum_x, maximum_y, maximum_z};

    cudaMemcpyToSymbol(Lfine, &finest_level, sizeof(finest_level));
    cudaMemcpyToSymbol(cmax, &maximum_h, sizeof(maximum_h));
    cudaMemcpyToSymbol(dmin, minimum, sizeof(minimum));
    cudaMemcpyToSymbol(dmax, maximum, sizeof(maximum));
    cudaMemcpyToSymbol(m_d, grid, sizeof(grid));

    return finest_level;
}

#endif