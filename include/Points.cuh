#ifndef NNPS_POINTS_CUH
#define NNPS_POINTS_CUH

#include "gpu_config.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

class Points {
public:
    std::size_t particle_num = 0;

    thrust::host_vector<float> host_x_coord;
    thrust::host_vector<float> host_y_coord;
    thrust::host_vector<float> host_z_coord;

    thrust::device_vector<float> device_x_coord;
    thrust::device_vector<float> device_y_coord;
    thrust::device_vector<float> device_z_coord;

    thrust::host_vector<float> host_h;
    thrust::device_vector<float> device_h;

    thrust::host_vector<int> host_level;
    thrust::device_vector<int> device_level;

    thrust::host_vector<unsigned> host_neighbor_num;
    thrust::device_vector<unsigned> device_neighbor_num;

    Points() {
        generatePointLattice();
    }

    explicit Points(std::size_t particle_num) {
        resize(particle_num);
    }

    void uploadPointData() {
        device_x_coord = host_x_coord;
        device_y_coord = host_y_coord;
        device_z_coord = host_z_coord;
        device_h = host_h;
        device_level = host_level;
        device_neighbor_num = host_neighbor_num;
    }

    void downloadPointData() {
        host_x_coord = device_x_coord;
        host_y_coord = device_y_coord;
        host_z_coord = device_z_coord;
        host_h = device_h;
        host_level = device_level;
        host_neighbor_num = device_neighbor_num;
    }

private:
    void generatePointLattice() {
        const std::size_t points_per_level = static_cast<std::size_t>(PARTICLE_PER_DIM * PARTICLE_PER_DIM * PARTICLE_PER_DIM);
        const int level_count = MultiLevel == 0 ? 1 : MaxLevel;

        particle_num = points_per_level * static_cast<std::size_t>(level_count);
        host_x_coord.reserve(particle_num);
        host_y_coord.reserve(particle_num);
        host_z_coord.reserve(particle_num);
        host_h.reserve(particle_num);
        host_level.reserve(particle_num);
        host_neighbor_num.resize(particle_num, 0);

        const float domain_min = static_cast<float>(MIN_);
        const float domain_width = static_cast<float>(2 * PARTICLE_PER_DIM * MAX_H_);

        for (int level = 0; level < level_count; ++level) {
            const float spacing = static_cast<float>(MAX_H_) /
                                  std::pow(2.0f, static_cast<float>(level));
            const float lattice_width = static_cast<float>(PARTICLE_PER_DIM) * spacing;
            const float first_point = domain_min +
                                      (domain_width - lattice_width) * 0.5f +
                                      spacing * 0.5f;

            for (int x = 0; x < PARTICLE_PER_DIM; ++x) {
                for (int y = 0; y < PARTICLE_PER_DIM; ++y) {
                    for (int z = 0; z < PARTICLE_PER_DIM; ++z) {
                        host_x_coord.push_back(first_point + x * spacing);
                        host_y_coord.push_back(first_point + y * spacing);
                        host_z_coord.push_back(first_point + z * spacing);
                        host_h.push_back(spacing);
                        host_level.push_back(level);
                    }
                }
            }
        }
    }

    void resize(std::size_t count) {
        particle_num = count;
        host_x_coord.resize(count);
        host_y_coord.resize(count);
        host_z_coord.resize(count);
        host_h.resize(count, static_cast<float>(MAX_H_));
        host_level.resize(count, 0);
        host_neighbor_num.resize(count, 0);
        uploadPointData();
    }
};

#endif