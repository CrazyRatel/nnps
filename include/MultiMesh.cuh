#ifndef NNPS_MULTI_MESH_CUH
#define NNPS_MULTI_MESH_CUH

#include "Points.cuh"

#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

class MultiMesh {
public:
    Points particles;

    thrust::device_vector<uint64_t> z_L;
    std::vector<thrust::device_vector<int>> cindexs;
    std::vector<thrust::device_vector<int>> findexs;
    int L_max = 0;

    std::vector<thrust::device_vector<int64_t>> hash_indexs_c;
    std::vector<thrust::device_vector<int64_t>> hash_indexs_f;

    std::vector<thrust::device_vector<int>> c_compact_begin_c;
    std::vector<thrust::device_vector<int>> c_compact_length_c;
    std::vector<thrust::device_vector<int>> c_compact_begin_f;
    std::vector<thrust::device_vector<int>> c_compact_length_f;

    std::vector<thrust::device_vector<int>> h_begin_c;
    std::vector<thrust::device_vector<int>> h_length_c;
    std::vector<thrust::device_vector<int>> h_begin_f;
    std::vector<thrust::device_vector<int>> h_length_f;

    uint64_t hash_table_size = 1;
    int L_fine = 0;

    MultiMesh() = default;

    explicit MultiMesh(const Points& input_particles, int input_fine_level = 0)
        : particles(input_particles), L_fine(std::max(0, input_fine_level)) {
        initializeMultiMesh();
    }

    void resetDS() {
        initializeStorage();
        fillIndexStorage();
    }

    void initializeMultiMesh() {
        const std::size_t particle_count = particles.particle_num;
        hash_table_size = std::max<uint64_t>(1, static_cast<uint64_t>(particle_count) * 2);
        L_max = L_fine;

        initializeStorage();
        fillIndexStorage();
    }

private:
    void initializeStorage() {
        const std::size_t particle_count = particles.particle_num;
        const std::size_t level_count = static_cast<std::size_t>(L_max + 1);

        z_L.resize(particle_count);
        cindexs.resize(level_count);
        findexs.resize(level_count);
        hash_indexs_c.resize(level_count);
        hash_indexs_f.resize(level_count);
        c_compact_begin_c.resize(level_count);
        c_compact_length_c.resize(level_count);
        c_compact_begin_f.resize(level_count);
        c_compact_length_f.resize(level_count);
        h_begin_c.resize(level_count);
        h_length_c.resize(level_count);
        h_begin_f.resize(level_count);
        h_length_f.resize(level_count);

        for (std::size_t level = 0; level < level_count; ++level) {
            cindexs[level].resize(particle_count);
            findexs[level].resize(particle_count);
            c_compact_begin_c[level].resize(particle_count);
            c_compact_length_c[level].resize(particle_count);
            c_compact_begin_f[level].resize(particle_count);
            c_compact_length_f[level].resize(particle_count);
            h_begin_c[level].resize(hash_table_size);
            h_length_c[level].resize(hash_table_size);
            h_begin_f[level].resize(hash_table_size);
            h_length_f[level].resize(hash_table_size);
        }
    }

    void fillIndexStorage() {
        for (std::size_t level = 0; level <= L_max; ++level) {
            thrust::fill(cindexs[level].begin(), cindexs[level].end(), -1);
            thrust::fill(findexs[level].begin(), findexs[level].end(), -1);
            thrust::fill(c_compact_begin_c[level].begin(), c_compact_begin_c[level].end(), -1);
            thrust::fill(c_compact_length_c[level].begin(), c_compact_length_c[level].end(), -1);
            thrust::fill(c_compact_begin_f[level].begin(), c_compact_begin_f[level].end(), -1);
            thrust::fill(c_compact_length_f[level].begin(), c_compact_length_f[level].end(), -1);
            thrust::fill(h_begin_c[level].begin(), h_begin_c[level].end(), -1);
            thrust::fill(h_length_c[level].begin(), h_length_c[level].end(), -1);
            thrust::fill(h_begin_f[level].begin(), h_begin_f[level].end(), -1);
            thrust::fill(h_length_f[level].begin(), h_length_f[level].end(), -1);
        }
    }
};

#endif