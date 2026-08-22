#ifndef NNPS_MULTI_MESH_CUH
#define NNPS_MULTI_MESH_CUH

#include "Points.cuh"

#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cmath>
#include <vector>

class MultiMesh {
public:
    Points particles;

    thrust::device_vector<unsigned long long> z_L;
    std::vector<thrust::device_vector<int>> m_cindexs;
    std::vector<thrust::device_vector<int>> m_findexs;
    int L_max = 0;

    std::vector<thrust::device_vector<long>> m_hash_indexs_c;
    std::vector<thrust::device_vector<long>> m_hash_indexs_f;

    std::vector<thrust::device_vector<int>> m_c_compact_begin_c;
    std::vector<thrust::device_vector<int>> m_c_compact_length_c;
    std::vector<thrust::device_vector<int>> m_c_compact_begin_f;
    std::vector<thrust::device_vector<int>> m_c_compact_length_f;

    std::vector<thrust::device_vector<int>> m_h_begin_c;
    std::vector<thrust::device_vector<int>> m_h_length_c;
    std::vector<thrust::device_vector<int>> m_h_begin_f;
    std::vector<thrust::device_vector<int>> m_h_length_f;

    unsigned long long hash_table_size = 1;
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
        hash_table_size = std::max<unsigned long long>(1, static_cast<unsigned long long>(particle_count) * 2);
        L_max = L_fine;

        initializeStorage();
        fillIndexStorage();
    }

private:
    void initializeStorage() {
        const std::size_t particle_count = particles.particle_num;
        const std::size_t level_count = static_cast<std::size_t>(L_max + 1);

        z_L.resize(particle_count);
        m_cindexs.resize(level_count);
        m_findexs.resize(level_count);
        m_hash_indexs_c.resize(level_count);
        m_hash_indexs_f.resize(level_count);
        m_c_compact_begin_c.resize(level_count);
        m_c_compact_length_c.resize(level_count);
        m_c_compact_begin_f.resize(level_count);
        m_c_compact_length_f.resize(level_count);
        m_h_begin_c.resize(level_count);
        m_h_length_c.resize(level_count);
        m_h_begin_f.resize(level_count);
        m_h_length_f.resize(level_count);

        for (std::size_t level = 0; level < level_count; ++level) {
            m_cindexs[level].resize(particle_count);
            m_findexs[level].resize(particle_count);
            m_c_compact_begin_c[level].resize(particle_count);
            m_c_compact_length_c[level].resize(particle_count);
            m_c_compact_begin_f[level].resize(particle_count);
            m_c_compact_length_f[level].resize(particle_count);
            m_h_begin_c[level].resize(hash_table_size);
            m_h_length_c[level].resize(hash_table_size);
            m_h_begin_f[level].resize(hash_table_size);
            m_h_length_f[level].resize(hash_table_size);
        }
    }

    void fillIndexStorage() {
        for (std::size_t level = 0; level <= L_max; ++level) {
            thrust::fill(m_cindexs[level].begin(), m_cindexs[level].end(), -1);
            thrust::fill(m_findexs[level].begin(), m_findexs[level].end(), -1);
            thrust::fill(m_c_compact_begin_c[level].begin(), m_c_compact_begin_c[level].end(), -1);
            thrust::fill(m_c_compact_length_c[level].begin(), m_c_compact_length_c[level].end(), -1);
            thrust::fill(m_c_compact_begin_f[level].begin(), m_c_compact_begin_f[level].end(), -1);
            thrust::fill(m_c_compact_length_f[level].begin(), m_c_compact_length_f[level].end(), -1);
            thrust::fill(m_h_begin_c[level].begin(), m_h_begin_c[level].end(), -1);
            thrust::fill(m_h_length_c[level].begin(), m_h_length_c[level].end(), -1);
            thrust::fill(m_h_begin_f[level].begin(), m_h_begin_f[level].end(), -1);
            thrust::fill(m_h_length_f[level].begin(), m_h_length_f[level].end(), -1);
        }
    }
};

#endif