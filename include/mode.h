#ifndef NNPS_MODE_H
#define NNPS_MODE_H

#include "Points.cuh"
#include "MultiMesh.cuh"
#include "particle_binary_writer.cuh"
#include "define_parameters.cuh"
#include "multi_morton_code.cuh"
#include "gpu_config.h"
#include "radix_sort.cuh"

inline void debugProcess() {
    std::cout << "NNPS (single GPU edition)" << std::endl;

    // automatically generate points information through settings in gpu_config.h
    Points points;
    points.uploadPointData();
    // ParticleBinaryWriter writer;
    // writer.Output_particle(0, points);

    const int finest_level = calculateLfineAndSetConstantMemory(points);
    std::cout << "Finest level: " << finest_level << std::endl;
    // initialize MultiMesh with points and Lfine
    MultiMesh multi_mesh(points, finest_level);

    // compute morton code
    calculateMultiMortonCode(multi_mesh);

    // sort points by morton code(cub::radixsort)
    sortParticles(getOrderByKey(multi_mesh.z_L), multi_mesh.particles);

    // found CF indexs

    // for every mesh:
    // found Ccompact
    // compute hash result of every elem in Ccompact
    // sort Ccompact by hash result(cub::radixsort)
    // found hash table for elem in Ccompact

    // search neightbor for every point in points

    std::cout << "Debug process completed." << std::endl;
}

#endif