#ifndef NNPS_MODE_H
#define NNPS_MODE_H

#include "utilities.h"
#include "Points.cuh"
#include "MultiMesh.cuh"
#include "particle_binary_writer.cuh"

inline void debugProcess() {
    std::cout << "NNPS (single GPU edition)\n" << std::endl;

    // automatically generate points information through settings in gpu_config.h
    Points points;
    // ParticleBinaryWriter writer;
    // writer.Output_particle(0, points);

    
    MultiMesh multi_mesh(input_particles=points, input_fine_level=0);

    // compute morton code

    // sort points by morton code(cub::radixsort)

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