#ifndef NNPS_MODE_H
#define NNPS_MODE_H

#include "Points.cuh"
#include "MultiMesh.cuh"
#include "particle_binary_writer.cuh"
#include "define_parameters.cuh"
#include "multi_morton_code.cuh"
#include "gpu_config.h"
#include "radix_sort.cuh"
#include "constructCFmesh.cuh"

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
    createCFindexs(multi_mesh);

    // for every mesh:
    for(int l = 0 ; l <= multi_mesh.L_max ; ++l){
        // found Ccompact
        calcCCompactMulti(multi_mesh, l);
        // compute hash result of every elem in Ccompact
        calcHashIndexMulti(multi_mesh, l);
        // sort Ccompact by hash result(cub::radixsort)
        resortCcompactByHashMulti(multi_mesh, l);
        // found hash table for elem in Ccompact
        createHtableMulti(multi_mesh, l);
    }

    // search neightbor for every point in points

    std::cout << "Debug process completed." << std::endl;
}

#endif