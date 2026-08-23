#ifndef NNPS_CONSTANT_MEMORY_CUH
#define NNPS_CONSTANT_MEMORY_CUH

#include "gpu_config.h"

#include <cuda_runtime.h>

__constant__ DATA_TYPE cmax;
__constant__ DATA_TYPE dmin[DIM_NUM];
__constant__ DATA_TYPE dmax[DIM_NUM];
__constant__ int m_d[DIM_NUM];
__constant__ int Lfine;

#endif