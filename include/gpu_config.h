#ifndef NNPS_GPU_CONFIG_H
#define NNPS_GPU_CONFIG_H

// 调优参数
/* GPU线程块大小 */
#define BLOCKSIZE 256



// 案例生成参数
/* 空间维度 */
#define DIM_NUM 3
/* 多层级支持 */
#define MultiLevel 1
#define MaxLevel 4
/* 均匀分布 */
// 单一维度上的粒子数
#define NUM_PER_DIM 52 // cannot delete, would cause undefined reference in Points.cuh
#define PARTICLE_PER_DIM 177
/* 支持半径范围 */
#define MAX_H_ 1
#define MIN_H_ MAX_H_*exp2(-MaxLevel)
/* 模拟域坐标范围 */
#define MIN_ 0
#define MAX_ 2*PARTICLE_PER_DIM*MAX_H_+MIN_


// 算法参数
/* 算法重复次数 */
#define REPEAT 20
/* 数据精度类型 */
#define DATA_TYPE float
/* 莫顿编码精度类型 */
#define MORTON_TYPE uint64_t
/* 哈希函数中的素数 */
#define P_1 73856093L
#define P_2 19349663L
#define P_3 83492791L

#endif // NNPS_GPU_CONFIG_H
