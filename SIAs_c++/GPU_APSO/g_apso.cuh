#ifndef _G_APSO_CUH
#define _G_APSO_CUH

#include <numeric>
#include <string>
#include <algorithm>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#include <chrono>

//constexpr auto inf = 9999.99f; //infinity
constexpr auto max_iters = 2048; //number of iterations
constexpr auto max_particles = 2048; //number of particles
constexpr auto M_PI = 3.14159265358979323846;  /* pi */

#define cudaCheckError()\
{\
	cudaError_t e = cudaGetLastError();\
	if(e != cudaSuccess)\
				{\
			printf("CUDA failure: %s%d: %s", __FILE__, __LINE__, cudaGetErrorString(e));\
			exit(EXIT_FAILURE);\
				}\
}

/*
 * Device code
 */
 //Kernel to initialize particles
 //Uses cuRAND to generate pseudorandom numbers on the GPU
__global__
void Initialize(float *d_pos, curandState *d_states,
  float const *d_xmin, float const *d_xmax);

/*
Calculate the fitness function for each particle
*/
__global__
void FitnessCalculate(float *d_pos, float *fitness, float const *d_inVector,
  float const *d_zern, const int dims, const int pixel_num);
 
/*
Kernel to obtain the min fitness and corresponding index in each block;
the min of each block are placed at the first blockIdx.x positions.
( 1. << <max_particles / 32, 32 >> > 2. << < 1, max_particles / 32 >> >)
*/
__global__
void ReduceFitnessMin(float *d_fitness, int *d_best_fitness_index,
  const int step, float *dh_fitness_all = nullptr, const int iter_count = 0);

/*
d_zern: pixel_num * dims [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim1,
(pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim2, ...)]
*/
__global__
void Iterate(float *d_pos, curandState *d_states, float *d_alpha0,
  const int iter_count, int *d_best_fitness_index,
  float const *d_xmin, float const *d_xmax, const int dims, const int pixel_num);

void ApsoOptimization(const std::vector<float> &in_vector, const std::vector<float> &xmin,
  const std::vector<float> &xmax, const std::vector<float> &zern, const int pixel_num,
  const int dims, float *g_best, float *g_best_pos);
 
#endif
 

