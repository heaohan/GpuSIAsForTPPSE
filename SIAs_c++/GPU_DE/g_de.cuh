#ifndef _G_DE_CUH
#define _G_DE_CUH
/*
Compared to GPUDE.cu, the d_pos and d_fitness is not updated
paralleled.
 */

/*#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "mat.h"
*/
#include <numeric>
//#include <string>
#include <algorithm>
//#include <fstream>
#include <vector>
#include <iostream>
//#include <cstdio>
//#include <cstdlib>
//#include <cmath>
//#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>
#include <limits>

constexpr auto inf = 1e10f;//9999.99f; //infinity
constexpr auto max_iters = 5000; //number of iterations
constexpr auto max_particles = 128;//2048; //number of particles
constexpr auto M_PI = 3.14159265358979323846f;  /* pi */


/*
 * Device code
 */
 //Kernel to initialize particles
 //Uses cuRAND to generate pseudorandom numbers on the GPU
__global__
void Initialize(unsigned long long seed, float *d_pos, curandState *d_states,
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
  const int step);
 
/*
Kernel to generate the random integers in the DE
*/
__device__
int RandomInteger(curandState *state, const int min_rand_int,
  const int max_rand_int);

/*
Calculate fitness for a single trail
*/
__device__
void TrailFitnessCalculate(float *d_trail, float *fitness_trail,
  float const *d_inVector, float const *d_zern, const int dims, const int pixel_num);

/*
d_zern: pixel_num * dims [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim1,
(pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim2, ...)]
*/
__global__
void Iterate(float *d_pos, curandState *d_states,
  float *d_fitness, float *d_trails, float *d_trail_fitness,
  float const *d_inVector, float const *d_zern,
  float const *d_xmin, float const *d_xmax, const int dims, const int pixel_num);

__global__
void PosUpdate(float *d_pos, const float *d_trails, const float *d_trail_fitness,
  const float *d_xmax, const float *d_xmin, const int dims);

//If iter_option is true, the DE terminates when it reaches the
//max_iters; otherwise, termination rule is used
void DE(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &crossover_rate, const float &difference_factor,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  bool iter_option, float &time_use, std::vector<float> &fit_rec, std::vector<float> &iter_rec);




 


#endif