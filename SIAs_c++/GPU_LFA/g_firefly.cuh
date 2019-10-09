#ifndef _G_FIREFLY_CUH
#define _G_FIREFLY_CUH


//#include <numeric>
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
#include <math.h>

constexpr auto inf = 1e10f; // 9999.99f; //infinity
constexpr auto max_iters = 400;//500; //number of iterations
constexpr auto max_particles = 128;//2048; //number of particles
constexpr auto M_PI = 3.14159265358979323846f;  /* pi */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Firefly {
public:
  //CUDA_CALLABLE_MEMBER Firefly(int in) {
    //pos = new float[in]; //dims
    //dims = in;
    //fitness = inf;
  //}
  CUDA_CALLABLE_MEMBER ~Firefly() {
    delete[] pos;
    pos = nullptr;
  }
  CUDA_CALLABLE_MEMBER Firefly & operator=(const Firefly &b) {
    if (this->dims != b.dims) {
      printf("dims conflict!\n");
      return *this;
    }
    if (this != &b) {
      for (int i = 0; i < this->dims; ++i)
        (this->pos)[i] = b.pos[i];
      this->fitness = b.fitness;
    }
    return *this;
  }

  float *pos;
  int dims;
  float fitness;
};

inline float Distance(const float *pos1, const float *pos2, const int &dims);

__device__
float GPUDistance(const float *pos1, const float *pos2, const int dims);

/*
Initialize d_states
*/
__global__
void InitializeStates(unsigned long long seed, curandState *d_states);

 //Kernel to initialize particles
__global__
void Initialize(Firefly *d_pop, Firefly *d_best_sol, curandState *d_states,
  float const *d_xmin, float const *d_xmax);

/*
Calculate the fitness function for each particle
*/
__global__
void FitnessCalculate(Firefly *d_pop, float const *d_inVector,
  float const *d_zern, const int dims, const int pixel_num);

/*
initialize the d_best_sol with the min fitness d_pop
*/
__global__
void BestSolInitialize(const Firefly *d_pop, Firefly *d_best_sol);

/*
Kernel to obtain the min newsols for each particles;
< << max_particles, max_particles >> >
The first max_particles will be the best newsol for each block
*/
__global__
void SelectNewSol(Firefly *d_newsols, const int dims);
 

/*
Calculate fitness for a single trail
*/
__device__
float TrailFitnessCalculate(float *d_trail, float const *d_inVector,
  float const *d_zern, const int dims, const int pixel_num);

/*
Obtain newsols for each ij combination of particles
< << max_particles * max_particles / 32, 32 >> >
*/
__global__
void ObtainNewSol(Firefly *d_pop, Firefly *d_newsols,
  const float lambda, const float sigma_square, const float dmax,
  curandState *d_states, const float *d_inVector, const float *d_zern,
  const float *d_xmin, const float *d_xmax, const int dims, const int pixel_num,
  const float beta0, const float gamma, const float alpha);

/*
Update pop with the newsols; the best max_particles in 
the first max_particles newsols and max_particles d_pop
will be selected, and stored in d_pop;
Update d_best_sol with the best particles in d_pop
*/
__global__
void UpdatePop(Firefly *d_pop, Firefly *d_newsols,
  Firefly *d_best_sol, int iter_count, float *d_best_fit_rec);

void FireflyOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &gamma, const float &beta0, float alpha,
  const float &alpha_damp, const int &pixel_num,
  const int &dims, float *g_best, float *g_best_pos, 
  float &time_use, std::vector<float> &fit_rec);

#endif // !_G_FIREFLY_CUH







 
 
