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

#include "mat_rw.h"

//constexpr auto inf = 9999.99f; //infinity
constexpr auto max_iters = 400;//200; //2048; //number of iterations
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
//<<<max_particles, dims>>>
__global__
void Initialize(unsigned long long seed, float *d_pos, curandState *d_states,
   float const *d_xmin, float const *d_xmax) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  //Adjust d_pos
  d_pos[index] = d_xmin[threadIdx.x] + (d_xmax[threadIdx.x] - d_xmin[threadIdx.x])
    * d_pos[index];

  //if (threadIdx.x == 9)
    //printf("%.2f\n", d_pos[index]);

  //Initializing up cuRAND
  ////Each thread gets a different seed, different sequence number and no offset
  //curand_init(index, index, 0, &d_states[index]);
  //Each thread gets one seed every run, but different sequence
  //number; the seed is also different in different run
  if(index < max_particles)//cudaMalloc((void**)&d_states, max_particles * sizeof(curandState));
    curand_init(seed, index, 0, &d_states[index]);
}

/*
Calculate the fitness function for each particle
*/
__global__
void FitnessCalculate(float *d_pos, float *fitness, float const *d_inVector,
  float const *d_zern, const int dims, const int pixel_num) {
  int index = blockDim.x * blockIdx.x + threadIdx.x; //particle index
  float tp_fitness = 0.0f;

  /*
  index: particle index
  d_zern: pixel_num * dims [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim1,
  (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim2, ...]
  d_inVector: [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_image1,
  (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_image2]
  d_pos: [store in this way (dim1, dim2, dim3, ...)_particle1,
  (dim1, dim2, dim3, ...)_particle2, ...] 
  */
  float phi;
  float delta;
  float tp1, tp2;
  for (int i = 0; i < pixel_num; ++i) {
    phi = 0.0f;
    delta = d_pos[index * dims + dims - 1];
    for (int j = 0; j < dims - 1; ++j) {
      phi += d_zern[j * pixel_num + i] * d_pos[index * dims + j];
    }
    //tp_fitness += abs(cos(phi) - d_inVector[i]) + abs(cos(phi + delta) - d_inVector[pixel_num + i]);
    tp1 = cos(phi) - d_inVector[i];
    tp2 = cos(phi + delta) - d_inVector[pixel_num + i];
    tp_fitness += tp1 * tp1 + tp2 * tp2;
  }
  fitness[index] = tp_fitness;
  //printf("index: %d, fitness: %.2f\n", index, fitness[index]);

  /*Test
    if (index == 0) {
    float tp_result = 0.0f;
    //float tp_pos[] = { 0.4820f, -0.2068f, 1.1224f, -0.0914f, 0.0067f, -9.5738f, -0.0742f, 0.0980f, 0.0987f, -0.0152f, 0.0301f };
    float tp_pos[] = { 0.479791f, -0.174059f, 1.09441f, 0.0210509f, 0.0434665f, - 9.51595f, - 0.0642395f, 0.0921565f, 0.0961314f, 0.0441319f, 0.0345671f };
    for (int i = 0; i < pixel_num; ++i) {
      phi = 0.0f;
      delta = tp_pos[dims - 1];
      for (int j = 0; j < dims - 1; ++j) {
        phi += d_zern[j * pixel_num + i] * tp_pos[j];
      }
      //tp_result += abs(cos(phi) - d_inVector[i]) + abs(cos(phi + delta) - d_inVector[pixel_num + i]);
      tp_result += (cos(phi) - d_inVector[i]) * (cos(phi) - d_inVector[i]) + (cos(phi + delta) - d_inVector[pixel_num + i]) * (cos(phi + delta) - d_inVector[pixel_num + i]);
    }
    printf("%f", tp_result);
  }
  */

}

/*
Kernel to obtain the min fitness and corresponding index in each block;
the min of each block are placed at the first blockIdx.x positions.
( 1. << <max_particles / 32, 32 >> > 2. << < 1, max_particles / 32 >> >)
*/
__global__
void ReduceFitnessMin(float *d_fitness, int *d_best_fitness_index,
  const int step, float *dh_fitness_all=nullptr, const int iter_count=0) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int tx = threadIdx.x;

  //Declare shared memory for staging the reduce phase
  __shared__ float stage[512]; //max particles number: 512 * 32
  __shared__ int best[512];

  //Copy PBestY to shared memory
  if (step == 1) {
    best[tx] = index;
  }
  else {
    best[tx] = d_best_fitness_index[index];
  }
  stage[tx] = d_fitness[index];
  __syncthreads();

  //Perform the actual reduce
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tx < s) {
      if (stage[tx] > stage[tx + s]) {
        stage[tx] = stage[tx + s];
        best[tx] = best[tx + s];
      }
    }
    __syncthreads();
  }

  //Copy results back into global memory
  if (tx == 0) {
    d_fitness[blockIdx.x] = stage[0];
    d_best_fitness_index[blockIdx.x] = best[0];
  }

  if (step == 2 && threadIdx.x == 0) {
    if (dh_fitness_all == nullptr) {
      printf("dh_fitness_all is not provided.\n");
    }
    dh_fitness_all[iter_count] = *d_fitness;
  }

  if (tx == 0 && step == 2)
    printf("*******\n Iteration: %d, Min: %.2f, index: %d\n",
      iter_count, d_fitness[0], *d_best_fitness_index);
}

/*
d_zern: pixel_num * dims [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim1,
(pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim2, ...)]
*/
//<<<max_particles / 32, 32>>>
__global__
void Iterate(float *d_pos, curandState *d_states, float *d_alpha0,
  const int iter_count, int *d_best_fitness_index, 
  float const *d_xmin, float const *d_xmax, const int dims, const int pixel_num) {
  //parameters
  int index = blockDim.x * blockIdx.x + threadIdx.x; //particle index
  curandState &local_state = d_states[index];
  float beta = 0.5f;
  float gamma = 0.97f;
  float r1;
  extern __shared__ float d_alpha[];
  //float tp;
  //if(iter_count == 1)
    //tp = powf(gamma, iter_count);
  for (int i = 0; i < dims; ++i) {
    d_alpha[i] = d_alpha0[i] * powf(gamma, iter_count);
    //if (iter_count == 0 && index == 0)
      //printf("\n alpha %d : %.2f\n", i, d_alpha[i]);
  }

  //Update the particle velocity and position
  for (int i = 0; i < dims; i++) {
    int id = index * dims + i;
    r1 = curand_uniform(&local_state);

    //if (index == 0 || index == 1)
      //printf("particle: %d\tdims: %d\t r1: %.4f\n", index, i, r1);

    d_pos[id] = (1.0f - beta) * d_pos[id] + 
      beta * d_pos[(*d_best_fitness_index) * dims + i] + 
      d_alpha[i] * (r1 - 0.5f);

    //Ensure position values are within range
    if (d_pos[id] > d_xmax[i])
      d_pos[id] = d_xmax[i];
    if (d_pos[id] < d_xmin[i])
      d_pos[id] = d_xmin[i];
  }

  ////Set the current state of the PRNG
  //d_states[index] = local_state;
}


//If iter_option is true, the PSO terminates when it reaches the
//max_iters; otherwise, termination rule is used
void ApsoOptimization(const std::vector<float> &in_vector, const std::vector<float> &xmin,
  const std::vector<float> &xmax, const std::vector<float> &zern, const int pixel_num,
  const int dims, float *g_best, float *g_best_pos, 
  bool iter_option, float &time_use, std::vector<float> &fit_rec) {
  /* Declare all variables.*/
  float const *d_inVector = nullptr;
  float const *d_zern = nullptr;
  float const *d_xmin = nullptr;
  float const *d_xmax = nullptr;

  float *d_pos = nullptr;
  float *d_alpha0 = nullptr;
  curandState *d_states = nullptr;
  float *d_fitness = nullptr;
  int *d_best_fitness_index = nullptr;
  float *dh_fitness_all = nullptr;

  /*
  *Allocat the memory on GPU for in_vector, zern, xmin, xmax
   */
  cudaMalloc((void**)&d_inVector, 2 * pixel_num * sizeof(float));
  cudaCheckError();
  cudaMemcpy((void*)d_inVector, (void*)in_vector.data(), 2 * pixel_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMalloc((void**)&d_zern, (dims - 1) * pixel_num * sizeof(float));
  cudaCheckError();
  cudaMemcpy((void*)d_zern, (void*)zern.data(), (dims - 1) * pixel_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMalloc((void**)&d_xmin, dims * sizeof(float));
  cudaCheckError();
  cudaMemcpy((void*)d_xmin, (void*)xmin.data(), dims * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMalloc((void**)&d_xmax, dims * sizeof(float));
  cudaCheckError();
  cudaMemcpy((void*)d_xmax, (void*)xmax.data(), dims * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();

   /* 
  Create the parameter for PSO optimization:
  (size = max_particles * dims)
    float *d_pos
    float *d_velocity
    float *d_p_best_x
    curandState *d_states
  (size = max_particles)
    float *d_p_best_y
    int *d_l_best_index
    int *d_best_index (used for reduce)
  */
  std::vector<float> alpha0(dims);
  for (int i = 0; i < dims; ++i) {
    alpha0[i] = (xmax[i] - xmin[i]) * 0.25f;
    //alpha0[i] = 1.0f;
  }


  cudaMalloc((void**)&d_pos, max_particles * dims * sizeof(float));
  cudaCheckError();
  //cudaMalloc((void**)&d_states, max_particles * dims * sizeof(curandState));
  cudaMalloc((void**)&d_states, max_particles * sizeof(curandState));
  cudaCheckError();
  cudaMalloc((void**)&d_alpha0, dims * sizeof(float));
  cudaCheckError();
  cudaMemcpy((void*)d_alpha0, (void*)alpha0.data(), dims * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMalloc((void**)&d_fitness, max_particles * sizeof(float));
  cudaCheckError();
  cudaMalloc((void**)&d_best_fitness_index, max_particles * sizeof(int));
  cudaCheckError();

  cudaMallocManaged(&dh_fitness_all, max_iters * sizeof(float));
  cudaCheckError();

  //Create PRNG
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

  //Initialize d_pos
  curandGenerateUniform(gen, d_pos, max_particles * dims);
  cudaCheckError();
  
  //Destroying cuRAND generator
  curandDestroyGenerator(gen);
  cudaCheckError();


  //Adjust the d_pos 
  //Generate the different seeds every time
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long long> distr;
  unsigned long long seed = distr(eng);
  Initialize << <max_particles, dims >> > (seed, d_pos, d_states, d_xmin, d_xmax);
  cudaCheckError();

  //Initialize 
  auto t_start = std::chrono::system_clock::now();//variables for time record
  for (int i = 0; i < max_iters; i++) {
   // ResetBestFitnessIndex << <max_particles / 32, 32 >> > (best_fitness_index);
    FitnessCalculate << <max_particles / 32, 32 >> > (d_pos, d_fitness, d_inVector,
      d_zern, dims, pixel_num); // Calculate the fitness for each particle
    
    //printf("Iteration: %d:\n", i);
    //cudaDeviceSynchronize();
    ReduceFitnessMin << <max_particles / 32, 32 >> >(d_fitness, d_best_fitness_index, 1);
    ReduceFitnessMin << <1, max_particles / 32 >> > (d_fitness, d_best_fitness_index, 2, dh_fitness_all, i);
    
    cudaDeviceSynchronize();
    //printf("Iteration: %d, current best: %.2f\n", i, dh_fitness_all[i]);
    fit_rec.push_back(dh_fitness_all[i]);
    if((!iter_option) && i >= 49 && abs(dh_fitness_all[i] - dh_fitness_all[i-49]) < 1e-2f)
      break;

    Iterate << <max_particles / 32, 32, dims * sizeof(float) >> > (d_pos,
       d_states, d_alpha0, i, d_best_fitness_index, d_xmin, d_xmax, dims, pixel_num);
  }//max_iters
  

  cudaDeviceSynchronize();
  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_from_start;
  elapsed_seconds_from_start = t_end - t_start;
  std::cout << "time cost: " << elapsed_seconds_from_start.count()
    << std::endl;
  time_use = elapsed_seconds_from_start.count();

  /*
  *Result Evaluation
  */
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  //Copy the best position and best values back
  int best_fitness_index; 
  cudaMemcpy((void*)&best_fitness_index, (void*)d_best_fitness_index, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckError();
  
  cudaMemcpy((void*)g_best_pos, (void*) (d_pos + best_fitness_index * dims), dims * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy((void*)g_best, (void*)d_fitness, sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError();

  /*
  *Clean
  */
  cudaFree((void *)d_inVector);
  cudaFree((void *)d_zern);
  cudaFree((void *)d_xmin);
  cudaFree((void *)d_xmax);
  cudaFree(d_pos);
  cudaFree(d_states);
  cudaFree(d_alpha0);
  cudaFree(d_fitness);
  cudaFree(d_best_fitness_index);
  cudaCheckError();

  return;

}
 
int main() {
  //READ data from MAT-file
  std::vector<float> in_vector;
  std::vector<float> xmax;
  std::vector<float> xmin;
  std::vector<float> zern;
  matread("for_c.mat", "in_vector", in_vector); //make sure .mat is float data
  matread("for_c.mat", "xmax", xmax); //make sure .mat is float data
  matread("for_c.mat", "xmin", xmin); //make sure .mat is float data
  matread("for_c.mat", "zern", zern); //make sure .mat is float data
  

  const int pixel_num = int(in_vector.size() / 2);
  const int dims = xmin.size();

  //Firefly
  float g_best;
  float *g_best_pos = new float[dims];
  float time_use;
  std::vector<float> fit_rec;
  
  ApsoOptimization(in_vector, xmin, xmax, zern,
    pixel_num, dims, &g_best, g_best_pos,
    true, time_use, fit_rec);

  //Result evaluation
  std::cout << "delta (pi): " << g_best_pos[dims - 1] / M_PI  << std::endl;

  //write to .mat 
  std::vector<float> tp, tp1;
  tp.push_back(g_best);
  for (int i = 0; i < dims; ++i) tp1.push_back(g_best_pos[i]);
  matwrite("g_best.mat", "g_best", 1, 1, tp);
  matwrite("g_best_pos.mat", "g_best_pos", dims, 1, tp1);
  tp[0] = time_use;
  matwrite("time_use.mat", "time_use", 1, 1, tp);
  matwrite("fit_rec.mat", "fit_rec", fit_rec.size(), 1, fit_rec);

  delete[] g_best_pos;
  g_best_pos = nullptr;

  //system("pause");
  return 0;
}
