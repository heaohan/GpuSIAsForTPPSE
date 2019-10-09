#include "g_de.cuh"
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
  //Each thread gets a different seed, different sequence number and no offset
  //curand_init(index, index, 0, &d_states[index]);
  //Each thread gets one seed every run, but different sequence
  //number; the seed is also different in different run
  if (index < max_particles)
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
  const int step) {
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
}

/*
Kernel to generate the random integers in the DE
*/
__device__
int RandomInteger(curandState *state, const int min_rand_int,
  const int max_rand_int) {
  float rand_f = curand_uniform(state);
  rand_f *= (max_rand_int - min_rand_int + 0.999999);
  rand_f += min_rand_int;
  return (int)truncf(rand_f);
}

/*
Calculate fitness for a single trail
*/
__device__
void TrailFitnessCalculate(float *d_trail, float *fitness_trail,
  float const *d_inVector, float const *d_zern, const int dims, const int pixel_num) {

  float tp_fitness = 0.0f;
  float phi;
  float delta;
  float tp1, tp2;
  for (int i = 0; i < pixel_num; ++i) {
    phi = 0.0f;
    delta = d_trail[dims - 1];
    for (int j = 0; j < dims - 1; ++j) {
      phi += d_zern[j * pixel_num + i] * d_trail[j];
    }
    //tp_fitness += abs(cos(phi) - d_inVector[i]) + abs(cos(phi + delta) - d_inVector[pixel_num + i]);
    tp1 = cos(phi) - d_inVector[i];
    tp2 = cos(phi + delta) - d_inVector[pixel_num + i];
    tp_fitness += tp1 * tp1 + tp2 * tp2;
  }
  *fitness_trail = tp_fitness;
}

/*
d_zern: pixel_num * dims [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim1,
(pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim2, ...)]
*/
//<<<max_particles/32, 32>>>
__global__
void Iterate(float *d_pos, curandState *d_states, 
  float *d_fitness, float *d_trails, float *d_trail_fitness,
  float const *d_inVector, float const *d_zern,
  float const *d_xmin, float const *d_xmax, const int dims, const int pixel_num) {
  //parameters
  int index = blockDim.x * blockIdx.x + threadIdx.x; //particle index
  curandState &local_state = d_states[index];

  const float difference_factor = 0.6f;
  const float crossover_rate = 0.5f;
  float F = difference_factor * curand_uniform(&local_state);

  //Choose 3 distinct particles (!= index) (range: [0, max_particles-1]
  int r0, r1, r2;
  r0 = RandomInteger(&local_state, 0, max_particles - 1);
  r1 = RandomInteger(&local_state, 0, max_particles - 1);
  r2 = RandomInteger(&local_state, 0, max_particles - 1);
  while (r0 == index || r1 == index || r2 == index ||
    r0 == r1 || r0 == r2 || r1 == r2) {
    r0 = RandomInteger(&local_state, 0, max_particles - 1);
    r1 = RandomInteger(&local_state, 0, max_particles - 1);
    r2 = RandomInteger(&local_state, 0, max_particles - 1);
  }

  //Mutation and crossover loops
  for (int k = 0; k < dims; ++k) {
    d_trails[index * dims + k] = d_pos[index * dims + k];
  }

  int j = RandomInteger(&local_state, 0, dims - 1);
  for (int k = 0; k < dims; ++k) {
    if (curand_uniform(&local_state) <= crossover_rate || k == j) {
      d_trails[index * dims + k] = d_pos[r0 * dims + k] + F *
        (d_pos[r1 * dims + k] - d_pos[r2 * dims + k]);

      //if (blockIdx.x == 0 && threadIdx.x == 0) {
        //printf("k = %d, j = %d\n", k, j);
      //}

      if (d_trails[index * dims + k] > d_xmax[k] ||
        d_trails[index * dims + k] < d_xmin[k]) //re-initialization
        d_trails[index * dims + k] = d_xmin[k] +
        curand_uniform(&local_state) * (d_xmax[k] - d_xmin[k]);
    }
  }

  //Fitness calculation of the trail and update d_pos if trail is better 
  float fitness_trail;
  TrailFitnessCalculate(&(d_trails[index * dims]), &fitness_trail, d_inVector,
    d_zern, dims, pixel_num);
  //bool accepted_flag = false;
  if (fitness_trail <= d_fitness[index]) { //trail accepted
    //accepted_flag = true;
    d_trail_fitness[index] = fitness_trail;
    d_fitness[index] = fitness_trail;
  }
  else
    d_trail_fitness[index] = inf;

  //printf("accepted? %d, %d, d_fitness[%d] = %.2f, d_trail_fitness[%d] = %.2f\n", accepted_flag, d_fitness[index] == d_trail_fitness[index],index, d_fitness[index], index, d_trail_fitness[index]);

  ////Set the current state of the PRNG
  //d_states[index] = local_state;
}

__global__
void PosUpdate(float *d_pos, const float *d_trails, const float *d_trail_fitness,
  const float *d_xmax, const float *d_xmin, const int dims) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (d_trail_fitness[index] < inf) {
    for (int k = 0; k < dims; ++k) {
      d_pos[index * dims + k] = d_trails[index * dims + k];
      //Ensure position values are within range
      if (d_pos[index * dims + k] > d_xmax[k])
        d_pos[index * dims + k] = d_xmax[k];
      if (d_pos[index * dims + k] < d_xmin[k])
        d_pos[index * dims + k] = d_xmin[k];
    }
  }
      
}

void DE(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &crossover_rate, const float &difference_factor,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  bool iter_option, float &time_use, std::vector<float> &fit_rec, std::vector<float> &iter_rec) {

  //GPU memory allocation
  float const *d_inVector;
  float const *d_zern;
  float const *d_xmin;
  float const *d_xmax;

  cudaMalloc((void**)&d_inVector, 2 * pixel_num * sizeof(float));
  cudaMemcpy((void*)d_inVector, (void*)in_vector, 2 * pixel_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_zern, (dims - 1) * pixel_num * sizeof(float));
  cudaMemcpy((void*)d_zern, (void*)zern, (dims - 1) * pixel_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_xmin, dims * sizeof(float));
  cudaMemcpy((void*)d_xmin, (void*)xmin, dims * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_xmax, dims * sizeof(float));
  cudaMemcpy((void*)d_xmax, (void*)xmax, dims * sizeof(float), cudaMemcpyHostToDevice);

  float *d_pos;
  curandState *d_states;
  float *d_fitness;
  int *d_best_fitness_index;
  cudaMalloc((void**)&d_pos, max_particles * dims * sizeof(float));
  //cudaMalloc((void**)&d_states, max_particles * dims * sizeof(curandState));
  cudaMalloc((void**)&d_states, max_particles * sizeof(curandState));
  cudaMalloc((void**)&d_fitness, max_particles * sizeof(float));
  cudaMalloc((void**)&d_best_fitness_index, max_particles * sizeof(int));

  //Initialize d_pos and d_states
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  curandGenerateUniform(gen, d_pos, max_particles * dims);
  curandDestroyGenerator(gen);


  //Adjust the d_pos 
  //Generate the different seeds every time
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long long> distr;
  unsigned long long seed = distr(eng);
  Initialize << <max_particles, dims >> > (seed, d_pos, d_states, d_xmin, d_xmax);

  //Initialize d_fitness, g_best, g_best_pos
  FitnessCalculate << <max_particles / 32, 32 >> > (d_pos, d_fitness, d_inVector,
    d_zern, dims, pixel_num); // Calculate the fitness for each particle

  float *tp_d_fitness;
  cudaMalloc((void**)&tp_d_fitness, max_particles * sizeof(float));
  cudaMemcpy(tp_d_fitness, d_fitness, max_particles * sizeof(float), cudaMemcpyDeviceToDevice);

  ReduceFitnessMin << <max_particles / 32, 32 >> > (tp_d_fitness, d_best_fitness_index, 1);
  ReduceFitnessMin << <1, max_particles / 32 >> > (tp_d_fitness, d_best_fitness_index, 2);

  int g_best_index;
  cudaMemcpy(g_best, tp_d_fitness, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&g_best_index, d_best_fitness_index, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(g_best_pos, &(d_pos[g_best_index * dims]), dims * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree((void *) tp_d_fitness);
  cudaFree((void *) d_best_fitness_index);

  //Iteration
  float *best_with_change = new float[1000]; //record the g_best change
  int count_best_with_change = 0;
  const int NE = 10; //Termination criteria
  const float TOLFUN = 0.001f; //Tolerance of the fitness change
  std::chrono::duration<double> elapsed_seconds_from_start;

  float *trail_fitness = new float[max_particles];
  float *d_trail_fitness;
  float *d_trails;
  cudaMalloc((void**)&d_trail_fitness, max_particles * sizeof(float));
  cudaMalloc((void**)&d_trails, dims * max_particles * sizeof(float));

  auto t_start = std::chrono::system_clock::now();//variables for time record
  for (int i = 0; i < max_iters; ++i) { //i
    //auto t_start_i = std::chrono::system_clock::now();
    //DE for each particles to obtain the trail and
    //d_trail_fitness, which records the fitness of the trail of each particles (= inf
    //if the trail is not accepted); it is then used for updating the best and best_pos.
    //printf("i = %d\n", i);
    Iterate << <max_particles / 32, 32 >> > (d_pos,
      d_states, d_fitness, d_trails, d_trail_fitness, d_inVector, d_zern, d_xmin, d_xmax, dims, pixel_num);

    //Kernel for updating the d_pos according to the d_trail_fitness (<inf means accepted)
    PosUpdate << <max_particles / 32, 32 >> > (d_pos, d_trails, d_trail_fitness, d_xmax, d_xmin, dims);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(trail_fitness, d_trail_fitness, max_particles * sizeof(float), cudaMemcpyDeviceToHost);

    //update g_best and g_best_pos by comparing each accepted trail (<= inf)
    int tp_min_index;
    bool updated_flag = false; //flag shows whether the g_best is updated by the accepted trails
    for (int index = 0; index < max_particles; ++index) {
      if (trail_fitness[index] <= *g_best && trail_fitness[index] < inf) {
        *g_best = trail_fitness[index];
        best_with_change[count_best_with_change++] = trail_fitness[index];
        tp_min_index = index;
        updated_flag = true;
        fit_rec.push_back(*g_best);
        iter_rec.push_back(i);
        printf("Changed best: %.3f, index: %d, iter: %d\n", *g_best, count_best_with_change - 1, i);
      }
    }
    if (updated_flag)
      cudaMemcpy(g_best_pos, &(d_pos[tp_min_index * dims]), dims * sizeof(float), cudaMemcpyDeviceToHost);
    
    //auto t_end_i = std::chrono::system_clock::now();
    //elapsed_seconds_from_start = t_end_i - t_start;
    //elapsed_seconds_from_start_i = t_end_i - t_start_i;
    //std::cout << "###########" << std::endl << "Iteration number: " << i <<
      //", time for the iteration (s): " << elapsed_seconds_from_start_i.count()
      //<< ", time since beginning of iteration (s): " 
      //<< elapsed_seconds_from_start.count() << std::endl << "###########" << std::endl;

    //Check the termination criterion
    if ( (!iter_option) && count_best_with_change > NE &&
      abs(best_with_change[count_best_with_change - 1] -
        best_with_change[count_best_with_change - 1 - NE]) < TOLFUN) {
      break;
    }

  }

  cudaDeviceSynchronize();
  auto t_end_i = std::chrono::system_clock::now();
  elapsed_seconds_from_start = t_end_i - t_start;
      std::cout << "*******" << std::endl << "Time for the iteration(s): " << 
        elapsed_seconds_from_start.count() << std::endl << "********" << std::endl;

  time_use = elapsed_seconds_from_start.count();
  //Clean
  cudaFree((void *) d_inVector);
  cudaFree((void *) d_zern);
  cudaFree((void *) d_xmin);
  cudaFree((void *) d_xmax);
  cudaFree((void *) d_pos);
  cudaFree((void *) d_states);
  cudaFree((void *) d_fitness);
  delete[] best_with_change;
  delete[] trail_fitness;
  cudaFree((void *) d_trail_fitness);
  cudaFree((void *)d_trails);

}


