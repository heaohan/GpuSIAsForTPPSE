/*
GPU version of Firefly
 */
#include "g_firefly.cuh"

/*
Calculate the norm between two vectors
*/
inline float Distance(const float *pos1, const float *pos2, const int &dims) {
  float distance = 0.0f;
  for (int i = 0; i < dims; ++i) {
    float tp = pos1[i] - pos2[i];
    distance += tp * tp;
  }
  return sqrt(distance);
}

__device__
float GPUDistance(const float *pos1, const float *pos2, const int dims) {
  float distance = 0.0f;
  for (int i = 0; i < dims; ++i) {
    float tp = pos1[i] - pos2[i];
    distance += tp * tp;
  }
  return sqrt(distance);
}

/*
Initialize d_states
*/
//<<<max_particles*max_particles/32, 32>>>
__global__
void InitializeStates(unsigned long long seed, curandState *d_states) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  //curand_init(index, index, 0, &d_states[index]);
  curand_init(seed, index, 0, &d_states[index]);
}

 //Kernel to initialize particles
//<<<max_particles, dims>>>
__global__
void Initialize(Firefly *d_pop, Firefly *d_best_sol, curandState *d_states,
   float const *d_xmin, float const *d_xmax) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; //d_states[max_particles * max_particles] should
  //be enough for max_particles * dims
  d_pop[blockIdx.x].pos[threadIdx.x] = curand_uniform(&(d_states[index]));
  d_pop[blockIdx.x].pos[threadIdx.x] = d_xmin[threadIdx.x] + (d_xmax[threadIdx.x] - d_xmin[threadIdx.x])
    * d_pop[blockIdx.x].pos[threadIdx.x];

  //if (threadIdx.x == 9)
    //printf("%.2f\n", d_pos[index]);

  //Initializing the fitness of d_pop and d_best_sol

  if (index == 0)
    d_best_sol->fitness = inf;
}


/*
Calculate the fitness function for each particle
*/
__global__
void FitnessCalculate(Firefly *d_pop, float const *d_inVector,
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
    delta = d_pop[index].pos[dims - 1];
    for (int j = 0; j < dims - 1; ++j) {
      phi += d_zern[j * pixel_num + i] * d_pop[index].pos[j];
    }
    //tp_fitness += abs(cos(phi) - d_inVector[i]) + abs(cos(phi + delta) - d_inVector[pixel_num + i]);
    tp1 = cos(phi) - d_inVector[i];
    tp2 = cos(phi + delta) - d_inVector[pixel_num + i];
    tp_fitness += tp1 * tp1 + tp2 * tp2;
  }
  d_pop[index].fitness = tp_fitness;
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
initialize the d_best_sol with the min fitness d_pop
*/
__global__
void BestSolInitialize(const Firefly *d_pop, Firefly *d_best_sol) {
  float tp_min = inf;
  int tp_index = 0;
  for (int i = 0; i < max_particles; ++i) {
    if (tp_min > d_pop[i].fitness) {
      tp_min = d_pop[i].fitness;
      tp_index = i;
    }
  }
  *d_best_sol = d_pop[tp_index];

  /*
  printf("fitness of d_best_sol: %.2f, fitness of d_pop[%d]: %.2f\n",
    d_best_sol->fitness, tp_index, d_pop[tp_index].fitness);
  for (int i = 0; i < 11; ++i) {
    printf("i = %d, %.2f, %.2f\n", i, d_pop[tp_index].pos[i], d_best_sol->pos[i]);
  }
  */
}


/*
Kernel to obtain the min newsols for each particles;
< << max_particles, max_particles >> >
The first max_particles will be the best newsol for each block
*/
__global__
void SelectNewSol(Firefly *d_newsols, const int dims) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int tx = threadIdx.x;

  //Declare shared memory for staging the reduce phase
  extern __shared__ Firefly s[];
  Firefly *stage = s;
  float *stage_pos = (float *)&stage[max_particles]; //max_particles Firefly
  stage[tx].dims = dims;
  stage[tx].pos = &(stage_pos[tx * dims]);

  //Copy PBestY to shared memory
  stage[tx] = d_newsols[index];
  __syncthreads();

  //Perform the actual reduce
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tx < s) {
      if (stage[tx].fitness > stage[tx + s].fitness) {
        stage[tx] = stage[tx + s];
      }
    }
    __syncthreads();
  }
  //Copy results back into global memory
  if (tx == 0) {
    /*
    if (index == 0) {
      for (int i = 0; i < dims; ++i) {
        printf("d_newsols[0]: dim = %d, %.2f\n", i, d_newsols[0].pos[i]);
      }
      for (int i = 0; i < dims; ++i) {
        printf("stage[0]: dim = %d, %.2f\n", i, stage[0].pos[i]);
      }
    }
    */  
    d_newsols[blockIdx.x] = stage[0];
    /*
    if (index == 0) {
      for (int i = 0; i < dims; ++i) {
        printf("new d_newsols[0]: dim = %d, %.2f\n", i, d_newsols[0].pos[i]);
      }
    }
    */
  }
}

/*
Update d_best_sol use the found index
*/
/*
__global__
void BestSolUpdate(Firefly *d_best_sol, const Firefly *d_pop,
  int *d_best_fitness_index) {
  *d_best_sol = d_pop[*d_best_fitness_index];
  return;
}
*/

/*
Calculate fitness for a single trail
*/
__device__
float TrailFitnessCalculate(float *d_trail, float const *d_inVector, 
  float const *d_zern, const int dims, const int pixel_num) {

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
  return tp_fitness;
}

/*
Obtain newsols for each ij combination of particles
< << max_particles * max_particles / 32, 32 >> >
*/
__global__
void ObtainNewSol(Firefly *d_pop, Firefly *d_newsols,
  const float lambda, const float sigma_square, const float dmax,
  curandState *d_states, const float *d_inVector, const float *d_zern,
  const float *d_xmin, const float *d_xmax, const int dims, const int pixel_num,
  const float beta0, const float gamma, const float alpha) {

  int index = blockDim.x * blockIdx.x + threadIdx.x; //particle index
  int i = index / max_particles;
  int j = index % max_particles;
  curandState &local_state = d_states[index];

  //printf("i = %d, j = %d, %.2f, %.2f\n", i, j, d_pop[i].fitness, d_pop[j].fitness);

  d_newsols[index].fitness = inf; //initialize d_newsols in each iteration

  if (d_pop[j].fitness < d_pop[i].fitness) {

    float r = GPUDistance(d_pop[j].pos, d_pop[i].pos, dims) / dmax;
    float beta = beta0 * exp(-gamma * pow(r, 2));

    for (int k = 0; k < dims; ++k) {
      float step = curand_normal(&local_state) * sigma_square /
        powf(abs(curand_normal(&local_state)), 1 / lambda);
      step = ((1.59922f - 1.0f) * expf(-step / 2.737f) + 1) * step;
      float e = (d_xmax[k] - d_xmin[k]) / 100.0f * step;
      d_newsols[index].pos[k] = d_pop[i].pos[k] + beta *
        curand_uniform(&local_state) *
        (d_pop[j].pos[k] - d_pop[i].pos[k]) +
        alpha * e;
      if (d_newsols[index].pos[k] > d_xmax[k])
        d_newsols[index].pos[k] = d_xmax[k];
      if (d_newsols[index].pos[k] < d_xmin[k])
        d_newsols[index].pos[k] = d_xmin[k];
    }

    d_newsols[index].fitness = TrailFitnessCalculate(d_newsols[index].pos,
      d_inVector, d_zern, dims, pixel_num);
  }

  ////Set the current state of the PRNG
  //d_states[index] = local_state;

//  if (i == 0 || i == 1)
  //  printf("i = %d, j = %d, %.2f, newsol: %.2f, %.2f\n", i, j, 
    //  d_pop[i].fitness, d_newsols[index].fitness, d_pop[j].fitness);

}

/*
Update pop with the newsols; the best max_particles in 
the first max_particles newsols and max_particles d_pop
will be selected, and stored in d_pop;
Update d_best_sol with the best particles in d_pop
*/
__global__
void UpdatePop(Firefly *d_pop, Firefly *d_newsols, 
  Firefly *d_best_sol, int iter_count, float *d_best_fit_rec) {

  //Test
  /*
  for (int i = 0; i < max_particles; ++i) {
    printf("i = %d, old d_pop: %.2f\n", i, d_pop[i].fitness);
  }

  for (int i = 0; i < max_particles; ++i) {
    printf("i = %d, d_newsols: %.2f\n", i, d_newsols[i].fitness);
  }
  */

  float low_bound = 0.0f;
  for (int i = 0; i < max_particles; ++i) {
    float tp_min = inf;
    int tp_min_index;
    bool old_or_new; //old = 0; new = 1;
    for (int j = 0; j < max_particles; ++j) {
      if (d_pop[j].fitness < tp_min &&
        d_pop[j].fitness > low_bound) {
        tp_min = d_pop[j].fitness;
        tp_min_index = j;
        old_or_new = false;
      }
      if (d_newsols[j].fitness < tp_min &&
        d_newsols[j].fitness > low_bound) {
        tp_min = d_newsols[j].fitness;
        tp_min_index = j;
        old_or_new = true;
      }
    }
    Firefly tp_particle = d_pop[i];
    if (!old_or_new) { //old
      d_pop[i] = d_pop[tp_min_index];
      d_pop[tp_min_index] = tp_particle;
    }
    else { //new
      d_pop[i] = d_newsols[tp_min_index];
      d_newsols[tp_min_index] = tp_particle;
    }
    low_bound = d_pop[i].fitness;
  }

  for (int i = 0; i < max_particles; ++i) {
    //printf("i = %d, new d_pop: %.2f\n", i, d_pop[i].fitness);
    if (d_pop[i].fitness < d_best_sol->fitness)
      *d_best_sol = d_pop[i];
  }
  
  d_best_fit_rec[iter_count] = d_best_sol->fitness;
  printf("Iteration: %d, Best = %.2f\n", 
    iter_count, d_best_sol->fitness);

}

/*
__global__
void FitnessTest(Firefly *d_pop, Firefly *d_best_sol) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("index = %d, fitness = %.2f\n", index, d_pop[index].fitness);
  if (index == 0) {
    for (int i = 0; i < 11; ++i) {
      printf("%.2f\n", d_pop[0].pos[i]);
    }
    printf("d_best_sol -> fitness: %.2f, d_best_sol ->pos: \n", d_best_sol->fitness);
    for (int i = 0; i < 11; ++i) {
      printf("%.2f\n", d_best_sol->pos[i]);
    }
  }


}

__global__
void SimpleTest(Firefly *d_newsols) {
  printf("d_newsols[0].fitness = %.2f, d_newsols[1].fitness = %.2f\n",
    d_newsols[0].fitness, d_newsols[1].fitness);
  printf("");
}
*/

void FireflyOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &gamma, const float &beta0, float alpha, 
  const float &alpha_damp, const int &pixel_num, 
  const int &dims, float *g_best, float *g_best_pos,
  float &time_use, std::vector<float> &fit_rec){

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

  Firefly *d_pop;
  curandState *d_states;
  Firefly *d_best_sol;
  float *d_best_fit_rec; //record the best fitness in each iteration

  cudaMalloc((void**)&d_pop, max_particles * sizeof(Firefly));
  cudaMalloc((void**)&d_states, max_particles * max_particles * sizeof(curandState));
  cudaMalloc((void**)&d_best_sol, sizeof(Firefly));
  cudaMalloc((void**)&d_best_fit_rec, sizeof(float) * max_iters);
  
  float *d_pop_pos[max_particles]; 
  float *d_best_sol_pos;
  for (int i = 0; i < max_particles; ++i) {
    cudaMalloc((void**)&(d_pop_pos[i]), dims * sizeof(float)); //allocate memory at GPU, and assigned it to d_pop_pos[i] 
    cudaMemcpy(&(d_pop[i].pos), &(d_pop_pos[i]), sizeof(float *), cudaMemcpyHostToDevice); //The memory at GPU pointed by d_pop_pos[i] is pointed by d_pop[i].pos, too
    cudaMemcpy(&(d_pop[i].dims), &(dims), sizeof(int), cudaMemcpyHostToDevice);
  }
  cudaMalloc((void**)&(d_best_sol_pos), dims * sizeof(float));
  cudaMemcpy(&(d_best_sol->pos), &(d_best_sol_pos), sizeof(float *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_best_sol->dims), &(dims), sizeof(int), cudaMemcpyHostToDevice);

  //Initialize d_pop and d_states
  
  //curandGenerator_t gen;
  //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  //curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  //for (int i = 0; i < max_particles; ++i)
    //curandGenerateUniform(gen, d_pop[i].pos, dims);
  //curandDestroyGenerator(gen);
  
  //Generate the different seeds every time
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long long> distr;
  unsigned long long seed = distr(eng);
  
  InitializeStates<<<max_particles*max_particles/32, 32>>>(seed, d_states);
  Initialize <<<max_particles, dims>>> (d_pop, d_best_sol, d_states, d_xmin, d_xmax);

  FitnessCalculate <<<max_particles / 32, 32>>> (d_pop, d_inVector,
    d_zern, dims, pixel_num); // Calculate the fitness for each particle

  //Initialize d_best_sol
  BestSolInitialize <<<1, 1>>> (d_pop, d_best_sol);
  //float *tp_d_fitness;
  //cudaMalloc((void**)&tp_d_fitness, max_particles * sizeof(float));
  //cudaMemcpy(tp_d_fitness, d_fitness, max_particles * sizeof(float), cudaMemcpyDeviceToDevice);

  //ReduceFitnessMin << <max_particles / 32, 32 >> > (tp_d_fitness, d_best_fitness_index, 1);
  //ReduceFitnessMin << <1, max_particles / 32 >> > (tp_d_fitness, d_best_fitness_index, 2);

  //BestSolUpdate << <1, 1 >> > (d_best_sol, d_pop, d_best_fitness_index);

  //cudaFree((void *) d_best_fitness_index);
  //cudaFree((void *) tp_d_fitness);

  //FitnessTest << <max_particles / 32, 32 >> > (d_pop, d_best_sol);
  //cudaDeviceSynchronize();
  
  //Iteration
  //auto t_start = std::chrono::system_clock::now();//variables for time record
  const float lambda = 1.5f;
  const float sigma_square = powf( tgammaf(1.0f + lambda) * sinf(M_PI * lambda / 2.0f) /
    ( tgammaf((1.0f + lambda) / 2.0f)*lambda * powf(2.0f, (lambda - 1.0f) / 2.0f) ), 1.0f / lambda);
  const float dmax = Distance(xmin, xmax, dims);

  Firefly *d_newsols;
  cudaMalloc((void**)&d_newsols, max_particles * max_particles * sizeof(Firefly));

  float *d_newsols_pos[max_particles * max_particles];
  for (int i = 0; i < max_particles * max_particles; ++i) {
    cudaMalloc((void**)&(d_newsols_pos[i]), dims * sizeof(float)); //allocate memory at GPU, and assigned it to d_newsols_pos[i] 
    cudaMemcpy(&(d_newsols[i].pos), &(d_newsols_pos[i]), sizeof(float *), cudaMemcpyHostToDevice); //The memory at GPU pointed by d_newsols_pos[i] is pointed by d_newsols[i].pos, too
    cudaMemcpy(&(d_newsols[i].dims), &(dims), sizeof(int), cudaMemcpyHostToDevice);
  }
  //FitnessTest << <max_particles / 32, 32 >> > (d_pop, d_best_sol);
  //printf("sizeof(Firefly) = %d\n", sizeof(Firefly));
  //cudaDeviceSynchronize();

  auto t_start = std::chrono::system_clock::now();//variables for time record

  for (int it = 0; it < max_iters; ++it) { //it
    //newsols for each ij is created (max_particles * max_particles)
    ObtainNewSol <<< max_particles * max_particles / 32, 32 >>>
      (d_pop, d_newsols, lambda, sigma_square, dmax, d_states,
      d_inVector, d_zern, d_xmin, d_xmax, dims, pixel_num, beta0, gamma, alpha);
    
    //only the newsol with smallest fitness for each i is needed; SelectNewSol selects the newsol with
    //the smallest fitness and put it at the [i-1] position of d_newsols 
    SelectNewSol <<< max_particles, max_particles, 
      max_particles * sizeof(Firefly) + max_particles * dims * sizeof(float)>>> (d_newsols, dims);
    //SimpleTest << <1, 1 >> > (d_newsols);
    //cudaDeviceSynchronize();
    UpdatePop <<< 1, 1 >>> (d_pop, d_newsols, d_best_sol, it, d_best_fit_rec);
    alpha = alpha * alpha_damp;

    //cudaDeviceSynchronize(); //For printf display
  }

  cudaDeviceSynchronize();
  auto t_end_i = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_from_start = t_end_i - t_start;
      std::cout << "*******" << std::endl << "Time for the iteration(s): " << 
        elapsed_seconds_from_start.count() << std::endl << "********" << std::endl;

  time_use = elapsed_seconds_from_start.count();
  
  cudaMemcpy((void *)g_best, &(d_best_sol->fitness), sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)g_best_pos, d_best_sol_pos, dims * sizeof(float), cudaMemcpyDeviceToHost);
  
  if (fit_rec.size() < max_iters) printf("Error: fit_rec are not big enough!\n");
  cudaMemcpy((void *)fit_rec.data(), d_best_fit_rec, max_iters * sizeof(float), cudaMemcpyDeviceToHost);

  //Clean
  cudaFree((void *) d_inVector);
  cudaFree((void *) d_zern);
  cudaFree((void *) d_xmin);
  cudaFree((void *) d_xmax);

  cudaFree((void *) d_pop);
  cudaFree((void *) d_states);
  cudaFree((void *) d_best_sol);
  cudaFree((void *) d_newsols);
  cudaFree((void *)d_pop_pos);
  cudaFree((void *)d_best_sol_pos);
  cudaFree((void *)d_newsols);
  cudaFree((void *)d_newsols_pos);
}

