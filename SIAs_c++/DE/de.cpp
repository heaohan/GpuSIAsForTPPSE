#include "de.h"

void FitnessCalculate(float *pos, float *fitness, const int num_particles,
  float const *in_vector, float const *zern, const int &dims, const int &pixel_num) {
  float phi;
  float delta;
  float tp1, tp2;
  for (int index = 0; index < num_particles; ++index) {
    float tp_fitness = 0.0f;
      for (int i = 0; i < pixel_num; ++i) {
        phi = 0.0f;
        delta = pos[index * dims + dims - 1];
        for (int j = 0; j < dims - 1; ++j) {
          phi += zern[j * pixel_num + i] * pos[index * dims + j];
        }
        //tp_fitness += abs(cos(phi) - d_inVector[i]) + abs(cos(phi + delta) - d_inVector[pixel_num + i]);
        tp1 = cos(phi) - in_vector[i];
        tp2 = cos(phi + delta) - in_vector[pixel_num + i];
        tp_fitness += tp1 * tp1 + tp2 * tp2;
      }
      fitness[index] = tp_fitness;
  }
}

void DeOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &crossover_rate, const float &difference_factor,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  bool iter_option, float &time_use, std::vector<float> &fit_rec, std::vector<float> &iter_rec) {

  float *pos = new float[max_particles * dims];

  //Initialize variables
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis_particle(0, max_particles-1);
  std::uniform_int_distribution<> dis_dim(0, dims-1);

  float *fitness = new float[max_particles];
  for (int i = 0; i < max_particles; ++i) {
    for (int j = 0; j < dims; ++j) {
      pos[i * dims + j] = xmin[j] +
        (xmax[j] - xmin[j]) * std::generate_canonical<float, 32>(gen);
    }
  }
  FitnessCalculate(pos, fitness, max_particles, in_vector, zern, dims, pixel_num);

  *g_best = fitness[0];
  int g_best_index = 0;
  for (int i = 1; i < max_particles; ++i) {
    if (*g_best > fitness[i]) {
      *g_best = fitness[i];
      g_best_index = i;
    }
  }
  for (int i = 0; i < dims; ++i)
    g_best_pos[i] = pos[g_best_index * dims + i];


  //Iteration
  float *trail = new float[dims];
  float *best_with_change = new float[1000];
  int count_best_with_change = 0;
  const int NE = 10; //Termination criteria
  const float TOLFUN = 1e-3f; //Tolerance of the fitness change
  auto t_start = std::chrono::system_clock::now();//variables for time record
  std::chrono::duration<double> elapsed_seconds_from_start,
    elapsed_seconds_from_start_i;

  for (int i = 0; i < max_iters; ++i) { //i
    auto t_start_i = std::chrono::system_clock::now();

    for (int index = 0; index < max_particles; ++index) { //index
      float F = difference_factor * 
        std::generate_canonical<float, 32>(gen);
      
      //Choose 3 distinct particles (!= index)
      int r[3];
      r[0] = dis_particle(gen); r[1] = dis_particle(gen); r[2] = dis_particle(gen);
      while ((r[0] == index || r[1] == index || r[2] == index) ||
        (r[0] == r[1] || r[0] == r[2] || r[1] == r[2])) {
        r[0] = dis_particle(gen); r[1] = dis_particle(gen); r[2] = dis_particle(gen);
      }

      //Mutation and crossover loops
      for (int k = 0; k < dims; ++k) {
        trail[k] = pos[index * dims + k];
      }
      int j = dis_dim(gen);
      for (int k = 0; k < dims; ++k) { //k
        if (std::generate_canonical<float, 32>(gen) <= crossover_rate
          || k == j) {
          trail[k] = pos[r[0] * dims + k] + F *
            (pos[r[1] * dims + k] - pos[r[2] * dims + k]);

          if (trail[k] > xmax[k] ||
            trail[k] < xmin[k]) //reinitialization
            trail[k] = xmin[k] +
            std::generate_canonical<float, 32>(gen)
            * (xmax[k] - xmin[k]);
        }
      }

      //Fitness calculation and selection
      float fitness_trail;
      FitnessCalculate(trail, &fitness_trail, 1,
        in_vector, zern, dims, pixel_num);
      if (fitness_trail <= fitness[index]) {
        for (int k = 0; k < dims; ++k)
          pos[index * dims + k] = trail[k];
        fitness[index] = fitness_trail;

        if (fitness_trail <= *g_best) {
          for (int k = 0; k < dims; ++k)
            g_best_pos[k] = trail[k];
          *g_best = fitness_trail;
          fit_rec.push_back(fitness_trail);
          iter_rec.push_back(i);
          printf("Changed best: %.2f, index: %d, iter: %d\n", *g_best, count_best_with_change + 1, i);
          best_with_change[count_best_with_change++] = fitness_trail;
        }
      }
    }
   
    auto t_end_i = std::chrono::system_clock::now();
    elapsed_seconds_from_start = t_end_i - t_start;
    elapsed_seconds_from_start_i = t_end_i - t_start_i;
    //std::cout << "###########" << std::endl << "Iteration number: " << i <<
      //", time for the iteration (s): " << elapsed_seconds_from_start_i.count()
      //<< ", time since beginning of iteration (s): " 
      //<< elapsed_seconds_from_start.count() << std::endl << "###########" << std::endl;

    //Check the termination criterion
    if ((!iter_option) && count_best_with_change > NE &&
      abs(best_with_change[count_best_with_change - 1] -
        best_with_change[count_best_with_change - 1 - NE]) < TOLFUN)
      break;
  }
  
  elapsed_seconds_from_start = std::chrono::system_clock::now()- t_start;
  time_use = elapsed_seconds_from_start.count();
  //Clean
  delete[] pos;
  delete[] fitness;
  delete[] trail;
  delete[] best_with_change;
}


