/*
The CPU version of Firefly; based on YPEA112 Firefly Algorithm
*/
//#define _CRT_SECURE_NO_WARNINGS

#include "firefly.h"

inline float Distance(const float *pos1, const float *pos2, const int dims) {
  float distance = 0.0f;
  for (int i = 0; i < dims; ++i) {
    float tp = pos1[i] - pos2[i];
    distance += tp * tp;
  }
  return sqrt(distance);
}

float FitnessCalculate(float *pos, float const *in_vector,
  float const *zern, const int &dims, const int &pixel_num) {
  float phi;
  float delta;
  float tp1, tp2;

  float tp_fitness = 0.0f;
  for (int i = 0; i < pixel_num; ++i) {
    phi = 0.0f;
    delta = pos[dims - 1];
    for (int j = 0; j < dims - 1; ++j) {
      phi += zern[j * pixel_num + i] * pos[j];
    }
    //tp_fitness += abs(cos(phi) - d_inVector[i]) + abs(cos(phi + delta) - d_inVector[pixel_num + i]);
    tp1 = cos(phi) - in_vector[i];
    tp2 = cos(phi + delta) - in_vector[pixel_num + i];
    tp_fitness += tp1 * tp1 + tp2 * tp2;
  }
  return tp_fitness;
}

void FireflyOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &gamma, const float &beta0,
  float alpha, const float &alpha_damp,
  const int &pixel_num, const int &dims, float *g_best, float *g_best_pos,
  float &time_use, std::vector<float> &fit_rec) {

  //Initialize population
  Firefly *pop = new Firefly[max_particles];
  for (int i = 0; i < max_particles; ++i) {
    pop[i].dims = dims;
    pop[i].pos = new float[dims];
  }

  Firefly best_sol;
  best_sol.dims = dims;
  best_sol.pos = new float[dims];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> uni_dist(0.0f, 1.0f);

  //best_sol test
  int tp_index;
  for (int i = 0; i < max_particles; ++i) {
    for (int j = 0; j < dims; ++j) {
      pop[i].pos[j] = xmin[j] +
        (xmax[j] - xmin[j]) * uni_dist(gen);
    }
    pop[i].fitness = FitnessCalculate(pop[i].pos, in_vector,
      zern, dims, pixel_num);
    if (pop[i].fitness <= best_sol.fitness) {
      best_sol = pop[i];
      tp_index = i;
    }
      
  }

  //pop and best_sol test
  for (int i = 0; i < max_particles; ++i) {
    printf("pop[%d].fitness: %.2f\n", i, pop[i].fitness);
  }
  printf("best_sol.fitness = %.2f\n", best_sol.fitness);
  for (int i = 0; i < dims; ++i) {
    printf("pop[0].pos[%d]: %.2f, pop[%d].pos[%d]: %.2f, best_sol.pos[%d]: %.2f\n",
      i, pop[0].pos[i], tp_index, i, pop[tp_index].pos[i], i, best_sol.pos[i]);
  }
  

  //Iteration
  auto t_start = std::chrono::system_clock::now();//variables for time record
  const float lambda = 1.5f;
  const float sigma_square = powf( tgammaf(1.0f + lambda) * sin(M_PI * lambda / 2.0f) /
    ( tgammaf((1.0f + lambda) / 2.0f)*lambda * powf(2.0f, (lambda - 1.0f) / 2.0f) ), 1.0f / lambda);
  std::normal_distribution<float> norm_dist(0.0f, 1.0f);
  const float dmax = Distance(xmin, xmax, dims);

  Firefly *newpop = new Firefly[max_particles];
  for (int i = 0; i < max_particles; ++i) {
    newpop[i].dims = dims;
    newpop[i].pos = new float[dims];
  }

  //float *e = new float[dims];
  Firefly newsol;
  newsol.dims = dims;
  newsol.pos = new float[dims];

  for (int it = 0; it < max_iters; ++it) { //it

    for (int i = 0; i < max_particles; ++i) { //i
      newpop[i].fitness = inf;
      //printf("i = %d\n", i);
      for (int j = 0; j < max_particles; ++j) { //j
        if (pop[j].fitness < pop[i].fitness) {
          float rij = Distance(pop[i].pos, pop[j].pos, dims) / dmax;
          float beta = beta0 * exp(-gamma * pow(rij, 2));

          for (int k = 0; k < dims; ++k) {
            float step = norm_dist(gen) * sigma_square /
              powf(abs(norm_dist(gen)), 1/lambda);
            step = ((1.59922f - 1.0f) * expf(-step / 2.737f) + 1) * step;
            float e = (xmax[k] - xmin[k]) / 100 * step;
            newsol.pos[k] = pop[i].pos[k] + beta *
              uni_dist(gen) *
              (pop[j].pos[k] - pop[i].pos[k]) +
              alpha * e;
            if (newsol.pos[k] > xmax[k])
              newsol.pos[k] = xmax[k];
            if (newsol.pos[k] < xmin[k])
              newsol.pos[k] = xmin[k];
          }
          newsol.fitness = FitnessCalculate(newsol.pos,
              in_vector, zern, dims, pixel_num);

          //newsol test
          //printf("Iteration: %d, i = %d, j = %d, newsol.fitness = %.2f\n",
            //it, i, j, newsol.fitness);

          if (newsol.fitness <= newpop[i].fitness) {
            newpop[i] = newsol;
            if (newpop[i].fitness <= best_sol.fitness)
              best_sol = newpop[i];
          }
        }
      }

      //newpop and best_sol test
      //printf("******\n newpop[%d].fitness = %.2f, best_sol.fitness = %.2f\n*******\n",
        //i, newpop[i].fitness, best_sol.fitness);
    }
    //Merge: select max_particles particles with min fitness
    //from pop and new_pop; pop will have the particles with
    //min fitness among pop and new_pop;

    //newpop and pop test
    //for (int i = 0; i < max_particles; ++i) {
      //printf("pop[%d].fitness = %.2f\n", i, pop[i].fitness);
      //printf("newpop[%d].fitness = %.2f\n", i, newpop[i].fitness);
    //}

    float low_bound = 0.0f;
    for (int i = 0; i < max_particles; ++i) {
      float tp_min = inf;
      int tp_min_index;
      bool old_or_new; //old = 0; new = 1;
      for (int j = 0; j < max_particles; ++j) {
        if (pop[j].fitness < tp_min &&
          pop[j].fitness > low_bound) {
          tp_min = pop[j].fitness;
          tp_min_index = j;
          old_or_new = false;
        }
        if (newpop[j].fitness < tp_min &&
          newpop[j].fitness > low_bound) {
          tp_min = newpop[j].fitness;
          tp_min_index = j;
          old_or_new = true;
        }
      }
      if (!old_or_new) { //old
        Firefly tp_particle;
        tp_particle.dims = dims;
        tp_particle.pos = new float[dims];

        tp_particle = pop[i];
        pop[i] = pop[tp_min_index];
        pop[tp_min_index] = tp_particle;
      }
      else { //new
        Firefly tp_particle;
        tp_particle.dims = dims;
        tp_particle.pos = new float[dims];

        tp_particle = pop[i];
        pop[i] = newpop[tp_min_index];
        newpop[tp_min_index] = tp_particle;    
      }
      low_bound = pop[i].fitness;
    }

    //pop test
    //for (int i = 0; i < max_particles; ++i) {
      //printf("pop[%d].fitness = %.2f\n", i, pop[i].fitness);
    //}

    /*
    for (int i = 0; i < max_particles; ++i) {
      printf("pop index: %d, pop fitness: %.2f, new pop fitness: %.2f\n", 
        i, pop[i].fitness, newpop[i].fitness);
    }
    */

    alpha = alpha * alpha_damp;
    //display
    fit_rec.push_back(best_sol.fitness);
    printf("Iteration: %d, best fitness = %.2f\n", it, best_sol.fitness);
  
  }
  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_from_start = t_end - t_start;
  printf("Iteration time: %.2f\n", elapsed_seconds_from_start.count());
  time_use = elapsed_seconds_from_start.count();
  *g_best = best_sol.fitness;
  for (int it = 0; it < dims; ++it)
    g_best_pos[it] = best_sol.pos[it];

  //Clean
  //delete[] newpop;
  //delete[] pop;
}



