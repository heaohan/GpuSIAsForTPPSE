#ifndef _DE_H
#define _DE_H

constexpr auto inf = 1e10f;//9999.99f; //infinity
constexpr auto max_iters = 5000; //2048 //number of iterations
constexpr auto max_particles = 128; //number of particles 2048; for DE, should be 5d to 10d
constexpr auto M_PI = 3.14159265358979323846f;  /* pi */

#define _CRT_SECURE_NO_WARNINGS

#include <random>
//#include <time.h>       /* time */
//#include <math.h>       /* atan2 */
//#include <fstream>      /* ofstream */
#include <chrono>
#include <algorithm>    /* std::min_element */
#include <iostream>

void FitnessCalculate(float *pos, float *fitness, const int num_particles,
  float const *in_vector, float const *zern, const int &dims, const int &pixel_num);

//If iter_option is true, the PSO terminates when it reaches the
//max_iters; otherwise, termination rule is used
void DeOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &crossover_rate, const float &difference_factor,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  bool iter_options, float &time_use, std::vector<float> &fit_rec,
  std::vector<float> &iter_rec);


 

#endif
