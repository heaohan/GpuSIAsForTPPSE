#ifndef _APSO_H
#define _APSO_H

/*
The CPU version of APSO; compared to APSO.cpp, APSOChange.cpp
changes the setting of the alpha: alpha0 will be set according to
the xmin and xmax
*/
constexpr auto inf = 1e10f; //9999.99f; //infinity
constexpr auto max_iters = 400;//2048; //number of iterations
constexpr auto max_particles = 2048; //number of particles 2048
constexpr auto chi = 0.729844f; //chi (constriction factor)
constexpr auto M_PI = 3.14159265358979323846;  /* pi */

//#define _CRT_SECURE_NO_WARNINGS

#include <random>
//#include <time.h>       /* time */
//#include <math.h>       /* atan2 */
//#include <fstream>      /* ofstream */
#include <chrono>
#include <algorithm>    /* std::min_element */
#include <iostream>

void ApsoIterate(float *pos,
  const float *in_vector, const float *zern,
  const float *xmin, const float *xmax, const int &dims, const int &pixel_num,
  std::mt19937 &gen, int &count_min_with_change,
  float *min_with_change, float *current_min_pos, float &current_min_fitness, float &global_min, float *alpha);


//If iter_option is true, the PSO terminates when it reaches the
//max_iters; otherwise, termination rule is used
void ApsoOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  bool iter_option, float &time_use, std::vector<float> &fit_rec);

#endif // !_APSO_H


