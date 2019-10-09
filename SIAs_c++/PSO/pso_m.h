/*
The CPU version PSO
*/

#ifndef _PSO_H
#define _PSO_H

constexpr auto inf = 1e10f; // 9999.99f; //infinity
constexpr auto max_iters = 400; // 1000; // 5000;// 400;//200;//2048; //number of iterations
constexpr auto max_particles = 2048;// 128;// 2048;//100; //2048;//2048; //number of particles
//constexpr auto chi = 0.729844f; //chi (constriction factor)
constexpr auto M_PI = 3.14159265358979323846;  /* pi */

#include <random>
//#include <time.h>       /* time */
//#include <math.h>       /* atan2 */
//#include <fstream>      /* ofstream */
#include <chrono>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

void PSO(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  float &time_use, std::vector<float> &fit_rec);

#endif
