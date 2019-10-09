/*
The CPU version of Firefly; based on YPEA112 Firefly Algorithm
*/
#ifndef _FIREFLY_H
#define _FIREFLY_H

#include <random>
//#include <time.h>       /* time */
//#include <math.h>       /* atan2 */
//#include <fstream>      /* ofstream */
#include <chrono>
#include <algorithm>    /* std::min_element */
//#include <tgmath.h>     /* gamma function */
#include <ctgmath>     /* gamma function */

constexpr auto inf = 1e10f; //infinity
constexpr auto max_iters = 400;//500; //2048 //number of iterations
constexpr auto max_particles = 128; //number of particles 2048; for DE, should be 5d to 10d
constexpr auto M_PI = 3.14159265358979323846f;  /* pi */

class Firefly {
public:
  Firefly() :fitness(inf), dims(inf), pos(nullptr) {
  }
  /*
  Firefly(int in) {
    pos = new float[in]; //dims
    dims = in;
    fitness = inf;
  }
  */
  ~Firefly() {
    delete[] pos;
    pos = nullptr;
  }
  Firefly & operator=(const Firefly &b) {
    if (this->dims != b.dims) {
      printf("dims conflict!\n");
      return *this;
    }
    if (pos == nullptr) {
      printf("pos is empty!\n");
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

inline float Distance(const float *pos1, const float *pos2, const int dims);

float FitnessCalculate(float *pos, float const *in_vector,
  float const *zern, const int &dims, const int &pixel_num);

void FireflyOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const float &gamma, const float &beta0,
  float alpha, const float &alpha_damp,
  const int &pixel_num, const int &dims, float *g_best, float *g_best_pos,
  float &time_use, std::vector<float> &fit_rec);

#endif
