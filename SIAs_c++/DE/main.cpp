/*
GPU version of Firefly
 */
#include "mat_rw.h"
#include "de.h"

/*
 * Host code
 */
int main()
{
  /*
  *Transfer the data from AUTO GENERATION
  */
  std::vector<float> in_vector;
  std::vector<float> xmin;
  std::vector<float> xmax;
  std::vector<float> zern;

  matread("for_c.mat", "in_vector", in_vector); //make sure .mat is float data
  matread("for_c.mat", "xmax", xmax); //make sure .mat is float data
  matread("for_c.mat", "xmin", xmin); //make sure .mat is float data
  matread("for_c.mat", "zern", zern); //make sure .mat is float data
  
  const int pixel_num = int(in_vector.size() / 2);
  const int dims = xmin.size();
  float crossover_rate = 0.5f; //[0, 1]
  float difference_factor = 0.6f; //[0, 1]
  
  float g_best;
  float *g_best_pos = new float[dims];
  float time_use;
  std::vector<float> fit_rec;
  std::vector<float> iter_rec;

  //If iter_option is true, the PSO terminates when it reaches the
  //max_iters; otherwise, termination rule is used
  DeOptimization(in_vector.data(), xmin.data(), xmax.data(), zern.data(), 
    crossover_rate, difference_factor, pixel_num, dims, &g_best, g_best_pos, 
    true, time_use, fit_rec, iter_rec);

  //Write to .mat
  printf("g_best = %.2f\n", g_best);
  std::vector<float> tp, tp1;
  tp.push_back(g_best);
  fit_rec.shrink_to_fit();
  iter_rec.shrink_to_fit();
  for (int i = 0; i < dims; ++i) tp1.push_back(g_best_pos[i]);
  matwrite("g_best.mat", "g_best", 1, 1, tp);
  matwrite("g_best_pos.mat", "g_best_pos", dims, 1, tp1);
  tp[0] = time_use;
  matwrite("time_use.mat", "time_use", 1, 1, tp);
  matwrite("fit_rec.mat", "fit_rec", fit_rec.size(), 1, fit_rec);
  matwrite("iter_rec.mat", "iter_rec", iter_rec.size(), 1, iter_rec);

  /*
  *Clean
  */
  delete[] g_best_pos;
  g_best_pos = nullptr;
  
  //system("pause");
  return 0;
}

