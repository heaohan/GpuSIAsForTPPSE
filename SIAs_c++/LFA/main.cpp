#include "mat_rw.h"
#include "firefly.h"

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
  float gamma = 1.0f; //Light Absorption Coefficient
  //float gamma = 1.0f / powf(Distance(xmin.data(), xmax.data(), dims) / 4, 2.0f); //Light Absorption Coefficient; 
  // characteristic length: 1/4 of the biggest solution distance; beta = 0.37 * beta0 at this length 
  float beta0 = 1.0f; //Attraction Coefficient Base Value
  float alpha = 1.0f; //0.2f; //Mutation Coefficient
  float alpha_damp = 0.98f; //Mutation Coefficient Damping Ratio

  FireflyOptimization(in_vector.data(), xmin.data(), xmax.data(), zern.data(),
    gamma, beta0, alpha, alpha_damp, pixel_num, dims, &g_best, g_best_pos, time_use, fit_rec);

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
