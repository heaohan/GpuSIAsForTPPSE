#include "apso.h"

void ApsoIterate(float *pos,
  const float *in_vector, const float *zern,
  const float *xmin, const float *xmax, const int &dims, const int &pixel_num,
  std::mt19937 &gen,  int &count_min_with_change,
  float *min_with_change, float *current_min_pos, float &current_min_fitness, float &global_min, float *alpha) {


  float beta = 0.5f;
  float r1;
  std::vector<float> fitness_all(max_particles, 0.0f);

  // Calculate fitness at time 't'; determine personal (particle) best x at 't' 
  for (int index = 0; index < max_particles; ++index) {
    //Calculate fitness of particle (shifted fringe patterns) at time 't'
    /*
    index: particle index
    d_zern: pixel_num * dims [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim1,
    (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_dim2, ...]
    d_inVector: [store in this way (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_image1,
    (pixel1, pixel2, pixel3, ..., pixel_pixel_num)_image2]
    d_pos: [store in this way (dim1, dim2, dim3, ...)_particle1,
    (dim1, dim2, dim3, ...)_particle2, ...]
    */
    for (int i = 0; i < pixel_num; ++i) {
      float phi = 0.0f;
      float delta = pos[index * dims + dims - 1];
      for (int j = 0; j < dims - 1; ++j) {
        phi += zern[j * pixel_num + i] * pos[index * dims + j];
      }
      float tp1 = cos(phi) - in_vector[i];
      float tp2 = cos(phi + delta) - in_vector[pixel_num + i];
      fitness_all[index] += tp1 * tp1 + tp2 * tp2;
      //fitness_all[index] += abs(cos(phi) - in_vector[i]) + abs(cos(phi + delta) - in_vector[pixel_num + i]);
    }


      /*For display the iteration process*/
      //Record the min with change; first iteration, all particles with inf value of p_best_y;
      //global_min is assigned with global min in second iteration initially in PSO
      //function; therefore, the "if" is not executed in the first iteration
      if (abs(global_min - inf) > 1e-4 && global_min > fitness_all[index]) {
        global_min = fitness_all[index];
        min_with_change[count_min_with_change++] = fitness_all[index];
        std::cout << "good change: " << fitness_all[index] << std::endl;
      }//the initial change is not recorded

    }

  //Determine global best x at 't'
  current_min_fitness = *std::min_element(fitness_all.begin(), fitness_all.end());
  int current_min_index;
  for (int index = 0; index < max_particles; ++index) {
    if (abs(fitness_all[index] - current_min_fitness) < 1e-4) {
      current_min_index = index;
      break;
    }
  }
  for (int i = 0; i < dims; ++i)
    current_min_pos[i] = pos[current_min_index * dims + i];
  
  //Calculate velocity and position at 't+1' and update 
  for (int index = 0; index < max_particles; ++index) {
    for (int i = 0; i < dims; i++) {
      int id = index * dims + i;
      r1 = std::generate_canonical<float, 32>(gen);
      pos[id] = (1.0f - beta) * pos[id] + beta * pos[current_min_index * dims + i] + alpha[i] * (r1 - 0.5f);

      //Ensure position values are within range
      if (pos[id] > xmax[i])
        pos[id] = xmax[i];
      if (pos[id] < xmin[i])
        pos[id] = xmin[i];
    }
  }
}

void ApsoOptimization(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  bool iter_option, float &time_use, std::vector<float> &fit_rec) {

  //Declare other variables for PSO
  float *pos = new float[max_particles * dims];

  //Initialize variables
  std::random_device rd;
  std::mt19937 gen(rd());

  for (int i = 0; i < max_particles; ++i) {
    for (int j = 0; j < dims; ++j) {
      pos[i * dims + j] = xmin[j] +
        (xmax[j] - xmin[j]) * std::generate_canonical<float, 32>(gen);
    }
  }

  //Iteration

  //variables for storing the final results
  float *current_min_pos = new float[dims];
  float current_min_fitness;

  //variables for display
  float display_min[max_iters]; //store the min in each iteration (each iteration, all particles are updated)
  float min_with_change[1000]{ 0 }; //store the min with change; 1000 should be enough
  float global_min = inf; //global min of all the time; used in iteration to control the stop condition
  int count_min_with_change = 0;
  const int count_min_with_change_thres = 10;
  const float diff_min_with_change_thres = 1e-2f;

  //variables for time record
  auto t_start = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_from_start,
    elapsed_seconds_from_start_i;

  float *alpha0 = new float[dims];
  float *alpha = new float[dims];
  for (int i = 0; i < dims; ++i)
    alpha0[i] = (xmax[i] - xmin[i]) / 4;

  float gamma = 0.97f;
  for (int i = 0; i < max_iters; ++i) {
    auto t_start_i = std::chrono::system_clock::now();
    for (int j = 0; j < dims; ++j) {
      alpha[j] = alpha0[j] * pow(gamma, i);
    }
      ApsoIterate(pos, in_vector,
      zern, xmin, xmax, dims, pixel_num,
      gen, count_min_with_change, min_with_change,
      current_min_pos, current_min_fitness, global_min, alpha);
    auto t_end_i = std::chrono::system_clock::now();
    

    if (i == 0)
      global_min = current_min_fitness;
    
    //display the results
    display_min[i] = global_min;
    fit_rec.push_back(global_min);
    
    elapsed_seconds_from_start = t_end_i - t_start;
    elapsed_seconds_from_start_i = t_end_i - t_start_i;
    std::cout << "Iteration number: " << i << "\t" << "global min: " << global_min 
      << "\t" << "current min: " << current_min_fitness
      << "\t" << "time for the iteration (s): " << elapsed_seconds_from_start_i.count()
      << "\t" << "time since beginning of iteration (s): " 
      << elapsed_seconds_from_start.count() <<std::endl;

    //stop the iteration when number of changes are big enough and the change is small enough
    std::cout << "diff min with change (10): " <<
      abs(min_with_change[count_min_with_change - 1] - min_with_change[count_min_with_change - 1 - count_min_with_change_thres])
      << std::endl;
    if ( (!iter_option) && count_min_with_change > count_min_with_change_thres && abs(min_with_change[count_min_with_change - 1] -
      min_with_change[count_min_with_change - 1 - count_min_with_change_thres]) < diff_min_with_change_thres )
      break;
    if ( (!iter_option) && i >= 99 && abs(display_min[i] - display_min[i - 99]) < diff_min_with_change_thres )
      break;
  }
  
  elapsed_seconds_from_start = std::chrono::system_clock::now()- t_start;
  time_use = elapsed_seconds_from_start.count();

  //Pick the results
  *g_best = current_min_fitness;
  for (int i = 0; i < dims; ++i) {
    g_best_pos[i] = current_min_pos[i];
  }

  delete[] pos;
  delete[] current_min_pos;
  delete[] alpha0;
  delete[] alpha;
  pos = nullptr;
  current_min_pos = nullptr;
  alpha0 = nullptr;
  alpha = nullptr;
  return;
}


