/*
The CPU version PSO
*/
#include "pso_m.h"

float FitnessCalculate(float *pos, const float *in_vector, const float *zern,
  const int &dims, const int &pixel_num) {
  float fitness = 0.0f; 
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
    float delta = pos[dims - 1];
    for (int j = 0; j < dims - 1; ++j) {
      phi += zern[j * pixel_num + i] * pos[j];
    }
    //fitness += abs(cos(phi) - in_vector[i]) + abs(cos(phi + delta) - in_vector[pixel_num + i]);
    float tp1 = cos(phi) - in_vector[i];
    float tp2 = cos(phi + delta) - in_vector[pixel_num + i];
    fitness += tp1 * tp1 + tp2 * tp2;
  }
  return fitness;
}

void PsoIterate(float *pos, float *velocity,
  float *p_best_x, float *p_best_y,
  const float *in_vector, const float *zern,
  const float *xmin, const float *xmax, const int &dims, const int &pixel_num,
  std::mt19937 &gen, std::vector<std::unordered_set<int>> &neighbors, float &global_min, float *global_min_pos,
  float inertia, bool &flag) {
  
  //float c1 = 2.05f, c2 = 2.05f; //2.05f 1.49618f
  float c1 = 1.49618f, c2 = 1.49618f; //2.05f 1.49618f
  std::uniform_int_distribution<> dis_particle(0, max_particles - 1);
  
  float *v_xmax = new float[dims];
  for (int i = 0; i < dims; ++i) {
    v_xmax[i] = (xmax[i] - xmin[i]) / 4.0f;
  }
  
  for (int index = 0; index < max_particles; ++index) {
    //generate a subset from max_particles as neighborhoods
    std::unordered_set<int>& neighborhood_indexes = neighbors[index];
    
    //search the g_x and g_y from the neighborhood
    float g_y = inf;
    int g_min_index;
    for (auto it = neighborhood_indexes.begin(); it != neighborhood_indexes.end(); ++it) {
      if (p_best_y[*it] < g_y) {
        g_y = p_best_y[*it];
        g_min_index = *it;
      }
    }
    float *g_x = &(p_best_x[g_min_index * dims]);
    
    //update velocity and pos for particle[index]
    for (int i = 0; i < dims; i++) {
      int id = index * dims + i;
      float r1 = std::generate_canonical<float, 32>(gen);
      float r2 = std::generate_canonical<float, 32>(gen);

      velocity[id] = inertia * velocity[id] + (c1 * r1 * (p_best_x[id] - pos[id])) +
        (c2 * r2 * (g_x[i] - pos[id]));
      
      if (velocity[id] < -v_xmax[i]) velocity[id] = -v_xmax[i];
      if (velocity[id] > v_xmax[i]) velocity[id] = v_xmax[i];
      
      pos[id] = pos[id] + velocity[id];

      //Ensure position values are within range
      if (pos[id] > xmax[i]) {
        pos[id] = xmax[i];
        if (velocity[id] > 0) velocity[id] = 0;
      }
      if (pos[id] < xmin[i]) {
        pos[id] = xmin[i];
        if (velocity[id] < 0) velocity[id] = 0;
      }
    }
  }
  
  delete[] v_xmax;
  v_xmax = nullptr;

  //update p_best_x, p_best_y for each particle, and check the global_min is updated or not
  float tp_min = inf;
  int tp_min_index;
  for (int index = 0; index < max_particles; ++index) {
     float tp = FitnessCalculate(&(pos[index * dims]), in_vector, zern, dims, pixel_num);
     if (tp < p_best_y[index]) {
       p_best_y[index] = tp;
       for (int i = 0; i < dims; ++i) p_best_x[index * dims + i] = pos[index * dims + i];
     }
     if (tp < tp_min) {
       tp_min = tp;
       tp_min_index = index;
     }
  }
  if (tp_min < global_min) {
    global_min = tp_min;
    flag = true;
    for (int i = 0; i < dims; ++i)
      global_min_pos[i] = pos[tp_min_index * dims + i];
  }
  else
    flag = false;
 
}

/*
void PsoIterate(float *pos, float *velocity,
  float *p_best_x, float *p_best_y,
  const float *in_vector, const float *zern,
  const float *xmin, const float *xmax, const int &dims, const int &pixel_num,
  std::mt19937 &gen, int neighborhood_size, float &global_min, float *global_min_pos,
  float inertia, bool &flag) {
  
  //float c1 = 2.05f, c2 = 2.05f; //2.05f 1.49618f
  float c1 = 1.49618f, c2 = 1.49618f; //2.05f 1.49618f
  std::uniform_int_distribution<> dis_particle(0, max_particles - 1);

  float *v_xmax = new float[dims];
  for (int i = 0; i < dims; ++i) {
    v_xmax[i] = (xmax[i] - xmin[i]) / 4.0f;
  }

  for (int index = 0; index < max_particles; ++index) {
    //generate a subset from max_particles as neighborhoods
    std::unordered_set<int> neighborhood_indexes;
    while (neighborhood_indexes.size() < neighborhood_size) {
      int tp_index = dis_particle(gen);
      if (tp_index != index)
        neighborhood_indexes.insert(tp_index);
    }
    
    //search the g_x and g_y from the neighborhood
    float g_y = inf;
    int g_min_index;
    for (auto it = neighborhood_indexes.begin(); it != neighborhood_indexes.end(); ++it) {
      if (p_best_y[*it] < g_y) {
        g_y = p_best_y[*it];
        g_min_index = *it;
      }
    }
    float *g_x = &(p_best_x[g_min_index * dims]);
    
    //update velocity and pos for particle[index]
    for (int i = 0; i < dims; i++) {
      int id = index * dims + i;
      float r1 = std::generate_canonical<float, 32>(gen);
      float r2 = std::generate_canonical<float, 32>(gen);

      velocity[id] = inertia * velocity[id] + (c1 * r1 * (p_best_x[id] - pos[id])) +
        (c2 * r2 * (g_x[i] - pos[id]));
      
      if (velocity[id] < -v_xmax[i]) velocity[id] = -v_xmax[i];
      if (velocity[id] > v_xmax[i]) velocity[id] = v_xmax[i];
      
      pos[id] = pos[id] + velocity[id];

      //Ensure position values are within range
      if (pos[id] > xmax[i]) {
        pos[id] = xmax[i];
        if (velocity[id] > 0) velocity[id] = 0;
      }
      if (pos[id] < xmin[i]) {
        pos[id] = xmin[i];
        if (velocity[id] < 0) velocity[id] = 0;
      }
    }
  }
  
  delete[] v_xmax;
  v_xmax = nullptr;

  //update p_best_x, p_best_y for each particle, and check the global_min is updated or not
  float tp_min = inf;
  int tp_min_index;
  for (int index = 0; index < max_particles; ++index) {
     float tp = FitnessCalculate(&(pos[index * dims]), in_vector, zern, dims, pixel_num);
     if (tp < p_best_y[index]) {
       p_best_y[index] = tp;
       for (int i = 0; i < dims; ++i) p_best_x[index * dims + i] = pos[index * dims + i];
     }
     if (tp < tp_min) {
       tp_min = tp;
       tp_min_index = index;
     }
  }
  if (tp_min < global_min) {
    global_min = tp_min;
    flag = true;
    for (int i = 0; i < dims; ++i)
      global_min_pos[i] = pos[tp_min_index * dims + i];
  }
  else
    flag = false;
 
}
*/

std::vector<std::unordered_set<int>> NeighborSelect(std::mt19937 &gen, int neighborhood_size) {
  std::vector<std::unordered_set<int>> result;
  std::uniform_int_distribution<> dis_particle(0, max_particles - 1);
  for (int index = 0; index < max_particles; ++index) {
    //generate a subset from max_particles as neighborhoods
    std::unordered_set<int> neighborhood_indexes;
    while (neighborhood_indexes.size() < neighborhood_size) {
      int tp_index = dis_particle(gen);
      if (tp_index != index)
        neighborhood_indexes.insert(tp_index);
    }
    result.push_back(neighborhood_indexes);
  }
  return result;
}

void PSO(const float * const in_vector, const float * const xmin,
  const float * const xmax, const float * const zern,
  const int pixel_num, const int dims, float *g_best, float *g_best_pos,
  float &time_use, std::vector<float> &fit_rec) {

  //Declare other variables for PSO
  float *pos = new float[max_particles * dims];
  float *velocity = new float[max_particles * dims]{ 0 };
  float *p_best_x = new float[max_particles * dims];
  float *p_best_y = new float[max_particles];
  
  //Initialize variables
  std::random_device rd;
  std::mt19937 gen(rd());

  for (int i = 0; i < max_particles; ++i) {
    for (int j = 0; j < dims; ++j) {
      pos[i * dims + j] = xmin[j] +
        (xmax[j] - xmin[j]) * std::generate_canonical<float, 32>(gen);
      p_best_x[i * dims + j] = pos[i * dims + j];
      //velocity[i * dims + j] = 
        //(2 * std::generate_canonical<float, 32>(gen) - 1) * (xmax[j] - xmin[j]);
      //velocity[i * dims + j] = 
        //(2 * std::generate_canonical<float, 32>(gen) - 1) * (xmax[j] - xmin[j]) / 4;
    }
    p_best_y[i] = FitnessCalculate(&(pos[i * dims]), in_vector, zern, dims, pixel_num);
    //printf("%d: %.2f\n", i, p_best_y[i]);
  }
  //for (int j = 0; j < dims; ++j) printf("%d: %.2f, %.2f\n", j, pos[100*dims+j], p_best_x[100*dims+j]);

  //Initiate g_best and g_best_pos
  float tp_min = inf;
  int tp_index;
  for (int i = 0; i < max_particles; ++i) {
    if (p_best_y[i] < tp_min) {
      tp_min = p_best_y[i];
      tp_index = i;
    }
  }
  *g_best = tp_min;
  for (int i = 0; i < dims; ++i) g_best_pos[i] = p_best_x[tp_index * dims + i];

  //for (int j = 0; j < dims; ++j) printf("%d: %.2f, %.2f, %.2f\n",
    //j, g_best_pos[j], pos[tp_index*dims+j], p_best_x[tp_index*dims+j]);
  
  //Iteration
  const int kMinNeighborhoodSize = (2 > max_particles / 4) ? 2 : max_particles / 4;
  int neighborhood_size = kMinNeighborhoodSize;
  const float kInertiaMin = 0.1f;
  const float kInertiaMax = 1.1f;
  float inertia = kInertiaMax;
  int stall_count = 0;

  auto t_start = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_from_start,
    elapsed_seconds_from_start_i;
  bool flag;
  std::unordered_map<int, std::vector<std::unordered_set<int>>> neighbor_cache;
  for (int i = 0; i < max_iters; ++i) {
    auto t_start_i = std::chrono::system_clock::now();

    if (!neighbor_cache.count(neighborhood_size))
      neighbor_cache[neighborhood_size] = NeighborSelect(gen, neighborhood_size);
  /*
    PsoIterate(pos, velocity,
      p_best_x, p_best_y, in_vector,
      zern, xmin, xmax, dims, pixel_num,
      gen, neighborhood_size, *g_best, g_best_pos, inertia, flag);
   */
    PsoIterate(pos, velocity,
      p_best_x, p_best_y, in_vector,
      zern, xmin, xmax, dims, pixel_num,
      gen, neighbor_cache[neighborhood_size], *g_best, g_best_pos, inertia, flag);
   

    auto t_end_i = std::chrono::system_clock::now();
    if (flag) { //g_best is updated
      stall_count = (0 > stall_count - 1) ? 0 : (stall_count - 1);
      neighborhood_size = kMinNeighborhoodSize;
      if (stall_count < 2) inertia = 2.0f * inertia;
      if (stall_count > 5) inertia = inertia / 2.0f;
      if (inertia > kInertiaMax) inertia = kInertiaMax;
      if (inertia < kInertiaMin) inertia = kInertiaMin;
    }
    else {
      stall_count++;
      neighborhood_size = ((neighborhood_size + kMinNeighborhoodSize < max_particles - 1) ?
        neighborhood_size + kMinNeighborhoodSize : max_particles - 1);
    }
    //display the results
    fit_rec.push_back(*g_best);

    elapsed_seconds_from_start = t_end_i - t_start;
    elapsed_seconds_from_start_i = t_end_i - t_start_i;
    std::cout << "Iteration number: " << i << "\t" << "global min: " << *g_best
      << "\t" << "time for the iteration (s): " << elapsed_seconds_from_start_i.count()
      << "\t" << "time since beginning of iteration (s): " 
      << elapsed_seconds_from_start.count() <<std::endl;
    
    //for (int j = 0; j < dims; ++j) printf("%d, %.2f, %.2f, %.2f\n", j, pos[j], p_best_x[j], g_best_pos[j]);
  }
  
  //Pick the results
  elapsed_seconds_from_start = std::chrono::system_clock::now()- t_start;
  time_use = elapsed_seconds_from_start.count();
  
  delete[] pos;
  delete[] velocity;
  delete[] p_best_x;
  delete[] p_best_y;
  pos = nullptr;
  velocity = nullptr;
  p_best_x = nullptr;
  p_best_y = nullptr;
}

