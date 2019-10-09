#ifndef _MAT_RW_H
#define _MAT_RW_H

#include "mat.h"
#include <iostream>
#include <vector>

void matread(const char *file, const char *var, std::vector<float>& v);

void matwrite(const char *file, const char *var,
  const int mRow, const int mCol, std::vector<float>& v);

#endif // !_MAT_RW_H
