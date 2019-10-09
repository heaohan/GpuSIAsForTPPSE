#include "mat_rw.h"

void matread(const char *file, const char *var, std::vector<float>& v)
{
  // open MAT-file
  MATFile *pmat = matOpen(file, "r");
  if (pmat == NULL) return;

  // extract the specified variable
  mxArray *arr = matGetVariable(pmat, var);
  if (arr != NULL && !mxIsEmpty(arr)) {
    // copy data
    mwSize num = mxGetNumberOfElements(arr);
    float *pr = (float *) mxGetPr(arr);
    if (pr != NULL) {
      v.reserve(num); //is faster than resize :-)
      v.assign(pr, pr + num);
    }
  }

  // cleanup
  mxDestroyArray(arr);
  matClose(pmat);
}

void matwrite(const char *file, const char *var,
  const int mRow, const int mCol, std::vector<float>& v) {
  // open MAT-file
  MATFile *pmat = matOpen(file, "w");
  if (pmat == NULL) return;
  
  //Create matrix
  if (mRow * mCol != v.size()) return;
  mxArray *pa = mxCreateNumericMatrix(mRow, mCol, mxSINGLE_CLASS, mxREAL);
  if (pa == NULL) return;
  
  //Copy data
  memcpy((void *)(mxGetPr(pa)), (void *)v.data(), v.size() * sizeof(float));
  
  //Copy data to MAT-file
  int status = matPutVariable(pmat, var, pa);
  if (status != 0) return;

  //Clean up
  mxDestroyArray(pa);
  matClose(pmat);
}