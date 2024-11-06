#include "hip/hip_runtime.h"

extern "C" __global__ void Kernel_test(int n, float factor, float *A)
  {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      A[idx] = factor*A[idx];

  }
