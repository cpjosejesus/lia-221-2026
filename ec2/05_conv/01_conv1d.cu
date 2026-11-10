#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int FILTER_RADIUS = 1;
const int BLOCK_DIM = 4;

__global__ void convolution_1D_basic_kernel(
    const float* N, const float* F, float* P, int r, int width) {
  int outCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  if (outCol < width) {
    float Pvalue = 0.0f;
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inCol = outCol - r + fCol;
      if (inCol >= 0 && inCol < width) {
        Pvalue += F[fCol] * N[inCol];
      }
    }
    P[outCol] = Pvalue;
  }
}

static std::vector<float> convolution1DOnCpu(const std::vector<float>& N,
                                             const std::vector<float>& F,
                                             int r) {
  const int width = (int)N.size();
  std::vector<float> P(width, 0.0f);
  for (int outCol = 0; outCol < width; outCol++) {
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inCol = outCol - r + fCol;
      if (inCol >= 0 && inCol < width) {
        P[outCol] += F[fCol] * N[inCol];
      }
    }
  }
  return P;
}

static float* copyToDevice(const std::vector<float>& host) {
  float* device;
  CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  return device;
}

static void printArray(const char* name, const std::vector<float>& values) {
  std::printf("%s =", name);
  for (float value : values) {
    std::printf(" %5.1f", value);
  }
  std::printf("\n");
}

int main() {
  const std::vector<float> N = {4, 1, 3, 2, 3};
  const std::vector<float> F = {2, 1, 4};
  const int width = (int)N.size();

  float* N_d = copyToDevice(N);
  float* F_d = copyToDevice(F);
  float* P_d = copyToDevice(std::vector<float>(width, 0.0f));

  launch(convolution_1D_basic_kernel,
         dim3(cdiv(width, BLOCK_DIM)),
         dim3(BLOCK_DIM),
         N_d,
         F_d,
         P_d,
         FILTER_RADIUS,
         width);
  CHECK(cudaGetLastError());

  std::vector<float> P(width);
  CHECK(cudaMemcpy(P.data(), P_d, width * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(F_d));
  CHECK(cudaFree(P_d));

  printArray("N", N);
  printArray("F", F);
  printArray("P", P);

  const bool passed = P == convolution1DOnCpu(N, F, FILTER_RADIUS);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
