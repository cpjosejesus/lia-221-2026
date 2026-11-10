#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int GRID_POINTS = 7;
const int BLOCK_DIM = 4;
const float TOLERANCE = 1e-5f;

using Samples = std::vector<float>;

__global__ void finite_difference_kernel(const float* F, float* FD, int n, float h) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= 1 && i < n - 1) {
    FD[i] = (F[i + 1] - F[i - 1]) / (2.0f * h);
  }
}

static Samples finiteDifferenceOnCpu(const Samples& F, float h) {
  const int n = (int)F.size();
  Samples FD(n, 0.0f);
  for (int i = 1; i < n - 1; i++) {
    FD[i] = (F[i + 1] - F[i - 1]) / (2.0f * h);
  }
  return FD;
}

static Samples sampleSine(int n, float h) {
  Samples F(n);
  for (int i = 0; i < n; i++) {
    F[i] = std::sin(i * h);
  }
  return F;
}

static Samples exactDerivative(int n, float h) {
  Samples derivative(n);
  for (int i = 0; i < n; i++) {
    derivative[i] = std::cos(i * h);
  }
  return derivative;
}

static void printAll(const char* name, const Samples& values) {
  std::printf("%-8s", name);
  for (float value : values) {
    std::printf(" %6.3f", value);
  }
  std::printf("\n");
}

static void printInterior(const char* name, const Samples& values) {
  const int n = (int)values.size();
  std::printf("%-8s %6s", name, "-");
  for (int i = 1; i < n - 1; i++) {
    std::printf(" %6.3f", values[i]);
  }
  std::printf(" %6s\n", "-");
}

static bool sameValues(const Samples& expected, const Samples& actual) {
  for (size_t i = 0; i < expected.size(); i++) {
    if (std::fabs(expected[i] - actual[i]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  const float pi = std::acos(-1.0f);
  const int n = GRID_POINTS;
  const float h = pi / (n - 1);
  const Samples F = sampleSine(n, h);
  const size_t bytes = n * sizeof(float);

  float* F_d;
  float* FD_d;
  CHECK(cudaMalloc(&F_d, bytes));
  CHECK(cudaMalloc(&FD_d, bytes));
  CHECK(cudaMemcpy(F_d, F.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(FD_d, 0, bytes));

  launch(finite_difference_kernel, dim3(cdiv(n, BLOCK_DIM)), dim3(BLOCK_DIM), F_d, FD_d, n, h);
  CHECK(cudaGetLastError());

  Samples FD(n);
  CHECK(cudaMemcpy(FD.data(), FD_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(F_d));
  CHECK(cudaFree(FD_d));

  printAll("F = sin", F);
  printInterior("FD", FD);
  printInterior("cos", exactDerivative(n, h));

  const bool passed = sameValues(finiteDifferenceOnCpu(F, h), FD);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
