#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int N = 8;

__global__ void vecAddKernel(const float* A, const float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

static void vecAdd(const float* A, const float* B, float* C, int n) {
  const size_t bytes = n * sizeof(float);

  float* A_d;
  float* B_d;
  float* C_d;
  cudaMalloc(&A_d, bytes);
  cudaMalloc(&B_d, bytes);
  cudaMalloc(&C_d, bytes);

  cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

  launch(vecAddKernel, dim3(1), dim3(n), A_d, B_d, C_d);

  cudaMemcpy(C, C_d, bytes, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

static void vecAddOnCpu(const float* A, const float* B, float* C, int n) {
  for (int i = 0; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}

static void printVector(const char* label, const std::vector<float>& v) {
  std::printf("%s =", label);
  for (float x : v) {
    std::printf(" %5.1f", x);
  }
  std::printf("\n");
}

int main() {
  std::vector<float> A(N);
  std::vector<float> B(N);
  for (int i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i);
    B[i] = 10.0f * (i + 1);
  }

  std::vector<float> C(N, 0.0f);
  vecAdd(A.data(), B.data(), C.data(), N);

  std::vector<float> expected(N);
  vecAddOnCpu(A.data(), B.data(), expected.data(), N);

  std::printf("one block of %d threads, thread i adds element i\n", N);
  printVector("A", A);
  printVector("B", B);
  printVector("C", C);

  const bool ok = C == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
