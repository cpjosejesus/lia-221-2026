#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <vector>

constexpr int M = 5;
constexpr int N = 6;

__host__ __device__ void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }
  while (i < m) {
    C[k++] = A[i++];
  }
  while (j < n) {
    C[k++] = B[j++];
  }
}

__global__ void merge_sequential_kernel(const int* A, int m, const int* B, int n, int* C) {
  merge_sequential(A, m, B, n, C);
}

static void printArray(const char* label, const std::vector<int>& values) {
  std::printf("%s", label);
  for (int value : values) {
    std::printf(" %3d", value);
  }
  std::printf("\n");
}

int main() {
  const std::vector<int> A = {3, 5, 5, 8, 12};
  const std::vector<int> B = {1, 5, 6, 8, 9, 14};
  std::vector<int> C(M + N);

  int* A_d = nullptr;
  int* B_d = nullptr;
  int* C_d = nullptr;
  CHECK(cudaMalloc(&A_d, M * sizeof(int)));
  CHECK(cudaMalloc(&B_d, N * sizeof(int)));
  CHECK(cudaMalloc(&C_d, (M + N) * sizeof(int)));
  CHECK(cudaMemcpy(A_d, A.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  launch(merge_sequential_kernel, dim3(1), dim3(1), A_d, M, B_d, N, C_d);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(C.data(), C_d, (M + N) * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> expected(M + N);
  std::merge(A.begin(), A.end(), B.begin(), B.end(), expected.begin());

  printArray("A:", A);
  printArray("B:", B);
  printArray("C:", C);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));

  const bool ok = C == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
