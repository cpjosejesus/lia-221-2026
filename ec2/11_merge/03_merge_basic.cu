#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <vector>

constexpr int M = 5;
constexpr int N = 6;
constexpr int THREADS = 4;

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

__host__ __device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
  int i = imin(k, m);
  int j = k - i;
  int i_low = imax(0, k - n);
  int j_low = imax(0, k - m);
  bool searching = true;
  while (searching) {
    const bool i_too_large = i > 0 && j < n && A[i - 1] > B[j];
    const bool i_too_small = j > 0 && i < m && B[j - 1] >= A[i];
    if (i_too_large) {
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      i -= delta;
      j += delta;
    } else if (i_too_small) {
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      searching = false;
    }
  }
  return i;
}

__global__ void merge_basic_kernel(const int* A, int m, const int* B, int n, int* C) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalThreads = gridDim.x * blockDim.x;
  const int elementsPerThread = (m + n + totalThreads - 1) / totalThreads;

  const int k_curr = imin(tid * elementsPerThread, m + n);
  const int k_next = imin(k_curr + elementsPerThread, m + n);
  const int i_curr = co_rank(k_curr, A, m, B, n);
  const int i_next = co_rank(k_next, A, m, B, n);
  const int j_curr = k_curr - i_curr;
  const int j_next = k_next - i_next;

  merge_sequential(A + i_curr, i_next - i_curr, B + j_curr, j_next - j_curr, C + k_curr);
}

static void printArray(const char* label, const std::vector<int>& values) {
  std::printf("%s", label);
  for (int value : values) {
    std::printf(" %3d", value);
  }
  std::printf("\n");
}

static void printThreadPieces(const std::vector<int>& A, const std::vector<int>& B) {
  const int elementsPerThread = (M + N + THREADS - 1) / THREADS;
  for (int tid = 0; tid < THREADS; ++tid) {
    const int k_curr = imin(tid * elementsPerThread, M + N);
    const int k_next = imin(k_curr + elementsPerThread, M + N);
    const int i_curr = co_rank(k_curr, A.data(), M, B.data(), N);
    const int i_next = co_rank(k_next, A.data(), M, B.data(), N);
    std::printf("thread %d: C[%2d, %2d) = merge(A[%d, %d), B[%d, %d))\n",
                tid,
                k_curr,
                k_next,
                i_curr,
                i_next,
                k_curr - i_curr,
                k_next - i_next);
  }
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

  launch(merge_basic_kernel, dim3(1), dim3(THREADS), A_d, M, B_d, N, C_d);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(C.data(), C_d, (M + N) * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> expected(M + N);
  std::merge(A.begin(), A.end(), B.begin(), B.end(), expected.begin());

  printArray("A:", A);
  printArray("B:", B);
  printThreadPieces(A, B);
  printArray("C:", C);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));

  const bool ok = C == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
