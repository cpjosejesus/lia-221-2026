#include "cuda_compat.h"

#include <cstdio>
#include <vector>

constexpr int M = 5;
constexpr int N = 6;
constexpr int TRACED_K = 6;

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

__global__ void co_rank_kernel(const int* A, int m, const int* B, int n, int* coRanks) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k <= m + n) {
    coRanks[k] = co_rank(k, A, m, B, n);
  }
}

static void traceCoRank(int k, const int* A, int m, const int* B, int n) {
  int i = imin(k, m);
  int j = k - i;
  int i_low = imax(0, k - n);
  int j_low = imax(0, k - m);
  std::printf("co_rank search for k = %d\n", k);
  bool searching = true;
  while (searching) {
    std::printf("  i = %d  j = %d  i_low = %d  j_low = %d  ", i, j, i_low, j_low);
    const bool i_too_large = i > 0 && j < n && A[i - 1] > B[j];
    const bool i_too_small = j > 0 && i < m && B[j - 1] >= A[i];
    if (i_too_large) {
      std::printf("A[%d]=%d > B[%d]=%d: i too large\n", i - 1, A[i - 1], j, B[j]);
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      i -= delta;
      j += delta;
    } else if (i_too_small) {
      std::printf("B[%d]=%d >= A[%d]=%d: i too small\n", j - 1, B[j - 1], i, A[i]);
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      std::printf("both conditions hold: stop\n");
      searching = false;
    }
  }
}

static void printArray(const char* label, const std::vector<int>& values) {
  std::printf("%s", label);
  for (int value : values) {
    std::printf(" %3d", value);
  }
  std::printf("\n");
}

static std::vector<int> coRanksOnHost(const std::vector<int>& A, const std::vector<int>& B) {
  std::vector<int> coRanks = {0};
  int i = 0;
  int j = 0;
  while (i < M || j < N) {
    const bool takeA = j == N || (i < M && A[i] <= B[j]);
    i += takeA ? 1 : 0;
    j += takeA ? 0 : 1;
    coRanks.push_back(i);
  }
  return coRanks;
}

int main() {
  const std::vector<int> A = {3, 5, 5, 8, 12};
  const std::vector<int> B = {1, 5, 6, 8, 9, 14};
  const int outputs = M + N;
  std::vector<int> coRanks(outputs + 1);

  int* A_d = nullptr;
  int* B_d = nullptr;
  int* coRanks_d = nullptr;
  CHECK(cudaMalloc(&A_d, M * sizeof(int)));
  CHECK(cudaMalloc(&B_d, N * sizeof(int)));
  CHECK(cudaMalloc(&coRanks_d, (outputs + 1) * sizeof(int)));
  CHECK(cudaMemcpy(A_d, A.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  launch(co_rank_kernel, dim3(1), dim3(outputs + 1), A_d, M, B_d, N, coRanks_d);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(coRanks.data(), coRanks_d, (outputs + 1) * sizeof(int), cudaMemcpyDeviceToHost));

  printArray("A:", A);
  printArray("B:", B);
  std::printf("\n");
  traceCoRank(TRACED_K, A.data(), M, B.data(), N);
  std::printf("\n k  i = co_rank(k)  j = k - i\n");
  for (int k = 0; k <= outputs; ++k) {
    std::printf("%2d  %14d  %9d\n", k, coRanks[k], k - coRanks[k]);
  }

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(coRanks_d));

  const bool ok = coRanks == coRanksOnHost(A, B);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
