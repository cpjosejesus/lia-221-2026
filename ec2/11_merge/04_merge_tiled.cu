#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <vector>

constexpr int TILE = 4;
constexpr int TILED_THREADS_PER_BLOCK = 2;
constexpr int TILE_OUTPUTS_PER_THREAD = TILE / TILED_THREADS_PER_BLOCK;
constexpr int BLOCKS = 2;
static_assert(TILE % TILED_THREADS_PER_BLOCK == 0, "every thread merges the same share of a tile");

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

__global__ void merge_tiled_kernel(const int* A, int m, const int* B, int n, int* C) {
  __shared__ int A_S[TILE];
  __shared__ int B_S[TILE];
  __shared__ int block_A_curr;
  __shared__ int block_A_next;

  const int outputsPerBlock = (m + n + gridDim.x - 1) / gridDim.x;
  const int C_curr = imin(blockIdx.x * outputsPerBlock, m + n);
  const int C_next = imin(C_curr + outputsPerBlock, m + n);
  if (threadIdx.x == 0) {
    block_A_curr = co_rank(C_curr, A, m, B, n);
    block_A_next = co_rank(C_next, A, m, B, n);
  }
  __syncthreads();

  const int A_curr = block_A_curr;
  const int A_next = block_A_next;
  const int B_curr = C_curr - A_curr;
  const int B_next = C_next - A_next;
  const int A_length = A_next - A_curr;
  const int B_length = B_next - B_curr;
  const int C_length = C_next - C_curr;

  int C_completed = 0;
  int A_consumed = 0;
  int B_consumed = 0;
  while (C_completed < C_length) {
    const int A_S_length = imin(TILE, A_length - A_consumed);
    const int B_S_length = imin(TILE, B_length - B_consumed);
    for (int x = threadIdx.x; x < A_S_length; x += blockDim.x) {
      A_S[x] = A[A_curr + A_consumed + x];
    }
    for (int x = threadIdx.x; x < B_S_length; x += blockDim.x) {
      B_S[x] = B[B_curr + B_consumed + x];
    }
    __syncthreads();

    const int C_step = imin(TILE, C_length - C_completed);
    const int c_curr = imin(threadIdx.x * TILE_OUTPUTS_PER_THREAD, C_step);
    const int c_next = imin(c_curr + TILE_OUTPUTS_PER_THREAD, C_step);
    const int a_curr = co_rank(c_curr, A_S, A_S_length, B_S, B_S_length);
    const int a_next = co_rank(c_next, A_S, A_S_length, B_S, B_S_length);
    const int b_curr = c_curr - a_curr;
    const int b_next = c_next - a_next;
    merge_sequential(A_S + a_curr,
                     a_next - a_curr,
                     B_S + b_curr,
                     b_next - b_curr,
                     C + C_curr + C_completed + c_curr);

    const int A_step = co_rank(C_step, A_S, A_S_length, B_S, B_S_length);
    A_consumed += A_step;
    B_consumed += C_step - A_step;
    C_completed += C_step;
    __syncthreads();
  }
}

static void printArray(const char* label, const std::vector<int>& values) {
  std::printf("%s", label);
  for (int value : values) {
    std::printf(" %3d", value);
  }
  std::printf("\n");
}

static void printBlockRanges(const std::vector<int>& A, const std::vector<int>& B) {
  const int m = A.size();
  const int n = B.size();
  const int outputsPerBlock = (m + n + BLOCKS - 1) / BLOCKS;
  for (int block = 0; block < BLOCKS; ++block) {
    const int C_curr = imin(block * outputsPerBlock, m + n);
    const int C_next = imin(C_curr + outputsPerBlock, m + n);
    const int A_curr = co_rank(C_curr, A.data(), m, B.data(), n);
    const int A_next = co_rank(C_next, A.data(), m, B.data(), n);
    std::printf("block %d: C[%2d, %2d) = merge(A[%2d, %2d), B[%2d, %2d)) in tiles of %d\n",
                block,
                C_curr,
                C_next,
                A_curr,
                A_next,
                C_curr - A_curr,
                C_next - A_next,
                TILE);
  }
}

int main() {
  const std::vector<int> A = {1, 3, 4, 4, 7, 9, 10, 12, 15, 18};
  const std::vector<int> B = {2, 4, 5, 6, 8, 11, 11, 13, 14, 16, 17, 20};
  const int m = A.size();
  const int n = B.size();
  std::vector<int> C(m + n);

  int* A_d = nullptr;
  int* B_d = nullptr;
  int* C_d = nullptr;
  CHECK(cudaMalloc(&A_d, m * sizeof(int)));
  CHECK(cudaMalloc(&B_d, n * sizeof(int)));
  CHECK(cudaMalloc(&C_d, (m + n) * sizeof(int)));
  CHECK(cudaMemcpy(A_d, A.data(), m * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  launch(merge_tiled_kernel, dim3(BLOCKS), dim3(TILED_THREADS_PER_BLOCK), A_d, m, B_d, n, C_d);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(C.data(), C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> expected(m + n);
  std::merge(A.begin(), A.end(), B.begin(), B.end(), expected.begin());

  printArray("A:", A);
  printArray("B:", B);
  printBlockRanges(A, B);
  printArray("C:", C);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));

  const bool ok = C == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
