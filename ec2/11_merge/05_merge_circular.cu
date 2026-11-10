#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <vector>

constexpr int TILE = 4;
constexpr int TILED_THREADS_PER_BLOCK = 2;
constexpr int TILE_OUTPUTS_PER_THREAD = TILE / TILED_THREADS_PER_BLOCK;
constexpr int BLOCKS = 2;
constexpr int TILE_MASK = TILE - 1;
static_assert(TILE % TILED_THREADS_PER_BLOCK == 0, "every thread merges the same share of a tile");
static_assert((TILE & TILE_MASK) == 0, "TILE must be a power of two");

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

__host__ __device__ int circular_slot(int index) {
  return index & TILE_MASK;
}

__host__ __device__ int co_rank_circular(
    int k, const int* A_S, int A_S_start, int m, const int* B_S, int B_S_start, int n) {
  int i = imin(k, m);
  int j = k - i;
  int i_low = imax(0, k - n);
  int j_low = imax(0, k - m);
  bool searching = true;
  while (searching) {
    const bool i_too_large =
        i > 0 && j < n && A_S[circular_slot(A_S_start + i - 1)] > B_S[circular_slot(B_S_start + j)];
    const bool i_too_small =
        j > 0 && i < m &&
        B_S[circular_slot(B_S_start + j - 1)] >= A_S[circular_slot(A_S_start + i)];
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

__host__ __device__ void merge_sequential_circular(
    const int* A_S, int A_S_start, int m, const int* B_S, int B_S_start, int n, int* C) {
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < m && j < n) {
    const int a = A_S[circular_slot(A_S_start + i)];
    const int b = B_S[circular_slot(B_S_start + j)];
    if (a <= b) {
      C[k++] = a;
      ++i;
    } else {
      C[k++] = b;
      ++j;
    }
  }
  while (i < m) {
    C[k++] = A_S[circular_slot(A_S_start + i++)];
  }
  while (j < n) {
    C[k++] = B_S[circular_slot(B_S_start + j++)];
  }
}

__global__ void merge_circular_kernel(const int* A, int m, const int* B, int n, int* C) {
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
  int A_loaded = 0;
  int B_loaded = 0;
  while (C_completed < C_length) {
    const int A_window_end = imin(A_consumed + TILE, A_length);
    const int B_window_end = imin(B_consumed + TILE, B_length);
    for (int x = A_loaded + threadIdx.x; x < A_window_end; x += blockDim.x) {
      A_S[circular_slot(x)] = A[A_curr + x];
    }
    for (int x = B_loaded + threadIdx.x; x < B_window_end; x += blockDim.x) {
      B_S[circular_slot(x)] = B[B_curr + x];
    }
    A_loaded = A_window_end;
    B_loaded = B_window_end;
    __syncthreads();

    const int A_S_length = A_loaded - A_consumed;
    const int B_S_length = B_loaded - B_consumed;
    const int C_step = imin(TILE, C_length - C_completed);
    const int c_curr = imin(threadIdx.x * TILE_OUTPUTS_PER_THREAD, C_step);
    const int c_next = imin(c_curr + TILE_OUTPUTS_PER_THREAD, C_step);
    const int a_curr =
        co_rank_circular(c_curr, A_S, A_consumed, A_S_length, B_S, B_consumed, B_S_length);
    const int a_next =
        co_rank_circular(c_next, A_S, A_consumed, A_S_length, B_S, B_consumed, B_S_length);
    const int b_curr = c_curr - a_curr;
    const int b_next = c_next - a_next;
    merge_sequential_circular(A_S,
                              A_consumed + a_curr,
                              a_next - a_curr,
                              B_S,
                              B_consumed + b_curr,
                              b_next - b_curr,
                              C + C_curr + C_completed + c_curr);

    const int A_step =
        co_rank_circular(C_step, A_S, A_consumed, A_S_length, B_S, B_consumed, B_S_length);
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

  launch(merge_circular_kernel, dim3(BLOCKS), dim3(TILED_THREADS_PER_BLOCK), A_d, m, B_d, n, C_d);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(C.data(), C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> expected(m + n);
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
