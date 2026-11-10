#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

constexpr int N = 16;
constexpr int RUN = 4;
constexpr int OUTPUTS_PER_THREAD = 3;
constexpr int THREADS = (N + OUTPUTS_PER_THREAD - 1) / OUTPUTS_PER_THREAD;
constexpr int RUNS = (N + RUN - 1) / RUN;
constexpr int LABEL_SIZE = 32;

__host__ __device__ int co_rank(int k, const unsigned* A, int m, const unsigned* B, int n) {
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

__host__ __device__ void merge_sequential(
    const unsigned* A, int m, const unsigned* B, int n, unsigned* C) {
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

__host__ __device__ void stable_insertion_sort(unsigned* keys, int begin, int end) {
  for (int i = begin + 1; i < end; ++i) {
    const unsigned key = keys[i];
    int j = i - 1;
    while (j >= begin && keys[j] > key) {
      keys[j + 1] = keys[j];
      --j;
    }
    keys[j + 1] = key;
  }
}

__global__ void sort_runs_kernel(unsigned* keys, int n, int run) {
  const int runStart = (blockIdx.x * blockDim.x + threadIdx.x) * run;
  if (runStart < n) {
    const int runEnd = imin(runStart + run, n);
    stable_insertion_sort(keys, runStart, runEnd);
  }
}

__global__ void merge_pass_kernel(
    const unsigned* keysIn, unsigned* keysOut, int total, int width, int outputsPerThread) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sliceBegin = imin(tid * outputsPerThread, total);
  const int sliceEnd = imin(sliceBegin + outputsPerThread, total);

  int k = sliceBegin;
  while (k < sliceEnd) {
    const int pairStart = (k / (2 * width)) * (2 * width);
    const int mid = imin(pairStart + width, total);
    const int pairEnd = imin(pairStart + 2 * width, total);
    const int pieceEnd = imin(sliceEnd, pairEnd);

    const int m = mid - pairStart;
    const int n = pairEnd - mid;
    const int k_curr = k - pairStart;
    const int k_next = pieceEnd - pairStart;
    const int i_curr = co_rank(k_curr, keysIn + pairStart, m, keysIn + mid, n);
    const int i_next = co_rank(k_next, keysIn + pairStart, m, keysIn + mid, n);
    const int j_curr = k_curr - i_curr;
    const int j_next = k_next - i_next;
    merge_sequential(keysIn + pairStart + i_curr,
                     i_next - i_curr,
                     keysIn + mid + j_curr,
                     j_next - j_curr,
                     keysOut + k);

    k = pieceEnd;
  }
}

static unsigned* toDevice(const std::vector<unsigned>& values) {
  unsigned* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, values.size() * sizeof(unsigned)));
  CHECK(
      cudaMemcpy(pointer, values.data(), values.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  return pointer;
}

static std::vector<unsigned> toHost(const unsigned* pointer, int count) {
  std::vector<unsigned> values(count);
  CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return values;
}

static void printRuns(const char* label, const std::vector<unsigned>& keys, int width) {
  std::printf("%-26s", label);
  for (int i = 0; i < N; ++i) {
    const bool startsRun = i > 0 && i % width == 0;
    std::printf("%s %2u", startsRun ? " |" : "", keys[i]);
  }
  std::printf("\n");
}

int main() {
  const std::vector<unsigned> keys = {9, 3, 14, 3, 7, 0, 12, 5, 9, 1, 15, 6, 2, 11, 7, 4};
  unsigned* keys_d[2] = {toDevice(keys), toDevice(keys)};

  printRuns("input:", keys, N);
  launch(sort_runs_kernel, dim3(1), dim3(RUNS), keys_d[0], N, RUN);
  CHECK(cudaGetLastError());
  printRuns("sorted runs of 4:", toHost(keys_d[0], N), RUN);

  for (int width = RUN; width < N; width *= 2) {
    launch(merge_pass_kernel,
           dim3(1),
           dim3(THREADS),
           keys_d[0],
           keys_d[1],
           N,
           width,
           OUTPUTS_PER_THREAD);
    CHECK(cudaGetLastError());
    std::swap(keys_d[0], keys_d[1]);
    char label[LABEL_SIZE];
    std::snprintf(label, sizeof label, "merged into runs of %d:", 2 * width);
    printRuns(label, toHost(keys_d[0], N), 2 * width);
  }

  const std::vector<unsigned> sortedKeys = toHost(keys_d[0], N);
  CHECK(cudaFree(keys_d[0]));
  CHECK(cudaFree(keys_d[1]));

  std::vector<unsigned> expected = keys;
  std::sort(expected.begin(), expected.end());
  const bool ok = sortedKeys == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
