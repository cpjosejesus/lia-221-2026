#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <random>
#include <vector>

__host__ __device__ inline void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
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

__host__ __device__ inline int co_rank(int k, const int* A, int m, const int* B, int n) {
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

constexpr int TILE = 1024;
constexpr int TILED_THREADS_PER_BLOCK = 128;
constexpr int TILE_OUTPUTS_PER_THREAD = TILE / TILED_THREADS_PER_BLOCK;
static_assert(TILE % TILED_THREADS_PER_BLOCK == 0, "every thread merges the same share of a tile");

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

constexpr int TILE_MASK = TILE - 1;
static_assert((TILE & TILE_MASK) == 0, "TILE must be a power of two");

__host__ __device__ inline int circular_slot(int index) {
  return index & TILE_MASK;
}

__host__ __device__ inline int co_rank_circular(
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

__host__ __device__ inline void merge_sequential_circular(
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

enum class MergeVariant { Basic, Tiled, Circular };

constexpr MergeVariant ALL_VARIANTS[] = {
    MergeVariant::Basic, MergeVariant::Tiled, MergeVariant::Circular};

static const char* variantName(MergeVariant variant) {
  switch (variant) {
    case MergeVariant::Basic:
      return "basic (global co-rank)";
    case MergeVariant::Tiled:
      return "tiled (shared memory)";
    case MergeVariant::Circular:
      return "circular buffer tiles";
  }
  return "unknown";
}

struct WorkSplit {
  int outputsPerThread;
  int outputsPerBlock;
};

constexpr int BASIC_THREADS_PER_BLOCK = 256;
constexpr WorkSplit DEFAULT_SPLIT = {64, 8 * TILE};
constexpr WorkSplit MANY_SMALL_SLICES_SPLIT = {5, 3 * TILE + 17};
constexpr int TIMED_OUTPUTS_PER_BLOCK = 16 * TILE;

constexpr unsigned RANDOM_SEED = 2024;
constexpr int FEW_DISTINCT_VALUES = 3;
constexpr int MANY_DISTINCT_VALUES = 1 << 30;
constexpr int LARGE_VALUE_RANGE = 1 << 26;
constexpr int LARGE_M = BY_BACKEND(1 << 24, 1 << 19);
constexpr int LARGE_N = LARGE_M + 12345;

constexpr int READS_AND_WRITES_PER_ELEMENT = 2;
constexpr double MS_PER_SECOND = 1e3;
constexpr double BYTES_PER_GB = 1e9;

struct DeviceMergeArrays {
  int* A;
  int* B;
  int* C;
  int m;
  int n;
};

static int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, std::max(count, 1) * sizeof(int)));
  return pointer;
}

static DeviceMergeArrays allocateDeviceArrays(int m, int n) {
  DeviceMergeArrays arrays;
  arrays.A = allocateDeviceInts(m);
  arrays.B = allocateDeviceInts(n);
  arrays.C = allocateDeviceInts(m + n);
  arrays.m = m;
  arrays.n = n;
  return arrays;
}

static void freeDeviceArrays(DeviceMergeArrays& arrays) {
  CHECK(cudaFree(arrays.A));
  CHECK(cudaFree(arrays.B));
  CHECK(cudaFree(arrays.C));
}

static void copyToDevice(int* destination, const std::vector<int>& source) {
  if (!source.empty()) {
    CHECK(cudaMemcpy(
        destination, source.data(), source.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
}

static void copyToHost(std::vector<int>& destination, const int* source) {
  if (!destination.empty()) {
    CHECK(cudaMemcpy(
        destination.data(), source, destination.size() * sizeof(int), cudaMemcpyDeviceToHost));
  }
}

static void launchBasicMerge(const DeviceMergeArrays& arrays, int outputsPerThread) {
  const int outputs = arrays.m + arrays.n;
  const unsigned threadsNeeded = cdiv(outputs, outputsPerThread);
  const unsigned blocks = cdiv(threadsNeeded, BASIC_THREADS_PER_BLOCK);
  launch(merge_basic_kernel,
         dim3(blocks),
         dim3(BASIC_THREADS_PER_BLOCK),
         arrays.A,
         arrays.m,
         arrays.B,
         arrays.n,
         arrays.C);
}

template <class TiledKernel>
static void launchTiledMerge(TiledKernel kernel,
                             const DeviceMergeArrays& arrays,
                             int outputsPerBlock) {
  const int outputs = arrays.m + arrays.n;
  const unsigned blocks = cdiv(outputs, outputsPerBlock);
  launch(kernel,
         dim3(blocks),
         dim3(TILED_THREADS_PER_BLOCK),
         arrays.A,
         arrays.m,
         arrays.B,
         arrays.n,
         arrays.C);
}

static void runMerge(MergeVariant variant, const DeviceMergeArrays& arrays, WorkSplit split) {
  if (arrays.m + arrays.n == 0) {
    return;
  }

  switch (variant) {
    case MergeVariant::Basic:
      launchBasicMerge(arrays, split.outputsPerThread);
      break;
    case MergeVariant::Tiled:
      launchTiledMerge(merge_tiled_kernel, arrays, split.outputsPerBlock);
      break;
    case MergeVariant::Circular:
      launchTiledMerge(merge_circular_kernel, arrays, split.outputsPerBlock);
      break;
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

static std::vector<int> sortedRandom(int count, int maxValue, std::mt19937& rng) {
  std::uniform_int_distribution<int> distribution(0, maxValue);
  std::vector<int> values(count);
  for (int& value : values) {
    value = distribution(rng);
  }
  std::sort(values.begin(), values.end());
  return values;
}

static std::vector<int> mergeOnHost(const std::vector<int>& A, const std::vector<int>& B) {
  std::vector<int> C(A.size() + B.size());
  std::merge(A.begin(), A.end(), B.begin(), B.end(), C.begin());
  return C;
}

static std::vector<int> mergeOnDevice(MergeVariant variant,
                                      const std::vector<int>& A,
                                      const std::vector<int>& B,
                                      WorkSplit split) {
  DeviceMergeArrays arrays =
      allocateDeviceArrays(static_cast<int>(A.size()), static_cast<int>(B.size()));
  copyToDevice(arrays.A, A);
  copyToDevice(arrays.B, B);

  runMerge(variant, arrays, split);

  std::vector<int> C(A.size() + B.size());
  copyToHost(C, arrays.C);
  freeDeviceArrays(arrays);
  return C;
}

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

static bool checkCoRankWorkedExample() {
  const int A[] = {3, 5, 5, 8, 12};
  const int B[] = {1, 5, 6, 8, 9, 14};
  const int expectedCoRank[] = {0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5};
  const int m = std::size(A);
  const int n = std::size(B);
  const int shownK = 6;

  bool ok = true;
  for (int k = 0; k <= m + n; ++k) {
    ok = ok && co_rank(k, A, m, B, n) == expectedCoRank[k];
  }

  const int shownI = co_rank(shownK, A, m, B, n);
  const int shownJ = shownK - shownI;
  std::printf("co_rank worked example (k = %d -> i = %d, j = %d): %s\n",
              shownK,
              shownI,
              shownJ,
              passOrFail(ok));
  return ok;
}

struct InputSizes {
  int m;
  int n;
};

static bool checkEdgeCases(std::mt19937& rng) {
  const InputSizes cases[] = {{0, 0},
                              {0, 7},
                              {9, 0},
                              {1, 1},
                              {5, 6},
                              {1000, 3},
                              {3, 1000},
                              {4097, 1},
                              {12345, 6789},
                              {20000, 20000}};
  const int valueRanges[] = {FEW_DISTINCT_VALUES, MANY_DISTINCT_VALUES};

  int passed = 0;
  int total = 0;
  for (const InputSizes& sizes : cases) {
    for (int maxValue : valueRanges) {
      const std::vector<int> A = sortedRandom(sizes.m, maxValue, rng);
      const std::vector<int> B = sortedRandom(sizes.n, maxValue, rng);
      const std::vector<int> expected = mergeOnHost(A, B);

      for (MergeVariant variant : ALL_VARIANTS) {
        const bool ok = mergeOnDevice(variant, A, B, MANY_SMALL_SLICES_SPLIT) == expected;
        passed += ok;
        ++total;
        if (!ok) {
          std::printf("  FAIL: %s with m=%d n=%d\n", variantName(variant), sizes.m, sizes.n);
        }
      }
    }
  }

  const bool allPassed = passed == total;
  std::printf("edge cases: %d/%d %s\n\n", passed, total, passOrFail(allPassed));
  return allPassed;
}

static double gigabytesPerSecond(int elements, double ms) {
  const double bytesMoved =
      static_cast<double>(elements) * READS_AND_WRITES_PER_ELEMENT * sizeof(int);
  const double seconds = ms / MS_PER_SECOND;
  return bytesMoved / seconds / BYTES_PER_GB;
}

static void printTimingRow(const char* label, double ms, int elements, const char* check) {
  std::printf("%-34s %10.2f %10.2f  %s\n", label, ms, gigabytesPerSecond(elements, ms), check);
}

struct TimedRun {
  MergeVariant variant;
  int outputsPerThread;
  const char* label;
};

static bool runTimedMerge(std::mt19937& rng) {
  const std::vector<int> A = sortedRandom(LARGE_M, LARGE_VALUE_RANGE, rng);
  const std::vector<int> B = sortedRandom(LARGE_N, LARGE_VALUE_RANGE, rng);
  const int outputs = LARGE_M + LARGE_N;

  std::vector<int> expected(outputs);
  Timer cpuTimer;
  std::merge(A.begin(), A.end(), B.begin(), B.end(), expected.begin());
  const double cpuMs = cpuTimer.ms();

  DeviceMergeArrays arrays = allocateDeviceArrays(LARGE_M, LARGE_N);
  copyToDevice(arrays.A, A);
  copyToDevice(arrays.B, B);
  runMerge(MergeVariant::Basic, arrays, DEFAULT_SPLIT);

  std::printf("merge of m = %d and n = %d ints (%d outputs)\n", LARGE_M, LARGE_N, outputs);
  std::printf("%-34s %10s %10s  %s\n", "variant", "time (ms)", "GB/s", "check");
  printTimingRow("std::merge (CPU, 1 thread)", cpuMs, outputs, "reference");

  const TimedRun runs[] = {{MergeVariant::Basic, 8, "basic, 8 outputs/thread"},
                           {MergeVariant::Basic, 64, "basic, 64 outputs/thread"},
                           {MergeVariant::Basic, 512, "basic, 512 outputs/thread"},
                           {MergeVariant::Tiled, 0, "tiled, 128 thr, TILE 1024"},
                           {MergeVariant::Circular, 0, "circular, 128 thr, TILE 1024"}};
  bool allPassed = true;
  std::vector<int> C(outputs);
  for (const TimedRun& run : runs) {
    CHECK(cudaMemset(arrays.C, 0, outputs * sizeof(int)));
    const WorkSplit split = {run.outputsPerThread, TIMED_OUTPUTS_PER_BLOCK};

    Timer gpuTimer;
    runMerge(run.variant, arrays, split);
    const double gpuMs = gpuTimer.ms();

    copyToHost(C, arrays.C);
    const bool ok = C == expected;
    allPassed = allPassed && ok;
    printTimingRow(run.label, gpuMs, outputs, passOrFail(ok));
  }
  freeDeviceArrays(arrays);

  std::printf("\n(GB/s counts one read and one write of every element.)\n");
  return allPassed;
}

int main() {
  std::printf("Parallel merge (chapter 13) on %s\n\n", CUDA_BACKEND);
  std::mt19937 rng(RANDOM_SEED);

  const bool workedExampleOk = checkCoRankWorkedExample();
  const bool edgeCasesOk = checkEdgeCases(rng);
  const bool timedMergeOk = runTimedMerge(rng);

  const bool allOk = workedExampleOk && edgeCasesOk && timedMergeOk;
  std::printf("%s\n", allOk ? "ALL PASS" : "SOME TESTS FAILED");
  return allOk ? 0 : 1;
}
