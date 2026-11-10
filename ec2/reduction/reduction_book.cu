#include "cuda_compat.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_DIM 1024
#define COARSE_FACTOR 4
static_assert(BLOCK_DIM == WARP_SIZE * WARP_SIZE, "the book's second stage needs 32 warp sums");

constexpr unsigned int FULL_WARP_MASK = 0xffffffff;
constexpr float SUM_IDENTITY = 0.0f;
constexpr int MAX_IDENTITY = INT_MIN;

constexpr unsigned SINGLE_BLOCK_INPUTS = 2 * BLOCK_DIM;
constexpr unsigned SEGMENT_LENGTH = 2 * BLOCK_DIM;
constexpr unsigned COARSENED_SEGMENT_LENGTH = COARSE_FACTOR * 2 * BLOCK_DIM;

#ifndef CPU_N
#define CPU_N (1u << 16)
#endif
const unsigned LARGE_INPUT_LENGTH = BY_BACKEND(1u << 25, CPU_N);
const int TIMED_REPETITIONS = BY_BACKEND(20, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);
const bool IS_CPU_EMULATION = BY_BACKEND(false, true);
static_assert(CPU_N % COARSENED_SEGMENT_LENGTH == 0, "CPU_N must be a multiple of 8192");
static_assert(CPU_N >= 2 * COARSENED_SEGMENT_LENGTH, "CPU_N must be at least 16384");

__global__ void SimpleSumReductionKernel(float* input, float* output) {
  unsigned int i = 2 * threadIdx.x;
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
  unsigned int i = threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void SharedMemorySumReductionKernel(float* input, float* output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int t = threadIdx.x;
  input_s[t] = input[t] + input[t + BLOCK_DIM];
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (threadIdx.x == 0) {
    *output = input_s[0];
  }
}

__device__ unsigned int warpIdx() {
  return threadIdx.x / WARP_SIZE;
}

__device__ unsigned int laneIdx() {
  return threadIdx.x % WARP_SIZE;
}

__device__ __inline__ float warp_reduce(float val) {
  float partialSum = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
  }
  return partialSum;
}

__global__ void WarpLevelSumReductionKernel(float* input, float* output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int t = threadIdx.x;
  input_s[t] = input[t] + input[t + BLOCK_DIM];
  for (unsigned int stride = blockDim.x / 2; stride >= WARP_SIZE; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  __syncthreads();
  if (warpIdx() == 0) {
    float partialSum = input_s[t];
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      *output = partialSum;
    }
  }
}

__global__ void TwoStageWarpLevelSumReductionKernel(float* input, float* output) {
  unsigned int t = threadIdx.x;
  float partialSum = input[t] + input[t + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float partialSum = partialSums_s[t];
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      *output = partialSum;
    }
  }
}

__global__ void reduce_kernel(float* input, float* output, unsigned int N) {
  (void)N;

  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i] + input[i + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
  if (threadIdx.x % WARP_SIZE == 0) {
    partialSums_s[threadIdx.x / WARP_SIZE] = partialSum;
  }
  __syncthreads();

  if (threadIdx.x < WARP_SIZE) {
    float partialSum = partialSums_s[threadIdx.x];
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
    }
  }
}

__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialSum += input[i + c * BLOCK_DIM];
  }

  partialSum = warp_reduce(partialSum);
  __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float blockSum = warp_reduce(partialSums_s[laneIdx()]);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(blockSum, cuda::memory_order_relaxed);
    }
  }
}

__global__ void ReversedConvergentSumReductionKernel(float* input, float* output) {
  unsigned int i = blockDim.x + threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x >= blockDim.x - stride) {
      input[i] += input[i - stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == blockDim.x - 1) {
    *output = input[2 * blockDim.x - 1];
  }
}

__device__ bool laneHasWarpResult() {
  return laneIdx() < BLOCK_DIM / WARP_SIZE;
}

__device__ __inline__ int warp_reduce_max(int val) {
  int partialMax = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialMax = imax(partialMax, __shfl_down_sync(FULL_WARP_MASK, partialMax, stride));
  }
  return partialMax;
}

__global__ void CoarsenedMaxReductionKernel(int* input, int* output) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  int partialMax = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialMax = imax(partialMax, input[i + c * BLOCK_DIM]);
  }

  partialMax = warp_reduce_max(partialMax);
  __shared__ int partialMaxs_s[BLOCK_DIM / WARP_SIZE];
  if (laneIdx() == 0) {
    partialMaxs_s[warpIdx()] = partialMax;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    int warpMax = laneHasWarpResult() ? partialMaxs_s[laneIdx()] : MAX_IDENTITY;
    int blockMax = warp_reduce_max(warpMax);
    if (threadIdx.x == 0) {
      atomicMax(output, blockMax);
    }
  }
}

__global__ void CoarsenedSumReductionKernelN(float* input, float* output, unsigned int N) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = SUM_IDENTITY;
  for (unsigned int c = 0; c < COARSE_FACTOR * 2; ++c) {
    if (i + c * BLOCK_DIM < N) {
      partialSum += input[i + c * BLOCK_DIM];
    }
  }

  partialSum = warp_reduce(partialSum);
  __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float warpSum = laneHasWarpResult() ? partialSums_s[laneIdx()] : SUM_IDENTITY;
    float blockSum = warp_reduce(warpSum);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(blockSum, cuda::memory_order_relaxed);
    }
  }
}

__device__ void copyOwnedAndNeverWrittenToTrace(const float* input,
                                                float* traceRow,
                                                unsigned int ownedPosition,
                                                unsigned int neverWrittenPosition) {
  traceRow[ownedPosition] = input[ownedPosition];
  traceRow[neverWrittenPosition] = input[neverWrittenPosition];
}

__global__ void SimpleSumReductionKernelTraced(float* input, float* output, float* trace) {
  unsigned int i = 2 * threadIdx.x;
  unsigned int step = 0;
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
    copyOwnedAndNeverWrittenToTrace(input, trace + step * 2 * blockDim.x, i, i + 1);
    ++step;
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void ConvergentSumReductionKernelTraced(float* input, float* output, float* trace) {
  unsigned int i = threadIdx.x;
  unsigned int step = 0;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
    copyOwnedAndNeverWrittenToTrace(input, trace + step * 2 * blockDim.x, i, i + blockDim.x);
    ++step;
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void ReversedConvergentSumReductionKernelTraced(float* input,
                                                           float* output,
                                                           float* trace) {
  unsigned int i = blockDim.x + threadIdx.x;
  unsigned int step = 0;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x >= blockDim.x - stride) {
      input[i] += input[i - stride];
    }
    __syncthreads();
    copyOwnedAndNeverWrittenToTrace(input, trace + step * 2 * blockDim.x, i, threadIdx.x);
    ++step;
  }
  if (threadIdx.x == blockDim.x - 1) {
    *output = input[2 * blockDim.x - 1];
  }
}

constexpr double SINGLE_BLOCK_TOLERANCE = 1e-5;
constexpr double MULTI_BLOCK_TOLERANCE = 1e-4;

constexpr uint64_t DATA_SEED = 10;
constexpr int RANDOM_INTEGER_MAGNITUDE = 1000000000;
constexpr int PLANTED_MAXIMUM = 1999999999;

constexpr unsigned MEMORY_SEGMENT_BYTES = 128;
constexpr unsigned BOOK_ANALYSIS_ELEMENTS = 256;
constexpr unsigned BOOK_ANALYSIS_THREADS = BOOK_ANALYSIS_ELEMENTS / 2;
constexpr unsigned BOOK_SIMPLE_RESOURCES = 864;
constexpr unsigned BOOK_CONVERGENT_RESOURCES = 384;
constexpr unsigned BOOK_COMMITTED_RESULTS = 255;
constexpr unsigned BOOK_SIMPLE_REQUESTS = 141;
constexpr unsigned BOOK_CONVERGENT_REQUESTS = 36;
constexpr unsigned BOOK_SHARED_MEMORY_REQUESTS = 9;

constexpr unsigned DIVERGENCE_EXERCISE_ELEMENTS = 1024;
constexpr unsigned DIVERGENCE_EXERCISE_THREADS = DIVERGENCE_EXERCISE_ELEMENTS / 2;
constexpr size_t FIFTH_ITERATION_INDEX = 4;
constexpr unsigned EXPECTED_SIMPLE_DIVERGENT_WARPS = 16;
constexpr unsigned EXPECTED_CONVERGENT_DIVERGENT_WARPS = 0;

constexpr unsigned TRACE_THREADS = 4;
constexpr unsigned TRACE_ELEMENTS = 2 * TRACE_THREADS;
constexpr unsigned TRACE_ITERATIONS = 3;
constexpr float TRACE_INPUT[TRACE_ELEMENTS] = {6, 2, 7, 4, 5, 8, 3, 1};
constexpr float TRACE_TOTAL = 36.0f;

class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {
  }

  uint64_t next() {
    state_ += GOLDEN_GAMMA;
    uint64_t mixed = state_;
    mixed = (mixed ^ (mixed >> 30)) * FIRST_MULTIPLIER;
    mixed = (mixed ^ (mixed >> 27)) * SECOND_MULTIPLIER;
    return mixed ^ (mixed >> 31);
  }

  float uniformBelowOne() {
    const uint64_t mantissa = next() >> (64 - FLOAT_MANTISSA_BITS);
    return static_cast<float>(mantissa * (1.0 / (1u << FLOAT_MANTISSA_BITS)));
  }

 private:
  static constexpr uint64_t GOLDEN_GAMMA = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MULTIPLIER = 0x94D049BB133111EBull;
  static constexpr int FLOAT_MANTISSA_BITS = 24;

  uint64_t state_;
};

struct InputData {
  std::vector<float> small;
  std::vector<float> large;
  std::vector<int> integers;
};

struct DeviceBuffers {
  float* small;
  float* large;
  float* sumOutput;
  int* integers;
  int* maxOutput;
};

enum class InputUse { ReadOnly, OverwrittenInPlace };

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

template <class T>
static T* allocateOnDevice(size_t count) {
  T* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

template <class T>
static void copyToDevice(T* destination, const T* source, size_t count) {
  CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
static void copyToHost(T* destination, const T* source, size_t count) {
  CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T>
static void setOnDevice(T* destination, T value) {
  copyToDevice(destination, &value, 1);
}

template <class T>
static T readFromDevice(const T* source) {
  T value{};
  copyToHost(&value, source, 1);
  return value;
}

template <class Work>
static double millisecondsPerRun(Work runOnce) {
  if (WARM_UP_BEFORE_TIMING) {
    runOnce();
  }
  Timer timer;
  for (int repetition = 0; repetition < TIMED_REPETITIONS; ++repetition) {
    runOnce();
  }
  const double milliseconds = timer.ms() / TIMED_REPETITIONS;
  CHECK(cudaGetLastError());
  return milliseconds;
}

static InputData makeInputData() {
  InputData data;
  data.small.resize(SINGLE_BLOCK_INPUTS);
  data.large.resize(LARGE_INPUT_LENGTH);
  data.integers.resize(LARGE_INPUT_LENGTH);

  SplitMix64 random(DATA_SEED);
  for (float& value : data.small) {
    value = random.uniformBelowOne();
  }
  for (float& value : data.large) {
    value = random.uniformBelowOne();
  }
  const uint64_t signedRangeSize = 2u * RANDOM_INTEGER_MAGNITUDE + 1u;
  for (int& value : data.integers) {
    value = static_cast<int>(random.next() % signedRangeSize) - RANDOM_INTEGER_MAGNITUDE;
  }
  data.integers[LARGE_INPUT_LENGTH / 3 + 5] = PLANTED_MAXIMUM;
  return data;
}

static double sumOfFirst(const std::vector<float>& values, unsigned count) {
  double sum = 0;
  for (unsigned k = 0; k < count; ++k) {
    sum += values[k];
  }
  return sum;
}

static int maximumOf(const std::vector<int>& values) {
  int maximum = MAX_IDENTITY;
  for (int value : values) {
    maximum = value > maximum ? value : maximum;
  }
  return maximum;
}

static DeviceBuffers allocateDeviceBuffers(const InputData& data) {
  DeviceBuffers device;
  device.small = allocateOnDevice<float>(SINGLE_BLOCK_INPUTS);
  device.large = allocateOnDevice<float>(LARGE_INPUT_LENGTH);
  device.sumOutput = allocateOnDevice<float>(1);
  device.integers = allocateOnDevice<int>(LARGE_INPUT_LENGTH);
  device.maxOutput = allocateOnDevice<int>(1);
  copyToDevice(device.large, data.large.data(), LARGE_INPUT_LENGTH);
  copyToDevice(device.integers, data.integers.data(), LARGE_INPUT_LENGTH);
  return device;
}

static void freeDeviceBuffers(const DeviceBuffers& device) {
  CHECK(cudaFree(device.small));
  CHECK(cudaFree(device.large));
  CHECK(cudaFree(device.sumOutput));
  CHECK(cudaFree(device.integers));
  CHECK(cudaFree(device.maxOutput));
}

static void printBanner() {
  std::printf("Reduction as the book writes it (chapter 10), backend: %s\n", CUDA_BACKEND);
  std::printf("BLOCK_DIM %d, WARP_SIZE %d, COARSE_FACTOR %d, %d timed run(s)\n",
              BLOCK_DIM,
              WARP_SIZE,
              COARSE_FACTOR,
              TIMED_REPETITIONS);
}

static void printTableHeader() {
  std::printf("\n%-40s %9s %6s %9s %16s %9s  %s\n",
              "kernel",
              "n",
              "blocks",
              "ms",
              "result",
              "rel.err",
              "check");
}

static void printReferenceRow(unsigned n, double reference) {
  std::printf("%-40s %9u %6s %9s %16.4f %9s  %s\n",
              "CPU, double accumulation",
              n,
              "-",
              "-",
              reference,
              "-",
              "reference");
}

static double relativeError(double value, double reference) {
  if (reference == 0) {
    return std::fabs(value);
  }
  return std::fabs(value - reference) / std::fabs(reference);
}

static bool printSumRow(const char* name,
                        unsigned n,
                        unsigned blocks,
                        double ms,
                        double value,
                        double reference,
                        double tolerance) {
  const double error = relativeError(value, reference);
  const bool ok = error <= tolerance;
  std::printf(
      "%-40s %9u %6u %9.3f %16.4f %9.2e  %s\n", name, n, blocks, ms, value, error, passOrFail(ok));
  return ok;
}

template <class Kernel>
static bool checkSingleBlockKernel(const char* name,
                                   Kernel kernel,
                                   InputUse inputUse,
                                   const InputData& data,
                                   const DeviceBuffers& device,
                                   double reference) {
  const bool overwritesInput = inputUse == InputUse::OverwrittenInPlace;
  if (!overwritesInput) {
    copyToDevice(device.small, data.small.data(), SINGLE_BLOCK_INPUTS);
  }
  const double ms = millisecondsPerRun([&] {
    if (overwritesInput) {
      copyToDevice(device.small, data.small.data(), SINGLE_BLOCK_INPUTS);
    }
    launch(kernel, dim3(1), dim3(BLOCK_DIM), device.small, device.sumOutput);
  });
  const float result = readFromDevice(device.sumOutput);
  return printSumRow(name, SINGLE_BLOCK_INPUTS, 1, ms, result, reference, SINGLE_BLOCK_TOLERANCE);
}

static bool checkSingleBlockKernels(const InputData& data, const DeviceBuffers& device) {
  const double reference = sumOfFirst(data.small, SINGLE_BLOCK_INPUTS);
  printReferenceRow(SINGLE_BLOCK_INPUTS, reference);

  const bool simpleOk = checkSingleBlockKernel("1 SimpleSumReductionKernel",
                                               SimpleSumReductionKernel,
                                               InputUse::OverwrittenInPlace,
                                               data,
                                               device,
                                               reference);
  const bool convergentOk = checkSingleBlockKernel("2 ConvergentSumReductionKernel",
                                                   ConvergentSumReductionKernel,
                                                   InputUse::OverwrittenInPlace,
                                                   data,
                                                   device,
                                                   reference);
  const bool reversedOk = checkSingleBlockKernel("ex3 ReversedConvergentSumReductionKernel",
                                                 ReversedConvergentSumReductionKernel,
                                                 InputUse::OverwrittenInPlace,
                                                 data,
                                                 device,
                                                 reference);
  const bool sharedMemoryOk = checkSingleBlockKernel("3 SharedMemorySumReductionKernel",
                                                     SharedMemorySumReductionKernel,
                                                     InputUse::ReadOnly,
                                                     data,
                                                     device,
                                                     reference);
  const bool warpLevelOk = checkSingleBlockKernel("5 WarpLevelSumReductionKernel",
                                                  WarpLevelSumReductionKernel,
                                                  InputUse::ReadOnly,
                                                  data,
                                                  device,
                                                  reference);
  const bool twoStageOk = checkSingleBlockKernel("6 TwoStageWarpLevelSumReductionKernel",
                                                 TwoStageWarpLevelSumReductionKernel,
                                                 InputUse::ReadOnly,
                                                 data,
                                                 device,
                                                 reference);
  return simpleOk && convergentOk && reversedOk && sharedMemoryOk && warpLevelOk && twoStageOk;
}

static bool checkReduceKernel(const DeviceBuffers& device, double reference) {
  const unsigned blocks = LARGE_INPUT_LENGTH / SEGMENT_LENGTH;
  const double ms = millisecondsPerRun([&] {
    setOnDevice(device.sumOutput, SUM_IDENTITY);
    launch(reduce_kernel,
           dim3(blocks),
           dim3(BLOCK_DIM),
           device.large,
           device.sumOutput,
           LARGE_INPUT_LENGTH);
  });
  const float result = readFromDevice(device.sumOutput);
  return printSumRow("7 reduce_kernel (atomic_ref)",
                     LARGE_INPUT_LENGTH,
                     blocks,
                     ms,
                     result,
                     reference,
                     MULTI_BLOCK_TOLERANCE);
}

static bool checkCoarsenedSumKernel(const DeviceBuffers& device, double reference) {
  const unsigned blocks = LARGE_INPUT_LENGTH / COARSENED_SEGMENT_LENGTH;
  const double ms = millisecondsPerRun([&] {
    setOnDevice(device.sumOutput, SUM_IDENTITY);
    launch(
        CoarsenedSumReductionKernel, dim3(blocks), dim3(BLOCK_DIM), device.large, device.sumOutput);
  });
  const float result = readFromDevice(device.sumOutput);
  return printSumRow("8 CoarsenedSumReductionKernel",
                     LARGE_INPUT_LENGTH,
                     blocks,
                     ms,
                     result,
                     reference,
                     MULTI_BLOCK_TOLERANCE);
}

static bool checkCoarsenedMaxKernel(const InputData& data, const DeviceBuffers& device) {
  const unsigned blocks = LARGE_INPUT_LENGTH / COARSENED_SEGMENT_LENGTH;
  const double ms = millisecondsPerRun([&] {
    setOnDevice(device.maxOutput, MAX_IDENTITY);
    launch(CoarsenedMaxReductionKernel,
           dim3(blocks),
           dim3(BLOCK_DIM),
           device.integers,
           device.maxOutput);
  });
  const int result = readFromDevice(device.maxOutput);
  const bool ok = result == maximumOf(data.integers);
  std::printf("%-40s %9u %6u %9.3f %16d %9s  %s\n",
              "ex4 CoarsenedMaxReductionKernel",
              LARGE_INPUT_LENGTH,
              blocks,
              ms,
              result,
              "exact",
              passOrFail(ok));
  return ok;
}

static bool checkAnyLengthKernel(const InputData& data, const DeviceBuffers& device) {
  const unsigned testLengths[] = {1u,
                                  1000u,
                                  COARSENED_SEGMENT_LENGTH - 1,
                                  COARSENED_SEGMENT_LENGTH + 1,
                                  LARGE_INPUT_LENGTH / 2 + 5u,
                                  LARGE_INPUT_LENGTH - 777u};
  bool ok = true;
  for (unsigned length : testLengths) {
    const unsigned blocks = cdiv(length, COARSENED_SEGMENT_LENGTH);
    const double ms = millisecondsPerRun([&] {
      setOnDevice(device.sumOutput, SUM_IDENTITY);
      launch(CoarsenedSumReductionKernelN,
             dim3(blocks),
             dim3(BLOCK_DIM),
             device.large,
             device.sumOutput,
             length);
    });
    const float result = readFromDevice(device.sumOutput);
    const double reference = sumOfFirst(data.large, length);
    const bool lengthOk = printSumRow("ex5 CoarsenedSumReductionKernelN",
                                      length,
                                      blocks,
                                      ms,
                                      result,
                                      reference,
                                      MULTI_BLOCK_TOLERANCE);
    ok = ok && lengthOk;
  }
  return ok;
}

static bool checkMultiBlockKernels(const InputData& data, const DeviceBuffers& device) {
  const double reference = sumOfFirst(data.large, LARGE_INPUT_LENGTH);
  printReferenceRow(LARGE_INPUT_LENGTH, reference);

  const bool reduceOk = checkReduceKernel(device, reference);
  const bool coarsenedOk = checkCoarsenedSumKernel(device, reference);
  const bool maxOk = checkCoarsenedMaxKernel(data, device);
  const bool anyLengthOk = checkAnyLengthKernel(data, device);
  return reduceOk && coarsenedOk && maxOk && anyLengthOk;
}

enum class Pattern { Simple, Convergent, Reversed };

static std::vector<unsigned> stridesOf(Pattern pattern, unsigned threads) {
  std::vector<unsigned> strides;
  if (pattern == Pattern::Simple) {
    for (unsigned stride = 1; stride <= threads; stride *= 2) {
      strides.push_back(stride);
    }
  } else {
    for (unsigned stride = threads; stride >= 1; stride /= 2) {
      strides.push_back(stride);
    }
  }
  return strides;
}

static bool isActive(Pattern pattern, unsigned thread, unsigned stride, unsigned threads) {
  switch (pattern) {
    case Pattern::Simple:
      return thread % stride == 0;
    case Pattern::Convergent:
      return thread < stride;
    case Pattern::Reversed:
      return thread >= threads - stride;
  }
  return false;
}

static unsigned ownedPosition(Pattern pattern, unsigned thread, unsigned threads) {
  switch (pattern) {
    case Pattern::Simple:
      return 2 * thread;
    case Pattern::Convergent:
      return thread;
    case Pattern::Reversed:
      return threads + thread;
  }
  return 0;
}

static unsigned partnerPosition(Pattern pattern,
                                unsigned thread,
                                unsigned stride,
                                unsigned threads) {
  const unsigned owned = ownedPosition(pattern, thread, threads);
  if (pattern == Pattern::Reversed) {
    return owned - stride;
  }
  return owned + stride;
}

static unsigned memorySegmentOf(unsigned floatPosition) {
  return static_cast<unsigned>(floatPosition * sizeof(float) / MEMORY_SEGMENT_BYTES);
}

struct WarpAccesses {
  unsigned activeLanes = 0;
  std::set<unsigned> ownSegments;
  std::set<unsigned> partnerSegments;
};

static WarpAccesses warpAccesses(Pattern pattern,
                                 unsigned threads,
                                 unsigned stride,
                                 unsigned warp) {
  WarpAccesses accesses;
  for (unsigned lane = 0; lane < WARP_SIZE && warp * WARP_SIZE + lane < threads; ++lane) {
    const unsigned thread = warp * WARP_SIZE + lane;
    if (!isActive(pattern, thread, stride, threads)) {
      continue;
    }
    ++accesses.activeLanes;
    accesses.ownSegments.insert(memorySegmentOf(ownedPosition(pattern, thread, threads)));
    accesses.partnerSegments.insert(
        memorySegmentOf(partnerPosition(pattern, thread, stride, threads)));
  }
  return accesses;
}

struct IterationStats {
  unsigned stride;
  unsigned activeThreads;
  unsigned activeWarps;
  unsigned divergentWarps;
  unsigned memoryRequests;
};

static IterationStats iterationStats(Pattern pattern, unsigned threads, unsigned stride) {
  IterationStats stats{stride, 0, 0, 0, 0};
  for (unsigned warp = 0; warp * WARP_SIZE < threads; ++warp) {
    const WarpAccesses accesses = warpAccesses(pattern, threads, stride, warp);
    const bool warpIsActive = accesses.activeLanes > 0;
    const bool warpIsDivergent = warpIsActive && accesses.activeLanes < WARP_SIZE;
    const unsigned readOwnRequests = static_cast<unsigned>(accesses.ownSegments.size());
    const unsigned readPartnerRequests = static_cast<unsigned>(accesses.partnerSegments.size());
    const unsigned writeOwnRequests = static_cast<unsigned>(accesses.ownSegments.size());

    stats.activeThreads += accesses.activeLanes;
    stats.activeWarps += warpIsActive ? 1 : 0;
    stats.divergentWarps += warpIsDivergent ? 1 : 0;
    stats.memoryRequests += readOwnRequests + readPartnerRequests + writeOwnRequests;
  }
  return stats;
}

struct PatternTotals {
  unsigned resources;
  unsigned committedResults;
  unsigned memoryRequests;
};

static PatternTotals patternTotals(Pattern pattern, unsigned threads) {
  PatternTotals totals{0, 0, 0};
  for (unsigned stride : stridesOf(pattern, threads)) {
    const IterationStats stats = iterationStats(pattern, threads, stride);
    totals.resources += stats.activeWarps * WARP_SIZE;
    totals.committedResults += stats.activeThreads;
    totals.memoryRequests += stats.memoryRequests;
  }
  return totals;
}

static unsigned sharedMemoryKernelRequests(unsigned threads) {
  unsigned loadRequests = 0;
  for (unsigned warp = 0; warp * WARP_SIZE < threads; ++warp) {
    std::set<unsigned> firstHalfSegments;
    std::set<unsigned> secondHalfSegments;
    for (unsigned lane = 0; lane < WARP_SIZE && warp * WARP_SIZE + lane < threads; ++lane) {
      const unsigned thread = warp * WARP_SIZE + lane;
      firstHalfSegments.insert(memorySegmentOf(thread));
      secondHalfSegments.insert(memorySegmentOf(thread + threads));
    }
    loadRequests += static_cast<unsigned>(firstHalfSegments.size() + secondHalfSegments.size());
  }
  const unsigned finalStoreRequests = 1;
  return loadRequests + finalStoreRequests;
}

static double efficiency(const PatternTotals& totals) {
  return static_cast<double>(totals.committedResults) / totals.resources;
}

static void printResourceTable(unsigned threads) {
  std::printf("\nExecution resources, %u elements = %u threads = %u warps\n",
              2 * threads,
              threads,
              threads / WARP_SIZE);
  std::printf("%5s | %-30s | %-30s\n",
              "iter",
              "simple: stride threads warps",
              "convergent: stride threads warps");
  const std::vector<unsigned> simpleStrides = stridesOf(Pattern::Simple, threads);
  const std::vector<unsigned> convergentStrides = stridesOf(Pattern::Convergent, threads);
  for (size_t k = 0; k < simpleStrides.size(); ++k) {
    const IterationStats simple = iterationStats(Pattern::Simple, threads, simpleStrides[k]);
    const IterationStats convergent =
        iterationStats(Pattern::Convergent, threads, convergentStrides[k]);
    std::printf("%5zu | %8u %8u %6u%7s | %8u %8u %6u\n",
                k + 1,
                simple.stride,
                simple.activeThreads,
                simple.activeWarps,
                "",
                convergent.stride,
                convergent.activeThreads,
                convergent.activeWarps);
  }
}

static bool checkExecutionResources() {
  printResourceTable(BOOK_ANALYSIS_THREADS);
  const PatternTotals simple = patternTotals(Pattern::Simple, BOOK_ANALYSIS_THREADS);
  const PatternTotals convergent = patternTotals(Pattern::Convergent, BOOK_ANALYSIS_THREADS);
  const PatternTotals reversed = patternTotals(Pattern::Reversed, BOOK_ANALYSIS_THREADS);

  std::printf("resources consumed (active warps x 32): simple %u, convergent %u, reversed %u\n",
              simple.resources,
              convergent.resources,
              reversed.resources);
  std::printf("results committed (active threads):     simple %u, convergent %u, reversed %u\n",
              simple.committedResults,
              convergent.committedResults,
              reversed.committedResults);
  std::printf("efficiency: simple %u/%u = %.2f, convergent %u/%u = %.2f (%.0f%%)\n",
              simple.committedResults,
              simple.resources,
              efficiency(simple),
              convergent.committedResults,
              convergent.resources,
              efficiency(convergent),
              100.0 * efficiency(convergent));

  const bool ok = simple.resources == BOOK_SIMPLE_RESOURCES &&
                  convergent.resources == BOOK_CONVERGENT_RESOURCES &&
                  simple.committedResults == BOOK_COMMITTED_RESULTS &&
                  convergent.committedResults == BOOK_COMMITTED_RESULTS &&
                  reversed.resources == BOOK_CONVERGENT_RESOURCES;
  std::printf("book: 864 and 384 resources, 255 results: %s\n", passOrFail(ok));
  return ok;
}

static bool checkMemoryRequests() {
  const PatternTotals simple = patternTotals(Pattern::Simple, BOOK_ANALYSIS_THREADS);
  const PatternTotals convergent = patternTotals(Pattern::Convergent, BOOK_ANALYSIS_THREADS);
  const PatternTotals reversed = patternTotals(Pattern::Reversed, BOOK_ANALYSIS_THREADS);
  const unsigned sharedMemory = sharedMemoryKernelRequests(BOOK_ANALYSIS_THREADS);

  std::printf("\nGlobal memory requests, %u elements (128-byte segments per warp access)\n",
              BOOK_ANALYSIS_ELEMENTS);
  std::printf("  simple %u, convergent %u, reversed (ex. 3) %u, shared memory %u (= 256/32 + 1)\n",
              simple.memoryRequests,
              convergent.memoryRequests,
              reversed.memoryRequests,
              sharedMemory);

  const bool ok = simple.memoryRequests == BOOK_SIMPLE_REQUESTS &&
                  convergent.memoryRequests == BOOK_CONVERGENT_REQUESTS &&
                  sharedMemory == BOOK_SHARED_MEMORY_REQUESTS;
  std::printf("book: 141, 36 and 9: %s\n", passOrFail(ok));
  return ok;
}

static bool checkDivergenceExercises() {
  const unsigned threads = DIVERGENCE_EXERCISE_THREADS;
  std::printf(
      "\nExercises 1 and 2: %u elements = %u threads = %u warps; divergent warp = some but"
      " not all lanes active\n",
      DIVERGENCE_EXERCISE_ELEMENTS,
      threads,
      threads / WARP_SIZE);
  std::printf("%5s | %-34s | %-34s\n",
              "iter",
              "simple: stride threads warps diverg",
              "convergent: stride threads warps diverg");

  const std::vector<unsigned> simpleStrides = stridesOf(Pattern::Simple, threads);
  const std::vector<unsigned> convergentStrides = stridesOf(Pattern::Convergent, threads);
  unsigned exerciseOneAnswer = 0;
  unsigned exerciseTwoAnswer = 0;
  for (size_t k = 0; k < simpleStrides.size(); ++k) {
    const IterationStats simple = iterationStats(Pattern::Simple, threads, simpleStrides[k]);
    const IterationStats convergent =
        iterationStats(Pattern::Convergent, threads, convergentStrides[k]);
    const bool isFifthIteration = k == FIFTH_ITERATION_INDEX;
    if (isFifthIteration) {
      exerciseOneAnswer = simple.divergentWarps;
      exerciseTwoAnswer = convergent.divergentWarps;
    }
    std::printf("%5zu | %8u %8u %6u %6u%5s | %8u %8u %6u %6u%s\n",
                k + 1,
                simple.stride,
                simple.activeThreads,
                simple.activeWarps,
                simple.divergentWarps,
                "",
                convergent.stride,
                convergent.activeThreads,
                convergent.activeWarps,
                convergent.divergentWarps,
                isFifthIteration ? "   <- 5th iteration" : "");
  }

  std::printf("exercise 1 (simple, 5th iteration, stride 16): %u divergent warps\n",
              exerciseOneAnswer);
  std::printf("exercise 2 (convergent, 5th iteration, stride 32): %u divergent warps\n",
              exerciseTwoAnswer);
  return exerciseOneAnswer == EXPECTED_SIMPLE_DIVERGENT_WARPS &&
         exerciseTwoAnswer == EXPECTED_CONVERGENT_DIVERGENT_WARPS;
}

static bool checkChapterAnalysis() {
  const bool resourcesOk = checkExecutionResources();
  const bool requestsOk = checkMemoryRequests();
  const bool exercisesOk = checkDivergenceExercises();
  return resourcesOk && requestsOk && exercisesOk;
}

struct TraceRun {
  std::vector<float> rows;
  float output;
};

template <class Kernel>
static TraceRun runTracedKernel(Kernel kernel) {
  float* input = allocateOnDevice<float>(TRACE_ELEMENTS);
  float* output = allocateOnDevice<float>(1);
  float* trace = allocateOnDevice<float>(TRACE_ITERATIONS * TRACE_ELEMENTS);
  copyToDevice(input, TRACE_INPUT, TRACE_ELEMENTS);

  launch(kernel, dim3(1), dim3(TRACE_THREADS), input, output, trace);
  CHECK(cudaGetLastError());

  TraceRun run;
  run.rows.resize(TRACE_ITERATIONS * TRACE_ELEMENTS);
  copyToHost(run.rows.data(), trace, run.rows.size());
  run.output = readFromDevice(output);

  CHECK(cudaFree(input));
  CHECK(cudaFree(output));
  CHECK(cudaFree(trace));
  return run;
}

static void replayIterationOnHost(Pattern pattern, unsigned stride, std::vector<float>& values) {
  for (unsigned thread = 0; thread < TRACE_THREADS; ++thread) {
    if (isActive(pattern, thread, stride, TRACE_THREADS)) {
      const unsigned owned = ownedPosition(pattern, thread, TRACE_THREADS);
      const unsigned partner = partnerPosition(pattern, thread, stride, TRACE_THREADS);
      values[owned] += values[partner];
    }
  }
}

static void printTraceRow(const char* label, const float* values) {
  std::printf("    %-16s", label);
  for (unsigned k = 0; k < TRACE_ELEMENTS; ++k) {
    std::printf(" %3.0f", values[k]);
  }
  std::printf("\n");
}

template <class Kernel>
static bool checkTrace(const char* title, Kernel kernel, Pattern pattern) {
  const TraceRun run = runTracedKernel(kernel);
  std::vector<float> expected(TRACE_INPUT, TRACE_INPUT + TRACE_ELEMENTS);

  std::printf("  %s\n", title);
  printTraceRow("input:", TRACE_INPUT);
  bool ok = true;
  const std::vector<unsigned> strides = stridesOf(pattern, TRACE_THREADS);
  for (unsigned iteration = 0; iteration < TRACE_ITERATIONS; ++iteration) {
    replayIterationOnHost(pattern, strides[iteration], expected);
    const float* traced = &run.rows[iteration * TRACE_ELEMENTS];
    char label[32];
    std::snprintf(label, sizeof label, "after stride %u:", strides[iteration]);
    printTraceRow(label, traced);
    for (unsigned k = 0; k < TRACE_ELEMENTS; ++k) {
      ok = ok && traced[k] == expected[k];
    }
  }

  ok = ok && run.output == TRACE_TOTAL;
  std::printf("    output %.0f  %s\n", run.output, passOrFail(ok));
  return ok;
}

static bool checkExerciseSixTraces() {
  std::printf(
      "\nExercise 6: {6, 2, 7, 4, 5, 8, 3, 1}, one block of 4 threads, array after each "
      "iteration\n");
  const bool simpleOk = checkTrace(
      "a. SimpleSumReductionKernel (traced copy)", SimpleSumReductionKernelTraced, Pattern::Simple);
  const bool convergentOk = checkTrace("b. ConvergentSumReductionKernel (traced copy)",
                                       ConvergentSumReductionKernelTraced,
                                       Pattern::Convergent);
  const bool reversedOk = checkTrace("ex. 3 ReversedConvergentSumReductionKernel (traced copy)",
                                     ReversedConvergentSumReductionKernelTraced,
                                     Pattern::Reversed);
  return simpleOk && convergentOk && reversedOk;
}

int main() {
  printBanner();

  const InputData data = makeInputData();
  const DeviceBuffers device = allocateDeviceBuffers(data);

  printTableHeader();
  const bool singleBlockOk = checkSingleBlockKernels(data, device);
  const bool multiBlockOk = checkMultiBlockKernels(data, device);
  const bool analysisOk = checkChapterAnalysis();
  const bool tracesOk = checkExerciseSixTraces();
  freeDeviceBuffers(device);

  const bool ok = singleBlockOk && multiBlockOk && analysisOk && tracesOk;
  std::printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  if (IS_CPU_EMULATION) {
    std::printf("(CPU emulation: the times measure thread start-up and barriers, not the GPU.)\n");
  }
  return ok ? 0 : 1;
}
