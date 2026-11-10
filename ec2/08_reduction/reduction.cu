#include "cuda_compat.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr unsigned SINGLE_BLOCK_THREADS = 1024;
constexpr unsigned SINGLE_BLOCK_INPUTS = 2 * SINGLE_BLOCK_THREADS;
constexpr unsigned BLOCK_DIM = 256;
constexpr unsigned WARP_SIZE = 32;
constexpr unsigned WARPS_IN_SINGLE_BLOCK = SINGLE_BLOCK_THREADS / WARP_SIZE;

constexpr float SUM_IDENTITY = 0.0f;
constexpr int MAX_IDENTITY = INT_MIN;

#ifndef CPU_N
#define CPU_N (1u << 18)
#endif
const unsigned LARGE_INPUT_LENGTH = BY_BACKEND(1u << 25, CPU_N);
const int TIMED_REPETITIONS = BY_BACKEND(20, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

constexpr unsigned SMALL_COARSE_FACTOR = 4;
constexpr unsigned LARGE_COARSE_FACTOR = 32;
constexpr unsigned MAX_COARSE_FACTOR = 8;

constexpr double SINGLE_BLOCK_TOLERANCE = 1e-5;
constexpr double MULTI_BLOCK_TOLERANCE = 1e-4;

constexpr uint64_t DATA_SEED = 10;
constexpr uint64_t RANDOM_INTEGER_LIMIT = 1000000000u;
constexpr int PLANTED_MAXIMUM = 1999999999;

__global__ void SimpleSumReductionKernel(float* input, float* output) {
  unsigned i = 2 * threadIdx.x;
  for (unsigned stride = 1; stride <= blockDim.x; stride *= 2) {
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
  unsigned i = threadIdx.x;
  for (unsigned stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void SharedMemorySumReductionKernel(const float* input, float* output) {
  __shared__ float input_s[SINGLE_BLOCK_THREADS];
  unsigned t = threadIdx.x;
  input_s[t] = input[t] + input[t + SINGLE_BLOCK_THREADS];
  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    *output = input_s[0];
  }
}

__device__ inline float valueOrSumIdentity(const float* input, unsigned index, unsigned n) {
  return index < n ? input[index] : SUM_IDENTITY;
}

__device__ inline int valueOrMaxIdentity(const int* input, unsigned index, unsigned n) {
  return index < n ? input[index] : MAX_IDENTITY;
}

__global__ void SegmentedSumReductionKernel(const float* input, float* output, unsigned n) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned segment = 2 * blockDim.x * blockIdx.x;
  unsigned i = segment + threadIdx.x;
  unsigned t = threadIdx.x;
  float firstHalf = valueOrSumIdentity(input, i, n);
  float secondHalf = valueOrSumIdentity(input, i + BLOCK_DIM, n);
  input_s[t] = firstHalf + secondHalf;

  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}

template <unsigned COARSE_FACTOR>
__global__ void CoarsenedSumReductionKernel(const float* input, float* output, unsigned n) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned i = segment + threadIdx.x;
  unsigned t = threadIdx.x;
  float partialSum = SUM_IDENTITY;
  for (unsigned tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
    partialSum += valueOrSumIdentity(input, i + tile * BLOCK_DIM, n);
  }
  input_s[t] = partialSum;

  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}

template <unsigned COARSE_FACTOR>
__global__ void CoarsenedMaxReductionKernel(const int* input, int* output, unsigned n) {
  __shared__ int input_s[BLOCK_DIM];
  unsigned segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned i = segment + threadIdx.x;
  unsigned t = threadIdx.x;
  int partialMax = MAX_IDENTITY;
  for (unsigned tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
    partialMax = imax(partialMax, valueOrMaxIdentity(input, i + tile * BLOCK_DIM, n));
  }
  input_s[t] = partialMax;

  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] = imax(input_s[t], input_s[t + stride]);
    }
  }

  if (t == 0) {
    atomicMax(output, input_s[0]);
  }
}

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

struct CpuReferences {
  double smallSum = 0;
  double largeSum = 0;
  double largeSumMs = 0;
  float sequentialFloatSum = 0;
  double sequentialFloatMs = 0;
  int maximum = MAX_IDENTITY;
  double maximumMs = 0;
};

struct DeviceBuffers {
  float* small;
  float* large;
  float* sumOutput;
  int* integers;
  int* maxOutput;
};

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
static void copyToDevice(T* destination, const std::vector<T>& source) {
  CHECK(cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
static void setOnDevice(T* destination, T value) {
  CHECK(cudaMemcpy(destination, &value, sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
static T readFromDevice(const T* source) {
  T value{};
  CHECK(cudaMemcpy(&value, source, sizeof(T), cudaMemcpyDeviceToHost));
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
  for (int& value : data.integers) {
    value = static_cast<int>(random.next() % RANDOM_INTEGER_LIMIT);
  }
  data.integers[LARGE_INPUT_LENGTH / 3] = PLANTED_MAXIMUM;
  return data;
}

static CpuReferences computeCpuReferences(const InputData& data) {
  CpuReferences references;
  for (float value : data.small) {
    references.smallSum += value;
  }

  Timer doubleTimer;
  for (float value : data.large) {
    references.largeSum += value;
  }
  references.largeSumMs = doubleTimer.ms();

  Timer floatTimer;
  for (float value : data.large) {
    references.sequentialFloatSum += value;
  }
  references.sequentialFloatMs = floatTimer.ms();

  Timer maximumTimer;
  for (int value : data.integers) {
    references.maximum = value > references.maximum ? value : references.maximum;
  }
  references.maximumMs = maximumTimer.ms();
  return references;
}

static DeviceBuffers allocateDeviceBuffers(const InputData& data) {
  DeviceBuffers device;
  device.small = allocateOnDevice<float>(SINGLE_BLOCK_INPUTS);
  device.large = allocateOnDevice<float>(LARGE_INPUT_LENGTH);
  device.sumOutput = allocateOnDevice<float>(1);
  device.integers = allocateOnDevice<int>(LARGE_INPUT_LENGTH);
  device.maxOutput = allocateOnDevice<int>(1);
  copyToDevice(device.large, data.large);
  copyToDevice(device.integers, data.integers);
  return device;
}

static void freeDeviceBuffers(const DeviceBuffers& device) {
  CHECK(cudaFree(device.small));
  CHECK(cudaFree(device.large));
  CHECK(cudaFree(device.sumOutput));
  CHECK(cudaFree(device.integers));
  CHECK(cudaFree(device.maxOutput));
}

static void printTableHeader() {
  std::printf("\n%-30s %10s %7s %10s %18s %10s  %s\n",
              "kernel",
              "n",
              "blocks",
              "ms",
              "result",
              "rel.err",
              "check");
}

static void printUntimedReferenceRow(const char* name, unsigned n, double value) {
  std::printf("%-30s %10u %7s %10s %18.4f %10s  %s\n", name, n, "-", "-", value, "-", "reference");
}

static void printTimedReferenceRow(const char* name, unsigned n, double ms, double value) {
  std::printf("%-30s %10u %7s %10.3f %18.4f %10s  %s\n", name, n, "-", ms, value, "-", "reference");
}

static bool printSumRow(const char* name,
                        unsigned n,
                        unsigned blocks,
                        double ms,
                        double value,
                        double reference,
                        double tolerance) {
  const double relativeError = std::fabs(value - reference) / std::fabs(reference);
  const bool ok = relativeError <= tolerance;
  std::printf("%-30s %10u %7u %10.3f %18.4f %10.2e  %s\n",
              name,
              n,
              blocks,
              ms,
              value,
              relativeError,
              passOrFail(ok));
  return ok;
}

template <class Kernel>
static bool checkInPlaceKernel(const char* name,
                               Kernel kernel,
                               const InputData& data,
                               const DeviceBuffers& device,
                               double reference) {
  const double ms = millisecondsPerRun([&] {
    copyToDevice(device.small, data.small);
    launch(kernel, dim3(1), dim3(SINGLE_BLOCK_THREADS), device.small, device.sumOutput);
  });
  const float result = readFromDevice(device.sumOutput);
  return printSumRow(name, SINGLE_BLOCK_INPUTS, 1, ms, result, reference, SINGLE_BLOCK_TOLERANCE);
}

static bool checkSharedMemoryKernel(const InputData& data,
                                    const DeviceBuffers& device,
                                    double reference) {
  copyToDevice(device.small, data.small);
  const float* input = device.small;
  const double ms = millisecondsPerRun([&] {
    launch(SharedMemorySumReductionKernel,
           dim3(1),
           dim3(SINGLE_BLOCK_THREADS),
           input,
           device.sumOutput);
  });
  const float result = readFromDevice(device.sumOutput);
  return printSumRow(
      "3 shared memory", SINGLE_BLOCK_INPUTS, 1, ms, result, reference, SINGLE_BLOCK_TOLERANCE);
}

static bool checkSingleBlockKernels(const InputData& data,
                                    const DeviceBuffers& device,
                                    const CpuReferences& references) {
  printUntimedReferenceRow("CPU double (small input)", SINGLE_BLOCK_INPUTS, references.smallSum);
  const bool simpleOk = checkInPlaceKernel(
      "1 simple (in place)", SimpleSumReductionKernel, data, device, references.smallSum);
  const bool convergentOk = checkInPlaceKernel(
      "2 convergent (in place)", ConvergentSumReductionKernel, data, device, references.smallSum);
  const bool sharedMemoryOk = checkSharedMemoryKernel(data, device, references.smallSum);
  return simpleOk && convergentOk && sharedMemoryOk;
}

template <class Kernel>
static bool checkMultiBlockSumKernel(const char* name,
                                     Kernel kernel,
                                     unsigned elementsPerBlock,
                                     const DeviceBuffers& device,
                                     double reference) {
  const unsigned blocks = cdiv(LARGE_INPUT_LENGTH, elementsPerBlock);
  const float* input = device.large;
  const double ms = millisecondsPerRun([&] {
    setOnDevice(device.sumOutput, SUM_IDENTITY);
    launch(kernel, dim3(blocks), dim3(BLOCK_DIM), input, device.sumOutput, LARGE_INPUT_LENGTH);
  });
  const float result = readFromDevice(device.sumOutput);
  return printSumRow(
      name, LARGE_INPUT_LENGTH, blocks, ms, result, reference, MULTI_BLOCK_TOLERANCE);
}

static void printLargeInputReferences(const CpuReferences& references) {
  printTimedReferenceRow(
      "CPU double (large input)", LARGE_INPUT_LENGTH, references.largeSumMs, references.largeSum);
  const double floatSum = static_cast<double>(references.sequentialFloatSum);
  const double floatRelativeError = std::fabs(floatSum - references.largeSum) / references.largeSum;
  std::printf("%-30s %10u %7s %10.3f %18.4f %10.2e  %s\n",
              "CPU float, sequential",
              LARGE_INPUT_LENGTH,
              "-",
              references.sequentialFloatMs,
              floatSum,
              floatRelativeError,
              "(for comparison)");
}

static bool checkMultiBlockSumKernels(const DeviceBuffers& device,
                                      const CpuReferences& references) {
  printLargeInputReferences(references);
  const bool segmentedOk = checkMultiBlockSumKernel("4 segmented + atomicAdd",
                                                    SegmentedSumReductionKernel,
                                                    2 * BLOCK_DIM,
                                                    device,
                                                    references.largeSum);
  const bool smallFactorOk =
      checkMultiBlockSumKernel("5 coarsened, factor 4",
                               CoarsenedSumReductionKernel<SMALL_COARSE_FACTOR>,
                               SMALL_COARSE_FACTOR * 2 * BLOCK_DIM,
                               device,
                               references.largeSum);
  const bool largeFactorOk =
      checkMultiBlockSumKernel("5 coarsened, factor 32",
                               CoarsenedSumReductionKernel<LARGE_COARSE_FACTOR>,
                               LARGE_COARSE_FACTOR * 2 * BLOCK_DIM,
                               device,
                               references.largeSum);
  return segmentedOk && smallFactorOk && largeFactorOk;
}

static bool checkMaxKernel(const DeviceBuffers& device, const CpuReferences& references) {
  const unsigned blocks = cdiv(LARGE_INPUT_LENGTH, MAX_COARSE_FACTOR * 2 * BLOCK_DIM);
  const int* input = device.integers;
  const double ms = millisecondsPerRun([&] {
    setOnDevice(device.maxOutput, MAX_IDENTITY);
    launch(CoarsenedMaxReductionKernel<MAX_COARSE_FACTOR>,
           dim3(blocks),
           dim3(BLOCK_DIM),
           input,
           device.maxOutput,
           LARGE_INPUT_LENGTH);
  });
  const int result = readFromDevice(device.maxOutput);
  const bool ok = result == references.maximum;

  std::printf("%-30s %10u %7s %10.3f %18d %10s  %s\n",
              "CPU max, sequential",
              LARGE_INPUT_LENGTH,
              "-",
              references.maximumMs,
              references.maximum,
              "-",
              "reference");
  std::printf("%-30s %10u %7u %10.3f %18d %10s  %s\n",
              "6 max, coarsened factor 8",
              LARGE_INPUT_LENGTH,
              blocks,
              ms,
              result,
              "exact",
              passOrFail(ok));
  return ok;
}

struct StepActivity {
  unsigned activeThreads = 0;
  unsigned busyWarps = 0;
};

static bool simpleKernelThreadIsActive(unsigned thread, unsigned stride) {
  return thread % stride == 0;
}

static bool convergentKernelThreadIsActive(unsigned thread, unsigned stride) {
  return thread < stride;
}

template <class IsActive>
static StepActivity countStepActivity(IsActive isActive, unsigned stride) {
  StepActivity activity;
  for (unsigned warp = 0; warp < WARPS_IN_SINGLE_BLOCK; ++warp) {
    unsigned activeLanes = 0;
    for (unsigned lane = 0; lane < WARP_SIZE; ++lane) {
      if (isActive(warp * WARP_SIZE + lane, stride)) {
        ++activeLanes;
      }
    }
    activity.activeThreads += activeLanes;
    if (activeLanes > 0) {
      ++activity.busyWarps;
    }
  }
  return activity;
}

static double laneUtilisationPercent(unsigned long long additions, unsigned long long warpSteps) {
  return 100.0 * additions / (warpSteps * static_cast<double>(WARP_SIZE));
}

static void printDivergenceAnalysis() {
  std::printf("\nDivergence analysis, one block of %u threads = %u warps, %u inputs\n",
              SINGLE_BLOCK_THREADS,
              WARPS_IN_SINGLE_BLOCK,
              SINGLE_BLOCK_INPUTS);
  std::printf(
      "%6s | %-26s | %-26s\n", "step", "simple: threads / warps", "convergent: threads / warps");

  unsigned long long additions = 0;
  unsigned long long simpleWarpSteps = 0;
  unsigned long long convergentWarpSteps = 0;
  for (unsigned step = 0; (1u << step) <= SINGLE_BLOCK_THREADS; ++step) {
    const unsigned simpleStride = 1u << step;
    const unsigned convergentStride = SINGLE_BLOCK_THREADS >> step;
    const StepActivity simple = countStepActivity(simpleKernelThreadIsActive, simpleStride);
    const StepActivity convergent =
        countStepActivity(convergentKernelThreadIsActive, convergentStride);

    additions += simple.activeThreads;
    simpleWarpSteps += simple.busyWarps;
    convergentWarpSteps += convergent.busyWarps;
    std::printf("%6u | %10u / %-13u | %10u / %-13u\n",
                step,
                simple.activeThreads,
                simple.busyWarps,
                convergent.activeThreads,
                convergent.busyWarps);
  }

  std::printf("total additions %llu; warp-steps: simple %llu, convergent %llu\n",
              additions,
              simpleWarpSteps,
              convergentWarpSteps);
  std::printf("lane utilisation (additions / (warp-steps*32)): simple %.1f%%, convergent %.1f%%\n",
              laneUtilisationPercent(additions, simpleWarpSteps),
              laneUtilisationPercent(additions, convergentWarpSteps));
  std::printf(
      "global traffic of kernels 1-2: %llu loads + %llu stores; kernel 3: %u loads + 1 store\n",
      2 * additions,
      additions,
      SINGLE_BLOCK_INPUTS);
}

int main() {
  std::printf("Parallel reduction (chapter 10), backend: %s\n", CUDA_BACKEND);

  const InputData data = makeInputData();
  const CpuReferences references = computeCpuReferences(data);
  const DeviceBuffers device = allocateDeviceBuffers(data);

  printTableHeader();
  const bool singleBlockOk = checkSingleBlockKernels(data, device, references);
  const bool multiBlockOk = checkMultiBlockSumKernels(device, references);
  const bool maxOk = checkMaxKernel(device, references);
  const bool ok = singleBlockOk && multiBlockOk && maxOk;

  printDivergenceAnalysis();

  freeDeviceBuffers(device);
  std::printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
