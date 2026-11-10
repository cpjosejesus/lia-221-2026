#include "cuda_compat.h"

#include <cstdio>
#include <vector>

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int FULL_WARP_MASK = 0xffffffff;
constexpr unsigned int BLOCK_DIM = 32;
constexpr unsigned int NUM_WARPS = BLOCK_DIM / WARP_SIZE;
constexpr unsigned int COARSE_FACTOR = 2;
constexpr unsigned int SEGMENT_LENGTH = COARSE_FACTOR * 2 * BLOCK_DIM;
constexpr unsigned int DIGITS = 10;
const unsigned int LENGTHS[] = {256, 200, 129, 100, 1};

__device__ unsigned int warpIdx() {
  return threadIdx.x / WARP_SIZE;
}

__device__ unsigned int laneIdx() {
  return threadIdx.x % WARP_SIZE;
}

__device__ __inline__ float warp_reduce(float val) {
  float partialSum = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialSum += __shfl_down_sync(FULL_WARP_MASK, partialSum, stride);
  }
  return partialSum;
}

__global__ void CoarsenedSumReductionKernel(float* input, float* output, unsigned int N) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = 0.0f;
  for (unsigned int c = 0; c < COARSE_FACTOR * 2; ++c) {
    if (i + c * BLOCK_DIM < N) {
      partialSum += input[i + c * BLOCK_DIM];
    }
  }

  partialSum = warp_reduce(partialSum);
  __shared__ float partialSums_s[NUM_WARPS];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float partialSum = laneIdx() < NUM_WARPS ? partialSums_s[laneIdx()] : 0.0f;
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
    }
  }
}

__global__ void CoarsenedSumReductionKernelPrinting(float* input, float* output, unsigned int N) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = 0.0f;
  for (unsigned int c = 0; c < COARSE_FACTOR * 2; ++c) {
    if (i + c * BLOCK_DIM < N) {
      partialSum += input[i + c * BLOCK_DIM];
    }
  }

  partialSum = warp_reduce(partialSum);
  __shared__ float partialSums_s[NUM_WARPS];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float partialSum = laneIdx() < NUM_WARPS ? partialSums_s[laneIdx()] : 0.0f;
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
      unsigned int segmentEnd = segment + SEGMENT_LENGTH < N ? segment + SEGMENT_LENGTH : N;
      printf("  block %u reads elements %3u..%3u (%3u of %u), adds %4g\n",
             blockIdx.x,
             segment,
             segmentEnd - 1,
             segmentEnd - segment,
             SEGMENT_LENGTH,
             partialSum);
    }
  }
}

float sum_sequential(const float* input, unsigned int n) {
  float sum = 0.0f;
  for (unsigned int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}

static std::vector<float> makeInput(unsigned int n) {
  std::vector<float> input(n);
  for (unsigned int i = 0; i < n; ++i) {
    input[i] = static_cast<float>((i * 7 + 3) % DIGITS);
  }
  return input;
}

template <class Kernel>
static float runKernel(Kernel kernel, float* input_d, float* output_d, unsigned int n) {
  CHECK(cudaMemset(output_d, 0, sizeof(float)));
  launch(kernel, dim3(cdiv(n, SEGMENT_LENGTH)), dim3(BLOCK_DIM), input_d, output_d, n);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

static bool reduceLength(unsigned int n) {
  const std::vector<float> input = makeInput(n);
  std::printf("N = %u: cdiv(%u, %u) = %u block%s\n",
              n,
              n,
              SEGMENT_LENGTH,
              cdiv(n, SEGMENT_LENGTH),
              cdiv(n, SEGMENT_LENGTH) == 1 ? "" : "s");
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, n * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const float printedSum = runKernel(CoarsenedSumReductionKernelPrinting, input_d, output_d, n);
  const float bookSum = runKernel(CoarsenedSumReductionKernel, input_d, output_d, n);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(input.data(), n);
  std::printf("  CoarsenedSumReductionKernel: %g, printing copy: %g, sequential: %g\n",
              bookSum,
              printedSum,
              expected);
  return bookSum == expected && printedSum == expected;
}

int main() {
  std::printf("BLOCK_DIM = %u threads, COARSE_FACTOR = %u: each block reduces up to %u elements\n",
              BLOCK_DIM,
              COARSE_FACTOR,
              SEGMENT_LENGTH);
  std::printf("input[i] = (7 i + 3) %% %u, and exactly N elements are allocated\n", DIGITS);

  bool ok = true;
  for (unsigned int n : LENGTHS) {
    ok = reduceLength(n) && ok;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
