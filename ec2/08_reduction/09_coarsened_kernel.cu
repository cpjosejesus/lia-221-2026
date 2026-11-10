#include "cuda_compat.h"

#include <cstdio>
#include <vector>

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int FULL_WARP_MASK = 0xffffffff;
constexpr unsigned int BLOCK_DIM = 32;
constexpr unsigned int NUM_WARPS = BLOCK_DIM / WARP_SIZE;
constexpr unsigned int COARSE_FACTOR = 2;
constexpr unsigned int NUM_BLOCKS = 2;
constexpr unsigned int INPUT_LENGTH = NUM_BLOCKS * COARSE_FACTOR * 2 * BLOCK_DIM;
constexpr unsigned int VALUES_PER_ROW = 16;
constexpr unsigned int DIGITS = 10;

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

__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialSum += input[i + c * BLOCK_DIM];
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

__global__ void CoarsenedSumReductionKernelPrinting(float* input, float* output) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialSum += input[i + c * BLOCK_DIM];
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
      float before = output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
      printf("block %u (elements %3u..%3u) adds %4g; fetch_add returned %4g\n",
             blockIdx.x,
             segment,
             segment + COARSE_FACTOR * 2 * BLOCK_DIM - 1,
             partialSum,
             before);
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

static std::vector<float> makeInput() {
  std::vector<float> input(INPUT_LENGTH);
  for (unsigned int i = 0; i < INPUT_LENGTH; ++i) {
    input[i] = static_cast<float>((i * 7 + 3) % DIGITS);
  }
  return input;
}

static void printInput(const std::vector<float>& input) {
  std::printf("input (%u elements):\n", INPUT_LENGTH);
  for (unsigned int i = 0; i < INPUT_LENGTH; ++i) {
    std::printf(" %3.0f", input[i]);
    if ((i + 1) % VALUES_PER_ROW == 0) {
      std::printf("\n");
    }
  }
}

template <class Kernel>
static float runKernel(Kernel kernel, float* input_d, float* output_d) {
  CHECK(cudaMemset(output_d, 0, sizeof(float)));
  launch(kernel, dim3(NUM_BLOCKS), dim3(BLOCK_DIM), input_d, output_d);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

int main() {
  const std::vector<float> input = makeInput();
  std::printf("%u blocks of BLOCK_DIM = %u threads, COARSE_FACTOR = %u: each reduces %u elements\n",
              NUM_BLOCKS,
              BLOCK_DIM,
              COARSE_FACTOR,
              COARSE_FACTOR * 2 * BLOCK_DIM);
  printInput(input);
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, INPUT_LENGTH * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, input.data(), INPUT_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

  const float printedSum = runKernel(CoarsenedSumReductionKernelPrinting, input_d, output_d);
  const float bookSum = runKernel(CoarsenedSumReductionKernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(input.data(), INPUT_LENGTH);
  std::printf("CoarsenedSumReductionKernel: %g, printing copy: %g, sequential: %g\n",
              bookSum,
              printedSum,
              expected);

  const bool ok = bookSum == expected && printedSum == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
