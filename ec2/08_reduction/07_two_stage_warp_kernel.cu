#include "cuda_compat.h"

#include <cstdio>
#include <vector>

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int FULL_WARP_MASK = 0xffffffff;
constexpr unsigned int BLOCK_DIM = 128;
constexpr unsigned int NUM_WARPS = BLOCK_DIM / WARP_SIZE;
constexpr unsigned int N = 2 * BLOCK_DIM;
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

__global__ void TwoStageWarpLevelSumReductionKernel(float* input, float* output) {
  unsigned int t = threadIdx.x;
  float partialSum = input[t] + input[t + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[NUM_WARPS];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float partialSum = t < NUM_WARPS ? partialSums_s[t] : 0.0f;
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      *output = partialSum;
    }
  }
}

__global__ void TwoStageWarpLevelSumReductionKernelPrinting(float* input, float* output) {
  unsigned int t = threadIdx.x;
  float partialSum = input[t] + input[t + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[NUM_WARPS];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    printf("stage 1, partialSums_s (one sum per warp):");
    for (unsigned int warp = 0; warp < NUM_WARPS; ++warp) {
      printf(" %g", partialSums_s[warp]);
    }
    printf("\n");
  }
  if (warpIdx() == 0) {
    float partialSum = t < NUM_WARPS ? partialSums_s[t] : 0.0f;
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      printf("stage 2, warp 0 reduces them to %g\n", partialSum);
      *output = partialSum;
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
  std::vector<float> input(N);
  for (unsigned int i = 0; i < N; ++i) {
    input[i] = static_cast<float>((i * 7 + 3) % DIGITS);
  }
  return input;
}

static void printInput(const std::vector<float>& input) {
  std::printf("input (%u elements):\n", N);
  for (unsigned int i = 0; i < N; ++i) {
    std::printf(" %3.0f", input[i]);
    if ((i + 1) % VALUES_PER_ROW == 0) {
      std::printf("\n");
    }
  }
}

template <class Kernel>
static float runKernel(Kernel kernel, float* input_d, float* output_d) {
  launch(kernel, dim3(1), dim3(BLOCK_DIM), input_d, output_d);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

int main() {
  const std::vector<float> input = makeInput();
  std::printf("one block of BLOCK_DIM = %u threads (%u warps)\n", BLOCK_DIM, NUM_WARPS);
  printInput(input);
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, N * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  const float printedSum =
      runKernel(TwoStageWarpLevelSumReductionKernelPrinting, input_d, output_d);
  const float bookSum = runKernel(TwoStageWarpLevelSumReductionKernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(input.data(), N);
  std::printf("TwoStageWarpLevelSumReductionKernel: %g, printing copy: %g, sequential: %g\n",
              bookSum,
              printedSum,
              expected);

  const bool ok = bookSum == expected && printedSum == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
