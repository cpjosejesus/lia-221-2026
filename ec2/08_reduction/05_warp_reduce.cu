#include "cuda_compat.h"

#include <cstdio>

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int FULL_WARP_MASK = 0xffffffff;
const float INPUT[WARP_SIZE] = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 8, 1, 9, 2, 4, 6,
                                5, 2, 0, 7, 3, 8, 1, 4, 6, 2, 9, 3, 0, 5, 7, 1};

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

__device__ void printLanes(unsigned int stride, float partialSum) {
  __shared__ float lanes_s[WARP_SIZE];
  lanes_s[laneIdx()] = partialSum;
  __syncwarp();
  if (laneIdx() == 0) {
    printf("after stride %2u:", stride);
    for (unsigned int lane = 0; lane < WARP_SIZE; ++lane) {
      printf(" %3.0f", lanes_s[lane]);
    }
    printf("\n");
  }
  __syncwarp();
}

__device__ float warp_reduce_printing(float val) {
  float partialSum = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialSum += __shfl_down_sync(FULL_WARP_MASK, partialSum, stride);
    printLanes(stride, partialSum);
  }
  return partialSum;
}

__global__ void WarpReduceKernel(float* input, float* output) {
  float partialSum = warp_reduce(input[threadIdx.x]);
  if (threadIdx.x == 0) {
    *output = partialSum;
  }
}

__global__ void WarpReduceKernelPrinting(float* input, float* output) {
  float partialSum = warp_reduce_printing(input[threadIdx.x]);
  if (threadIdx.x == 0) {
    *output = partialSum;
  }
}

float sum_sequential(const float* input, unsigned int n) {
  float sum = 0.0f;
  for (unsigned int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}

template <class Kernel>
static float runKernel(Kernel kernel, float* input_d, float* output_d) {
  launch(kernel, dim3(1), dim3(WARP_SIZE), input_d, output_d);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

int main() {
  std::printf(
      "one warp of %u lanes, one value per lane; lane i adds the value of lane i + stride\n",
      WARP_SIZE);
  std::printf("%-16s", "lane values");
  for (unsigned int lane = 0; lane < WARP_SIZE; ++lane) {
    std::printf(" %3.0f", INPUT[lane]);
  }
  std::printf("\n");
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, WARP_SIZE * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, INPUT, WARP_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  const float printedSum = runKernel(WarpReduceKernelPrinting, input_d, output_d);
  const float bookSum = runKernel(WarpReduceKernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(INPUT, WARP_SIZE);
  std::printf("warp_reduce: %g in lane 0, printing copy: %g, sequential: %g\n",
              bookSum,
              printedSum,
              expected);

  const bool ok = bookSum == expected && printedSum == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
