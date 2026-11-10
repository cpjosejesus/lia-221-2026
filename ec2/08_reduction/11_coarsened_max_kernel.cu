#include "cuda_compat.h"

#include <cmath>
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
const float MAX_IDENTITY = -INFINITY;

__device__ unsigned int warpIdx() {
  return threadIdx.x / WARP_SIZE;
}

__device__ unsigned int laneIdx() {
  return threadIdx.x % WARP_SIZE;
}

__device__ float warp_max(float val) {
  float partialMax = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialMax = fmaxf(partialMax, __shfl_down_sync(FULL_WARP_MASK, partialMax, stride));
  }
  return partialMax;
}

__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
  unsigned int i = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x + threadIdx.x;
  float partialMax = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialMax = fmaxf(partialMax, input[i + c * BLOCK_DIM]);
  }

  partialMax = warp_max(partialMax);
  __shared__ float partialMaxs_s[NUM_WARPS];
  if (laneIdx() == 0) {
    partialMaxs_s[warpIdx()] = partialMax;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    partialMax = warp_max(threadIdx.x < NUM_WARPS ? partialMaxs_s[threadIdx.x] : -INFINITY);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      float old = output_ref.load(cuda::memory_order_relaxed);
      while (old < partialMax &&
             !output_ref.compare_exchange_strong(old, partialMax, cuda::memory_order_relaxed)) {
      }
    }
  }
}

__global__ void CoarsenedMaxReductionKernelPrinting(float* input, float* output) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialMax = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialMax = fmaxf(partialMax, input[i + c * BLOCK_DIM]);
  }

  partialMax = warp_max(partialMax);
  __shared__ float partialMaxs_s[NUM_WARPS];
  if (laneIdx() == 0) {
    partialMaxs_s[warpIdx()] = partialMax;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    partialMax = warp_max(threadIdx.x < NUM_WARPS ? partialMaxs_s[threadIdx.x] : -INFINITY);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      float old = output_ref.load(cuda::memory_order_relaxed);
      while (old < partialMax &&
             !output_ref.compare_exchange_strong(old, partialMax, cuda::memory_order_relaxed)) {
      }
      printf("block %u (elements %3u..%3u) max %4g; output was %4g, %s\n",
             blockIdx.x,
             segment,
             segment + COARSE_FACTOR * 2 * BLOCK_DIM - 1,
             partialMax,
             old,
             old < partialMax ? "replaced by compare_exchange" : "kept");
    }
  }
}

float max_sequential(const float* input, unsigned int n) {
  float acc = MAX_IDENTITY;
  for (unsigned int i = 0; i < n; ++i) {
    acc = std::fmax(acc, input[i]);
  }
  return acc;
}

static std::vector<float> makeInput() {
  std::vector<float> input(INPUT_LENGTH);
  for (unsigned int i = 0; i < INPUT_LENGTH; ++i) {
    input[i] = static_cast<float>((i * 37 + 11) % 257) - 300.0f;
  }
  return input;
}

static void printInput(const std::vector<float>& input) {
  std::printf("input (%u elements, all negative):\n", INPUT_LENGTH);
  for (unsigned int i = 0; i < INPUT_LENGTH; ++i) {
    std::printf(" %4.0f", input[i]);
    if ((i + 1) % VALUES_PER_ROW == 0) {
      std::printf("\n");
    }
  }
}

template <class Kernel>
static float runKernel(Kernel kernel, float* input_d, float* output_d) {
  CHECK(cudaMemcpy(output_d, &MAX_IDENTITY, sizeof(float), cudaMemcpyHostToDevice));
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
  std::printf("the host sets *output = %g, the identity of max\n", MAX_IDENTITY);
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, INPUT_LENGTH * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, input.data(), INPUT_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

  const float printedMax = runKernel(CoarsenedMaxReductionKernelPrinting, input_d, output_d);
  const float bookMax = runKernel(CoarsenedMaxReductionKernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = max_sequential(input.data(), INPUT_LENGTH);
  std::printf("CoarsenedMaxReductionKernel: %g, printing copy: %g, sequential: %g\n",
              bookMax,
              printedMax,
              expected);

  const bool ok = bookMax == expected && printedMax == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
