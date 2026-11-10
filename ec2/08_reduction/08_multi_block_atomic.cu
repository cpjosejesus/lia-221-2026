#include "cuda_compat.h"

#include <cstdio>
#include <vector>

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int FULL_WARP_MASK = 0xffffffff;
constexpr unsigned int BLOCK_DIM = 32;
constexpr unsigned int NUM_WARPS = BLOCK_DIM / WARP_SIZE;
constexpr unsigned int NUM_BLOCKS = 4;
constexpr unsigned int INPUT_LENGTH = NUM_BLOCKS * 2 * BLOCK_DIM;
constexpr unsigned int VALUES_PER_ROW = 16;
constexpr unsigned int DIGITS = 10;

__device__ __inline__ float warp_reduce(float val) {
  float partialSum = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialSum += __shfl_down_sync(FULL_WARP_MASK, partialSum, stride);
  }
  return partialSum;
}

__global__ void reduce_kernel(float* input, float* output, unsigned int N) {
  (void)N;

  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i] + input[i + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[NUM_WARPS];
  if (threadIdx.x % WARP_SIZE == 0) {
    partialSums_s[threadIdx.x / WARP_SIZE] = partialSum;
  }
  __syncthreads();

  if (threadIdx.x < WARP_SIZE) {
    float partialSum = threadIdx.x < NUM_WARPS ? partialSums_s[threadIdx.x] : 0.0f;
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
    }
  }
}

__global__ void reduce_kernel_printing(float* input, float* output, unsigned int N) {
  (void)N;

  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i] + input[i + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[NUM_WARPS];
  if (threadIdx.x % WARP_SIZE == 0) {
    partialSums_s[threadIdx.x / WARP_SIZE] = partialSum;
  }
  __syncthreads();

  if (threadIdx.x < WARP_SIZE) {
    float partialSum = threadIdx.x < NUM_WARPS ? partialSums_s[threadIdx.x] : 0.0f;
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      float before = output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
      printf("block %u (elements %3u..%3u) adds %4g; fetch_add returned %4g\n",
             blockIdx.x,
             segment,
             segment + 2 * BLOCK_DIM - 1,
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
  launch(kernel, dim3(NUM_BLOCKS), dim3(BLOCK_DIM), input_d, output_d, INPUT_LENGTH);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

int main() {
  const std::vector<float> input = makeInput();
  std::printf("%u blocks of BLOCK_DIM = %u threads, each reduces a segment of %u elements\n",
              NUM_BLOCKS,
              BLOCK_DIM,
              2 * BLOCK_DIM);
  printInput(input);
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, INPUT_LENGTH * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, input.data(), INPUT_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

  const float printedSum = runKernel(reduce_kernel_printing, input_d, output_d);
  const float bookSum = runKernel(reduce_kernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(input.data(), INPUT_LENGTH);
  std::printf(
      "reduce_kernel: %g, printing copy: %g, sequential: %g\n", bookSum, printedSum, expected);

  const bool ok = bookSum == expected && printedSum == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
