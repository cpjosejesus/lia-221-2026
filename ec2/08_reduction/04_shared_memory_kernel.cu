#include "cuda_compat.h"

#include <cstdio>

constexpr unsigned int BLOCK_DIM = 8;
constexpr unsigned int N = 2 * BLOCK_DIM;
const float INPUT[N] = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 8, 1, 9, 2, 4, 6};

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

__device__ void printValues(const float* values, unsigned int n) {
  for (unsigned int k = 0; k < n; ++k) {
    printf(" %3.0f", values[k]);
  }
  printf("\n");
}

__global__ void SharedMemorySumReductionKernelPrinting(float* input, float* output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int t = threadIdx.x;
  input_s[t] = input[t] + input[t + BLOCK_DIM];
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("input_s, loaded:  ");
    printValues(input_s, BLOCK_DIM);
  }

  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[t] += input_s[t + stride];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      printf("after stride %2u:  ", stride);
      printValues(input_s, BLOCK_DIM);
    }
  }
  if (threadIdx.x == 0) {
    *output = input_s[0];
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
  launch(kernel, dim3(1), dim3(BLOCK_DIM), input_d, output_d);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

int main() {
  std::printf(
      "one block of BLOCK_DIM = %u threads, %u elements; thread t owns input_s[t]\n", BLOCK_DIM, N);
  std::printf("%-18s", "input");
  for (unsigned int k = 0; k < N; ++k) {
    std::printf(" %3.0f", INPUT[k]);
  }
  std::printf("\n");
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, N * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));
  CHECK(cudaMemcpy(input_d, INPUT, N * sizeof(float), cudaMemcpyHostToDevice));

  const float printedSum = runKernel(SharedMemorySumReductionKernelPrinting, input_d, output_d);
  const float bookSum = runKernel(SharedMemorySumReductionKernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(INPUT, N);
  std::printf("SharedMemorySumReductionKernel: %g, printing copy: %g, sequential: %g\n",
              bookSum,
              printedSum,
              expected);

  const bool ok = bookSum == expected && printedSum == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
