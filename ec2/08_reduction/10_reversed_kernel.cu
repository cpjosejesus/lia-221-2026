#include "cuda_compat.h"

#include <cstdio>

constexpr unsigned int N = 16;
constexpr unsigned int THREADS = N / 2;
const float INPUT[N] = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 8, 1, 9, 2, 4, 6};

__global__ void ReversedSumReductionKernel(float* input, float* output) {
  unsigned int i = blockDim.x + threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x >= blockDim.x - stride) {
      input[i] += input[i - stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == blockDim.x - 1) {
    *output = input[i];
  }
}

__device__ void printArray(const char* label,
                           unsigned int stride,
                           const float* values,
                           unsigned int n) {
  printf("%s %2u:", label, stride);
  for (unsigned int k = 0; k < n; ++k) {
    printf(" %3.0f", values[k]);
  }
  printf("\n");
}

__global__ void ReversedSumReductionKernelPrinting(float* input, float* output) {
  unsigned int i = blockDim.x + threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x >= blockDim.x - stride) {
      input[i] += input[i - stride];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      printArray("after stride", stride, input, 2 * blockDim.x);
    }
    __syncthreads();
  }
  if (threadIdx.x == blockDim.x - 1) {
    *output = input[i];
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
  CHECK(cudaMemcpy(input_d, INPUT, N * sizeof(float), cudaMemcpyHostToDevice));
  launch(kernel, dim3(1), dim3(THREADS), input_d, output_d);
  CHECK(cudaGetLastError());

  float output = 0.0f;
  CHECK(cudaMemcpy(&output, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

int main() {
  std::printf(
      "one block of %u threads, %u elements; thread t owns input[%u + t]\n", THREADS, N, THREADS);
  std::printf("%-15s", "input");
  for (unsigned int k = 0; k < N; ++k) {
    std::printf(" %3.0f", INPUT[k]);
  }
  std::printf("\n");
  std::fflush(stdout);

  float* input_d = nullptr;
  float* output_d = nullptr;
  CHECK(cudaMalloc(&input_d, N * sizeof(float)));
  CHECK(cudaMalloc(&output_d, sizeof(float)));

  const float printedSum = runKernel(ReversedSumReductionKernelPrinting, input_d, output_d);
  const float bookSum = runKernel(ReversedSumReductionKernel, input_d, output_d);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(output_d));

  const float expected = sum_sequential(INPUT, N);
  std::printf("ReversedSumReductionKernel: %g, printing copy: %g, sequential: %g\n",
              bookSum,
              printedSum,
              expected);

  const bool ok = bookSum == expected && printedSum == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
