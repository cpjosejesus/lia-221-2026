#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int BLOCK_SIZE = 8;
const int ITERATIONS = 3;
const float STEP = 10.0f;

__device__ float f(float v) {
  return v + STEP;
}

__global__ void oneBufferKernel(const float* in, float* out) {
  __shared__ float buffer[BLOCK_SIZE];
  unsigned myThreadID = threadIdx.x;
  unsigned anotherThreadID = (threadIdx.x + 1) % blockDim.x;

  buffer[myThreadID] = in[myThreadID];
  __syncthreads();

  for (unsigned i = 0; i < ITERATIONS; ++i) {
    float v = buffer[anotherThreadID];
    __syncthreads();
    buffer[myThreadID] = f(v);
    __syncthreads();
  }
  out[myThreadID] = buffer[myThreadID];
}

__global__ void twoBufferKernel(const float* in, float* out) {
  __shared__ float bufferA[BLOCK_SIZE];
  __shared__ float bufferB[BLOCK_SIZE];
  float* inBuffer = bufferA;
  float* outBuffer = bufferB;
  unsigned myThreadID = threadIdx.x;
  unsigned anotherThreadID = (threadIdx.x + 1) % blockDim.x;

  inBuffer[myThreadID] = in[myThreadID];
  __syncthreads();

  for (unsigned i = 0; i < ITERATIONS; ++i) {
    float v = inBuffer[anotherThreadID];
    outBuffer[myThreadID] = f(v);
    __syncthreads();

    float* swap = inBuffer;
    inBuffer = outBuffer;
    outBuffer = swap;
  }
  out[myThreadID] = inBuffer[myThreadID];
}

template <class Kernel>
static std::vector<float> runOnDevice(Kernel kernel, const float* in_d) {
  std::vector<float> out(BLOCK_SIZE);
  float* out_d;
  CHECK(cudaMalloc(&out_d, BLOCK_SIZE * sizeof(float)));

  launch(kernel, dim3(1), dim3(BLOCK_SIZE), in_d, out_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(out.data(), out_d, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(out_d));
  return out;
}

static std::vector<float> runOnCpu(const std::vector<float>& in) {
  std::vector<float> values(in);
  for (int i = 0; i < ITERATIONS; ++i) {
    std::vector<float> next(BLOCK_SIZE);
    for (int t = 0; t < BLOCK_SIZE; ++t) {
      next[t] = values[(t + 1) % BLOCK_SIZE] + STEP;
    }
    values = next;
  }
  return values;
}

static void printVector(const char* name, const std::vector<float>& values, int barriers) {
  std::printf("%-12s =", name);
  for (float value : values) {
    std::printf("%4.0f", value);
  }
  if (barriers > 0) {
    std::printf("   %d barriers in the loop", barriers);
  }
  std::printf("\n");
}

int main() {
  std::vector<float> in(BLOCK_SIZE);
  for (int t = 0; t < BLOCK_SIZE; ++t) {
    in[t] = static_cast<float>(t);
  }

  float* in_d;
  CHECK(cudaMalloc(&in_d, BLOCK_SIZE * sizeof(float)));
  CHECK(cudaMemcpy(in_d, in.data(), BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  const std::vector<float> oneBuffer = runOnDevice(oneBufferKernel, in_d);
  const std::vector<float> twoBuffers = runOnDevice(twoBufferKernel, in_d);
  CHECK(cudaFree(in_d));

  const std::vector<float> expected = runOnCpu(in);
  printVector("in", in, 0);
  printVector("one buffer", oneBuffer, 2 * ITERATIONS);
  printVector("two buffers", twoBuffers, ITERATIONS);

  const bool pass = oneBuffer == expected && twoBuffers == expected;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
