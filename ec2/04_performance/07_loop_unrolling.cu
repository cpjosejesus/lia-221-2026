#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int ELEMENT_COUNT = 16;
const int COARSE_FACTOR = 4;
const int BLOCK_SIZE = 4;

__global__ void chunkScanKernel(const float* in, float* out, int n) {
  unsigned iStart = COARSE_FACTOR * (blockIdx.x * blockDim.x + threadIdx.x);
  if (iStart + COARSE_FACTOR > static_cast<unsigned>(n)) {
    return;
  }

  float x[COARSE_FACTOR];
#pragma unroll
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    x[c] = in[iStart + c];
  }

#pragma unroll
  for (unsigned c = 1; c < COARSE_FACTOR; ++c) {
    x[c] += x[c - 1];
  }

#pragma unroll
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    out[iStart + c] = x[c];
  }
}

__global__ void chunkScanNotUnrolledKernel(const float* in, float* out, int n) {
  unsigned iStart = COARSE_FACTOR * (blockIdx.x * blockDim.x + threadIdx.x);
  if (iStart + COARSE_FACTOR > static_cast<unsigned>(n)) {
    return;
  }

  float x[COARSE_FACTOR];
#pragma unroll 1
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    x[c] = in[iStart + c];
  }

#pragma unroll 1
  for (unsigned c = 1; c < COARSE_FACTOR; ++c) {
    x[c] += x[c - 1];
  }

#pragma unroll 1
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    out[iStart + c] = x[c];
  }
}

template <class Kernel>
static std::vector<float> scanChunksOnDevice(Kernel kernel, const float* in_d) {
  std::vector<float> out(ELEMENT_COUNT);
  float* out_d;
  CHECK(cudaMalloc(&out_d, ELEMENT_COUNT * sizeof(float)));

  const int threads = ELEMENT_COUNT / COARSE_FACTOR;
  launch(kernel, dim3(cdiv(threads, BLOCK_SIZE)), dim3(BLOCK_SIZE), in_d, out_d, ELEMENT_COUNT);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(out.data(), out_d, ELEMENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(out_d));
  return out;
}

static std::vector<float> scanChunksOnCpu(const std::vector<float>& in) {
  std::vector<float> out(in);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    if (i % COARSE_FACTOR != 0) {
      out[i] += out[i - 1];
    }
  }
  return out;
}

static void printVector(const char* name, const std::vector<float>& values) {
  std::printf("%-13s =", name);
  for (float value : values) {
    std::printf("%4.0f", value);
  }
  std::printf("\n");
}

int main() {
  std::vector<float> in(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    in[i] = static_cast<float>(1 + i % 3);
  }

  float* in_d;
  CHECK(cudaMalloc(&in_d, ELEMENT_COUNT * sizeof(float)));
  CHECK(cudaMemcpy(in_d, in.data(), ELEMENT_COUNT * sizeof(float), cudaMemcpyHostToDevice));
  const std::vector<float> unrolled = scanChunksOnDevice(chunkScanKernel, in_d);
  const std::vector<float> notUnrolled = scanChunksOnDevice(chunkScanNotUnrolledKernel, in_d);
  CHECK(cudaFree(in_d));

  const std::vector<float> expected = scanChunksOnCpu(in);
  printVector("in", in);
  printVector("unrolled", unrolled);
  printVector("not unrolled", notUnrolled);

  const bool pass = unrolled == expected && notUnrolled == expected;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
