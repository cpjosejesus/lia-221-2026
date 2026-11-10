#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int ELEMENT_COUNT = 16;
const int COARSE_FACTOR = 4;
const int BLOCK_SIZE = 4;
const float SCALE = 10.0f;

__device__ void scaleOneUnit(unsigned i, const float* in, float* out, int* owner, int n) {
  if (i < static_cast<unsigned>(n)) {
    out[i] = SCALE * in[i];
    owner[i] = blockIdx.x * blockDim.x + threadIdx.x;
  }
}

__global__ void scaleKernel(const float* in, float* out, int* owner, int n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  scaleOneUnit(i, in, out, owner, n);
}

__global__ void scaleCoarsenedKernel(const float* in, float* out, int* owner, int n) {
  unsigned iStart = COARSE_FACTOR * (blockIdx.x * blockDim.x + threadIdx.x);
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    unsigned i = iStart + c;
    scaleOneUnit(i, in, out, owner, n);
  }
}

struct Result {
  std::vector<float> out;
  std::vector<int> owner;
};

template <class Kernel>
static Result scaleOnDevice(Kernel kernel, const float* in_d, int threads) {
  Result result{std::vector<float>(ELEMENT_COUNT), std::vector<int>(ELEMENT_COUNT)};
  float* out_d;
  int* owner_d;
  CHECK(cudaMalloc(&out_d, ELEMENT_COUNT * sizeof(float)));
  CHECK(cudaMalloc(&owner_d, ELEMENT_COUNT * sizeof(int)));

  launch(kernel,
         dim3(cdiv(threads, BLOCK_SIZE)),
         dim3(BLOCK_SIZE),
         in_d,
         out_d,
         owner_d,
         ELEMENT_COUNT);
  CHECK(cudaGetLastError());

  CHECK(
      cudaMemcpy(result.out.data(), out_d, ELEMENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(
      result.owner.data(), owner_d, ELEMENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(out_d));
  CHECK(cudaFree(owner_d));
  return result;
}

static void printResult(const char* name, int threads, const Result& result) {
  std::printf("%s, %d threads\n  out    =", name, threads);
  for (float value : result.out) {
    std::printf("%4.0f", value);
  }
  std::printf("\n  thread =");
  for (int thread : result.owner) {
    std::printf("%4d", thread);
  }
  std::printf("\n");
}

static bool scaledCorrectly(const std::vector<float>& in, const Result& result) {
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    if (result.out[i] != SCALE * in[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  std::vector<float> in(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    in[i] = static_cast<float>(i % 5);
  }

  float* in_d;
  CHECK(cudaMalloc(&in_d, ELEMENT_COUNT * sizeof(float)));
  CHECK(cudaMemcpy(in_d, in.data(), ELEMENT_COUNT * sizeof(float), cudaMemcpyHostToDevice));

  const int fineThreads = ELEMENT_COUNT;
  const int coarseThreads = cdiv(ELEMENT_COUNT, COARSE_FACTOR);
  const Result fine = scaleOnDevice(scaleKernel, in_d, fineThreads);
  const Result coarse = scaleOnDevice(scaleCoarsenedKernel, in_d, coarseThreads);
  CHECK(cudaFree(in_d));

  std::printf("input\n  in     =");
  for (float value : in) {
    std::printf("%4.0f", value);
  }
  std::printf("\n");
  printResult("one unit per thread", fineThreads, fine);
  printResult("coarsening factor 4", coarseThreads, coarse);

  const bool pass = scaledCorrectly(in, fine) && scaledCorrectly(in, coarse);
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
