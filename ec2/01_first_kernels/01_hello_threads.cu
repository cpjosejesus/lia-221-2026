#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int BLOCKS = 2;
const int THREADS_PER_BLOCK = 4;
const int TOTAL_THREADS = BLOCKS * THREADS_PER_BLOCK;

__global__ void helloKernel(int* globalIndexOf) {
  int block = blockIdx.x;
  int thread = threadIdx.x;
  int i = block * blockDim.x + thread;

  printf("hello from block %d, thread %d: global index %d\n", block, thread, i);
  globalIndexOf[i] = i;
}

static std::vector<int> runHello() {
  const size_t bytes = TOTAL_THREADS * sizeof(int);
  std::vector<int> globalIndexOf(TOTAL_THREADS, -1);

  int* globalIndexOf_d;
  cudaMalloc(&globalIndexOf_d, bytes);

  launch(helloKernel, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), globalIndexOf_d);

  cudaMemcpy(globalIndexOf.data(), globalIndexOf_d, bytes, cudaMemcpyDeviceToHost);
  cudaFree(globalIndexOf_d);
  return globalIndexOf;
}

static bool everyThreadWroteItsIndex(const std::vector<int>& globalIndexOf) {
  for (int i = 0; i < TOTAL_THREADS; ++i) {
    if (globalIndexOf[i] != i) {
      return false;
    }
  }
  return true;
}

int main() {
  std::printf("backend: %s\n", CUDA_BACKEND);
  std::printf(
      "grid: %d blocks of %d threads = %d threads\n\n", BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);

  const std::vector<int> globalIndexOf = runHello();

  std::printf("\nglobal index written by each thread:");
  for (int index : globalIndexOf) {
    std::printf(" %d", index);
  }
  std::printf("\n");

  const bool ok = everyThreadWroteItsIndex(globalIndexOf);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
