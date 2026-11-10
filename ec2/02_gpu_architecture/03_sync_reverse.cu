#include "cuda_compat.h"

#include <cstdio>
#include <vector>

#define BLOCK 8

const int CHUNKS = 2;
const int N = CHUNKS * BLOCK;

__global__ void reverseChunks(int* d) {
  __shared__ int s[BLOCK];
  int t = threadIdx.x;
  int base = blockIdx.x * BLOCK;

  s[t] = d[base + t];
  __syncthreads();
  d[base + t] = s[BLOCK - 1 - t];
}

static void reverseOnDevice(std::vector<int>& data) {
  const size_t bytes = data.size() * sizeof(int);

  int* d_d;
  CHECK(cudaMalloc(&d_d, bytes));
  CHECK(cudaMemcpy(d_d, data.data(), bytes, cudaMemcpyHostToDevice));

  launch(reverseChunks, dim3(N / BLOCK), dim3(BLOCK), d_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(data.data(), d_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_d));
}

static std::vector<int> reverseOnCpu(const std::vector<int>& data) {
  std::vector<int> reversed(data.size());
  for (int chunk = 0; chunk < CHUNKS; ++chunk) {
    const int base = chunk * BLOCK;
    for (int t = 0; t < BLOCK; ++t) {
      reversed[base + t] = data[base + BLOCK - 1 - t];
    }
  }
  return reversed;
}

static void printData(const char* label, const std::vector<int>& data) {
  std::printf("%-8s", label);
  for (int i = 0; i < N; ++i) {
    if (i > 0 && i % BLOCK == 0) {
      std::printf(" |");
    }
    std::printf(" %3d", data[i]);
  }
  std::printf("\n");
}

int main() {
  std::vector<int> data(N);
  for (int i = 0; i < N; ++i) {
    data[i] = 10 * i;
  }
  const std::vector<int> expected = reverseOnCpu(data);

  std::printf("%d blocks of %d threads, each block reverses its own chunk\n", N / BLOCK, BLOCK);
  printData("before:", data);
  reverseOnDevice(data);
  printData("after:", data);

  const bool ok = data == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
