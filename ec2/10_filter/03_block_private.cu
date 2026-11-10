#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "cuda_compat.h"

constexpr int THRESHOLD = 100;
constexpr int BLOCK_SIZE = 4;

__host__ __device__ inline bool keep(int v, int threshold) {
  return v < threshold;
}

__global__ void filter_block_private(const int* in, int* out, int* counter, int n, int threshold) {
  __shared__ int blockCount;
  __shared__ int blockStart;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  const int v = (i < n) ? in[i] : 0;
  const bool k = (i < n) && keep(v, threshold);
  int local = 0;
  if (k) {
    local = atomicAdd(&blockCount, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    blockStart = atomicAdd(counter, blockCount);
    printf("  block %d keeps %d, reserves out[%d..%d)\n",
           blockIdx.x,
           blockCount,
           blockStart,
           blockStart + blockCount);
  }
  __syncthreads();

  if (k) {
    out[blockStart + local] = v;
  }
}

void printRow(const char* label, const std::vector<int>& values) {
  std::printf("  %-6s", label);
  for (int value : values) {
    std::printf("%5d", value);
  }
  std::printf("\n");
}

int* deviceCopyOf(const std::vector<int>& host) {
  int* device = nullptr;
  CHECK(cudaMalloc(&device, host.size() * sizeof(int)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
  return device;
}

std::vector<int> hostCopyOf(const int* device, int count) {
  std::vector<int> host(count);
  CHECK(cudaMemcpy(host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost));
  return host;
}

std::vector<int> cpuFilter(const std::vector<int>& in) {
  std::vector<int> out;
  std::copy_if(in.begin(), in.end(), std::back_inserter(out), [](int v) {
    return keep(v, THRESHOLD);
  });
  return out;
}

bool sameElements(std::vector<int> a, std::vector<int> b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

int main() {
  const std::vector<int> in = {
      812, 45, 30, 17, 999, 60, 431, 8, 77, 650, 230, 905, 140, 91, 3, 520};
  const int n = static_cast<int>(in.size());
  const int blocks = static_cast<int>(cdiv(n, BLOCK_SIZE));

  int* d_in = deviceCopyOf(in);
  int* d_out = deviceCopyOf(std::vector<int>(n, -1));
  int* d_counter = deviceCopyOf({0});

  std::printf("keep v < %d, %d blocks of %d threads\n", THRESHOLD, blocks, BLOCK_SIZE);
  printRow("in", in);
  launch(
      filter_block_private, dim3(blocks), dim3(BLOCK_SIZE), d_in, d_out, d_counter, n, THRESHOLD);
  CHECK(cudaGetLastError());
  const int count = hostCopyOf(d_counter, 1)[0];
  const std::vector<int> out = hostCopyOf(d_out, count);
  const std::vector<int> expected = cpuFilter(in);

  printRow("out", out);
  printRow("seq", expected);
  std::printf("  same order as the sequential filter: %s\n", out == expected ? "yes" : "no");

  const bool pass = sameElements(out, expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_counter));
  return pass ? 0 : 1;
}
