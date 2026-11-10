#include <cstdio>
#include <numeric>
#include <vector>

#include "cuda_compat.h"

constexpr int SECTION_SIZE = 4;

__global__ void kogge_stone_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int XY[SECTION_SIZE];
  const int t = threadIdx.x;
  const int i = blockIdx.x * SECTION_SIZE + t;
  XY[t] = (i < n) ? X[i] : 0;

  for (int stride = 1; stride < SECTION_SIZE; stride *= 2) {
    __syncthreads();
    int temp = XY[t];
    if (t >= stride) {
      temp += XY[t - stride];
    }
    __syncthreads();
    XY[t] = temp;
  }

  if (i < n) {
    Y[i] = XY[t];
  }
  if (t == SECTION_SIZE - 1) {
    sums[blockIdx.x] = XY[t];
  }
}

__global__ void add_block_offsets(int* Y, const int* scannedSums, int n) {
  const int i = blockIdx.x * SECTION_SIZE + threadIdx.x;
  if (blockIdx.x > 0 && i < n) {
    Y[i] += scannedSums[blockIdx.x - 1];
  }
}

int* deviceInts(int count) {
  int* device = nullptr;
  CHECK(cudaMalloc(&device, count * sizeof(int)));
  return device;
}

std::vector<int> hostCopyOf(const int* device, int count) {
  std::vector<int> host(count);
  CHECK(cudaMemcpy(host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost));
  return host;
}

void printRow(int level, const char* label, const std::vector<int>& values) {
  std::printf("  level %d  %-12s", level, label);
  for (int value : values) {
    std::printf("%4d", value);
  }
  std::printf("\n");
}

void printDevice(int level, const char* label, const int* device, int count) {
  printRow(level, label, hostCopyOf(device, count));
}

void scanAnyLength(const int* d_x, int* d_y, int n, int level) {
  const int blocks = static_cast<int>(cdiv(n, SECTION_SIZE));
  int* d_sums = deviceInts(blocks);
  launch(kogge_stone_scan, dim3(blocks), dim3(SECTION_SIZE), d_x, d_y, d_sums, n);
  printDevice(level, "local scans", d_y, n);

  if (blocks > 1) {
    printDevice(level, "block sums", d_sums, blocks);
    int* d_scannedSums = deviceInts(blocks);
    scanAnyLength(d_sums, d_scannedSums, blocks, level + 1);

    launch(add_block_offsets, dim3(blocks), dim3(SECTION_SIZE), d_y, d_scannedSums, n);
    printDevice(level, "add offsets", d_y, n);
    CHECK(cudaFree(d_scannedSums));
  }
  CHECK(cudaFree(d_sums));
}

int main() {
  const std::vector<int> x = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 0, 8, 1, 1, 9, 2, 4, 6, 1};
  const int n = static_cast<int>(x.size());

  int* d_x = deviceInts(n);
  int* d_y = deviceInts(n);
  CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  printRow(0, "x", x);
  scanAnyLength(d_x, d_y, n, 0);
  CHECK(cudaGetLastError());
  const std::vector<int> y = hostCopyOf(d_y, n);

  std::vector<int> expected(n);
  std::inclusive_scan(x.begin(), x.end(), expected.begin());
  const bool pass = (y == expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_y));
  return pass ? 0 : 1;
}
