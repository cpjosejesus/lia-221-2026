#include <cstdio>
#include <numeric>
#include <vector>

#include "cuda_compat.h"

constexpr int SECTION_SIZE = 8;

__device__ void print_after_stride(int stride, const int* XY) {
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("  stride %-3d", stride);
    for (int j = 0; j < SECTION_SIZE; ++j) {
      printf("%4d", XY[j]);
    }
    printf("\n");
  }
}

__global__ void kogge_stone_scan(const int* X, int* Y, int n) {
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
    print_after_stride(stride, XY);
  }

  if (i < n) {
    Y[i] = XY[t];
  }
}

void printRow(const char* label, const std::vector<int>& values) {
  std::printf("  %-10s", label);
  for (int value : values) {
    std::printf("%4d", value);
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

int main() {
  const std::vector<int> x = {3, 1, 7, 0, 4, 1, 6, 3};
  const int n = static_cast<int>(x.size());

  int* d_x = deviceCopyOf(x);
  int* d_y = deviceCopyOf(std::vector<int>(n));

  printRow("x", x);
  launch(kogge_stone_scan, dim3(1), dim3(SECTION_SIZE), d_x, d_y, n);
  CHECK(cudaGetLastError());
  const std::vector<int> y = hostCopyOf(d_y, n);
  printRow("inclusive", y);

  std::vector<int> expected(n);
  std::inclusive_scan(x.begin(), x.end(), expected.begin());
  const bool pass = (y == expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_y));
  return pass ? 0 : 1;
}
