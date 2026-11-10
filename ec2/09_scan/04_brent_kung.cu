#include <cstdio>
#include <numeric>
#include <vector>

#include "cuda_compat.h"

constexpr int BK_THREADS = 4;
constexpr int BK_SECTION = 2 * BK_THREADS;

__device__ void print_after_step(const char* sweep, int stride, const int* XY) {
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("  %s %d ", sweep, stride);
    for (int j = 0; j < BK_SECTION; ++j) {
      printf("%4d", XY[j]);
    }
    printf("\n");
  }
}

__global__ void brent_kung_scan(const int* X, int* Y, int n) {
  __shared__ int XY[BK_SECTION];
  const int t = threadIdx.x;
  const int base = blockIdx.x * BK_SECTION;
  XY[t] = (base + t < n) ? X[base + t] : 0;
  XY[t + BK_THREADS] = (base + t + BK_THREADS < n) ? X[base + t + BK_THREADS] : 0;

  for (int stride = 1; stride <= BK_THREADS; stride *= 2) {
    __syncthreads();
    const int index = (t + 1) * 2 * stride - 1;
    if (index < BK_SECTION) {
      XY[index] += XY[index - stride];
    }
    print_after_step("up   stride", stride, XY);
  }

  for (int stride = BK_SECTION / 4; stride > 0; stride /= 2) {
    __syncthreads();
    const int index = (t + 1) * 2 * stride - 1;
    if (index + stride < BK_SECTION) {
      XY[index + stride] += XY[index];
    }
    print_after_step("down stride", stride, XY);
  }

  __syncthreads();
  if (base + t < n) {
    Y[base + t] = XY[t];
  }
  if (base + t + BK_THREADS < n) {
    Y[base + t + BK_THREADS] = XY[t + BK_THREADS];
  }
}

void printRow(const char* label, const std::vector<int>& values) {
  std::printf("  %-14s", label);
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
  launch(brent_kung_scan, dim3(1), dim3(BK_THREADS), d_x, d_y, n);
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
