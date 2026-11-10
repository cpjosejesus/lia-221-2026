#include <cstdio>
#include <numeric>
#include <vector>

#include "cuda_compat.h"

__global__ void sequential_inclusive_scan(const int* X, int* Y, int n) {
  int run = 0;
  for (int i = 0; i < n; ++i) {
    run += X[i];
    Y[i] = run;
  }
}

__global__ void sequential_exclusive_scan(const int* X, int* Y, int n) {
  int run = 0;
  for (int i = 0; i < n; ++i) {
    Y[i] = run;
    run += X[i];
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
  int* d_inclusive = deviceCopyOf(std::vector<int>(n));
  int* d_exclusive = deviceCopyOf(std::vector<int>(n));

  launch(sequential_inclusive_scan, dim3(1), dim3(1), d_x, d_inclusive, n);
  launch(sequential_exclusive_scan, dim3(1), dim3(1), d_x, d_exclusive, n);
  CHECK(cudaGetLastError());

  const std::vector<int> inclusive = hostCopyOf(d_inclusive, n);
  const std::vector<int> exclusive = hostCopyOf(d_exclusive, n);
  printRow("x", x);
  printRow("inclusive", inclusive);
  printRow("exclusive", exclusive);

  std::vector<int> expectedInclusive(n);
  std::vector<int> expectedExclusive(n);
  std::inclusive_scan(x.begin(), x.end(), expectedInclusive.begin());
  std::exclusive_scan(x.begin(), x.end(), expectedExclusive.begin(), 0);
  const bool pass = (inclusive == expectedInclusive) && (exclusive == expectedExclusive);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_inclusive));
  CHECK(cudaFree(d_exclusive));
  return pass ? 0 : 1;
}
