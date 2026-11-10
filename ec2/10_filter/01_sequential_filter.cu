#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "cuda_compat.h"

constexpr int THRESHOLD = 100;

__host__ __device__ inline bool keep(int v, int threshold) {
  return v < threshold;
}

__global__ void sequential_filter(const int* in, int* out, int* counter, int n, int threshold) {
  int kept = 0;
  for (int i = 0; i < n; ++i) {
    if (keep(in[i], threshold)) {
      out[kept] = in[i];
      ++kept;
    }
  }
  *counter = kept;
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

int main() {
  const std::vector<int> in = {
      812, 45, 30, 17, 999, 60, 431, 8, 77, 650, 230, 905, 140, 91, 3, 520};
  const int n = static_cast<int>(in.size());

  int* d_in = deviceCopyOf(in);
  int* d_out = deviceCopyOf(std::vector<int>(n, -1));
  int* d_counter = deviceCopyOf({0});

  launch(sequential_filter, dim3(1), dim3(1), d_in, d_out, d_counter, n, THRESHOLD);
  CHECK(cudaGetLastError());
  const int count = hostCopyOf(d_counter, 1)[0];
  const std::vector<int> out = hostCopyOf(d_out, count);

  std::printf("keep v < %d\n", THRESHOLD);
  printRow("in", in);
  printRow("out", out);
  std::printf("  count = %d\n", count);

  const bool pass = (out == cpuFilter(in));
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_counter));
  return pass ? 0 : 1;
}
