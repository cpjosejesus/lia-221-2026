#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "cuda_compat.h"

constexpr int THRESHOLD = 100;
constexpr int SECTION_SIZE = 16;

__host__ __device__ inline bool keep(int v, int threshold) {
  return v < threshold;
}

__global__ void make_flags(const int* in, int* flags, int n, int threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    flags[i] = keep(in[i], threshold) ? 1 : 0;
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
  }

  if (i < n) {
    Y[i] = XY[t];
  }
}

__global__ void scatter_kept(const int* in, const int* flags, const int* pos, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) {
    out[pos[i] - 1] = in[i];
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

int main() {
  const std::vector<int> in = {
      812, 45, 30, 17, 999, 60, 431, 8, 77, 650, 230, 905, 140, 91, 3, 520};
  const int n = static_cast<int>(in.size());

  int* d_in = deviceCopyOf(in);
  int* d_flags = deviceCopyOf(std::vector<int>(n));
  int* d_pos = deviceCopyOf(std::vector<int>(n));
  int* d_out = deviceCopyOf(std::vector<int>(n, -1));

  launch(make_flags, dim3(1), dim3(SECTION_SIZE), d_in, d_flags, n, THRESHOLD);
  launch(kogge_stone_scan, dim3(1), dim3(SECTION_SIZE), d_flags, d_pos, n);
  launch(scatter_kept, dim3(1), dim3(SECTION_SIZE), d_in, d_flags, d_pos, d_out, n);
  CHECK(cudaGetLastError());

  const std::vector<int> pos = hostCopyOf(d_pos, n);
  const int count = pos[n - 1];
  const std::vector<int> out = hostCopyOf(d_out, count);

  std::printf("keep v < %d\n", THRESHOLD);
  printRow("in", in);
  printRow("flag", hostCopyOf(d_flags, n));
  printRow("pos", pos);
  printRow("out", out);
  std::printf("  count = pos[%d] = %d\n", n - 1, count);

  const bool pass = (out == cpuFilter(in));
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_flags));
  CHECK(cudaFree(d_pos));
  CHECK(cudaFree(d_out));
  return pass ? 0 : 1;
}
