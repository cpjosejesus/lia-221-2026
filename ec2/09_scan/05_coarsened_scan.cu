#include <cstdio>
#include <numeric>
#include <vector>

#include "cuda_compat.h"

constexpr int CO_THREADS = 4;
constexpr int CO_COARSEN = 4;
constexpr int CO_SECTION = CO_THREADS * CO_COARSEN;

__device__ void print_shared(const char* label, const int* values, int count) {
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("  %-10s", label);
    for (int j = 0; j < count; ++j) {
      printf("%4d", values[j]);
    }
    printf("\n");
  }
}

__device__ void load_section_coalesced(const int* X, int n, int base, int* XY) {
  for (int k = 0; k < CO_COARSEN; ++k) {
    const int j = k * CO_THREADS + threadIdx.x;
    XY[j] = (base + j < n) ? X[base + j] : 0;
  }
}

__device__ void store_section_coalesced(const int* XY, int n, int base, int* Y) {
  for (int k = 0; k < CO_COARSEN; ++k) {
    const int j = k * CO_THREADS + threadIdx.x;
    if (base + j < n) {
      Y[base + j] = XY[j];
    }
  }
}

__device__ void phase1_scan_own_subsection(int* XY, int first) {
  for (int k = 1; k < CO_COARSEN; ++k) {
    XY[first + k] += XY[first + k - 1];
  }
}

__device__ void phase2_kogge_stone_over_subsection_totals(int* part) {
  const int t = threadIdx.x;
  for (int stride = 1; stride < CO_THREADS; stride *= 2) {
    __syncthreads();
    int temp = part[t];
    if (t >= stride) {
      temp += part[t - stride];
    }
    __syncthreads();
    part[t] = temp;
  }
}

__device__ void phase3_add_total_of_subsections_to_the_left(int* XY, const int* part, int first) {
  const int t = threadIdx.x;
  if (t > 0) {
    const int add = part[t - 1];
    for (int k = 0; k < CO_COARSEN; ++k) {
      XY[first + k] += add;
    }
  }
}

__global__ void coarsened_scan(const int* X, int* Y, int n) {
  __shared__ int XY[CO_SECTION];
  __shared__ int part[CO_THREADS];
  const int t = threadIdx.x;
  const int first = t * CO_COARSEN;
  const int base = blockIdx.x * CO_SECTION;

  load_section_coalesced(X, n, base, XY);
  __syncthreads();

  phase1_scan_own_subsection(XY, first);
  part[t] = XY[first + CO_COARSEN - 1];
  print_shared("phase 1", XY, CO_SECTION);

  phase2_kogge_stone_over_subsection_totals(part);
  print_shared("phase 2", part, CO_THREADS);

  phase3_add_total_of_subsections_to_the_left(XY, part, first);
  print_shared("phase 3", XY, CO_SECTION);

  store_section_coalesced(XY, n, base, Y);
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
  const std::vector<int> x = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 0, 8, 1, 1, 9, 2};
  const int n = static_cast<int>(x.size());

  int* d_x = deviceCopyOf(x);
  int* d_y = deviceCopyOf(std::vector<int>(n));

  printRow("x", x);
  launch(coarsened_scan, dim3(1), dim3(CO_THREADS), d_x, d_y, n);
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
