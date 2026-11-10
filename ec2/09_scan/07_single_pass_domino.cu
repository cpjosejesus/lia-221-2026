#include <cstdio>
#include <numeric>
#include <vector>

#include "cuda_compat.h"

constexpr int SECTION_SIZE = 4;
constexpr int FLAG_READY = 1;

__device__ void kogge_stone_in_shared(int* XY) {
  const int t = threadIdx.x;
  for (int stride = 1; stride < SECTION_SIZE; stride *= 2) {
    __syncthreads();
    int temp = XY[t];
    if (t >= stride) {
      temp += XY[t - stride];
    }
    __syncthreads();
    XY[t] = temp;
  }
  __syncthreads();
}

__device__ int next_ticket(int* ticket) {
  return atomicAdd(ticket, 1);
}

__device__ int atomic_read(int* address) {
  return atomicAdd(address, 0);
}

__device__ void wait_until_ready(int* flag) {
  int state = atomic_read(flag);
  while (state != FLAG_READY) {
    state = atomic_read(flag);
  }
}

__device__ int wait_for_sum_of_previous_sections(int* flags, int* prefix, int bid) {
  if (bid == 0) {
    return 0;
  }
  wait_until_ready(&flags[bid]);
  return atomic_read(&prefix[bid]);
}

__device__ void publish_sum_of_sections_up_to(int bid, int sum, int* flags, int* prefix) {
  atomicExch(&prefix[bid + 1], sum);
  __threadfence();
  atomicExch(&flags[bid + 1], FLAG_READY);
}

__global__ void domino_scan(const int* X, int* Y, int* flags, int* prefix, int* ticket, int n) {
  __shared__ int XY[SECTION_SIZE];
  __shared__ int bid_s;
  __shared__ int before_s;
  const int t = threadIdx.x;

  if (t == 0) {
    bid_s = next_ticket(ticket);
  }
  __syncthreads();
  const int bid = bid_s;
  const int i = bid * SECTION_SIZE + t;

  XY[t] = (i < n) ? X[i] : 0;
  kogge_stone_in_shared(XY);

  if (t == 0) {
    const int before = wait_for_sum_of_previous_sections(flags, prefix, bid);
    const int sum = before + XY[SECTION_SIZE - 1];
    publish_sum_of_sections_up_to(bid, sum, flags, prefix);
    printf("  block %d got ticket %d: sum before = %3d, publishes %3d\n",
           blockIdx.x,
           bid,
           before,
           sum);
    before_s = before;
  }
  __syncthreads();

  if (i < n) {
    Y[i] = XY[t] + before_s;
  }
}

int* deviceZeros(int count) {
  int* device = nullptr;
  CHECK(cudaMalloc(&device, count * sizeof(int)));
  CHECK(cudaMemset(device, 0, count * sizeof(int)));
  return device;
}

void printRow(const char* label, const std::vector<int>& values) {
  std::printf("  %-10s", label);
  for (int value : values) {
    std::printf("%4d", value);
  }
  std::printf("\n");
}

int main() {
  const std::vector<int> x = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 0, 8, 1, 1, 9, 2, 4, 6, 1};
  const int n = static_cast<int>(x.size());
  const int sections = static_cast<int>(cdiv(n, SECTION_SIZE));

  int* d_x = deviceZeros(n);
  int* d_y = deviceZeros(n);
  int* d_flags = deviceZeros(sections + 1);
  int* d_prefix = deviceZeros(sections + 1);
  int* d_ticket = deviceZeros(1);
  CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  printRow("x", x);
  launch(domino_scan, dim3(sections), dim3(SECTION_SIZE), d_x, d_y, d_flags, d_prefix, d_ticket, n);
  CHECK(cudaGetLastError());
  std::vector<int> y(n);
  CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(int), cudaMemcpyDeviceToHost));
  printRow("inclusive", y);

  std::vector<int> expected(n);
  std::inclusive_scan(x.begin(), x.end(), expected.begin());
  const bool pass = (y == expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_y));
  CHECK(cudaFree(d_flags));
  CHECK(cudaFree(d_prefix));
  CHECK(cudaFree(d_ticket));
  return pass ? 0 : 1;
}
