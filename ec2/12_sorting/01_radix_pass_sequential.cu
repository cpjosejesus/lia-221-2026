#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <vector>

constexpr int N = 8;
constexpr int BIT = 0;

__host__ __device__ unsigned bit_of(unsigned key, int bit) {
  return (key >> bit) & 1u;
}

__host__ __device__ void radix_pass_sequential(const unsigned* keysIn,
                                               const unsigned* valsIn,
                                               int n,
                                               int bit,
                                               unsigned* keysOut,
                                               unsigned* valsOut) {
  unsigned zeros = 0;
  for (int i = 0; i < n; ++i) {
    zeros += 1u - bit_of(keysIn[i], bit);
  }

  unsigned zerosBefore = 0;
  unsigned onesBefore = 0;
  for (int i = 0; i < n; ++i) {
    const bool isOne = bit_of(keysIn[i], bit) == 1u;
    const unsigned dst = isOne ? zeros + onesBefore : zerosBefore;
    keysOut[dst] = keysIn[i];
    valsOut[dst] = valsIn[i];
    onesBefore += isOne ? 1u : 0u;
    zerosBefore += isOne ? 0u : 1u;
  }
}

__global__ void radix_pass_sequential_kernel(const unsigned* keysIn,
                                             const unsigned* valsIn,
                                             int n,
                                             int bit,
                                             unsigned* keysOut,
                                             unsigned* valsOut) {
  radix_pass_sequential(keysIn, valsIn, n, bit, keysOut, valsOut);
}

struct Pair {
  unsigned key;
  unsigned val;
};

static unsigned* toDevice(const std::vector<unsigned>& values) {
  unsigned* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, values.size() * sizeof(unsigned)));
  CHECK(
      cudaMemcpy(pointer, values.data(), values.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  return pointer;
}

static std::vector<unsigned> toHost(const unsigned* pointer, int count) {
  std::vector<unsigned> values(count);
  CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return values;
}

static void printPairs(const char* label,
                       const std::vector<unsigned>& keys,
                       const std::vector<unsigned>& vals) {
  std::printf("%-14s", label);
  for (int i = 0; i < N; ++i) {
    std::printf("  %u(%u)", keys[i], vals[i]);
  }
  std::printf("\n");
}

static std::vector<Pair> stablePartitionOnHost(const std::vector<unsigned>& keys,
                                               const std::vector<unsigned>& vals) {
  std::vector<Pair> pairs(N);
  for (int i = 0; i < N; ++i) {
    pairs[i] = {keys[i], vals[i]};
  }
  std::stable_partition(pairs.begin(), pairs.end(), [](const Pair& pair) {
    return bit_of(pair.key, BIT) == 0u;
  });
  return pairs;
}

int main() {
  const std::vector<unsigned> keys = {5, 2, 7, 2, 4, 1, 6, 3};
  const std::vector<unsigned> vals = {0, 1, 2, 3, 4, 5, 6, 7};
  unsigned* keysIn_d = toDevice(keys);
  unsigned* valsIn_d = toDevice(vals);
  unsigned* keysOut_d = toDevice(keys);
  unsigned* valsOut_d = toDevice(vals);

  launch(radix_pass_sequential_kernel,
         dim3(1),
         dim3(1),
         keysIn_d,
         valsIn_d,
         N,
         BIT,
         keysOut_d,
         valsOut_d);
  CHECK(cudaGetLastError());
  const std::vector<unsigned> keysOut = toHost(keysOut_d, N);
  const std::vector<unsigned> valsOut = toHost(valsOut_d, N);

  printPairs("input:", keys, vals);
  printPairs("after bit 0:", keysOut, valsOut);

  CHECK(cudaFree(keysIn_d));
  CHECK(cudaFree(valsIn_d));
  CHECK(cudaFree(keysOut_d));
  CHECK(cudaFree(valsOut_d));

  const std::vector<Pair> expected = stablePartitionOnHost(keys, vals);
  bool ok = true;
  for (int i = 0; i < N; ++i) {
    ok = ok && keysOut[i] == expected[i].key && valsOut[i] == expected[i].val;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
