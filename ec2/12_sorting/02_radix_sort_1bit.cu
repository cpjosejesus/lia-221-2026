#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

constexpr int N = 8;
constexpr int KEY_BITS = 3;

__host__ __device__ unsigned bit_of(unsigned key, int bit) {
  return (key >> bit) & 1u;
}

__global__ void radix_bit_kernel(const unsigned* keys, unsigned* bits, int n, int bit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    bits[i] = bit_of(keys[i], bit);
  }
}

__global__ void exclusive_scan_kernel(const unsigned* in, unsigned* out, int n) {
  __shared__ unsigned buffer[2][N];
  const int t = threadIdx.x;
  buffer[0][t] = t > 0 && t < n ? in[t - 1] : 0u;
  __syncthreads();

  int readHalf = 0;
  for (int stride = 1; stride < N; stride *= 2) {
    const int writeHalf = 1 - readHalf;
    unsigned sum = buffer[readHalf][t];
    if (t >= stride) {
      sum += buffer[readHalf][t - stride];
    }
    buffer[writeHalf][t] = sum;
    readHalf = writeHalf;
    __syncthreads();
  }

  if (t < n) {
    out[t] = buffer[readHalf][t];
  }
}

__global__ void radix_scatter_kernel(const unsigned* keysIn,
                                     const unsigned* valsIn,
                                     const unsigned* bits,
                                     const unsigned* onesBefore,
                                     int n,
                                     unsigned* keysOut,
                                     unsigned* valsOut) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const unsigned totalOnes = onesBefore[n - 1] + bits[n - 1];
    const unsigned zeros = n - totalOnes;
    const unsigned dst = bits[i] ? zeros + onesBefore[i] : i - onesBefore[i];
    keysOut[dst] = keysIn[i];
    valsOut[dst] = valsIn[i];
  }
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

static void printValues(const char* label, const std::vector<unsigned>& values) {
  std::printf("%-14s", label);
  for (unsigned value : values) {
    std::printf("  %4u", value);
  }
  std::printf("\n");
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

static void radixPass(
    unsigned* keys_d[2], unsigned* vals_d[2], unsigned* bits_d, unsigned* onesBefore_d, int bit) {
  launch(radix_bit_kernel, dim3(1), dim3(N), keys_d[0], bits_d, N, bit);
  launch(exclusive_scan_kernel, dim3(1), dim3(N), bits_d, onesBefore_d, N);
  launch(radix_scatter_kernel,
         dim3(1),
         dim3(N),
         keys_d[0],
         vals_d[0],
         bits_d,
         onesBefore_d,
         N,
         keys_d[1],
         vals_d[1]);
  CHECK(cudaGetLastError());
  std::swap(keys_d[0], keys_d[1]);
  std::swap(vals_d[0], vals_d[1]);
}

int main() {
  const std::vector<unsigned> keys = {5, 2, 7, 2, 4, 1, 6, 3};
  const std::vector<unsigned> vals = {0, 1, 2, 3, 4, 5, 6, 7};
  unsigned* keys_d[2] = {toDevice(keys), toDevice(keys)};
  unsigned* vals_d[2] = {toDevice(vals), toDevice(vals)};
  unsigned* bits_d = toDevice(keys);
  unsigned* onesBefore_d = toDevice(keys);

  printPairs("input:", keys, vals);
  for (int bit = 0; bit < KEY_BITS; ++bit) {
    radixPass(keys_d, vals_d, bits_d, onesBefore_d, bit);
    std::printf("\nbit %d\n", bit);
    printValues("bits:", toHost(bits_d, N));
    printValues("onesBefore:", toHost(onesBefore_d, N));
    printPairs("after:", toHost(keys_d[0], N), toHost(vals_d[0], N));
  }

  const std::vector<unsigned> sortedKeys = toHost(keys_d[0], N);
  const std::vector<unsigned> sortedVals = toHost(vals_d[0], N);
  for (int buffer = 0; buffer < 2; ++buffer) {
    CHECK(cudaFree(keys_d[buffer]));
    CHECK(cudaFree(vals_d[buffer]));
  }
  CHECK(cudaFree(bits_d));
  CHECK(cudaFree(onesBefore_d));

  std::vector<Pair> expected(N);
  for (int i = 0; i < N; ++i) {
    expected[i] = {keys[i], vals[i]};
  }
  std::stable_sort(expected.begin(), expected.end(), [](const Pair& a, const Pair& b) {
    return a.key < b.key;
  });
  bool ok = true;
  for (int i = 0; i < N; ++i) {
    ok = ok && sortedKeys[i] == expected[i].key && sortedVals[i] == expected[i].val;
  }
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
