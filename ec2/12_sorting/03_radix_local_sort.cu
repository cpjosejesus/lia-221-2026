#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

constexpr int N = 8;
constexpr int KEY_BITS = 3;
constexpr int TILE = 4;
constexpr int BLOCKS = (N + TILE - 1) / TILE;
constexpr int BUCKETS = 2;
constexpr int TABLE_SIZE = BUCKETS * BLOCKS;
constexpr unsigned PADDING_KEY = 0xFFFFFFFFu;

__host__ __device__ unsigned bit_of(unsigned key, int bit) {
  return (key >> bit) & 1u;
}

__host__ __device__ int bucket_major_index(int bucket, int block, int numBlocks) {
  return bucket * numBlocks + block;
}

__host__ __device__ int block_major_index(int block, int bucket, int buckets) {
  return block * buckets + bucket;
}

__device__ unsigned block_inclusive_scan(unsigned value, unsigned* buf, unsigned* total) {
  const int t = threadIdx.x;
  const int threads = blockDim.x;
  buf[t] = value;
  __syncthreads();

  int readHalf = 0;
  for (int stride = 1; stride < threads; stride *= 2) {
    const int writeHalf = 1 - readHalf;
    unsigned sum = buf[readHalf * threads + t];
    if (t >= stride) {
      sum += buf[readHalf * threads + t - stride];
    }
    buf[writeHalf * threads + t] = sum;
    readHalf = writeHalf;
    __syncthreads();
  }

  *total = buf[readHalf * threads + threads - 1];
  const unsigned inclusivePrefix = buf[readHalf * threads + t];
  __syncthreads();
  return inclusivePrefix;
}

__global__ void radix_local_sort_kernel(const unsigned* keysIn,
                                        int n,
                                        int bit,
                                        unsigned* keysTmp,
                                        unsigned* counts,
                                        unsigned* localStart) {
  __shared__ unsigned keys_S[TILE];
  __shared__ unsigned buf[2 * TILE];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  const bool isRealElement = t < tileCount;
  const unsigned key = isRealElement ? keysIn[tileBase + t] : PADDING_KEY;
  const unsigned myBit = bit_of(key, bit);

  unsigned totalOnes;
  const unsigned onesBefore = block_inclusive_scan(myBit, buf, &totalOnes) - myBit;
  const unsigned zeros = TILE - totalOnes;
  const unsigned dst = myBit ? zeros + onesBefore : t - onesBefore;
  keys_S[dst] = key;
  __syncthreads();

  if (isRealElement) {
    keysTmp[tileBase + t] = keys_S[t];
  }
  if (t == 0) {
    counts[bucket_major_index(0, blockIdx.x, gridDim.x)] = zeros;
    counts[bucket_major_index(1, blockIdx.x, gridDim.x)] = tileCount - zeros;
    localStart[block_major_index(blockIdx.x, 0, BUCKETS)] = 0;
    localStart[block_major_index(blockIdx.x, 1, BUCKETS)] = zeros;
  }
}

__global__ void scan_table_kernel(const unsigned* counts, unsigned* offsets, int size) {
  __shared__ unsigned buf[2 * TABLE_SIZE];
  const int t = threadIdx.x;
  const unsigned count = t < size ? counts[t] : 0u;
  unsigned total;
  const unsigned inclusivePrefix = block_inclusive_scan(count, buf, &total);
  if (t < size) {
    offsets[t] = inclusivePrefix - count;
  }
}

__global__ void radix_scatter_coalesced_kernel(const unsigned* keysTmp,
                                               int n,
                                               int bit,
                                               const unsigned* offsets,
                                               const unsigned* localStart,
                                               unsigned* keysOut) {
  __shared__ int localToGlobal[BUCKETS];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  for (int d = t; d < BUCKETS; d += blockDim.x) {
    const int globalStart = offsets[bucket_major_index(d, blockIdx.x, gridDim.x)];
    const int bucketLocalStart = localStart[block_major_index(blockIdx.x, d, BUCKETS)];
    localToGlobal[d] = globalStart - bucketLocalStart;
  }
  __syncthreads();

  if (t < tileCount) {
    const unsigned key = keysTmp[tileBase + t];
    const int dst = localToGlobal[bit_of(key, bit)] + t;
    keysOut[dst] = key;
  }
}

struct SortBuffers {
  unsigned* keys[2];
  unsigned* keysTmp;
  unsigned* counts;
  unsigned* offsets;
  unsigned* localStart;
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
  std::printf("%-24s", label);
  for (unsigned value : values) {
    std::printf(" %2u", value);
  }
  std::printf("\n");
}

static void radixPass(SortBuffers& buffers, int bit) {
  launch(radix_local_sort_kernel,
         dim3(BLOCKS),
         dim3(TILE),
         buffers.keys[0],
         N,
         bit,
         buffers.keysTmp,
         buffers.counts,
         buffers.localStart);
  launch(scan_table_kernel, dim3(1), dim3(TABLE_SIZE), buffers.counts, buffers.offsets, TABLE_SIZE);
  launch(radix_scatter_coalesced_kernel,
         dim3(BLOCKS),
         dim3(TILE),
         buffers.keysTmp,
         N,
         bit,
         buffers.offsets,
         buffers.localStart,
         buffers.keys[1]);
  CHECK(cudaGetLastError());
  std::swap(buffers.keys[0], buffers.keys[1]);
}

static void printPass(const SortBuffers& buffers, int bit) {
  std::printf("\nbit %d\n", bit);
  printValues("tiles after local sort:", toHost(buffers.keysTmp, N));
  printValues("counts (bucket-major):", toHost(buffers.counts, TABLE_SIZE));
  printValues("offsets (scan):", toHost(buffers.offsets, TABLE_SIZE));
  printValues("after scatter:", toHost(buffers.keys[0], N));
}

int main() {
  const std::vector<unsigned> keys = {5, 2, 7, 2, 4, 1, 6, 3};
  const std::vector<unsigned> table(TABLE_SIZE, 0u);
  SortBuffers buffers = {{toDevice(keys), toDevice(keys)},
                         toDevice(keys),
                         toDevice(table),
                         toDevice(table),
                         toDevice(table)};

  printValues("input:", keys);
  for (int bit = 0; bit < KEY_BITS; ++bit) {
    radixPass(buffers, bit);
    printPass(buffers, bit);
  }

  const std::vector<unsigned> sortedKeys = toHost(buffers.keys[0], N);
  unsigned* allBuffers[] = {buffers.keys[0],
                            buffers.keys[1],
                            buffers.keysTmp,
                            buffers.counts,
                            buffers.offsets,
                            buffers.localStart};
  for (unsigned* buffer : allBuffers) {
    CHECK(cudaFree(buffer));
  }

  std::vector<unsigned> expected = keys;
  std::sort(expected.begin(), expected.end());
  const bool ok = sortedKeys == expected;
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
