#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

constexpr int N = 16;
constexpr int KEY_BITS = 8;
constexpr unsigned KEY_RANGE = 1u << KEY_BITS;
constexpr unsigned DISTINCT_KEYS = 12;
constexpr unsigned KEY_STEP = 0x5b;
constexpr unsigned FIRST_KEY = 0x3a;
constexpr int RADIX_BITS = 4;
constexpr int TILE = 8;
constexpr int BLOCKS = (N + TILE - 1) / TILE;
constexpr int BUCKETS = 1 << RADIX_BITS;
constexpr int TABLE_SIZE = BUCKETS * BLOCKS;
constexpr unsigned PADDING_KEY = 0xFFFFFFFFu;

__host__ __device__ unsigned bit_of(unsigned key, int bit) {
  return (key >> bit) & 1u;
}

__host__ __device__ unsigned digit_of(unsigned key, int shift) {
  return (key >> shift) & (BUCKETS - 1u);
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
                                        int shift,
                                        unsigned* keysTmp,
                                        unsigned* counts,
                                        unsigned* localStart) {
  __shared__ unsigned keys_S[TILE];
  __shared__ unsigned buf[2 * TILE];
  __shared__ unsigned bucketStart[BUCKETS];
  __shared__ unsigned bucketEnd[BUCKETS];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  for (int d = t; d < BUCKETS; d += blockDim.x) {
    bucketStart[d] = 0;
    bucketEnd[d] = 0;
  }
  keys_S[t] = t < tileCount ? keysIn[tileBase + t] : PADDING_KEY;
  __syncthreads();

  for (int bitInDigit = 0; bitInDigit < RADIX_BITS; ++bitInDigit) {
    const unsigned key = keys_S[t];
    const unsigned myBit = bit_of(key, shift + bitInDigit);
    unsigned totalOnes;
    const unsigned onesBefore = block_inclusive_scan(myBit, buf, &totalOnes) - myBit;
    const unsigned zeros = TILE - totalOnes;
    const unsigned dst = myBit ? zeros + onesBefore : t - onesBefore;
    keys_S[dst] = key;
    __syncthreads();
  }

  if (t < tileCount) {
    const unsigned d = digit_of(keys_S[t], shift);
    const bool startsBucket = t == 0 || digit_of(keys_S[t - 1], shift) != d;
    const bool endsBucket = t == tileCount - 1 || digit_of(keys_S[t + 1], shift) != d;
    if (startsBucket) {
      bucketStart[d] = t;
    }
    if (endsBucket) {
      bucketEnd[d] = t + 1;
    }
    keysTmp[tileBase + t] = keys_S[t];
  }
  __syncthreads();

  for (int d = t; d < BUCKETS; d += blockDim.x) {
    counts[bucket_major_index(d, blockIdx.x, gridDim.x)] = bucketEnd[d] - bucketStart[d];
    localStart[block_major_index(blockIdx.x, d, BUCKETS)] = bucketStart[d];
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
                                               int shift,
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
    const int dst = localToGlobal[digit_of(key, shift)] + t;
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

static void printKeys(const char* label, const std::vector<unsigned>& keys) {
  std::printf("%-34s", label);
  for (unsigned key : keys) {
    std::printf(" %02x", key);
  }
  std::printf("\n");
}

static void printCounts(const std::vector<unsigned>& counts, int block) {
  std::printf("counts of block %d, buckets 0..f:  ", block);
  for (int d = 0; d < BUCKETS; ++d) {
    std::printf(" %2u", counts[bucket_major_index(d, block, BLOCKS)]);
  }
  std::printf("\n");
}

static void radixPass(SortBuffers& buffers, int shift) {
  launch(radix_local_sort_kernel,
         dim3(BLOCKS),
         dim3(TILE),
         buffers.keys[0],
         N,
         shift,
         buffers.keysTmp,
         buffers.counts,
         buffers.localStart);
  launch(scan_table_kernel, dim3(1), dim3(TABLE_SIZE), buffers.counts, buffers.offsets, TABLE_SIZE);
  launch(radix_scatter_coalesced_kernel,
         dim3(BLOCKS),
         dim3(TILE),
         buffers.keysTmp,
         N,
         shift,
         buffers.offsets,
         buffers.localStart,
         buffers.keys[1]);
  CHECK(cudaGetLastError());
  std::swap(buffers.keys[0], buffers.keys[1]);
}

static void printPass(const SortBuffers& buffers, int shift) {
  std::printf("\ndigit of bits %d..%d\n", shift, shift + RADIX_BITS - 1);
  printKeys("tiles after local sort:", toHost(buffers.keysTmp, N));
  const std::vector<unsigned> counts = toHost(buffers.counts, TABLE_SIZE);
  for (int block = 0; block < BLOCKS; ++block) {
    printCounts(counts, block);
  }
  printKeys("after scatter:", toHost(buffers.keys[0], N));
}

static std::vector<unsigned> makeKeys() {
  std::vector<unsigned> keys(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = (i % DISTINCT_KEYS * KEY_STEP + FIRST_KEY) % KEY_RANGE;
  }
  return keys;
}

int main() {
  const std::vector<unsigned> keys = makeKeys();
  const std::vector<unsigned> table(TABLE_SIZE, 0u);
  SortBuffers buffers = {{toDevice(keys), toDevice(keys)},
                         toDevice(keys),
                         toDevice(table),
                         toDevice(table),
                         toDevice(table)};

  printKeys("input (hex):", keys);
  for (int shift = 0; shift < KEY_BITS; shift += RADIX_BITS) {
    radixPass(buffers, shift);
    printPass(buffers, shift);
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
