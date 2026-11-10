#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
constexpr int UNSIGNED_BITS = 32;
constexpr unsigned PADDING_KEY = 0xFFFFFFFFu;

__device__ inline int coalesced_position(int item) {
  return item * BLOCK_THREADS + threadIdx.x;
}

__device__ inline int thread_owned_position(int item) {
  return threadIdx.x * ITEMS_PER_THREAD + item;
}

__device__ inline unsigned block_inclusive_scan(unsigned value, unsigned* buf, unsigned* total) {
  const int t = threadIdx.x;
  buf[t] = value;
  __syncthreads();

  int readHalf = 0;
  for (int stride = 1; stride < BLOCK_THREADS; stride *= 2) {
    const int writeHalf = 1 - readHalf;
    unsigned sum = buf[readHalf * BLOCK_THREADS + t];
    if (t >= stride) {
      sum += buf[readHalf * BLOCK_THREADS + t - stride];
    }
    buf[writeHalf * BLOCK_THREADS + t] = sum;
    readHalf = writeHalf;
    __syncthreads();
  }

  *total = buf[readHalf * BLOCK_THREADS + BLOCK_THREADS - 1];
  const unsigned inclusivePrefix = buf[readHalf * BLOCK_THREADS + t];
  __syncthreads();
  return inclusivePrefix;
}

__global__ void scan_block_kernel(const unsigned* in, unsigned* out, int n, unsigned* blockSums) {
  __shared__ unsigned tile[TILE];
  __shared__ unsigned buf[2 * BLOCK_THREADS];
  const int base = blockIdx.x * TILE;

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    tile[pos] = base + pos < n ? in[base + pos] : 0u;
  }
  __syncthreads();

  unsigned threadSum = 0;
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    threadSum += tile[thread_owned_position(item)];
  }

  unsigned blockTotal;
  unsigned running = block_inclusive_scan(threadSum, buf, &blockTotal) - threadSum;
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = thread_owned_position(item);
    const unsigned value = tile[pos];
    tile[pos] = running;
    running += value;
  }
  __syncthreads();

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    if (base + pos < n) {
      out[base + pos] = tile[pos];
    }
  }
  if (threadIdx.x == 0) {
    blockSums[blockIdx.x] = blockTotal;
  }
}

__global__ void add_block_offsets_kernel(unsigned* out, int n, const unsigned* blockOffsets) {
  const unsigned offset = blockOffsets[blockIdx.x];
  const int base = blockIdx.x * TILE;
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = base + coalesced_position(item);
    if (index < n) {
      out[index] += offset;
    }
  }
}

constexpr size_t SCAN_WORKSPACE_SLACK = 2;

static size_t scanWorkspace(int n) {
  size_t words = SCAN_WORKSPACE_SLACK;
  for (int blocks = cdiv(n, TILE); blocks > 1; blocks = cdiv(blocks, TILE)) {
    const size_t sumsAndScannedSums = 2 * blocks;
    words += sumsAndScannedSums;
  }
  return words + SCAN_WORKSPACE_SLACK;
}

static void exclusiveScan(const unsigned* in, unsigned* out, int n, unsigned* workspace) {
  if (n <= 0) {
    return;
  }

  const int blocks = cdiv(n, TILE);
  unsigned* sums = workspace;
  unsigned* sumsScanned = sums + blocks;
  unsigned* deeperWorkspace = sumsScanned + blocks;
  launch(scan_block_kernel, dim3(blocks), dim3(BLOCK_THREADS), in, out, n, sums);
  if (blocks > 1) {
    exclusiveScan(sums, sumsScanned, blocks, deeperWorkspace);
    launch(add_block_offsets_kernel, dim3(blocks), dim3(BLOCK_THREADS), out, n, sumsScanned);
  }
}

__host__ __device__ inline unsigned digit_of(unsigned key, int shift, int r) {
  return shift < UNSIGNED_BITS ? (key >> shift) & ((1u << r) - 1u) : 0u;
}

__host__ __device__ inline unsigned bit_of(unsigned key, int bit) {
  return digit_of(key, bit, 1);
}

__global__ void radix_bit_kernel(const unsigned* keys, unsigned* bits, int n, int bit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    bits[i] = bit_of(keys[i], bit);
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

__host__ __device__ inline int bucket_major_index(int bucket, int block, int numBlocks) {
  return bucket * numBlocks + block;
}

__host__ __device__ inline int block_major_index(int block, int bucket, int buckets) {
  return block * buckets + bucket;
}

template <int RADIX_BITS>
__global__ void radix_local_sort_kernel(const unsigned* keysIn,
                                        const unsigned* valsIn,
                                        int n,
                                        int shift,
                                        unsigned* keysTmp,
                                        unsigned* valsTmp,
                                        unsigned* counts,
                                        unsigned* localStart) {
  constexpr int BUCKETS = 1 << RADIX_BITS;
  __shared__ unsigned keys_S[2][TILE];
  __shared__ unsigned vals_S[2][TILE];
  __shared__ unsigned buf[2 * BLOCK_THREADS];
  __shared__ unsigned bucketStart[BUCKETS];
  __shared__ unsigned bucketEnd[BUCKETS];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    const bool isRealElement = pos < tileCount;
    keys_S[0][pos] = isRealElement ? keysIn[tileBase + pos] : PADDING_KEY;
    vals_S[0][pos] = isRealElement ? valsIn[tileBase + pos] : 0u;
  }
  for (int d = t; d < BUCKETS; d += BLOCK_THREADS) {
    bucketStart[d] = 0;
    bucketEnd[d] = 0;
  }
  __syncthreads();

  int current = 0;
  for (int bitInDigit = 0; bitInDigit < RADIX_BITS; ++bitInDigit) {
    const int bit = shift + bitInDigit;
    const int next = 1 - current;
    unsigned myBits[ITEMS_PER_THREAD];
    unsigned myOnes = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      myBits[item] = bit_of(keys_S[current][thread_owned_position(item)], bit);
      myOnes += myBits[item];
    }

    unsigned totalOnes;
    unsigned onesBefore = block_inclusive_scan(myOnes, buf, &totalOnes) - myOnes;
    const unsigned zeros = TILE - totalOnes;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const unsigned pos = thread_owned_position(item);
      const unsigned dst = myBits[item] ? zeros + onesBefore : pos - onesBefore;
      onesBefore += myBits[item];
      keys_S[next][dst] = keys_S[current][pos];
      vals_S[next][dst] = vals_S[current][pos];
    }
    __syncthreads();
    current = next;
  }

  const unsigned* sortedKeys = keys_S[current];
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = thread_owned_position(item);
    if (pos < tileCount) {
      const unsigned d = digit_of(sortedKeys[pos], shift, RADIX_BITS);
      const bool startsBucket = pos == 0 || digit_of(sortedKeys[pos - 1], shift, RADIX_BITS) != d;
      const bool endsBucket =
          pos == tileCount - 1 || digit_of(sortedKeys[pos + 1], shift, RADIX_BITS) != d;
      if (startsBucket) {
        bucketStart[d] = pos;
      }
      if (endsBucket) {
        bucketEnd[d] = pos + 1;
      }
    }
  }
  __syncthreads();

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    if (pos < tileCount) {
      keysTmp[tileBase + pos] = keys_S[current][pos];
      valsTmp[tileBase + pos] = vals_S[current][pos];
    }
  }
  for (int d = t; d < BUCKETS; d += BLOCK_THREADS) {
    counts[bucket_major_index(d, blockIdx.x, gridDim.x)] = bucketEnd[d] - bucketStart[d];
    localStart[block_major_index(blockIdx.x, d, BUCKETS)] = bucketStart[d];
  }
}

template <int RADIX_BITS>
__global__ void radix_scatter_coalesced_kernel(const unsigned* keysTmp,
                                               const unsigned* valsTmp,
                                               int n,
                                               int shift,
                                               const unsigned* offsets,
                                               const unsigned* localStart,
                                               unsigned* keysOut,
                                               unsigned* valsOut) {
  constexpr int BUCKETS = 1 << RADIX_BITS;
  __shared__ int localToGlobal[BUCKETS];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  for (int d = t; d < BUCKETS; d += BLOCK_THREADS) {
    const int globalStart = offsets[bucket_major_index(d, blockIdx.x, gridDim.x)];
    const int bucketLocalStart = localStart[block_major_index(blockIdx.x, d, BUCKETS)];
    localToGlobal[d] = globalStart - bucketLocalStart;
  }
  __syncthreads();

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    if (pos < tileCount) {
      const unsigned key = keysTmp[tileBase + pos];
      const int dst = localToGlobal[digit_of(key, shift, RADIX_BITS)] + pos;
      keysOut[dst] = key;
      valsOut[dst] = valsTmp[tileBase + pos];
    }
  }
}

__host__ __device__ inline int co_rank(int k, const unsigned* A, int m, const unsigned* B, int n) {
  int i = imin(k, m);
  int j = k - i;
  int i_low = imax(0, k - n);
  int j_low = imax(0, k - m);
  bool searching = true;
  while (searching) {
    const bool i_too_large = i > 0 && j < n && A[i - 1] > B[j];
    const bool i_too_small = j > 0 && i < m && B[j - 1] >= A[i];
    if (i_too_large) {
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      i -= delta;
      j += delta;
    } else if (i_too_small) {
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      searching = false;
    }
  }
  return i;
}

__host__ __device__ inline void merge_sequential_kv(const unsigned* Ak,
                                                    const unsigned* Av,
                                                    int m,
                                                    const unsigned* Bk,
                                                    const unsigned* Bv,
                                                    int n,
                                                    unsigned* Ck,
                                                    unsigned* Cv) {
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < m && j < n) {
    if (Ak[i] <= Bk[j]) {
      Ck[k] = Ak[i];
      Cv[k++] = Av[i++];
    } else {
      Ck[k] = Bk[j];
      Cv[k++] = Bv[j++];
    }
  }
  while (i < m) {
    Ck[k] = Ak[i];
    Cv[k++] = Av[i++];
  }
  while (j < n) {
    Ck[k] = Bk[j];
    Cv[k++] = Bv[j++];
  }
}

__host__ __device__ inline void stable_insertion_sort(unsigned* keys,
                                                      unsigned* vals,
                                                      int begin,
                                                      int end) {
  for (int i = begin + 1; i < end; ++i) {
    const unsigned key = keys[i];
    const unsigned val = vals[i];
    int j = i - 1;
    while (j >= begin && keys[j] > key) {
      keys[j + 1] = keys[j];
      vals[j + 1] = vals[j];
      --j;
    }
    keys[j + 1] = key;
    vals[j + 1] = val;
  }
}

__global__ void sort_runs_kernel(unsigned* keys, unsigned* vals, int n, int run) {
  const int runStart = (blockIdx.x * blockDim.x + threadIdx.x) * run;
  if (runStart < n) {
    const int runEnd = imin(runStart + run, n);
    stable_insertion_sort(keys, vals, runStart, runEnd);
  }
}

__global__ void merge_pass_kernel(const unsigned* keysIn,
                                  const unsigned* valsIn,
                                  unsigned* keysOut,
                                  unsigned* valsOut,
                                  int total,
                                  int width,
                                  int outputsPerThread) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sliceBegin = imin(tid * outputsPerThread, total);
  const int sliceEnd = imin(sliceBegin + outputsPerThread, total);

  int k = sliceBegin;
  while (k < sliceEnd) {
    const int pairStart = (k / (2 * width)) * (2 * width);
    const int mid = imin(pairStart + width, total);
    const int pairEnd = imin(pairStart + 2 * width, total);
    const int pieceEnd = imin(sliceEnd, pairEnd);

    const int m = mid - pairStart;
    const int n = pairEnd - mid;
    const int k_curr = k - pairStart;
    const int k_next = pieceEnd - pairStart;
    const int i_curr = co_rank(k_curr, keysIn + pairStart, m, keysIn + mid, n);
    const int i_next = co_rank(k_next, keysIn + pairStart, m, keysIn + mid, n);
    const int j_curr = k_curr - i_curr;
    const int j_next = k_next - i_next;
    merge_sequential_kv(keysIn + pairStart + i_curr,
                        valsIn + pairStart + i_curr,
                        i_next - i_curr,
                        keysIn + mid + j_curr,
                        valsIn + mid + j_curr,
                        j_next - j_curr,
                        keysOut + k,
                        valsOut + k);

    k = pieceEnd;
  }
}

constexpr int SIMPLE_KERNEL_THREADS = 256;
constexpr int MAX_RADIX_BITS = 8;
constexpr int MAX_BUCKETS = 1 << MAX_RADIX_BITS;
constexpr int MERGE_SORT_RUN = 16;
constexpr int MERGE_OUTPUTS_PER_THREAD = 64;

struct SortBuffers {
  unsigned* keys[2];
  unsigned* vals[2];
  unsigned* bits;
  unsigned* scan;
  unsigned* counts;
  unsigned* offsets;
  unsigned* localStart;
  unsigned* workspace;
};

static unsigned* allocateDeviceWords(size_t count) {
  unsigned* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(unsigned)));
  return pointer;
}

static SortBuffers allocateBuffers(int n) {
  const int tableEntries = MAX_BUCKETS * cdiv(n, TILE);
  SortBuffers buffers;
  for (int side = 0; side < 2; ++side) {
    buffers.keys[side] = allocateDeviceWords(n);
    buffers.vals[side] = allocateDeviceWords(n);
  }
  buffers.bits = allocateDeviceWords(n);
  buffers.scan = allocateDeviceWords(n);
  buffers.counts = allocateDeviceWords(tableEntries);
  buffers.offsets = allocateDeviceWords(tableEntries);
  buffers.localStart = allocateDeviceWords(tableEntries);
  buffers.workspace = allocateDeviceWords(scanWorkspace(imax(n, tableEntries)));
  return buffers;
}

static void freeBuffers(SortBuffers& buffers) {
  for (int side = 0; side < 2; ++side) {
    CHECK(cudaFree(buffers.keys[side]));
    CHECK(cudaFree(buffers.vals[side]));
  }
  CHECK(cudaFree(buffers.bits));
  CHECK(cudaFree(buffers.scan));
  CHECK(cudaFree(buffers.counts));
  CHECK(cudaFree(buffers.offsets));
  CHECK(cudaFree(buffers.localStart));
  CHECK(cudaFree(buffers.workspace));
}

static int otherSide(int side) {
  return 1 - side;
}

static int radixSortNaive(SortBuffers& buffers, int n, int keyBits) {
  const unsigned blocks = cdiv(n, SIMPLE_KERNEL_THREADS);
  int source = 0;
  for (int bit = 0; bit < keyBits; ++bit) {
    const int destination = otherSide(source);
    launch(radix_bit_kernel,
           dim3(blocks),
           dim3(SIMPLE_KERNEL_THREADS),
           buffers.keys[source],
           buffers.bits,
           n,
           bit);
    exclusiveScan(buffers.bits, buffers.scan, n, buffers.workspace);
    launch(radix_scatter_kernel,
           dim3(blocks),
           dim3(SIMPLE_KERNEL_THREADS),
           buffers.keys[source],
           buffers.vals[source],
           buffers.bits,
           buffers.scan,
           n,
           buffers.keys[destination],
           buffers.vals[destination]);
    source = destination;
  }
  return source;
}

template <int RADIX_BITS>
static int radixSortLocal(SortBuffers& buffers, int n, int keyBits) {
  static_assert(RADIX_BITS <= MAX_RADIX_BITS, "the bucket table is sized for MAX_RADIX_BITS");
  constexpr int BUCKETS = 1 << RADIX_BITS;
  const unsigned blocks = cdiv(n, TILE);
  unsigned* tileSortedKeys = buffers.bits;
  unsigned* tileSortedVals = buffers.scan;
  int source = 0;
  for (int shift = 0; shift < keyBits; shift += RADIX_BITS) {
    const int destination = otherSide(source);
    launch(radix_local_sort_kernel<RADIX_BITS>,
           dim3(blocks),
           dim3(BLOCK_THREADS),
           buffers.keys[source],
           buffers.vals[source],
           n,
           shift,
           tileSortedKeys,
           tileSortedVals,
           buffers.counts,
           buffers.localStart);
    exclusiveScan(buffers.counts, buffers.offsets, BUCKETS * blocks, buffers.workspace);
    launch(radix_scatter_coalesced_kernel<RADIX_BITS>,
           dim3(blocks),
           dim3(BLOCK_THREADS),
           tileSortedKeys,
           tileSortedVals,
           n,
           shift,
           buffers.offsets,
           buffers.localStart,
           buffers.keys[destination],
           buffers.vals[destination]);
    source = destination;
  }
  return source;
}

static int mergeSort(SortBuffers& buffers, int n) {
  const unsigned runs = cdiv(n, MERGE_SORT_RUN);
  launch(sort_runs_kernel,
         dim3(cdiv(runs, SIMPLE_KERNEL_THREADS)),
         dim3(SIMPLE_KERNEL_THREADS),
         buffers.keys[0],
         buffers.vals[0],
         n,
         MERGE_SORT_RUN);

  const unsigned mergeThreads = cdiv(n, MERGE_OUTPUTS_PER_THREAD);
  int source = 0;
  for (int width = MERGE_SORT_RUN; width < n; width *= 2) {
    const int destination = otherSide(source);
    launch(merge_pass_kernel,
           dim3(cdiv(mergeThreads, SIMPLE_KERNEL_THREADS)),
           dim3(SIMPLE_KERNEL_THREADS),
           buffers.keys[source],
           buffers.vals[source],
           buffers.keys[destination],
           buffers.vals[destination],
           n,
           width,
           MERGE_OUTPUTS_PER_THREAD);
    source = destination;
  }
  return source;
}

enum class SortVariant { Radix1Naive, Radix1Local, Radix4Local, Radix8Local, MergeSort };

constexpr SortVariant ALL_VARIANTS[] = {SortVariant::Radix1Naive,
                                        SortVariant::Radix1Local,
                                        SortVariant::Radix4Local,
                                        SortVariant::Radix8Local,
                                        SortVariant::MergeSort};

static const char* variantName(SortVariant variant) {
  switch (variant) {
    case SortVariant::Radix1Naive:
      return "radix 1 bit, naive scatter";
    case SortVariant::Radix1Local:
      return "radix 1 bit, local sort";
    case SortVariant::Radix4Local:
      return "radix 4 bits, local sort";
    case SortVariant::Radix8Local:
      return "radix 8 bits, local sort";
    case SortVariant::MergeSort:
      return "merge sort (co-rank merge)";
  }
  return "unknown";
}

static int runVariant(SortVariant variant, SortBuffers& buffers, int n, int keyBits) {
  int resultSide = 0;
  switch (variant) {
    case SortVariant::Radix1Naive:
      resultSide = radixSortNaive(buffers, n, keyBits);
      break;
    case SortVariant::Radix1Local:
      resultSide = radixSortLocal<1>(buffers, n, keyBits);
      break;
    case SortVariant::Radix4Local:
      resultSide = radixSortLocal<4>(buffers, n, keyBits);
      break;
    case SortVariant::Radix8Local:
      resultSide = radixSortLocal<8>(buffers, n, keyBits);
      break;
    case SortVariant::MergeSort:
      resultSide = mergeSort(buffers, n);
      break;
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  return resultSide;
}

struct KeyValuePairs {
  std::vector<unsigned> keys;
  std::vector<unsigned> vals;
};

static bool operator==(const KeyValuePairs& a, const KeyValuePairs& b) {
  return a.keys == b.keys && a.vals == b.vals;
}

static void copyToDevice(unsigned* destination, const std::vector<unsigned>& source) {
  CHECK(cudaMemcpy(
      destination, source.data(), source.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
}

static void copyToHost(std::vector<unsigned>& destination, const unsigned* source) {
  CHECK(cudaMemcpy(
      destination.data(), source, destination.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
}

static double sortOnDevice(SortVariant variant,
                           SortBuffers& buffers,
                           KeyValuePairs& pairs,
                           int keyBits) {
  const int n = static_cast<int>(pairs.keys.size());
  copyToDevice(buffers.keys[0], pairs.keys);
  copyToDevice(buffers.vals[0], pairs.vals);

  Timer timer;
  const int resultSide = runVariant(variant, buffers, n, keyBits);
  const double ms = timer.ms();

  copyToHost(pairs.keys, buffers.keys[resultSide]);
  copyToHost(pairs.vals, buffers.vals[resultSide]);
  return ms;
}

static KeyValuePairs randomKeysWithPositions(int n, int keyBits, unsigned seed) {
  std::mt19937 rng(seed);
  const unsigned keyMask = keyBits >= UNSIGNED_BITS ? ~0u : (1u << keyBits) - 1u;
  KeyValuePairs pairs;
  pairs.keys.resize(n);
  pairs.vals.resize(n);
  for (int i = 0; i < n; ++i) {
    pairs.keys[i] = rng() & keyMask;
    pairs.vals[i] = i;
  }
  return pairs;
}

static KeyValuePairs stableSortOnHost(const std::vector<unsigned>& keys) {
  const int n = static_cast<int>(keys.size());
  std::vector<std::pair<unsigned, unsigned>> keyPosition(n);
  for (int i = 0; i < n; ++i) {
    keyPosition[i] = {keys[i], static_cast<unsigned>(i)};
  }
  std::stable_sort(keyPosition.begin(), keyPosition.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  KeyValuePairs sorted;
  sorted.keys.resize(n);
  sorted.vals.resize(n);
  for (int i = 0; i < n; ++i) {
    sorted.keys[i] = keyPosition[i].first;
    sorted.vals[i] = keyPosition[i].second;
  }
  return sorted;
}

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

constexpr int SMALL_CASE_CAPACITY = 3000;
constexpr int SMALL_CASE_KEY_BITS = 6;
constexpr unsigned SMALL_CASE_SEED_BASE = 100;

static bool checkSmallCases() {
  const int sizes[] = {1, 2, 1000, 2500};
  SortBuffers buffers = allocateBuffers(SMALL_CASE_CAPACITY);
  int passed = 0;
  int total = 0;
  for (int size : sizes) {
    const KeyValuePairs input =
        randomKeysWithPositions(size, SMALL_CASE_KEY_BITS, SMALL_CASE_SEED_BASE + size);
    const KeyValuePairs expected = stableSortOnHost(input.keys);
    for (SortVariant variant : ALL_VARIANTS) {
      KeyValuePairs pairs = input;
      sortOnDevice(variant, buffers, pairs, SMALL_CASE_KEY_BITS);
      const bool ok = pairs == expected;
      passed += ok;
      ++total;
      if (!ok) {
        std::printf("  FAIL: %s with n = %d\n", variantName(variant), size);
      }
    }
  }
  freeBuffers(buffers);

  const bool allPassed = passed == total;
  std::printf("small cases: %d/%d %s\n\n", passed, total, passOrFail(allPassed));
  return allPassed;
}

constexpr int DEFAULT_PAIRS = BY_BACKEND(1 << 24, 1 << 14);
constexpr int TIMED_KEY_BITS = BY_BACKEND(32, 16);
constexpr unsigned TIMED_SEED = 7;

constexpr double MS_PER_SECOND = 1e3;
constexpr double PAIRS_PER_MILLION = 1e6;

static double millionPairsPerSecond(int n, double ms) {
  const double seconds = ms / MS_PER_SECOND;
  return n / seconds / PAIRS_PER_MILLION;
}

static void printHostRow(const char* label, int n, double ms, const char* check) {
  std::printf("%-30s %10.2f %12.2f  %s\n", label, ms, millionPairsPerSecond(n, ms), check);
}

static void printDeviceRow(const char* label, int n, double ms, bool keysOk, bool stable) {
  std::printf("%-30s %10.2f %12.2f  %s, %s\n",
              label,
              ms,
              millionPairsPerSecond(n, ms),
              passOrFail(keysOk),
              stable ? "stable" : "NOT STABLE");
}

static bool runTimedComparison(int n, int keyBits) {
  const KeyValuePairs input = randomKeysWithPositions(n, keyBits, TIMED_SEED);

  std::vector<unsigned> keysOnly = input.keys;
  Timer sortTimer;
  std::sort(keysOnly.begin(), keysOnly.end());
  const double sortMs = sortTimer.ms();

  Timer stableTimer;
  const KeyValuePairs expected = stableSortOnHost(input.keys);
  const double stableMs = stableTimer.ms();

  SortBuffers buffers = allocateBuffers(n);
  KeyValuePairs warmUp = input;
  sortOnDevice(SortVariant::MergeSort, buffers, warmUp, keyBits);

  std::printf("sorting n = %d (key, value) pairs, keys of %d bits\n", n, keyBits);
  std::printf("%-30s %10s %12s  %s\n", "variant", "time (ms)", "Mpairs/s", "check (keys, stable)");
  printHostRow("std::sort (CPU, keys only)", n, sortMs, "-");
  printHostRow("std::stable_sort (CPU, pairs)", n, stableMs, "reference");

  bool allPassed = true;
  for (SortVariant variant : ALL_VARIANTS) {
    KeyValuePairs pairs = input;
    const double ms = sortOnDevice(variant, buffers, pairs, keyBits);
    const bool keysOk = pairs.keys == expected.keys;
    const bool stable = pairs.vals == expected.vals;
    allPassed = allPassed && keysOk && stable;
    printDeviceRow(variantName(variant), n, ms, keysOk, stable);
  }
  freeBuffers(buffers);
  return allPassed;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : DEFAULT_PAIRS;
  std::printf("Parallel sorting (chapter 14) on %s\n\n", CUDA_BACKEND);

  const bool smallCasesOk = checkSmallCases();
  const bool timedOk = runTimedComparison(n, TIMED_KEY_BITS);

  const bool allOk = smallCasesOk && timedOk;
  std::printf("\n%s\n", allOk ? "ALL PASS" : "SOME TESTS FAILED");
  return allOk ? 0 : 1;
}
