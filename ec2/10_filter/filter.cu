#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_compat.h"

constexpr int SIMPLE_THREADS = 256;
constexpr int BK_THREADS = 256;
constexpr int BK_SECTION = 2 * BK_THREADS;
constexpr int F_THREADS = 256;
constexpr int F_COARSEN = 8;
constexpr int F_SECTION = F_THREADS * F_COARSEN;

constexpr int OFFSET_THREADS = 256;
constexpr int FLAG_READY = 1;

constexpr int DEFAULT_ELEMENTS = BY_BACKEND(16000003, 65537);
constexpr int TIMED_RUNS = BY_BACKEND(5, 1);
constexpr int WARMUP_RUNS = BY_BACKEND(1, 0);
constexpr unsigned INPUT_SEED = 2025;
constexpr int VALUE_RANGE = 1000;
constexpr int THRESHOLD_KEEPING_ABOUT_10_PERCENT = 100;
constexpr int THRESHOLD_KEEPING_ABOUT_50_PERCENT = 500;
constexpr int VALUES_SHOWN = 10;

__host__ __device__ inline bool keep(int v, int threshold) {
  return v < threshold;
}

__global__ void filter_global_atomic(const int* in, int* out, int* counter, int n, int threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int v = in[i];
    if (keep(v, threshold)) {
      out[atomicAdd(counter, 1)] = v;
    }
  }
}

__global__ void filter_block_private(const int* in, int* out, int* counter, int n, int threshold) {
  __shared__ int blockCount;
  __shared__ int blockStart;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  const int v = (i < n) ? in[i] : 0;
  const bool k = (i < n) && keep(v, threshold);
  int local = 0;
  if (k) {
    local = atomicAdd(&blockCount, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    blockStart = atomicAdd(counter, blockCount);
  }
  __syncthreads();

  if (k) {
    out[blockStart + local] = v;
  }
}

__global__ void brent_kung_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int XY[BK_SECTION];
  const int t = threadIdx.x;
  const int base = blockIdx.x * BK_SECTION;
  XY[t] = (base + t < n) ? X[base + t] : 0;
  XY[t + BK_THREADS] = (base + t + BK_THREADS < n) ? X[base + t + BK_THREADS] : 0;

  for (int stride = 1; stride <= BK_THREADS; stride *= 2) {
    __syncthreads();
    const int index = (t + 1) * 2 * stride - 1;
    if (index < BK_SECTION) {
      XY[index] += XY[index - stride];
    }
  }

  for (int stride = BK_SECTION / 4; stride > 0; stride /= 2) {
    __syncthreads();
    const int index = (t + 1) * 2 * stride - 1;
    if (index + stride < BK_SECTION) {
      XY[index + stride] += XY[index];
    }
  }

  __syncthreads();
  if (base + t < n) {
    Y[base + t] = XY[t];
  }
  if (base + t + BK_THREADS < n) {
    Y[base + t + BK_THREADS] = XY[t + BK_THREADS];
  }
  if (t == 0) {
    sums[blockIdx.x] = XY[BK_SECTION - 1];
  }
}

__global__ void add_block_offsets(int* Y, const int* scannedSums, int n, int section) {
  if (blockIdx.x > 0) {
    const int add = scannedSums[blockIdx.x - 1];
    for (int j = threadIdx.x; j < section; j += blockDim.x) {
      const long long idx = (long long)blockIdx.x * section + j;
      if (idx < n) {
        Y[idx] += add;
      }
    }
  }
}

__global__ void make_flags(const int* in, int* flags, int n, int threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    flags[i] = keep(in[i], threshold) ? 1 : 0;
  }
}

__global__ void scatter_kept(const int* in, const int* flags, const int* pos, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) {
    out[pos[i] - 1] = in[i];
  }
}

__host__ __device__ inline int exclusive_from_inclusive(const int* inclusive, int i) {
  return (i > 0) ? inclusive[i - 1] : 0;
}

__device__ void load_section_coalesced(const int* in, int n, int base, int* vals) {
  for (int k = 0; k < F_COARSEN; ++k) {
    const int j = k * F_THREADS + threadIdx.x;
    vals[j] = (base + j < n) ? in[base + j] : 0;
  }
}

__device__ bool is_real_and_kept(const int* vals, int j, int base, int n, int threshold) {
  return base + j < n && keep(vals[j], threshold);
}

__device__ void kogge_stone_in_shared(int* part) {
  const int t = threadIdx.x;
  for (int stride = 1; stride < F_THREADS; stride *= 2) {
    __syncthreads();
    int temp = part[t];
    if (t >= stride) {
      temp += part[t - stride];
    }
    __syncthreads();
    part[t] = temp;
  }
}

__device__ void copy_out_coalesced(const int* stage, int count, int* out) {
  for (int k = threadIdx.x; k < count; k += F_THREADS) {
    out[k] = stage[k];
  }
}

__device__ int block_compact(
    const int* in, int n, int base, int threshold, int* vals, int* stage, int* part) {
  const int t = threadIdx.x;
  const int first = t * F_COARSEN;

  load_section_coalesced(in, n, base, vals);
  __syncthreads();

  int count = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    if (is_real_and_kept(vals, first + k, base, n, threshold)) {
      ++count;
    }
  }
  part[t] = count;

  kogge_stone_in_shared(part);
  __syncthreads();

  int pos = exclusive_from_inclusive(part, t);
  for (int k = 0; k < F_COARSEN; ++k) {
    if (is_real_and_kept(vals, first + k, base, n, threshold)) {
      stage[pos++] = vals[first + k];
    }
  }
  __syncthreads();
  return part[F_THREADS - 1];
}

__global__ void count_kept(const int* in, int* counts, int n, int threshold) {
  __shared__ int blockCount;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  const int base = blockIdx.x * F_SECTION;
  int c = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    const int j = base + k * F_THREADS + threadIdx.x;
    if (j < n && keep(in[j], threshold)) {
      ++c;
    }
  }
  atomicAdd(&blockCount, c);
  __syncthreads();

  if (threadIdx.x == 0) {
    counts[blockIdx.x] = blockCount;
  }
}

__global__ void compact_staged(
    const int* in, int* out, const int* scannedCounts, int n, int threshold) {
  __shared__ int vals[F_SECTION];
  __shared__ int stage[F_SECTION];
  __shared__ int part[F_THREADS];
  const int count = block_compact(in, n, blockIdx.x * F_SECTION, threshold, vals, stage, part);
  const int start = exclusive_from_inclusive(scannedCounts, blockIdx.x);
  copy_out_coalesced(stage, count, out + start);
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

__global__ void compact_single_pass(
    const int* in, int* out, int* flags, int* prefix, int* ticket, int n, int threshold) {
  __shared__ int vals[F_SECTION];
  __shared__ int stage[F_SECTION];
  __shared__ int part[F_THREADS];
  __shared__ int bid_s;
  __shared__ int start_s;

  if (threadIdx.x == 0) {
    bid_s = next_ticket(ticket);
  }
  __syncthreads();
  const int bid = bid_s;

  const int count = block_compact(in, n, bid * F_SECTION, threshold, vals, stage, part);

  if (threadIdx.x == 0) {
    const int before = wait_for_sum_of_previous_sections(flags, prefix, bid);
    publish_sum_of_sections_up_to(bid, before + count, flags, prefix);
    start_s = before;
  }
  __syncthreads();

  copy_out_coalesced(stage, count, out + start_s);
}

enum class FilterVariant {
  GlobalAtomic,
  BlockPrivate,
  FlagsScanScatter,
  CountScanCompact,
  SinglePass
};

struct VariantInfo {
  FilterVariant variant;
  const char* name;
  bool preservesOrder;
};

const VariantInfo FILTER_VARIANTS[] = {
    {FilterVariant::GlobalAtomic, "1 global atomic per kept element", false},
    {FilterVariant::BlockPrivate, "2 block-private counter", false},
    {FilterVariant::FlagsScanScatter, "3 flags + scan + scatter", true},
    {FilterVariant::CountScanCompact, "4 count + scan + staged compact", true},
    {FilterVariant::SinglePass, "5 single pass (domino)", true},
};

struct Buffers {
  int* input;
  int* output;
  int* counter;
  int* flags;
  int* positions;
  int* scratch;
  int* sectionCounts;
  int* scannedSectionCounts;
  int* chainFlags;
  int* chainPrefix;
  int* ticket;
};

int blocksFor(int n, int blockSize) {
  return static_cast<int>(cdiv(n, blockSize));
}

int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(int)));
  return pointer;
}

int copyIntToHost(const int* source) {
  int value = 0;
  CHECK(cudaMemcpy(&value, source, sizeof(int), cudaMemcpyDeviceToHost));
  return value;
}

int scratchIntsForHierarchicalScan(int n) {
  const int blocks = blocksFor(n, BK_SECTION);
  if (blocks == 1) {
    return 1;
  }
  return 2 * blocks + scratchIntsForHierarchicalScan(blocks);
}

void scanInclusive(const int* d_in, int* d_out, int n, int* scratch) {
  const int blocks = blocksFor(n, BK_SECTION);
  int* blockSums = scratch;
  launch(brent_kung_scan, dim3(blocks), dim3(BK_THREADS), d_in, d_out, blockSums, n);

  if (blocks > 1) {
    int* scannedBlockSums = scratch + blocks;
    int* scratchForNextLevel = scratch + 2 * blocks;
    scanInclusive(blockSums, scannedBlockSums, blocks, scratchForNextLevel);
    launch(add_block_offsets,
           dim3(blocks),
           dim3(OFFSET_THREADS),
           d_out,
           scannedBlockSums,
           n,
           BK_SECTION);
  }
}

Buffers allocateBuffers(int n) {
  const int sections = blocksFor(n, F_SECTION);
  Buffers buffers;
  buffers.input = allocateDeviceInts(n);
  buffers.output = allocateDeviceInts(n);
  buffers.counter = allocateDeviceInts(1);
  buffers.flags = allocateDeviceInts(n);
  buffers.positions = allocateDeviceInts(n);
  buffers.scratch = allocateDeviceInts(scratchIntsForHierarchicalScan(n));
  buffers.sectionCounts = allocateDeviceInts(sections);
  buffers.scannedSectionCounts = allocateDeviceInts(sections);
  buffers.chainFlags = allocateDeviceInts(sections + 1);
  buffers.chainPrefix = allocateDeviceInts(sections + 1);
  buffers.ticket = allocateDeviceInts(1);
  return buffers;
}

void freeBuffers(const Buffers& buffers) {
  CHECK(cudaFree(buffers.input));
  CHECK(cudaFree(buffers.output));
  CHECK(cudaFree(buffers.counter));
  CHECK(cudaFree(buffers.flags));
  CHECK(cudaFree(buffers.positions));
  CHECK(cudaFree(buffers.scratch));
  CHECK(cudaFree(buffers.sectionCounts));
  CHECK(cudaFree(buffers.scannedSectionCounts));
  CHECK(cudaFree(buffers.chainFlags));
  CHECK(cudaFree(buffers.chainPrefix));
  CHECK(cudaFree(buffers.ticket));
}

int filterWithGlobalAtomic(const Buffers& buffers, int n, int threshold) {
  CHECK(cudaMemset(buffers.counter, 0, sizeof(int)));
  launch(filter_global_atomic,
         dim3(blocksFor(n, SIMPLE_THREADS)),
         dim3(SIMPLE_THREADS),
         buffers.input,
         buffers.output,
         buffers.counter,
         n,
         threshold);
  return copyIntToHost(buffers.counter);
}

int filterWithBlockPrivateCounter(const Buffers& buffers, int n, int threshold) {
  CHECK(cudaMemset(buffers.counter, 0, sizeof(int)));
  launch(filter_block_private,
         dim3(blocksFor(n, SIMPLE_THREADS)),
         dim3(SIMPLE_THREADS),
         buffers.input,
         buffers.output,
         buffers.counter,
         n,
         threshold);
  return copyIntToHost(buffers.counter);
}

int filterWithFlagsScanScatter(const Buffers& buffers, int n, int threshold) {
  const int blocks = blocksFor(n, SIMPLE_THREADS);
  launch(
      make_flags, dim3(blocks), dim3(SIMPLE_THREADS), buffers.input, buffers.flags, n, threshold);
  scanInclusive(buffers.flags, buffers.positions, n, buffers.scratch);
  launch(scatter_kept,
         dim3(blocks),
         dim3(SIMPLE_THREADS),
         buffers.input,
         buffers.flags,
         buffers.positions,
         buffers.output,
         n);
  return copyIntToHost(buffers.positions + (n - 1));
}

int filterWithCountScanCompact(const Buffers& buffers, int n, int threshold) {
  const int sections = blocksFor(n, F_SECTION);
  launch(count_kept,
         dim3(sections),
         dim3(F_THREADS),
         buffers.input,
         buffers.sectionCounts,
         n,
         threshold);
  scanInclusive(buffers.sectionCounts, buffers.scannedSectionCounts, sections, buffers.scratch);
  launch(compact_staged,
         dim3(sections),
         dim3(F_THREADS),
         buffers.input,
         buffers.output,
         buffers.scannedSectionCounts,
         n,
         threshold);
  return copyIntToHost(buffers.scannedSectionCounts + (sections - 1));
}

void resetDominoChain(const Buffers& buffers, int sections) {
  const int chainEntries = sections + 1;
  CHECK(cudaMemset(buffers.chainFlags, 0, chainEntries * sizeof(int)));
  CHECK(cudaMemset(buffers.chainPrefix, 0, chainEntries * sizeof(int)));
  CHECK(cudaMemset(buffers.ticket, 0, sizeof(int)));
}

int filterInSinglePass(const Buffers& buffers, int n, int threshold) {
  const int sections = blocksFor(n, F_SECTION);
  resetDominoChain(buffers, sections);
  launch(compact_single_pass,
         dim3(sections),
         dim3(F_THREADS),
         buffers.input,
         buffers.output,
         buffers.chainFlags,
         buffers.chainPrefix,
         buffers.ticket,
         n,
         threshold);
  return copyIntToHost(buffers.chainPrefix + sections);
}

int runVariant(FilterVariant variant, const Buffers& buffers, int n, int threshold) {
  int keptCount = 0;
  switch (variant) {
    case FilterVariant::GlobalAtomic:
      keptCount = filterWithGlobalAtomic(buffers, n, threshold);
      break;
    case FilterVariant::BlockPrivate:
      keptCount = filterWithBlockPrivateCounter(buffers, n, threshold);
      break;
    case FilterVariant::FlagsScanScatter:
      keptCount = filterWithFlagsScanScatter(buffers, n, threshold);
      break;
    case FilterVariant::CountScanCompact:
      keptCount = filterWithCountScanCompact(buffers, n, threshold);
      break;
    case FilterVariant::SinglePass:
      keptCount = filterInSinglePass(buffers, n, threshold);
      break;
  }
  CHECK(cudaGetLastError());
  return keptCount;
}

std::vector<int> randomValues(int n) {
  std::mt19937 rng(INPUT_SEED);
  std::uniform_int_distribution<int> value(0, VALUE_RANGE - 1);
  std::vector<int> values(n);
  for (int& v : values) {
    v = value(rng);
  }
  return values;
}

std::vector<int> cpuFilter(const std::vector<int>& input, int threshold) {
  std::vector<int> kept;
  kept.reserve(input.size());
  for (int v : input) {
    if (keep(v, threshold)) {
      kept.push_back(v);
    }
  }
  return kept;
}

std::vector<int> sorted(std::vector<int> values) {
  std::sort(values.begin(), values.end());
  return values;
}

void fillWithMinusOne(int* d_values, int count) {
  CHECK(cudaMemset(d_values, 0xff, count * sizeof(int)));
}

std::vector<int> readOutput(const Buffers& buffers, int count) {
  std::vector<int> output(count);
  CHECK(cudaMemcpy(output.data(), buffers.output, count * sizeof(int), cudaMemcpyDeviceToHost));
  return output;
}

template <class Work>
double averageMs(int runs, Work work) {
  Timer timer;
  for (int run = 0; run < runs; ++run) {
    work();
  }
  return timer.ms() / runs;
}

const char* yesNo(bool value) {
  return value ? "yes" : "no";
}

struct Expected {
  std::vector<int> inInputOrder;
  std::vector<int> sortedValues;
};

bool measureAndCheckVariant(const VariantInfo& info,
                            const Buffers& buffers,
                            int n,
                            int threshold,
                            const Expected& expected,
                            double cpuMs) {
  auto run = [&] {
    return runVariant(info.variant, buffers, n, threshold);
  };
  for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
    run();
  }
  fillWithMinusOne(buffers.output, n);
  CHECK(cudaDeviceSynchronize());

  int keptCount = 0;
  const double ms = averageMs(TIMED_RUNS, [&] {
    keptCount = run();
  });

  const int expectedCount = static_cast<int>(expected.inInputOrder.size());
  bool pass = (keptCount == expectedCount);
  bool inInputOrder = false;
  if (pass) {
    const std::vector<int> output = readOutput(buffers, keptCount);
    inInputOrder = (output == expected.inInputOrder);
    pass = info.preservesOrder ? inInputOrder : (sorted(output) == expected.sortedValues);
  }

  std::printf("  %-34s %10.3f %7.2fx %8d %9s  %s\n",
              info.name,
              ms,
              cpuMs / ms,
              keptCount,
              yesNo(inInputOrder),
              pass ? "PASS" : "FAIL");
  return pass;
}

bool runAtThreshold(const std::vector<int>& input, const Buffers& buffers, int threshold) {
  const int n = static_cast<int>(input.size());
  Expected expected;
  const double cpuMs = averageMs(TIMED_RUNS, [&] {
    expected.inInputOrder = cpuFilter(input, threshold);
  });
  expected.sortedValues = sorted(expected.inInputOrder);
  const int expectedCount = static_cast<int>(expected.inInputOrder.size());

  std::printf("threshold %d: %d of %d kept (%.1f %%)\n",
              threshold,
              expectedCount,
              n,
              100.0 * expectedCount / n);
  std::printf("  %-34s %10s %8s %8s %9s  %s\n",
              "variant",
              "time [ms]",
              "vs CPU",
              "kept",
              "in order",
              "check");
  std::printf("  %-34s %10.3f %8s %8d %9s  %s\n",
              "CPU sequential loop (reference)",
              cpuMs,
              "1.00x",
              expectedCount,
              "yes",
              "-");

  bool allPass = true;
  for (const VariantInfo& info : FILTER_VARIANTS) {
    const bool pass = measureAndCheckVariant(info, buffers, n, threshold, expected, cpuMs);
    allPass = allPass && pass;
  }
  std::printf("\n");
  return allPass;
}

void printValues(const std::vector<int>& values) {
  for (int v : values) {
    std::printf("%4d", v);
  }
}

void showWhatUnorderedMeans(const std::vector<int>& input, const Buffers& buffers) {
  const int n = static_cast<int>(input.size());
  const int threshold = THRESHOLD_KEEPING_ABOUT_10_PERCENT;
  std::vector<int> firstInInputOrder = cpuFilter(input, threshold);
  firstInInputOrder.resize(imin(static_cast<int>(firstInInputOrder.size()), VALUES_SHOWN));

  const int keptCount = runVariant(FilterVariant::GlobalAtomic, buffers, n, threshold);
  const std::vector<int> firstFromAtomic = readOutput(buffers, imin(keptCount, VALUES_SHOWN));

  std::printf("First kept values, threshold %d:\n  input order       ", threshold);
  printValues(firstInInputOrder);
  std::printf("\n  global atomic     ");
  printValues(firstFromAtomic);
  std::printf("\n  (variants 3-5 always reproduce the input order; 1-2 only by chance)\n");
}

void printEmulationNote() {
#ifndef __CUDACC__
  std::printf(
      "\nNote: CPU emulation times measure thread barriers, not a GPU; run on a GPU for real "
      "timings.\n");
#endif
}

int elementCount(int argc, char** argv) {
  return (argc > 1) ? std::atoi(argv[1]) : DEFAULT_ELEMENTS;
}

int main(int argc, char** argv) {
  const int n = elementCount(argc, argv);
  if (n < 1) {
    std::fprintf(stderr, "n must be >= 1\n");
    return 1;
  }
  std::printf("Filtering / stream compaction (chapter 12)  backend: %s\n", CUDA_BACKEND);
  std::printf("n = %d ints uniform in [0, %d), keep value < threshold, %d timed run(s) each\n\n",
              n,
              VALUE_RANGE,
              TIMED_RUNS);

  const std::vector<int> input = randomValues(n);
  const Buffers buffers = allocateBuffers(n);
  CHECK(cudaMemcpy(buffers.input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  bool allPass = true;
  const int thresholds[] = {THRESHOLD_KEEPING_ABOUT_10_PERCENT, THRESHOLD_KEEPING_ABOUT_50_PERCENT};
  for (int threshold : thresholds) {
    const bool pass = runAtThreshold(input, buffers, threshold);
    allPass = allPass && pass;
  }

  showWhatUnorderedMeans(input, buffers);
  printEmulationNote();
  std::printf("\n%s\n", allPass ? "ALL PASS" : "SOME CHECKS FAILED");

  freeBuffers(buffers);
  return allPass ? 0 : 1;
}
