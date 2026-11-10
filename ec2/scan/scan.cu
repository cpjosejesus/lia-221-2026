#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_compat.h"

constexpr int KS_SECTION = 256;
constexpr int KS_THREADS = KS_SECTION;

constexpr int BK_THREADS = 256;
constexpr int BK_ELEMENTS_PER_THREAD = 2;
constexpr int BK_SECTION = BK_ELEMENTS_PER_THREAD * BK_THREADS;

constexpr int CO_THREADS = 256;
constexpr int CO_COARSEN = 8;
constexpr int CO_SECTION = CO_THREADS * CO_COARSEN;

constexpr int OFFSET_THREADS = 256;
constexpr int SHIFT_THREADS = 256;
constexpr int FLAG_READY = 1;

constexpr int DEFAULT_ELEMENTS = BY_BACKEND(16000003, 100003);
constexpr int TIMED_RUNS = BY_BACKEND(5, 1);
constexpr int WARMUP_RUNS = BY_BACKEND(1, 0);
constexpr unsigned INPUT_SEED = 2025;
constexpr int MAX_DIGIT = 9;

static_assert(static_cast<long long>(DEFAULT_ELEMENTS) * MAX_DIGIT < INT_MAX,
              "the scan of the default input must fit in an int");

__global__ void kogge_stone_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int XY[KS_SECTION];
  const int t = threadIdx.x;
  const int i = blockIdx.x * KS_SECTION + t;
  XY[t] = (i < n) ? X[i] : 0;

  for (int stride = 1; stride < KS_SECTION; stride *= 2) {
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
  if (t == KS_SECTION - 1) {
    sums[blockIdx.x] = XY[t];
  }
}

__global__ void kogge_stone_double_buffer_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int buf[2][KS_SECTION];
  const int t = threadIdx.x;
  const int i = blockIdx.x * KS_SECTION + t;
  int src = 0;
  buf[src][t] = (i < n) ? X[i] : 0;

  for (int stride = 1; stride < KS_SECTION; stride *= 2) {
    __syncthreads();
    int v = buf[src][t];
    if (t >= stride) {
      v += buf[src][t - stride];
    }
    const int dst = 1 - src;
    buf[dst][t] = v;
    src = dst;
  }

  if (i < n) {
    Y[i] = buf[src][t];
  }
  if (t == KS_SECTION - 1) {
    sums[blockIdx.x] = buf[src][t];
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

__device__ void load_section_coalesced(const int* X, int n, int base, int* XY) {
  for (int k = 0; k < CO_COARSEN; ++k) {
    const int j = k * CO_THREADS + threadIdx.x;
    XY[j] = (base + j < n) ? X[base + j] : 0;
  }
}

__device__ void store_section_coalesced(const int* XY, int offset, int n, int base, int* Y) {
  for (int k = 0; k < CO_COARSEN; ++k) {
    const int j = k * CO_THREADS + threadIdx.x;
    if (base + j < n) {
      Y[base + j] = XY[j] + offset;
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

__device__ void coarsened_block_scan(const int* X, int n, int base, int* XY, int* part) {
  const int t = threadIdx.x;
  const int first = t * CO_COARSEN;

  load_section_coalesced(X, n, base, XY);
  __syncthreads();

  phase1_scan_own_subsection(XY, first);
  part[t] = XY[first + CO_COARSEN - 1];

  phase2_kogge_stone_over_subsection_totals(part);
  __syncthreads();

  phase3_add_total_of_subsections_to_the_left(XY, part, first);
  __syncthreads();
}

__global__ void coarsened_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int XY[CO_SECTION];
  __shared__ int part[CO_THREADS];
  const int base = blockIdx.x * CO_SECTION;

  coarsened_block_scan(X, n, base, XY, part);

  store_section_coalesced(XY, 0, n, base, Y);
  if (threadIdx.x == 0) {
    sums[blockIdx.x] = part[CO_THREADS - 1];
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
  __shared__ int XY[CO_SECTION];
  __shared__ int part[CO_THREADS];
  __shared__ int bid_s;
  __shared__ int before_s;

  if (threadIdx.x == 0) {
    bid_s = next_ticket(ticket);
  }
  __syncthreads();
  const int bid = bid_s;
  const int base = bid * CO_SECTION;

  coarsened_block_scan(X, n, base, XY, part);

  if (threadIdx.x == 0) {
    const int before = wait_for_sum_of_previous_sections(flags, prefix, bid);
    publish_sum_of_sections_up_to(bid, before + part[CO_THREADS - 1], flags, prefix);
    before_s = before;
  }
  __syncthreads();

  store_section_coalesced(XY, before_s, n, base, Y);
}

__global__ void inclusive_to_exclusive(const int* Y, int* E, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    E[i] = (i == 0) ? 0 : Y[i - 1];
  }
}

using BlockScanKernel = void (*)(const int*, int*, int*, int);

enum class ScanStrategy { Hierarchical, SinglePass };

struct ScanVariant {
  const char* name;
  ScanStrategy strategy;
  BlockScanKernel blockScan;
  int section;
  int threads;
};

const ScanVariant SCAN_VARIANTS[] = {
    {"hierarchical + Kogge-Stone",
     ScanStrategy::Hierarchical,
     kogge_stone_scan,
     KS_SECTION,
     KS_THREADS},
    {"hierarchical + Kogge-Stone 2-buffer",
     ScanStrategy::Hierarchical,
     kogge_stone_double_buffer_scan,
     KS_SECTION,
     KS_THREADS},
    {"hierarchical + Brent-Kung",
     ScanStrategy::Hierarchical,
     brent_kung_scan,
     BK_SECTION,
     BK_THREADS},
    {"hierarchical + coarsened 3-phase",
     ScanStrategy::Hierarchical,
     coarsened_scan,
     CO_SECTION,
     CO_THREADS},
    {"single-pass domino (coarsened)", ScanStrategy::SinglePass, nullptr, CO_SECTION, CO_THREADS},
};

struct ScanBuffers {
  int* input;
  int* output;
  int* scratch;
  int* flags;
  int* prefix;
  int* ticket;
};

int blocksFor(int n, int section) {
  return static_cast<int>(cdiv(n, section));
}

int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(int)));
  return pointer;
}

void copyToDevice(int* destination, const std::vector<int>& source) {
  CHECK(
      cudaMemcpy(destination, source.data(), source.size() * sizeof(int), cudaMemcpyHostToDevice));
}

void copyToHost(std::vector<int>& destination, const int* source) {
  CHECK(cudaMemcpy(
      destination.data(), source, destination.size() * sizeof(int), cudaMemcpyDeviceToHost));
}

int scratchIntsForHierarchicalScan(int n, int section) {
  const int blocks = blocksFor(n, section);
  if (blocks == 1) {
    return 1;
  }
  return 2 * blocks + scratchIntsForHierarchicalScan(blocks, section);
}

void scanHierarchical(BlockScanKernel blockScan,
                      int section,
                      int threads,
                      const int* d_in,
                      int* d_out,
                      int n,
                      int* scratch) {
  const int blocks = blocksFor(n, section);
  int* blockSums = scratch;
  launch(blockScan, dim3(blocks), dim3(threads), d_in, d_out, blockSums, n);

  if (blocks > 1) {
    int* scannedBlockSums = scratch + blocks;
    int* scratchForNextLevel = scratch + 2 * blocks;
    scanHierarchical(
        blockScan, section, threads, blockSums, scannedBlockSums, blocks, scratchForNextLevel);
    launch(
        add_block_offsets, dim3(blocks), dim3(OFFSET_THREADS), d_out, scannedBlockSums, n, section);
  }
}

void resetDominoChain(const ScanBuffers& buffers, int sections) {
  const int chainEntries = sections + 1;
  CHECK(cudaMemset(buffers.flags, 0, chainEntries * sizeof(int)));
  CHECK(cudaMemset(buffers.prefix, 0, chainEntries * sizeof(int)));
  CHECK(cudaMemset(buffers.ticket, 0, sizeof(int)));
}

void scanSinglePass(const ScanBuffers& buffers, int n) {
  const int sections = blocksFor(n, CO_SECTION);
  resetDominoChain(buffers, sections);
  launch(domino_scan,
         dim3(sections),
         dim3(CO_THREADS),
         buffers.input,
         buffers.output,
         buffers.flags,
         buffers.prefix,
         buffers.ticket,
         n);
}

void runScanVariant(const ScanVariant& variant, const ScanBuffers& buffers, int n) {
  switch (variant.strategy) {
    case ScanStrategy::Hierarchical:
      scanHierarchical(variant.blockScan,
                       variant.section,
                       variant.threads,
                       buffers.input,
                       buffers.output,
                       n,
                       buffers.scratch);
      break;
    case ScanStrategy::SinglePass:
      scanSinglePass(buffers, n);
      break;
  }
}

void shiftToExclusive(const int* d_inclusive, int* d_exclusive, int n) {
  launch(inclusive_to_exclusive,
         dim3(blocksFor(n, SHIFT_THREADS)),
         dim3(SHIFT_THREADS),
         d_inclusive,
         d_exclusive,
         n);
}

void cpuInclusiveScan(const std::vector<int>& x, std::vector<int>& y) {
  int run = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    run += x[i];
    y[i] = run;
  }
}

bool isExclusiveScanOf(const std::vector<int>& exclusive, const std::vector<int>& inclusive) {
  if (exclusive[0] != 0) {
    return false;
  }
  for (size_t i = 1; i < exclusive.size(); ++i) {
    if (exclusive[i] != inclusive[i - 1]) {
      return false;
    }
  }
  return true;
}

int firstMismatch(const std::vector<int>& actual, const std::vector<int>& expected) {
  return static_cast<int>(std::mismatch(actual.begin(), actual.end(), expected.begin()).first -
                          actual.begin());
}

template <class Work>
double averageMs(int runs, Work work) {
  Timer timer;
  for (int run = 0; run < runs; ++run) {
    work();
  }
  return timer.ms() / runs;
}

const char* verdict(bool pass) {
  return pass ? "PASS" : "FAIL";
}

void printRow(const char* label, const std::vector<int>& values) {
  std::printf("  %-10s", label);
  for (int value : values) {
    std::printf("%4d", value);
  }
  std::printf("\n");
}

bool runTinyExample() {
  const std::vector<int> input = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 0, 8, 1, 1, 9, 2};
  const int m = static_cast<int>(input.size());
  std::vector<int> inclusive(m);
  std::vector<int> exclusive(m);

  int* d_input = allocateDeviceInts(m);
  int* d_inclusive = allocateDeviceInts(m);
  int* d_exclusive = allocateDeviceInts(m);
  int* d_blockSum = allocateDeviceInts(1);
  copyToDevice(d_input, input);

  launch(brent_kung_scan, dim3(1), dim3(BK_THREADS), d_input, d_inclusive, d_blockSum, m);
  shiftToExclusive(d_inclusive, d_exclusive, m);
  CHECK(cudaGetLastError());
  copyToHost(inclusive, d_inclusive);
  copyToHost(exclusive, d_exclusive);

  std::printf("Tiny example (Brent-Kung, one block, %d elements padded with zeros to %d):\n",
              m,
              BK_SECTION);
  printRow("input", input);
  printRow("inclusive", inclusive);
  printRow("exclusive", exclusive);

  std::vector<int> reference(m);
  cpuInclusiveScan(input, reference);
  const bool pass = (reference == inclusive) && isExclusiveScanOf(exclusive, reference);
  std::printf("  check: %s\n\n", verdict(pass));

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_inclusive));
  CHECK(cudaFree(d_exclusive));
  CHECK(cudaFree(d_blockSum));
  return pass;
}

int log2Floor(int value) {
  int exponent = 0;
  while ((1 << (exponent + 1)) <= value) {
    ++exponent;
  }
  return exponent;
}

int sequentialAdds(int N) {
  return N - 1;
}

int koggeStoneAdds(int N) {
  return N * log2Floor(N) - (N - 1);
}

int koggeStoneSteps(int N) {
  return log2Floor(N);
}

int brentKungAdds(int N) {
  return 2 * N - 2 - log2Floor(N);
}

int brentKungSteps(int N) {
  return 2 * log2Floor(N) - 1;
}

void printWorkAndStepsTable() {
  const int sectionSizes[] = {16, 256, 512, 2048};
  std::printf("Work (additions) and steps to scan one section of N elements:\n");
  std::printf("  %6s | %10s %6s | %10s %6s | %10s %6s\n",
              "N",
              "seq adds",
              "steps",
              "KS adds",
              "steps",
              "BK adds",
              "steps");
  for (int N : sectionSizes) {
    std::printf("  %6d | %10d %6d | %10d %6d | %10d %6d\n",
                N,
                sequentialAdds(N),
                sequentialAdds(N),
                koggeStoneAdds(N),
                koggeStoneSteps(N),
                brentKungAdds(N),
                brentKungSteps(N));
  }
  std::printf("\n");
}

std::vector<int> randomDigits(int n) {
  std::mt19937 rng(INPUT_SEED);
  std::uniform_int_distribution<int> digit(0, MAX_DIGIT);
  std::vector<int> values(n);
  for (int& value : values) {
    value = digit(rng);
  }
  return values;
}

int scratchIntsForEveryVariant(int n) {
  int largest = 0;
  for (const ScanVariant& variant : SCAN_VARIANTS) {
    largest = imax(largest, scratchIntsForHierarchicalScan(n, variant.section));
  }
  return largest;
}

ScanBuffers allocateScanBuffers(int n) {
  const int chainEntries = blocksFor(n, CO_SECTION) + 1;
  ScanBuffers buffers;
  buffers.input = allocateDeviceInts(n);
  buffers.output = allocateDeviceInts(n);
  buffers.scratch = allocateDeviceInts(scratchIntsForEveryVariant(n));
  buffers.flags = allocateDeviceInts(chainEntries);
  buffers.prefix = allocateDeviceInts(chainEntries);
  buffers.ticket = allocateDeviceInts(1);
  return buffers;
}

void freeScanBuffers(const ScanBuffers& buffers) {
  CHECK(cudaFree(buffers.input));
  CHECK(cudaFree(buffers.output));
  CHECK(cudaFree(buffers.scratch));
  CHECK(cudaFree(buffers.flags));
  CHECK(cudaFree(buffers.prefix));
  CHECK(cudaFree(buffers.ticket));
}

void clearOutputSoNoStaleResultCanPass(const ScanBuffers& buffers, int n) {
  CHECK(cudaMemset(buffers.output, 0, n * sizeof(int)));
  CHECK(cudaDeviceSynchronize());
}

void printVariantTableHeader(int n, double cpuMs) {
  std::printf("Inclusive scan of n = %d ints (values 0..%d), %d timed run(s) each:\n",
              n,
              MAX_DIGIT,
              TIMED_RUNS);
  std::printf("  %-37s %8s %8s %10s %8s  %s\n",
              "variant",
              "section",
              "blocks",
              "time [ms]",
              "vs CPU",
              "check");
  std::printf("  %-37s %8s %8s %10.3f %8s  %s\n",
              "CPU sequential loop (reference)",
              "-",
              "-",
              cpuMs,
              "1.00x",
              "-");
}

bool measureAndCheckVariant(const ScanVariant& variant,
                            const ScanBuffers& buffers,
                            const std::vector<int>& reference,
                            double cpuMs) {
  const int n = static_cast<int>(reference.size());
  auto run = [&] {
    runScanVariant(variant, buffers, n);
  };

  for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
    run();
  }
  clearOutputSoNoStaleResultCanPass(buffers, n);
  const double ms = averageMs(TIMED_RUNS, run);
  CHECK(cudaGetLastError());

  std::vector<int> output(n);
  copyToHost(output, buffers.output);
  const int mismatch = firstMismatch(output, reference);
  const bool pass = (mismatch == n);

  std::printf("  %-37s %8d %8d %10.3f %7.2fx  %s",
              variant.name,
              variant.section,
              blocksFor(n, variant.section),
              ms,
              cpuMs / ms,
              verdict(pass));
  if (!pass) {
    std::printf(" (Y[%d] = %d, expected %d)", mismatch, output[mismatch], reference[mismatch]);
  }
  std::printf("\n");
  return pass;
}

bool checkExclusiveScan(const ScanBuffers& buffers, const std::vector<int>& reference) {
  const int n = static_cast<int>(reference.size());
  scanHierarchical(
      coarsened_scan, CO_SECTION, CO_THREADS, buffers.input, buffers.output, n, buffers.scratch);

  int* d_exclusive = allocateDeviceInts(n);
  shiftToExclusive(buffers.output, d_exclusive, n);
  CHECK(cudaGetLastError());
  std::vector<int> exclusive(n);
  copyToHost(exclusive, d_exclusive);
  CHECK(cudaFree(d_exclusive));

  const bool pass = isExclusiveScanOf(exclusive, reference);
  std::printf("  %-37s %8s %8s %10s %8s  %s\n",
              "exclusive scan (shifted inclusive)",
              "-",
              "-",
              "-",
              "-",
              verdict(pass));
  return pass;
}

void printEmulationNote() {
#ifndef __CUDACC__
  std::printf(
      "\nNote: CPU emulation times measure thread barriers, not a GPU; use them only to see that\n"
      "every variant runs. Run on a GPU (Colab) for meaningful timings.\n");
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
  std::printf("Parallel scan (chapter 11)  backend: %s\n\n", CUDA_BACKEND);

  bool allPass = runTinyExample();
  printWorkAndStepsTable();

  const std::vector<int> input = randomDigits(n);
  std::vector<int> reference(n);
  const double cpuMs = averageMs(TIMED_RUNS, [&] {
    cpuInclusiveScan(input, reference);
  });

  const ScanBuffers buffers = allocateScanBuffers(n);
  copyToDevice(buffers.input, input);

  printVariantTableHeader(n, cpuMs);
  for (const ScanVariant& variant : SCAN_VARIANTS) {
    const bool pass = measureAndCheckVariant(variant, buffers, reference, cpuMs);
    allPass = allPass && pass;
  }
  const bool exclusivePass = checkExclusiveScan(buffers, reference);
  allPass = allPass && exclusivePass;

  printEmulationNote();
  std::printf("\nTotal of all elements: %d\n", reference[n - 1]);
  std::printf("%s\n", allPass ? "ALL PASS" : "SOME CHECKS FAILED");

  freeScanBuffers(buffers);
  return allPass ? 0 : 1;
}
