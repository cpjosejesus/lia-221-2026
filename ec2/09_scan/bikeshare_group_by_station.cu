#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_compat.h"

constexpr int TRIPS_PER_BLOCK = 256;
constexpr int BK_THREADS = 256;
constexpr int BK_SECTION = 2 * BK_THREADS;
constexpr int MAX_STATIONS = BK_SECTION;

constexpr int OFFSET_THREADS = 256;
constexpr int STATION_THREADS = 64;
constexpr int MAX_HISTOGRAM_BLOCKS = BY_BACKEND(1024, 16);

#ifndef CPU_TRIPS
#define CPU_TRIPS 60000
#endif
constexpr int DEFAULT_TRIPS = BY_BACKEND(4000000, CPU_TRIPS);
constexpr int STATIONS = 48;
constexpr int TIMED_RUNS = BY_BACKEND(5, 1);
constexpr int WARMUP_RUNS = BY_BACKEND(1, 0);
constexpr uint64_t TRIP_LOG_SEED = 17;

constexpr int POPULARITY_SCALE = 1200;
constexpr int QUIET_STATION_WEIGHT = 15;

constexpr int MS_PER_SECOND = 1000;
constexpr int MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr int MS_PER_HOUR = 60 * MS_PER_MINUTE;
constexpr int HOURS_PER_DAY = 24;
constexpr int HOUR_WEIGHTS[HOURS_PER_DAY] = {1, 1, 1, 1, 1, 2,  5,  12, 20, 12, 6, 5,
                                             7, 7, 5, 5, 7, 13, 20, 13, 7,  4,  3, 2};

constexpr int BUSIEST_STATIONS_SHOWN = 5;
constexpr int FIRST_TRIPS_SHOWN = 3;

static_assert(STATIONS <= MAX_STATIONS,
              "the starts of all stations are scanned by one Brent-Kung block");

__device__ void brent_kung_in_shared(int* XY) {
  const int t = threadIdx.x;
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
}

__global__ void brent_kung_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int XY[BK_SECTION];
  const int t = threadIdx.x;
  const int base = blockIdx.x * BK_SECTION;
  XY[t] = (base + t < n) ? X[base + t] : 0;
  XY[t + BK_THREADS] = (base + t + BK_THREADS < n) ? X[base + t + BK_THREADS] : 0;

  brent_kung_in_shared(XY);

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

__host__ __device__ inline int exclusive_from_inclusive(const int* inclusive, int i) {
  return (i > 0) ? inclusive[i - 1] : 0;
}

__host__ __device__ inline int station_major_cell(int station, int block, int blocks) {
  return station * blocks + block;
}

__device__ void clear_histogram(int* histogram, int stations) {
  for (int s = threadIdx.x; s < stations; s += blockDim.x) {
    histogram[s] = 0;
  }
}

__global__ void count_per_station(const int* startStation, int* counts, int n, int stations) {
  __shared__ int histogram[MAX_STATIONS];
  clear_histogram(histogram, stations);
  __syncthreads();

  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    atomicAdd(&histogram[startStation[i]], 1);
  }
  __syncthreads();

  for (int s = threadIdx.x; s < stations; s += blockDim.x) {
    if (histogram[s] > 0) {
      atomicAdd(&counts[s], histogram[s]);
    }
  }
}

__global__ void bucket_starts(const int* counts, int* start, int stations) {
  __shared__ int XY[BK_SECTION];
  const int t = threadIdx.x;
  XY[t] = (t < stations) ? counts[t] : 0;
  XY[t + BK_THREADS] = (t + BK_THREADS < stations) ? counts[t + BK_THREADS] : 0;

  brent_kung_in_shared(XY);

  if (t == 0) {
    start[0] = 0;
  }
  if (t < stations) {
    start[t + 1] = XY[t];
  }
  if (t + BK_THREADS < stations) {
    start[t + BK_THREADS + 1] = XY[t + BK_THREADS];
  }
}

__global__ void scatter_by_cursor(const int* startStation, int* cursor, int* grouped, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int slot = atomicAdd(&cursor[startStation[i]], 1);
    grouped[slot] = i;
  }
}

__global__ void block_histograms(const int* startStation, int* counts, int n, int stations) {
  __shared__ int histogram[MAX_STATIONS];
  clear_histogram(histogram, stations);
  __syncthreads();

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&histogram[startStation[i]], 1);
  }
  __syncthreads();

  for (int s = threadIdx.x; s < stations; s += blockDim.x) {
    counts[station_major_cell(s, blockIdx.x, gridDim.x)] = histogram[s];
  }
}

__device__ int earlier_trips_from_same_station_in_block(const int* stationOfThread, int t) {
  int rank = 0;
  for (int j = 0; j < t; ++j) {
    if (stationOfThread[j] == stationOfThread[t]) {
      ++rank;
    }
  }
  return rank;
}

__global__ void stable_scatter(const int* startStation, const int* scanned, int* grouped, int n) {
  __shared__ int stationOfThread[TRIPS_PER_BLOCK];
  const int t = threadIdx.x;
  const int i = blockIdx.x * TRIPS_PER_BLOCK + t;
  const int s = (i < n) ? startStation[i] : -1;
  stationOfThread[t] = s;
  __syncthreads();

  if (i < n) {
    const int rank = earlier_trips_from_same_station_in_block(stationOfThread, t);
    const int cell = station_major_cell(s, blockIdx.x, gridDim.x);
    grouped[exclusive_from_inclusive(scanned, cell) + rank] = i;
  }
}

__global__ void starts_from_matrix(const int* scanned, int* start, int stations, int blocks) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < stations) {
    start[s] = exclusive_from_inclusive(scanned, station_major_cell(s, 0, blocks));
  }
  if (s == 0) {
    start[stations] = scanned[station_major_cell(stations, 0, blocks) - 1];
  }
}

class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {
  }

  uint64_t next() {
    state_ += GOLDEN_RATIO_INCREMENT;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * FIRST_MIX_MULTIPLIER;
    z = (z ^ (z >> 27)) * SECOND_MIX_MULTIPLIER;
    return z ^ (z >> 31);
  }

  int below(int bound) {
    return static_cast<int>(next() % static_cast<uint64_t>(bound));
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

class WeightedPick {
 public:
  explicit WeightedPick(const std::vector<int>& weights) : cumulative_(weights.size()) {
    int total = 0;
    for (size_t k = 0; k < weights.size(); ++k) {
      total += weights[k];
      cumulative_[k] = total;
    }
  }

  int pick(SplitMix64& rng) const {
    const int ticket = rng.below(cumulative_.back());
    const auto chosen = std::upper_bound(cumulative_.begin(), cumulative_.end(), ticket);
    return static_cast<int>(chosen - cumulative_.begin());
  }

 private:
  std::vector<int> cumulative_;
};

struct TripLog {
  std::vector<int> startStation;
  std::vector<int> startMs;
};

struct Grouping {
  std::vector<int> start;
  std::vector<int> grouped;
};

struct DeviceBuffers {
  int* startStation;
  int* counts;
  int* start;
  int* cursor;
  int* grouped;
  int* matrix;
  int* scannedMatrix;
  int* scratch;
};

int blocksFor(int n, int blockSize) {
  return static_cast<int>(cdiv(n, blockSize));
}

int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(int)));
  return pointer;
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

std::vector<int> stationPopularity() {
  std::vector<int> weights(STATIONS);
  for (int s = 0; s < STATIONS; ++s) {
    weights[s] = POPULARITY_SCALE / (s + 1) + QUIET_STATION_WEIGHT;
  }
  return weights;
}

std::vector<int> sortedStartTimes(SplitMix64& rng, int trips) {
  const WeightedPick pickHour(std::vector<int>(std::begin(HOUR_WEIGHTS), std::end(HOUR_WEIGHTS)));
  std::vector<int> startMs(trips);
  for (int i = 0; i < trips; ++i) {
    startMs[i] = pickHour.pick(rng) * MS_PER_HOUR + rng.below(MS_PER_HOUR);
  }
  std::sort(startMs.begin(), startMs.end());
  return startMs;
}

TripLog makeTripLog(int trips) {
  SplitMix64 rng(TRIP_LOG_SEED);
  const WeightedPick pickStation(stationPopularity());

  TripLog log;
  log.startMs = sortedStartTimes(rng, trips);
  log.startStation.resize(trips);
  for (int i = 0; i < trips; ++i) {
    log.startStation[i] = pickStation.pick(rng);
  }
  return log;
}

Grouping cpuStableCountingSort(const std::vector<int>& startStation) {
  const int trips = static_cast<int>(startStation.size());
  std::vector<int> counts(STATIONS, 0);
  for (int s : startStation) {
    ++counts[s];
  }

  Grouping grouping;
  grouping.start.assign(STATIONS + 1, 0);
  for (int s = 0; s < STATIONS; ++s) {
    grouping.start[s + 1] = grouping.start[s] + counts[s];
  }

  std::vector<int> cursor(grouping.start.begin(), grouping.start.end() - 1);
  grouping.grouped.resize(trips);
  for (int i = 0; i < trips; ++i) {
    grouping.grouped[cursor[startStation[i]]++] = i;
  }
  return grouping;
}

DeviceBuffers allocateDeviceBuffers(int trips) {
  const int matrixCells = STATIONS * blocksFor(trips, TRIPS_PER_BLOCK);
  DeviceBuffers buffers;
  buffers.startStation = allocateDeviceInts(trips);
  buffers.counts = allocateDeviceInts(STATIONS);
  buffers.start = allocateDeviceInts(STATIONS + 1);
  buffers.cursor = allocateDeviceInts(STATIONS + 1);
  buffers.grouped = allocateDeviceInts(trips);
  buffers.matrix = allocateDeviceInts(matrixCells);
  buffers.scannedMatrix = allocateDeviceInts(matrixCells);
  buffers.scratch = allocateDeviceInts(scratchIntsForHierarchicalScan(matrixCells));
  return buffers;
}

void freeDeviceBuffers(const DeviceBuffers& buffers) {
  CHECK(cudaFree(buffers.startStation));
  CHECK(cudaFree(buffers.counts));
  CHECK(cudaFree(buffers.start));
  CHECK(cudaFree(buffers.cursor));
  CHECK(cudaFree(buffers.grouped));
  CHECK(cudaFree(buffers.matrix));
  CHECK(cudaFree(buffers.scannedMatrix));
  CHECK(cudaFree(buffers.scratch));
}

void groupWithAtomicCursor(const DeviceBuffers& buffers, int trips) {
  const int tripBlocks = blocksFor(trips, TRIPS_PER_BLOCK);
  const int histogramBlocks = imin(tripBlocks, MAX_HISTOGRAM_BLOCKS);
  const size_t startBytes = (STATIONS + 1) * sizeof(int);

  CHECK(cudaMemset(buffers.counts, 0, STATIONS * sizeof(int)));
  launch(count_per_station,
         dim3(histogramBlocks),
         dim3(TRIPS_PER_BLOCK),
         buffers.startStation,
         buffers.counts,
         trips,
         STATIONS);
  launch(bucket_starts, dim3(1), dim3(BK_THREADS), buffers.counts, buffers.start, STATIONS);
  CHECK(cudaMemcpy(buffers.cursor, buffers.start, startBytes, cudaMemcpyDeviceToDevice));
  launch(scatter_by_cursor,
         dim3(tripBlocks),
         dim3(TRIPS_PER_BLOCK),
         buffers.startStation,
         buffers.cursor,
         buffers.grouped,
         trips);
}

void groupStably(const DeviceBuffers& buffers, int trips) {
  const int tripBlocks = blocksFor(trips, TRIPS_PER_BLOCK);
  const int matrixCells = STATIONS * tripBlocks;

  launch(block_histograms,
         dim3(tripBlocks),
         dim3(TRIPS_PER_BLOCK),
         buffers.startStation,
         buffers.matrix,
         trips,
         STATIONS);
  scanInclusive(buffers.matrix, buffers.scannedMatrix, matrixCells, buffers.scratch);
  launch(stable_scatter,
         dim3(tripBlocks),
         dim3(TRIPS_PER_BLOCK),
         buffers.startStation,
         buffers.scannedMatrix,
         buffers.grouped,
         trips);
  launch(starts_from_matrix,
         dim3(blocksFor(STATIONS, STATION_THREADS)),
         dim3(STATION_THREADS),
         buffers.scannedMatrix,
         buffers.start,
         STATIONS,
         tripBlocks);
}

void fillWithMinusOne(int* d_values, int count) {
  CHECK(cudaMemset(d_values, 0xff, count * sizeof(int)));
}

template <class Work>
double averageMs(int runs, Work work) {
  Timer timer;
  for (int run = 0; run < runs; ++run) {
    work();
  }
  return timer.ms() / runs;
}

template <class Work>
double measureGrouping(Work work, const DeviceBuffers& buffers, int trips) {
  for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
    work();
  }
  fillWithMinusOne(buffers.grouped, trips);
  CHECK(cudaDeviceSynchronize());
  const double ms = averageMs(TIMED_RUNS, work);
  CHECK(cudaGetLastError());
  return ms;
}

Grouping readGrouping(const DeviceBuffers& buffers, int trips) {
  Grouping grouping;
  grouping.start.resize(STATIONS + 1);
  grouping.grouped.resize(trips);
  CHECK(cudaMemcpy(
      grouping.start.data(), buffers.start, (STATIONS + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(
      grouping.grouped.data(), buffers.grouped, trips * sizeof(int), cudaMemcpyDeviceToHost));
  return grouping;
}

std::vector<int> sliceOf(const Grouping& grouping, int station) {
  return std::vector<int>(grouping.grouped.begin() + grouping.start[station],
                          grouping.grouped.begin() + grouping.start[station + 1]);
}

struct SliceCheck {
  bool sameTripsPerSlice;
  int slicesByStartTime;
};

SliceCheck compareSlicesAsSets(const Grouping& actual, const Grouping& expected) {
  SliceCheck check = {actual.start == expected.start, 0};
  for (int s = 0; s < STATIONS && check.sameTripsPerSlice; ++s) {
    std::vector<int> slice = sliceOf(actual, s);
    if (std::is_sorted(slice.begin(), slice.end())) {
      ++check.slicesByStartTime;
    }
    std::sort(slice.begin(), slice.end());
    check.sameTripsPerSlice = (slice == sliceOf(expected, s));
  }
  return check;
}

const char* verdict(bool pass) {
  return pass ? "PASS" : "FAIL";
}

void printClock(int ms) {
  std::printf("%02d:%02d:%02d", ms / MS_PER_HOUR, ms / MS_PER_MINUTE % 60, ms / MS_PER_SECOND % 60);
}

int sliceLength(const Grouping& grouping, int station) {
  return grouping.start[station + 1] - grouping.start[station];
}

std::vector<int> stationsByTripCount(const Grouping& grouping) {
  std::vector<int> stations(STATIONS);
  for (int s = 0; s < STATIONS; ++s) {
    stations[s] = s;
  }
  std::stable_sort(stations.begin(), stations.end(), [&](int a, int b) {
    return sliceLength(grouping, a) > sliceLength(grouping, b);
  });
  return stations;
}

void printBusiestStations(const Grouping& expected, const Grouping& grouping, const TripLog& log) {
  const std::vector<int> busiest = stationsByTripCount(expected);
  std::printf("\nBusiest start stations (slice of the grouped array from version B):\n");
  for (int k = 0; k < BUSIEST_STATIONS_SHOWN; ++k) {
    const int s = busiest[k];
    const int first = grouping.start[s];
    const int shownEnd = imin(first + FIRST_TRIPS_SHOWN, grouping.start[s + 1]);
    std::printf("  s%-3d %7d trips  start %8d  first:", s, sliceLength(grouping, s), first);
    for (int j = first; j < shownEnd; ++j) {
      const int trip = grouping.grouped[j];
      std::printf("  t%d @ ", trip);
      printClock(log.startMs[trip]);
    }
    std::printf("\n");
  }
}

void printEmulationNote() {
#ifndef __CUDACC__
  std::printf("\nNote: CPU emulation times measure thread barriers, not a GPU.\n");
#endif
}

int tripCount(int argc, char** argv) {
  return (argc > 1) ? std::atoi(argv[1]) : DEFAULT_TRIPS;
}

void printAtomicRow(double cpuMs, double atomicMs, const SliceCheck& check) {
  std::printf("  %-40s %10.3f %7.2fx  %s (same trips per slice; %d of %d slices by start time)\n",
              "A  histogram + scan + atomic cursor",
              atomicMs,
              cpuMs / atomicMs,
              verdict(check.sameTripsPerSlice),
              check.slicesByStartTime,
              STATIONS);
}

void printStableRow(double cpuMs, double stableMs, bool pass) {
  std::printf("  %-40s %10.3f %7.2fx  %s (identical to the CPU, every slice by start time)\n",
              "B  block histograms + scan + stable rank",
              stableMs,
              cpuMs / stableMs,
              verdict(pass));
}

int main(int argc, char** argv) {
  const int trips = tripCount(argc, argv);
  if (trips < 1) {
    std::fprintf(stderr, "need at least one trip\n");
    return 1;
  }
  std::printf(
      "Group trips by start station (counting sort = histogram + exclusive scan + scatter)\n");
  std::printf("backend: %s, %d trips, %d stations\n\n", CUDA_BACKEND, trips, STATIONS);

  const TripLog log = makeTripLog(trips);
  Grouping expected;
  const double cpuMs = averageMs(TIMED_RUNS, [&] {
    expected = cpuStableCountingSort(log.startStation);
  });

  const DeviceBuffers buffers = allocateDeviceBuffers(trips);
  CHECK(cudaMemcpy(
      buffers.startStation, log.startStation.data(), trips * sizeof(int), cudaMemcpyHostToDevice));

  std::printf("  %-40s %10s %8s  %s\n", "version", "time [ms]", "vs CPU", "check");
  std::printf(
      "  %-40s %10.3f %8s  %s\n", "CPU stable counting sort (reference)", cpuMs, "1.00x", "-");

  const double atomicMs = measureGrouping(
      [&] {
        groupWithAtomicCursor(buffers, trips);
      },
      buffers,
      trips);
  const SliceCheck atomicCheck = compareSlicesAsSets(readGrouping(buffers, trips), expected);
  printAtomicRow(cpuMs, atomicMs, atomicCheck);

  const double stableMs = measureGrouping(
      [&] {
        groupStably(buffers, trips);
      },
      buffers,
      trips);
  const Grouping stable = readGrouping(buffers, trips);
  const bool stablePass = (stable.start == expected.start) && (stable.grouped == expected.grouped);
  printStableRow(cpuMs, stableMs, stablePass);

  printBusiestStations(expected, stable, log);
  printEmulationNote();

  const bool allPass = atomicCheck.sameTripsPerSlice && stablePass;
  std::printf("\n%s\n", allPass ? "ALL PASS" : "SOME CHECKS FAILED");

  freeDeviceBuffers(buffers);
  return allPass ? 0 : 1;
}
