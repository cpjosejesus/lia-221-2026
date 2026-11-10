#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_compat.h"

constexpr int BK_THREADS = 256;
constexpr int BK_SECTION = 2 * BK_THREADS;
constexpr int F_THREADS = 256;
constexpr int F_COARSEN = 8;
constexpr int F_SECTION = F_THREADS * F_COARSEN;

constexpr int OFFSET_THREADS = 256;
constexpr int SIMPLE_THREADS = 256;
constexpr int GATHER_THREADS = 256;

#ifndef CPU_TRIPS
#define CPU_TRIPS 60000
#endif
constexpr int DEFAULT_TRIPS = BY_BACKEND(4000000, CPU_TRIPS);
constexpr int TIMED_RUNS = BY_BACKEND(5, 1);
constexpr int WARMUP_RUNS = BY_BACKEND(1, 0);
constexpr uint64_t TRIP_LOG_SEED = 7;

constexpr int MS_PER_SECOND = 1000;
constexpr int MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr int MS_PER_HOUR = 60 * MS_PER_MINUTE;
constexpr int INCLUDED_MINUTES = 45;
constexpr int INCLUDED_MS = INCLUDED_MINUTES * MS_PER_MINUTE;
constexpr int EVENING_RUSH_START_MS = 16 * MS_PER_HOUR;
constexpr int EVENING_RUSH_LENGTH_MS = 4 * MS_PER_HOUR;

constexpr int PERCENT = 100;
constexpr int LEISURE_RIDE_PERCENT = 15;
constexpr int SHORTEST_COMMUTE_MS = 6 * MS_PER_MINUTE;
constexpr int COMMUTE_PARTS = 4;
constexpr int LONGEST_COMMUTE_PART_MS = 10 * MS_PER_MINUTE;
constexpr int SHORTEST_LEISURE_MS = 30 * MS_PER_MINUTE;
constexpr int LONGEST_LEISURE_MS = 150 * MS_PER_MINUTE;

constexpr int BASE_FARE_CENTS = 300;
constexpr int FARE_PER_STARTED_BLOCK_CENTS = 100;
constexpr int FARE_BLOCK_MS = 10 * MS_PER_MINUTE;
constexpr int CENTS_PER_BOLIVIANO = 100;

constexpr int STATIONS = 48;
constexpr int BIKES = 1500;
constexpr int REPORT_LINES_SHOWN = 6;

__host__ __device__ inline bool isLate(int startMs, int endMs, int includedMs) {
  return endMs - startMs > includedMs;
}

__host__ __device__ inline int fareCents(int rideMs) {
  const int startedBlocks = (rideMs + FARE_BLOCK_MS - 1) / FARE_BLOCK_MS;
  return BASE_FARE_CENTS + FARE_PER_STARTED_BLOCK_CENTS * startedBlocks;
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

__host__ __device__ inline int exclusive_from_inclusive(const int* inclusive, int i) {
  return (i > 0) ? inclusive[i - 1] : 0;
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

__device__ int count_own_flags(const int* flag, int first) {
  int count = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    count += flag[first + k];
  }
  return count;
}

__device__ void pack_own_flagged_indices(
    const int* flag, int first, int base, int pos, int* stage) {
  for (int k = 0; k < F_COARSEN; ++k) {
    if (flag[first + k]) {
      stage[pos++] = base + first + k;
    }
  }
}

__device__ void copy_out_coalesced(const int* stage, int count, int* out) {
  for (int k = threadIdx.x; k < count; k += F_THREADS) {
    out[k] = stage[k];
  }
}

__global__ void count_late(
    const int* startMs, const int* endMs, int* counts, int n, int includedMs) {
  __shared__ int blockCount;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  int c = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    const int i = blockIdx.x * F_SECTION + k * F_THREADS + threadIdx.x;
    if (i < n && isLate(startMs[i], endMs[i], includedMs)) {
      ++c;
    }
  }
  atomicAdd(&blockCount, c);
  __syncthreads();

  if (threadIdx.x == 0) {
    counts[blockIdx.x] = blockCount;
  }
}

__global__ void compact_late(const int* startMs,
                             const int* endMs,
                             const int* scannedCounts,
                             int* lateIdx,
                             int n,
                             int includedMs) {
  __shared__ int late[F_SECTION];
  __shared__ int stage[F_SECTION];
  __shared__ int part[F_THREADS];
  const int t = threadIdx.x;
  const int base = blockIdx.x * F_SECTION;
  const int first = t * F_COARSEN;

  for (int k = 0; k < F_COARSEN; ++k) {
    const int j = k * F_THREADS + t;
    const int i = base + j;
    late[j] = (i < n && isLate(startMs[i], endMs[i], includedMs)) ? 1 : 0;
  }
  __syncthreads();

  part[t] = count_own_flags(late, first);
  kogge_stone_in_shared(part);
  __syncthreads();

  pack_own_flagged_indices(late, first, base, exclusive_from_inclusive(part, t), stage);
  __syncthreads();

  const int blockCount = part[F_THREADS - 1];
  const int start = exclusive_from_inclusive(scannedCounts, blockIdx.x);
  copy_out_coalesced(stage, blockCount, lateIdx + start);
}

__global__ void gather_report(const int* lateIdx,
                              int count,
                              const int* tripId,
                              const int* bikeId,
                              const int* startStation,
                              const int* startMs,
                              const int* endMs,
                              int* repTrip,
                              int* repBike,
                              int* repStation,
                              int* repEnd,
                              int* repDuration,
                              int* repFare) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < count) {
    const int i = lateIdx[k];
    const int duration = endMs[i] - startMs[i];
    repTrip[k] = tripId[i];
    repBike[k] = bikeId[i];
    repStation[k] = startStation[i];
    repEnd[k] = endMs[i];
    repDuration[k] = duration;
    repFare[k] = fareCents(duration);
  }
}

__global__ void late_by_atomics(
    const int* startMs, const int* endMs, int* lateIdx, int* counter, int n, int includedMs) {
  __shared__ int blockCount;
  __shared__ int blockStart;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  const bool k = i < n && isLate(startMs[i], endMs[i], includedMs);
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
    lateIdx[blockStart + local] = i;
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

  int between(int lowest, int highest) {
    return lowest + below(highest - lowest + 1);
  }

  bool percentChance(int percent) {
    return below(PERCENT) < percent;
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

struct Trip {
  int tripId;
  int bikeId;
  int startStation;
  int startMs;
  int endMs;
};

bool dockedEarlier(const Trip& a, const Trip& b) {
  if (a.endMs != b.endMs) {
    return a.endMs < b.endMs;
  }
  return a.tripId < b.tripId;
}

struct TripLog {
  std::vector<int> tripId;
  std::vector<int> bikeId;
  std::vector<int> startStation;
  std::vector<int> startMs;
  std::vector<int> endMs;
};

struct ReportTable {
  std::vector<int> lateIndex;
  std::vector<int> trip;
  std::vector<int> bike;
  std::vector<int> station;
  std::vector<int> endMs;
  std::vector<int> duration;
  std::vector<int> fare;
};

struct DeviceLog {
  int* tripId;
  int* bikeId;
  int* startStation;
  int* startMs;
  int* endMs;
};

struct DeviceReport {
  int* trip;
  int* bike;
  int* station;
  int* endMs;
  int* duration;
  int* fare;
};

struct DeviceWork {
  int* sectionCounts;
  int* scannedSectionCounts;
  int* scratch;
  int* lateIndex;
  int* counter;
};

int blocksFor(int n, int blockSize) {
  return static_cast<int>(cdiv(n, blockSize));
}

int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(int)));
  return pointer;
}

int* uploadInts(const std::vector<int>& values) {
  int* pointer = allocateDeviceInts(static_cast<int>(values.size()));
  CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice));
  return pointer;
}

std::vector<int> downloadInts(const int* source, int count) {
  std::vector<int> values(count);
  if (count > 0) {
    CHECK(cudaMemcpy(values.data(), source, count * sizeof(int), cudaMemcpyDeviceToHost));
  }
  return values;
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

int syntheticRideMs(SplitMix64& rng) {
  if (rng.percentChance(LEISURE_RIDE_PERCENT)) {
    return rng.between(SHORTEST_LEISURE_MS, LONGEST_LEISURE_MS);
  }
  int rideMs = SHORTEST_COMMUTE_MS;
  for (int part = 0; part < COMMUTE_PARTS; ++part) {
    rideMs += rng.below(LONGEST_COMMUTE_PART_MS + 1);
  }
  return rideMs;
}

TripLog makeTripLog(int trips) {
  SplitMix64 rng(TRIP_LOG_SEED);
  std::vector<Trip> rows(trips);
  for (int i = 0; i < trips; ++i) {
    const int startMs = EVENING_RUSH_START_MS + rng.below(EVENING_RUSH_LENGTH_MS);
    const int endMs = startMs + syntheticRideMs(rng);
    rows[i] = {i, rng.below(BIKES), rng.below(STATIONS), startMs, endMs};
  }
  std::sort(rows.begin(), rows.end(), dockedEarlier);

  TripLog log;
  for (const Trip& row : rows) {
    log.tripId.push_back(row.tripId);
    log.bikeId.push_back(row.bikeId);
    log.startStation.push_back(row.startStation);
    log.startMs.push_back(row.startMs);
    log.endMs.push_back(row.endMs);
  }
  return log;
}

std::vector<int> cpuLateIndices(const TripLog& log) {
  std::vector<int> lateIndices;
  lateIndices.reserve(log.startMs.size());
  for (size_t i = 0; i < log.startMs.size(); ++i) {
    if (isLate(log.startMs[i], log.endMs[i], INCLUDED_MS)) {
      lateIndices.push_back(static_cast<int>(i));
    }
  }
  return lateIndices;
}

DeviceLog uploadLog(const TripLog& log) {
  DeviceLog device;
  device.tripId = uploadInts(log.tripId);
  device.bikeId = uploadInts(log.bikeId);
  device.startStation = uploadInts(log.startStation);
  device.startMs = uploadInts(log.startMs);
  device.endMs = uploadInts(log.endMs);
  return device;
}

DeviceReport allocateDeviceReport(int trips) {
  DeviceReport report;
  report.trip = allocateDeviceInts(trips);
  report.bike = allocateDeviceInts(trips);
  report.station = allocateDeviceInts(trips);
  report.endMs = allocateDeviceInts(trips);
  report.duration = allocateDeviceInts(trips);
  report.fare = allocateDeviceInts(trips);
  return report;
}

DeviceWork allocateDeviceWork(int trips) {
  const int sections = blocksFor(trips, F_SECTION);
  DeviceWork work;
  work.sectionCounts = allocateDeviceInts(sections);
  work.scannedSectionCounts = allocateDeviceInts(sections);
  work.scratch = allocateDeviceInts(scratchIntsForHierarchicalScan(sections));
  work.lateIndex = allocateDeviceInts(trips);
  work.counter = allocateDeviceInts(1);
  return work;
}

void freeDevice(const DeviceLog& log, const DeviceReport& report, const DeviceWork& work) {
  int* const buffers[] = {log.tripId,
                          log.bikeId,
                          log.startStation,
                          log.startMs,
                          log.endMs,
                          work.sectionCounts,
                          work.scannedSectionCounts,
                          work.scratch,
                          work.lateIndex,
                          work.counter,
                          report.trip,
                          report.bike,
                          report.station,
                          report.endMs,
                          report.duration,
                          report.fare};
  for (int* buffer : buffers) {
    CHECK(cudaFree(buffer));
  }
}

int compactLateInLogSequence(const DeviceLog& log, const DeviceWork& work, int trips) {
  const int sections = blocksFor(trips, F_SECTION);
  launch(count_late,
         dim3(sections),
         dim3(F_THREADS),
         log.startMs,
         log.endMs,
         work.sectionCounts,
         trips,
         INCLUDED_MS);
  scanInclusive(work.sectionCounts, work.scannedSectionCounts, sections, work.scratch);
  launch(compact_late,
         dim3(sections),
         dim3(F_THREADS),
         log.startMs,
         log.endMs,
         work.scannedSectionCounts,
         work.lateIndex,
         trips,
         INCLUDED_MS);
  return copyIntToHost(work.scannedSectionCounts + (sections - 1));
}

void gatherReport(const DeviceLog& log,
                  const DeviceWork& work,
                  const DeviceReport& report,
                  int lateCount) {
  launch(gather_report,
         dim3(blocksFor(imax(lateCount, 1), GATHER_THREADS)),
         dim3(GATHER_THREADS),
         work.lateIndex,
         lateCount,
         log.tripId,
         log.bikeId,
         log.startStation,
         log.startMs,
         log.endMs,
         report.trip,
         report.bike,
         report.station,
         report.endMs,
         report.duration,
         report.fare);
}

int compactLateWithAtomics(const DeviceLog& log, const DeviceWork& work, int trips) {
  CHECK(cudaMemset(work.counter, 0, sizeof(int)));
  launch(late_by_atomics,
         dim3(blocksFor(trips, SIMPLE_THREADS)),
         dim3(SIMPLE_THREADS),
         log.startMs,
         log.endMs,
         work.lateIndex,
         work.counter,
         trips,
         INCLUDED_MS);
  return copyIntToHost(work.counter);
}

ReportTable downloadReport(const DeviceReport& report, const DeviceWork& work, int lateCount) {
  ReportTable table;
  table.lateIndex = downloadInts(work.lateIndex, lateCount);
  table.trip = downloadInts(report.trip, lateCount);
  table.bike = downloadInts(report.bike, lateCount);
  table.station = downloadInts(report.station, lateCount);
  table.endMs = downloadInts(report.endMs, lateCount);
  table.duration = downloadInts(report.duration, lateCount);
  table.fare = downloadInts(report.fare, lateCount);
  return table;
}

bool rowMatchesLog(const ReportTable& table, int k, int i, const TripLog& log) {
  const int duration = log.endMs[i] - log.startMs[i];
  return table.lateIndex[k] == i && table.trip[k] == log.tripId[i] &&
         table.bike[k] == log.bikeId[i] && table.station[k] == log.startStation[i] &&
         table.endMs[k] == log.endMs[i] && table.duration[k] == duration &&
         table.fare[k] == fareCents(duration);
}

bool reportMatchesLog(const ReportTable& table,
                      const std::vector<int>& expectedIndices,
                      const TripLog& log) {
  for (size_t k = 0; k < table.lateIndex.size(); ++k) {
    if (!rowMatchesLog(table, static_cast<int>(k), expectedIndices[k], log)) {
      return false;
    }
  }
  return true;
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
void warmUp(Work work) {
  for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
    work();
  }
}

template <class Work>
double timeRuns(Work work) {
  CHECK(cudaDeviceSynchronize());
  const double ms = averageMs(TIMED_RUNS, work);
  CHECK(cudaGetLastError());
  return ms;
}

void fillWithMinusOne(int* d_values, int count) {
  CHECK(cudaMemset(d_values, 0xff, count * sizeof(int)));
}

const char* yesNo(bool value) {
  return value ? "yes" : "no";
}

const char* verdict(bool pass) {
  return pass ? "PASS" : "FAIL";
}

int indexOfLargest(const std::vector<int>& values) {
  return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

void printClock(int ms) {
  std::printf("%02d:%02d:%02d", ms / MS_PER_HOUR, ms / MS_PER_MINUTE % 60, ms / MS_PER_SECOND % 60);
}

void printReportLine(const ReportTable& table, int k) {
  std::printf(
      "  trip t%-6d bike b%-5d from s%-3d docked ", table.trip[k], table.bike[k], table.station[k]);
  printClock(table.endMs[k]);
  std::printf("  ride %3d min  fare Bs %d.%02d\n",
              table.duration[k] / MS_PER_MINUTE,
              table.fare[k] / CENTS_PER_BOLIVIANO,
              table.fare[k] % CENTS_PER_BOLIVIANO);
}

void printWorstStationAndBike(const ReportTable& table) {
  std::vector<int> lateByStation(STATIONS, 0);
  std::vector<int> lateByBike(BIKES, 0);
  for (size_t k = 0; k < table.trip.size(); ++k) {
    ++lateByStation[table.station[k]];
    ++lateByBike[table.bike[k]];
  }
  const int worstStation = indexOfLargest(lateByStation);
  const int worstBike = indexOfLargest(lateByBike);
  std::printf("  most late returns: start station s%d (%d), bike b%d (%d)\n",
              worstStation,
              lateByStation[worstStation],
              worstBike,
              lateByBike[worstBike]);
}

void printReport(const ReportTable& table, int trips) {
  const int lateCount = static_cast<int>(table.trip.size());
  std::printf("\n%d of %d trips over the included %d min (%.1f %%). First lines, by dock time:\n",
              lateCount,
              trips,
              INCLUDED_MINUTES,
              100.0 * lateCount / trips);
  for (int k = 0; k < imin(lateCount, REPORT_LINES_SHOWN); ++k) {
    printReportLine(table, k);
  }
  printWorstStationAndBike(table);
}

void printEmulationNote() {
#ifndef __CUDACC__
  std::printf("\nNote: CPU emulation times measure thread barriers, not a GPU.\n");
#endif
}

int tripCount(int argc, char** argv) {
  return (argc > 1) ? std::atoi(argv[1]) : DEFAULT_TRIPS;
}

void printResultTable(double cpuMs,
                      int expectedCount,
                      double sequenceMs,
                      int lateCount,
                      bool sequencePass,
                      double atomicMs,
                      int atomicCount,
                      bool atomicInLogSequence,
                      bool atomicPass) {
  std::printf("  %-43s %10s %8s %8s %9s  %s\n",
              "version",
              "time [ms]",
              "vs CPU",
              "late",
              "by time",
              "check");
  std::printf("  %-43s %10.3f %8s %8d %9s  %s\n",
              "CPU loop (reference)",
              cpuMs,
              "1.00x",
              expectedCount,
              "yes",
              "-");
  std::printf("  %-43s %10.3f %7.2fx %8d %9s  %s\n",
              "GPU count + scan + staged compact + gather",
              sequenceMs,
              cpuMs / sequenceMs,
              lateCount,
              yesNo(sequencePass),
              verdict(sequencePass));
  std::printf("  %-43s %10.3f %7.2fx %8d %9s  %s\n",
              "GPU block-private atomics (no gather)",
              atomicMs,
              cpuMs / atomicMs,
              atomicCount,
              yesNo(atomicInLogSequence),
              verdict(atomicPass));
}

int main(int argc, char** argv) {
  const int trips = tripCount(argc, argv);
  if (trips < 1) {
    std::fprintf(stderr, "need at least one trip\n");
    return 1;
  }
  std::printf("Late returns report (sequence-preserving filter + gather)\n");
  std::printf("backend: %s, %d finished trips, late = more than %d min\n\n",
              CUDA_BACKEND,
              trips,
              INCLUDED_MINUTES);

  const TripLog log = makeTripLog(trips);
  std::vector<int> expectedIndices;
  const double cpuMs = averageMs(TIMED_RUNS, [&] {
    expectedIndices = cpuLateIndices(log);
  });
  const int expectedCount = static_cast<int>(expectedIndices.size());

  const DeviceLog deviceLog = uploadLog(log);
  const DeviceReport deviceReport = allocateDeviceReport(trips);
  const DeviceWork work = allocateDeviceWork(trips);

  int lateCount = 0;
  auto runInSequence = [&] {
    lateCount = compactLateInLogSequence(deviceLog, work, trips);
    gatherReport(deviceLog, work, deviceReport, lateCount);
  };
  warmUp(runInSequence);
  fillWithMinusOne(work.lateIndex, trips);
  const double sequenceMs = timeRuns(runInSequence);

  bool sequencePass = (lateCount == expectedCount);
  ReportTable table;
  if (sequencePass && lateCount > 0) {
    table = downloadReport(deviceReport, work, lateCount);
    sequencePass = reportMatchesLog(table, expectedIndices, log);
  }

  int atomicCount = 0;
  auto runWithAtomics = [&] {
    atomicCount = compactLateWithAtomics(deviceLog, work, trips);
  };
  warmUp(runWithAtomics);
  const double atomicMs = timeRuns(runWithAtomics);
  bool atomicPass = (atomicCount == expectedCount);
  bool atomicInLogSequence = false;
  if (atomicPass && atomicCount > 0) {
    std::vector<int> found = downloadInts(work.lateIndex, atomicCount);
    atomicInLogSequence = (found == expectedIndices);
    std::sort(found.begin(), found.end());
    atomicPass = (found == expectedIndices);
  }

  printResultTable(cpuMs,
                   expectedCount,
                   sequenceMs,
                   lateCount,
                   sequencePass,
                   atomicMs,
                   atomicCount,
                   atomicInLogSequence,
                   atomicPass);

  if (sequencePass && lateCount > 0) {
    printReport(table, trips);
  }
  printEmulationNote();

  const bool allPass = sequencePass && atomicPass;
  std::printf("\n%s\n", allPass ? "ALL PASS" : "SOME CHECKS FAILED");

  freeDevice(deviceLog, deviceReport, work);
  return allPass ? 0 : 1;
}
