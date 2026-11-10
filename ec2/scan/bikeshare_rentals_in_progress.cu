#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_compat.h"

constexpr int CO_THREADS = 256;
constexpr int CO_COARSEN = 8;
constexpr int CO_SECTION = CO_THREADS * CO_COARSEN;

constexpr int OFFSET_THREADS = 256;
constexpr int REDUCTION_THREADS = 256;
constexpr int MAX_REDUCTION_BLOCKS = BY_BACKEND(1024, 8);

#ifndef CPU_TRIPS
#define CPU_TRIPS 40000
#endif
constexpr int DEFAULT_TRIPS = BY_BACKEND(4000000, CPU_TRIPS);
constexpr int TIMED_RUNS = BY_BACKEND(5, 1);
constexpr int WARMUP_RUNS = BY_BACKEND(1, 0);
constexpr uint64_t TRIP_LOG_SEED = 42;

constexpr int MS_PER_SECOND = 1000;
constexpr int MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr int MS_PER_HOUR = 60 * MS_PER_MINUTE;
constexpr int HOURS_PER_DAY = 24;
constexpr int DAY_MS = HOURS_PER_DAY * MS_PER_HOUR;
constexpr int END_OF_RUN_MS = DAY_MS + 3 * MS_PER_HOUR;

constexpr int PERCENT = 100;
constexpr int FAULTY_UNDOCK_PERCENT = 3;
constexpr int LEISURE_RIDE_PERCENT = 12;
constexpr int MAX_FAULTY_CHECK_SECONDS = 90;
constexpr int MIN_COMMUTE_MINUTES = 4;
constexpr int MAX_COMMUTE_MINUTES = 35;
constexpr int MIN_LEISURE_MINUTES = 30;
constexpr int MAX_LEISURE_MINUTES = 150;

constexpr int HOUR_WEIGHTS[HOURS_PER_DAY] = {1, 1, 1, 1, 1, 2,  5,  12, 20, 12, 6, 5,
                                             7, 7, 5, 5, 7, 13, 20, 13, 7,  4,  3, 2};

constexpr int BIKE_LEAVES_DOCK = +1;
constexpr int BIKE_RETURNS_TO_DOCK = -1;
constexpr int BAR_LENGTH_AT_PEAK = 50;

__device__ void load_section_coalesced(const int* X, int n, int base, int* XY) {
  for (int k = 0; k < CO_COARSEN; ++k) {
    const int j = k * CO_THREADS + threadIdx.x;
    XY[j] = (base + j < n) ? X[base + j] : 0;
  }
}

__device__ void store_section_coalesced(const int* XY, int n, int base, int* Y) {
  for (int k = 0; k < CO_COARSEN; ++k) {
    const int j = k * CO_THREADS + threadIdx.x;
    if (base + j < n) {
      Y[base + j] = XY[j];
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

__global__ void coarsened_scan(const int* X, int* Y, int* sums, int n) {
  __shared__ int XY[CO_SECTION];
  __shared__ int part[CO_THREADS];
  const int t = threadIdx.x;
  const int base = blockIdx.x * CO_SECTION;
  const int first = t * CO_COARSEN;

  load_section_coalesced(X, n, base, XY);
  __syncthreads();

  phase1_scan_own_subsection(XY, first);
  part[t] = XY[first + CO_COARSEN - 1];

  phase2_kogge_stone_over_subsection_totals(part);
  __syncthreads();

  phase3_add_total_of_subsections_to_the_left(XY, part, first);
  __syncthreads();

  store_section_coalesced(XY, n, base, Y);
  if (t == 0) {
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

__global__ void max_value(const int* v, int* result, int n) {
  __shared__ int blockMax;
  if (threadIdx.x == 0) {
    blockMax = INT_MIN;
  }
  __syncthreads();

  int threadMax = INT_MIN;
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    threadMax = imax(threadMax, v[i]);
  }
  atomicMax(&blockMax, threadMax);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMax(result, blockMax);
  }
}

__global__ void first_index_of(const int* v, int value, int* result, int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    if (v[i] == value) {
      atomicMin(result, i);
    }
  }
}

enum class DockEventKind { Undock, Dock };

struct DockEvent {
  int timeMs;
  DockEventKind kind;
  int delta;
};

bool happensBefore(const DockEvent& a, const DockEvent& b) {
  if (a.timeMs != b.timeMs) {
    return a.timeMs < b.timeMs;
  }
  return a.kind < b.kind;
}

struct EventStream {
  std::vector<int> deltas;
  std::vector<int> timesMs;
  int faultyUndocks;
};

struct PeakLoad {
  int rentalsOut;
  int firstEvent;
};

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

int blocksFor(int n, int blockSize) {
  return static_cast<int>(cdiv(n, blockSize));
}

int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(int)));
  return pointer;
}

void copyIntToDevice(int* destination, int value) {
  CHECK(cudaMemcpy(destination, &value, sizeof(int), cudaMemcpyHostToDevice));
}

int copyIntToHost(const int* source) {
  int value = 0;
  CHECK(cudaMemcpy(&value, source, sizeof(int), cudaMemcpyDeviceToHost));
  return value;
}

int scratchIntsForHierarchicalScan(int n) {
  const int blocks = blocksFor(n, CO_SECTION);
  if (blocks == 1) {
    return 1;
  }
  return 2 * blocks + scratchIntsForHierarchicalScan(blocks);
}

void scanInclusive(const int* d_in, int* d_out, int n, int* scratch) {
  const int blocks = blocksFor(n, CO_SECTION);
  int* blockSums = scratch;
  launch(coarsened_scan, dim3(blocks), dim3(CO_THREADS), d_in, d_out, blockSums, n);

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
           CO_SECTION);
  }
}

int minutesToMs(int minutes) {
  return minutes * MS_PER_MINUTE;
}

class SyntheticDay {
 public:
  SyntheticDay() : rng_(TRIP_LOG_SEED), hourTotal_(0) {
    for (int hour = 0; hour < HOURS_PER_DAY; ++hour) {
      hourTotal_ += HOUR_WEIGHTS[hour];
      cumulativeHourWeight_[hour] = hourTotal_;
    }
  }

  void addTrip(std::vector<DockEvent>& events, int& faultyUndocks) {
    const int undockMs = pickHour() * MS_PER_HOUR + rng_.below(MS_PER_HOUR);
    int dockMs = 0;
    if (rng_.percentChance(FAULTY_UNDOCK_PERCENT)) {
      dockMs = undockMs + rng_.below(MAX_FAULTY_CHECK_SECONDS * MS_PER_SECOND);
      ++faultyUndocks;
    } else {
      dockMs = undockMs + rideMs();
    }

    events.push_back({undockMs, DockEventKind::Undock, BIKE_LEAVES_DOCK});
    events.push_back({imin(dockMs, END_OF_RUN_MS), DockEventKind::Dock, BIKE_RETURNS_TO_DOCK});
  }

 private:
  int pickHour() {
    const int ticket = rng_.below(hourTotal_);
    int hour = 0;
    while (cumulativeHourWeight_[hour] <= ticket) {
      ++hour;
    }
    return hour;
  }

  int rideMs() {
    if (rng_.percentChance(LEISURE_RIDE_PERCENT)) {
      return rng_.between(minutesToMs(MIN_LEISURE_MINUTES), minutesToMs(MAX_LEISURE_MINUTES));
    }
    return rng_.between(minutesToMs(MIN_COMMUTE_MINUTES), minutesToMs(MAX_COMMUTE_MINUTES));
  }

  SplitMix64 rng_;
  int hourTotal_;
  int cumulativeHourWeight_[HOURS_PER_DAY];
};

EventStream makeEventStream(int trips) {
  SyntheticDay day;
  std::vector<DockEvent> events;
  events.reserve(2 * trips);
  int faultyUndocks = 0;
  for (int trip = 0; trip < trips; ++trip) {
    day.addTrip(events, faultyUndocks);
  }
  std::sort(events.begin(), events.end(), happensBefore);

  EventStream stream;
  stream.faultyUndocks = faultyUndocks;
  for (const DockEvent& event : events) {
    stream.deltas.push_back(event.delta);
    stream.timesMs.push_back(event.timeMs);
  }
  return stream;
}

PeakLoad cpuRunningSumAndPeak(const std::vector<int>& deltas, std::vector<int>& out) {
  PeakLoad peak = {INT_MIN, -1};
  int run = 0;
  for (size_t i = 0; i < deltas.size(); ++i) {
    run += deltas[i];
    out[i] = run;
    if (run > peak.rentalsOut) {
      peak.rentalsOut = run;
      peak.firstEvent = static_cast<int>(i);
    }
  }
  return peak;
}

struct DeviceBuffers {
  int* deltas;
  int* out;
  int* scratch;
  int* peak;
  int* peakEvent;
};

DeviceBuffers allocateDeviceBuffers(int events) {
  DeviceBuffers buffers;
  buffers.deltas = allocateDeviceInts(events);
  buffers.out = allocateDeviceInts(events);
  buffers.scratch = allocateDeviceInts(scratchIntsForHierarchicalScan(events));
  buffers.peak = allocateDeviceInts(1);
  buffers.peakEvent = allocateDeviceInts(1);
  return buffers;
}

void freeDeviceBuffers(const DeviceBuffers& buffers) {
  CHECK(cudaFree(buffers.deltas));
  CHECK(cudaFree(buffers.out));
  CHECK(cudaFree(buffers.scratch));
  CHECK(cudaFree(buffers.peak));
  CHECK(cudaFree(buffers.peakEvent));
}

PeakLoad gpuRentalsOutAndPeak(const DeviceBuffers& buffers, int events) {
  const int blocks = imin(blocksFor(events, REDUCTION_THREADS), MAX_REDUCTION_BLOCKS);
  scanInclusive(buffers.deltas, buffers.out, events, buffers.scratch);

  copyIntToDevice(buffers.peak, INT_MIN);
  copyIntToDevice(buffers.peakEvent, INT_MAX);
  launch(max_value, dim3(blocks), dim3(REDUCTION_THREADS), buffers.out, buffers.peak, events);

  PeakLoad peak;
  peak.rentalsOut = copyIntToHost(buffers.peak);
  launch(first_index_of,
         dim3(blocks),
         dim3(REDUCTION_THREADS),
         buffers.out,
         peak.rentalsOut,
         buffers.peakEvent,
         events);
  peak.firstEvent = copyIntToHost(buffers.peakEvent);
  return peak;
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

void printClock(int ms) {
  std::printf("%02d:%02d", ms / MS_PER_HOUR, ms / MS_PER_MINUTE % 60);
}

int rentalsOutAtTime(int timeMs, const std::vector<int>& timesMs, const std::vector<int>& out) {
  const auto firstEventAtOrAfter = std::lower_bound(timesMs.begin(), timesMs.end(), timeMs);
  const int lastEventBefore = static_cast<int>(firstEventAtOrAfter - timesMs.begin()) - 1;
  return (lastEventBefore >= 0) ? out[lastEventBefore] : 0;
}

void printHourlyCurve(const EventStream& stream, const std::vector<int>& out, int peakRentals) {
  const int rentalsPerMark = imax(1, peakRentals / BAR_LENGTH_AT_PEAK);
  std::printf("\nBikes out on a rental at each full hour (each # = %d bikes):\n", rentalsPerMark);
  for (int hour = 1; hour <= HOURS_PER_DAY; ++hour) {
    const int rentalsOut = rentalsOutAtTime(hour * MS_PER_HOUR, stream.timesMs, out);
    std::printf("  %02d:00 %6d ", hour, rentalsOut);
    for (int mark = 0; mark < rentalsOut / rentalsPerMark; ++mark) {
      std::printf("#");
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

void printPeak(int trips, const EventStream& stream, const PeakLoad& peak) {
  std::printf("\n%d trips (%d faulty undocks returned within %d s). Peak: %d bikes out at once, ",
              trips,
              stream.faultyUndocks,
              MAX_FAULTY_CHECK_SECONDS,
              peak.rentalsOut);
  std::printf("first reached at ");
  printClock(stream.timesMs[peak.firstEvent]);
  std::printf(" (event %d).\n", peak.firstEvent);
}

int main(int argc, char** argv) {
  const int trips = tripCount(argc, argv);
  if (trips < 1) {
    std::fprintf(stderr, "need at least one trip\n");
    return 1;
  }
  const int events = 2 * trips;
  std::printf("Bikes out on a rental over one day (+1 undock, -1 dock, inclusive scan)\n");
  std::printf("backend: %s, %d trips -> %d events\n\n", CUDA_BACKEND, trips, events);

  const EventStream stream = makeEventStream(trips);

  std::vector<int> expectedOut(events);
  PeakLoad expectedPeak = {};
  const double cpuMs = averageMs(TIMED_RUNS, [&] {
    expectedPeak = cpuRunningSumAndPeak(stream.deltas, expectedOut);
  });

  const DeviceBuffers buffers = allocateDeviceBuffers(events);
  CHECK(cudaMemcpy(
      buffers.deltas, stream.deltas.data(), events * sizeof(int), cudaMemcpyHostToDevice));

  PeakLoad peak = {};
  auto run = [&] {
    peak = gpuRentalsOutAndPeak(buffers, events);
  };
  for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
    run();
  }
  CHECK(cudaMemset(buffers.out, 0, events * sizeof(int)));
  CHECK(cudaDeviceSynchronize());
  const double gpuMs = averageMs(TIMED_RUNS, run);
  CHECK(cudaGetLastError());

  std::vector<int> out(events);
  CHECK(cudaMemcpy(out.data(), buffers.out, events * sizeof(int), cudaMemcpyDeviceToHost));

  const bool scanPass = (out == expectedOut);
  const bool peakPass =
      (peak.rentalsOut == expectedPeak.rentalsOut) && (peak.firstEvent == expectedPeak.firstEvent);
  const bool everyBikeDocked = (out[events - 1] == 0);

  std::printf("  %-34s %10s %8s  %s\n", "step", "time [ms]", "vs CPU", "check");
  std::printf("  %-34s %10.3f %8s  %s\n", "CPU running sum + max (reference)", cpuMs, "1.00x", "-");
  std::printf("  %-34s %10.3f %7.2fx  scan %s, peak %s, ends at 0 %s\n",
              "GPU scan + max + first index",
              gpuMs,
              cpuMs / gpuMs,
              verdict(scanPass),
              verdict(peakPass),
              verdict(everyBikeDocked));

  printPeak(trips, stream, peak);
  printHourlyCurve(stream, out, expectedPeak.rentalsOut);
  printEmulationNote();

  const bool allPass = scanPass && peakPass && everyBikeDocked;
  std::printf("\n%s\n", allPass ? "ALL PASS" : "SOME CHECKS FAILED");

  freeDeviceBuffers(buffers);
  return allPass ? 0 : 1;
}
