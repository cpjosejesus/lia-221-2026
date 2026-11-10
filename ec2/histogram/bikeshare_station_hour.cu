#include "cuda_compat.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr unsigned HOURS = 24;
constexpr unsigned STATION_COLUMNS = 8;
constexpr unsigned STATION_ROWS = 6;
constexpr unsigned NUM_STATIONS = STATION_COLUMNS * STATION_ROWS;
constexpr unsigned BINS_B = NUM_STATIONS * HOURS;
constexpr unsigned MS_PER_HOUR = 60 * 60 * 1000;
constexpr unsigned THREADS_PER_BLOCK = 256;
constexpr unsigned CFACTOR = 64;

#ifndef CPU_RENTALS
#define CPU_RENTALS (1u << 19)
#endif
const unsigned NUM_SYNTHETIC_RENTALS = BY_BACKEND(1u << 24, CPU_RENTALS);
const int TIMED_RUNS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

__host__ __device__ inline unsigned hourOf(unsigned tMs) {
  return (tMs / MS_PER_HOUR) % HOURS;
}

__host__ __device__ inline unsigned stationHourBin(unsigned station, unsigned tMs) {
  return station * HOURS + hourOf(tMs);
}

__global__ void perHourInterleaved(const unsigned* startMs, unsigned n, unsigned* histo) {
  __shared__ unsigned histo_s[HOURS];
  for (unsigned b = threadIdx.x; b < HOURS; b += blockDim.x) {
    histo_s[b] = 0u;
  }
  __syncthreads();

  unsigned accumulator = 0;
  int prevHour = -1;
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int hour = static_cast<int>(hourOf(startMs[i]));
    if (hour == prevHour) {
      ++accumulator;
    } else {
      if (accumulator > 0) {
        atomicAdd(&histo_s[prevHour], accumulator);
      }
      accumulator = 1;
      prevHour = hour;
    }
  }
  if (accumulator > 0) {
    atomicAdd(&histo_s[prevHour], accumulator);
  }
  __syncthreads();

  for (unsigned b = threadIdx.x; b < HOURS; b += blockDim.x) {
    if (histo_s[b] > 0) {
      atomicAdd(&histo[b], histo_s[b]);
    }
  }
}

__global__ void perHourContiguous(const unsigned* startMs, unsigned n, unsigned* histo) {
  __shared__ unsigned histo_s[HOURS];
  for (unsigned b = threadIdx.x; b < HOURS; b += blockDim.x) {
    histo_s[b] = 0u;
  }
  __syncthreads();

  unsigned accumulator = 0;
  int prevHour = -1;
  unsigned begin = (blockIdx.x * blockDim.x + threadIdx.x) * CFACTOR;
  unsigned end = begin + CFACTOR < n ? begin + CFACTOR : n;
  for (unsigned i = begin; i < end; ++i) {
    int hour = static_cast<int>(hourOf(startMs[i]));
    if (hour == prevHour) {
      ++accumulator;
    } else {
      if (accumulator > 0) {
        atomicAdd(&histo_s[prevHour], accumulator);
      }
      accumulator = 1;
      prevHour = hour;
    }
  }
  if (accumulator > 0) {
    atomicAdd(&histo_s[prevHour], accumulator);
  }
  __syncthreads();

  for (unsigned b = threadIdx.x; b < HOURS; b += blockDim.x) {
    if (histo_s[b] > 0) {
      atomicAdd(&histo[b], histo_s[b]);
    }
  }
}

__global__ void perStationHour(const unsigned* startMs,
                               const unsigned* startStation,
                               unsigned n,
                               unsigned* histo) {
  __shared__ unsigned histo_s[BINS_B];
  for (unsigned b = threadIdx.x; b < BINS_B; b += blockDim.x) {
    histo_s[b] = 0u;
  }
  __syncthreads();

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(&histo_s[stationHourBin(startStation[i], startMs[i])], 1u);
  }
  __syncthreads();

  for (unsigned b = threadIdx.x; b < BINS_B; b += blockDim.x) {
    if (histo_s[b] > 0) {
      atomicAdd(&histo[b], histo_s[b]);
    }
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

  double uniformOpenInterval() {
    const uint64_t mantissa = next() >> (64 - DOUBLE_MANTISSA_BITS);
    return (static_cast<double>(mantissa) + 0.5) * (1.0 / TWO_TO_THE_MANTISSA_BITS);
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;
  static constexpr int DOUBLE_MANTISSA_BITS = 53;
  static constexpr double TWO_TO_THE_MANTISSA_BITS =
      static_cast<double>(1ull << DOUBLE_MANTISSA_BITS);

  uint64_t state_;
};

struct Position {
  int x;
  int y;
};

struct Rentals {
  std::vector<unsigned> startMs;
  std::vector<unsigned> startStation;
};

constexpr double RELATIVE_RENTAL_RATE_BY_HOUR[HOURS] = {
    0.10, 0.05, 0.03, 0.03, 0.05, 0.20, 0.80, 2.20, 3.00, 1.60, 0.90, 0.90,
    1.20, 1.10, 0.90, 1.00, 1.50, 2.40, 3.10, 2.00, 1.10, 0.70, 0.40, 0.20,
};
constexpr int CITY_SIDE_M = 6000;
constexpr int STATION_SPACING_X_M = CITY_SIDE_M / STATION_COLUMNS;
constexpr int STATION_SPACING_Y_M = CITY_SIDE_M / STATION_ROWS;
constexpr double METRES_PER_KM = 1000;
constexpr double BASE_POPULARITY = 1.0;
constexpr double CENTRE_BONUS = 6.0;
constexpr unsigned SYNTHETIC_DAYS = 7;
constexpr uint64_t SYNTHETIC_SEED = 7;

static Position stationPosition(unsigned station) {
  const int column = static_cast<int>(station % STATION_COLUMNS);
  const int row = static_cast<int>(station / STATION_COLUMNS);
  return {column * STATION_SPACING_X_M + STATION_SPACING_X_M / 2,
          row * STATION_SPACING_Y_M + STATION_SPACING_Y_M / 2};
}

static double popularity(unsigned station) {
  const Position position = stationPosition(station);
  const double dxKm = (position.x - CITY_SIDE_M / 2) / METRES_PER_KM;
  const double dyKm = (position.y - CITY_SIDE_M / 2) / METRES_PER_KM;
  const double distanceToCentreKm = std::sqrt(dxKm * dxKm + dyKm * dyKm);
  return BASE_POPULARITY + CENTRE_BONUS / (1.0 + distanceToCentreKm);
}

static std::vector<double> cumulativePopularity() {
  std::vector<double> cumulative(NUM_STATIONS);
  double sum = 0;
  for (unsigned s = 0; s < NUM_STATIONS; ++s) {
    sum += popularity(s);
    cumulative[s] = sum;
  }
  return cumulative;
}

static unsigned pickStationByPopularity(SplitMix64& rng, const std::vector<double>& cumulative) {
  const double draw = rng.uniformOpenInterval() * cumulative.back();
  for (unsigned s = 0; s < NUM_STATIONS; ++s) {
    if (draw < cumulative[s]) {
      return s;
    }
  }
  return NUM_STATIONS - 1;
}

static double nextRentalMs(double nowMs, double meanGapMs, SplitMix64& rng) {
  const double exponentialGap = -std::log(rng.uniformOpenInterval()) * meanGapMs;
  const double rateNow = RELATIVE_RENTAL_RATE_BY_HOUR[hourOf(static_cast<unsigned>(nowMs))];
  return nowMs + exponentialGap / rateNow;
}

static Rentals syntheticRentals(unsigned count, uint64_t seed) {
  SplitMix64 rng(seed);
  const std::vector<double> cumulative = cumulativePopularity();
  Rentals rentals;
  rentals.startMs.resize(count);
  rentals.startStation.resize(count);

  const double meanGapMs = static_cast<double>(SYNTHETIC_DAYS) * HOURS * MS_PER_HOUR / count;
  double nowMs = 0;
  for (unsigned i = 0; i < count; ++i) {
    nowMs = nextRentalMs(nowMs, meanGapMs, rng);
    rentals.startMs[i] = static_cast<unsigned>(nowMs);
    rentals.startStation[i] = pickStationByPopularity(rng, cumulative);
  }
  return rentals;
}

using Histogram = std::vector<unsigned>;

struct References {
  Histogram perHour;
  Histogram stationHour;
  double cpuMs;
};

static References sequentialReferences(const Rentals& rentals) {
  References references;
  references.perHour.assign(HOURS, 0);
  references.stationHour.assign(BINS_B, 0);
  Timer timer;
  for (size_t i = 0; i < rentals.startMs.size(); ++i) {
    references.perHour[hourOf(rentals.startMs[i])]++;
    references.stationHour[stationHourBin(rentals.startStation[i], rentals.startMs[i])]++;
  }
  references.cpuMs = timer.ms();
  return references;
}

enum class Partition { Interleaved, Contiguous };

static unsigned long long countAggregatedAtomics(const std::vector<unsigned>& startMs,
                                                 unsigned threadCount,
                                                 Partition partition) {
  const unsigned count = static_cast<unsigned>(startMs.size());
  const unsigned distanceToPreviousOfSameThread =
      partition == Partition::Contiguous ? 1 : threadCount;
  unsigned long long runs = 0;
  for (unsigned i = 0; i < count; ++i) {
    const bool startsThreadWork =
        partition == Partition::Contiguous ? i % CFACTOR == 0 : i < threadCount;
    if (startsThreadWork ||
        hourOf(startMs[i]) != hourOf(startMs[i - distanceToPreviousOfSameThread])) {
      ++runs;
    }
  }
  return runs;
}

struct DeviceBuffers {
  unsigned* startMs;
  unsigned* startStation;
  unsigned* perHour;
  unsigned* stationHour;
  unsigned count;
  unsigned blocks;
};

static unsigned atLeastOne(unsigned count) {
  return count > 0 ? count : 1;
}

static DeviceBuffers copyRentalsToDevice(const Rentals& rentals) {
  DeviceBuffers device;
  device.count = static_cast<unsigned>(rentals.startMs.size());
  device.blocks = device.count > 0 ? cdiv(device.count, THREADS_PER_BLOCK * CFACTOR) : 1;

  const size_t inputBytes = atLeastOne(device.count) * sizeof(unsigned);
  CHECK(cudaMalloc(&device.startMs, inputBytes));
  CHECK(cudaMalloc(&device.startStation, inputBytes));
  CHECK(cudaMalloc(&device.perHour, HOURS * sizeof(unsigned)));
  CHECK(cudaMalloc(&device.stationHour, BINS_B * sizeof(unsigned)));

  if (device.count > 0) {
    const size_t bytes = device.count * sizeof(unsigned);
    CHECK(cudaMemcpy(device.startMs, rentals.startMs.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        device.startStation, rentals.startStation.data(), bytes, cudaMemcpyHostToDevice));
  }
  return device;
}

static void freeDeviceBuffers(const DeviceBuffers& device) {
  CHECK(cudaFree(device.startMs));
  CHECK(cudaFree(device.startStation));
  CHECK(cudaFree(device.perHour));
  CHECK(cudaFree(device.stationHour));
}

template <class ClearAndLaunch>
static double timeLaunches(ClearAndLaunch clearAndLaunch) {
  if (WARM_UP_BEFORE_TIMING) {
    clearAndLaunch();
  }
  Timer timer;
  for (int run = 0; run < TIMED_RUNS; ++run) {
    clearAndLaunch();
  }
  const double ms = timer.ms() / TIMED_RUNS;
  CHECK(cudaGetLastError());
  return ms;
}

static Histogram copyToHost(const unsigned* d_histo, unsigned bins) {
  Histogram histo(bins);
  CHECK(cudaMemcpy(histo.data(), d_histo, bins * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return histo;
}

static void printRow(
    const char* name, unsigned blocks, double ms, unsigned long long sharedAtomics, bool correct) {
  std::printf(
      "%-34s %7u %10.3f %15llu  %s\n", name, blocks, ms, sharedAtomics, correct ? "PASS" : "FAIL");
}

using PerHourKernel = void (*)(const unsigned*, unsigned, unsigned*);

static bool runPerHourKernel(const char* name,
                             PerHourKernel kernel,
                             Partition partition,
                             const DeviceBuffers& device,
                             const Rentals& rentals,
                             const Histogram& reference,
                             Histogram& perHour) {
  const unsigned* startMs = device.startMs;
  const double ms = timeLaunches([&] {
    CHECK(cudaMemset(device.perHour, 0, HOURS * sizeof(unsigned)));
    launch(kernel,
           dim3(device.blocks),
           dim3(THREADS_PER_BLOCK),
           startMs,
           device.count,
           device.perHour);
  });

  perHour = copyToHost(device.perHour, HOURS);
  const bool correct = perHour == reference;
  const unsigned threadCount = device.blocks * THREADS_PER_BLOCK;
  printRow(name,
           device.blocks,
           ms,
           countAggregatedAtomics(rentals.startMs, threadCount, partition),
           correct);
  return correct;
}

static bool runPerStationHourKernel(const DeviceBuffers& device,
                                    const Histogram& reference,
                                    Histogram& stationHourBins) {
  const unsigned* startMs = device.startMs;
  const unsigned* startStation = device.startStation;
  const double ms = timeLaunches([&] {
    CHECK(cudaMemset(device.stationHour, 0, BINS_B * sizeof(unsigned)));
    launch(perStationHour,
           dim3(device.blocks),
           dim3(THREADS_PER_BLOCK),
           startMs,
           startStation,
           device.count,
           device.stationHour);
  });

  stationHourBins = copyToHost(device.stationHour, BINS_B);
  const bool correct = stationHourBins == reference;
  const unsigned long long oneAtomicPerRental = device.count;
  printRow("B station x hour, shared priv.", device.blocks, ms, oneAtomicPerRental, correct);
  return correct;
}

struct GpuHistograms {
  Histogram perHour;
  Histogram stationHour;
};

static bool runKernels(const Rentals& rentals, const References& references, GpuHistograms& gpu) {
  const DeviceBuffers device = copyRentalsToDevice(rentals);

  std::printf("\n%-34s %7s %10s %15s  %s\n", "kernel", "blocks", "ms", "shared atomics", "check");
  std::printf("%-34s %7s %10.3f %15s  %s\n",
              "CPU sequential (A and B)",
              "-",
              references.cpuMs,
              "-",
              "reference");

  const bool interleavedCorrect = runPerHourKernel("A per hour, interleaved + aggr.",
                                                   perHourInterleaved,
                                                   Partition::Interleaved,
                                                   device,
                                                   rentals,
                                                   references.perHour,
                                                   gpu.perHour);
  const bool contiguousCorrect = runPerHourKernel("A per hour, contiguous + aggr.",
                                                  perHourContiguous,
                                                  Partition::Contiguous,
                                                  device,
                                                  rentals,
                                                  references.perHour,
                                                  gpu.perHour);
  const bool stationHourCorrect =
      runPerStationHourKernel(device, references.stationHour, gpu.stationHour);

  freeDeviceBuffers(device);
  return interleavedCorrect && contiguousCorrect && stationHourCorrect;
}

constexpr int BAR_WIDTH = 50;

static void printBar(unsigned count, unsigned peak) {
  const int width =
      peak ? static_cast<int>(BAR_WIDTH * static_cast<double>(count) / peak + 0.5) : 0;
  for (int j = 0; j < width; ++j) {
    std::putchar('#');
  }
  std::putchar('\n');
}

static unsigned largest(const Histogram& histo) {
  unsigned peak = 0;
  for (unsigned count : histo) {
    if (count > peak) {
      peak = count;
    }
  }
  return peak;
}

static unsigned binCount(const Histogram& stationHour, unsigned station, unsigned hour) {
  return stationHour[station * HOURS + hour];
}

static Histogram rowSums(const Histogram& stationHour) {
  Histogram perStation(NUM_STATIONS, 0);
  for (unsigned s = 0; s < NUM_STATIONS; ++s) {
    for (unsigned h = 0; h < HOURS; ++h) {
      perStation[s] += binCount(stationHour, s, h);
    }
  }
  return perStation;
}

static bool columnSumsEqual(const Histogram& stationHour, const Histogram& perHour) {
  for (unsigned h = 0; h < HOURS; ++h) {
    unsigned columnSum = 0;
    for (unsigned s = 0; s < NUM_STATIONS; ++s) {
      columnSum += binCount(stationHour, s, h);
    }
    if (columnSum != perHour[h]) {
      return false;
    }
  }
  return true;
}

struct StationHour {
  unsigned station;
  unsigned hour;
};

static StationHour busiestStationHour(const Histogram& stationHour) {
  StationHour busiest = {0, 0};
  for (unsigned s = 0; s < NUM_STATIONS; ++s) {
    for (unsigned h = 0; h < HOURS; ++h) {
      const unsigned busiestCount = binCount(stationHour, busiest.station, busiest.hour);
      if (binCount(stationHour, s, h) > busiestCount) {
        busiest = {s, h};
      }
    }
  }
  return busiest;
}

static void printRentalsPerHour(const Histogram& perHour) {
  const unsigned peak = largest(perHour);
  std::printf("\nhour   rentals\n");
  for (unsigned h = 0; h < HOURS; ++h) {
    std::printf("%02u  %9u  ", h, perHour[h]);
    printBar(perHour[h], peak);
  }
}

static void printRentalsPerStationAsMap(const Histogram& perStation) {
  std::printf("\nrentals per station, laid out as on the map (row 0 is the south edge)\n");
  for (int row = STATION_ROWS - 1; row >= 0; --row) {
    std::printf("row %d ", row);
    for (unsigned column = 0; column < STATION_COLUMNS; ++column) {
      std::printf(" %7u", perStation[row * STATION_COLUMNS + column]);
    }
    std::printf("\n");
  }
}

static bool printReport(const Rentals& rentals, const GpuHistograms& gpu) {
  printRentalsPerHour(gpu.perHour);
  printRentalsPerStationAsMap(rowSums(gpu.stationHour));

  if (!rentals.startMs.empty()) {
    const StationHour busiest = busiestStationHour(gpu.stationHour);
    std::printf("\nbusiest station-hour: s%02u at %02u:00 with %u rentals\n",
                busiest.station,
                busiest.hour,
                binCount(gpu.stationHour, busiest.station, busiest.hour));
  }

  const bool consistent = columnSumsEqual(gpu.stationHour, gpu.perHour);
  std::printf("columns of B equal A: %s\n", consistent ? "yes" : "NO");
  return consistent;
}

int main() {
  std::printf("Bike rentals per hour and per station (chapter 9 use case), backend: %s\n",
              CUDA_BACKEND);
  const Rentals rentals = syntheticRentals(NUM_SYNTHETIC_RENTALS, SYNTHETIC_SEED);
  std::printf("synthetic week, %zu rentals at %u stations (seed %llu)\n",
              rentals.startMs.size(),
              NUM_STATIONS,
              static_cast<unsigned long long>(SYNTHETIC_SEED));
  const References references = sequentialReferences(rentals);

  GpuHistograms gpu;
  const bool kernelsCorrect = runKernels(rentals, references, gpu);
  const bool consistent = printReport(rentals, gpu);

  const bool allPass = kernelsCorrect && consistent;
  std::printf("%s\n", allPass ? "ALL PASS" : "SOME TESTS FAILED");
  return allPass ? 0 : 1;
}
