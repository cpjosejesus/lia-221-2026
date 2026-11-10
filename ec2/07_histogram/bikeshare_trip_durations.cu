#include "cuda_compat.h"

#include <cstdint>
#include <cstdio>
#include <vector>

constexpr unsigned MS_PER_SECOND = 1000;
constexpr unsigned MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr unsigned MINUTES_WITH_OWN_BUCKET = 60;
constexpr unsigned NUM_BUCKETS = MINUTES_WITH_OWN_BUCKET + 1;
constexpr unsigned OVERFLOW_BUCKET = NUM_BUCKETS - 1;
constexpr unsigned MS_PER_BUCKET = MS_PER_MINUTE;
constexpr unsigned THREADS_PER_BLOCK = 256;
constexpr unsigned CFACTOR = 32;

#ifndef CPU_TRIPS
#define CPU_TRIPS (1u << 18)
#endif
const unsigned NUM_SYNTHETIC_TRIPS = BY_BACKEND(1u << 24, CPU_TRIPS);
const int TIMED_RUNS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

__host__ __device__ inline unsigned bucketOf(unsigned ms) {
  unsigned b = ms / MS_PER_BUCKET;
  return b < OVERFLOW_BUCKET ? b : OVERFLOW_BUCKET;
}

__global__ void tripHistoBasic(const unsigned* tripMs, unsigned n, unsigned* histo) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&histo[bucketOf(tripMs[i])], 1u);
  }
}

__global__ void tripHistoPrivate(const unsigned* tripMs, unsigned n, unsigned* histo) {
  __shared__ unsigned histo_s[NUM_BUCKETS];
  for (unsigned b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) {
    histo_s[b] = 0u;
  }
  __syncthreads();

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(&histo_s[bucketOf(tripMs[i])], 1u);
  }
  __syncthreads();

  for (unsigned b = threadIdx.x; b < NUM_BUCKETS; b += blockDim.x) {
    unsigned v = histo_s[b];
    if (v > 0) {
      atomicAdd(&histo[b], v);
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

  bool oneIn(unsigned n) {
    return next() % n == 0;
  }

  bool percentChance(unsigned percent) {
    return next() % 100 < percent;
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

struct Range {
  unsigned lowest;
  unsigned highest;
};

static unsigned randomIn(SplitMix64& rng, Range range) {
  return range.lowest + static_cast<unsigned>(rng.next() % (range.highest - range.lowest + 1));
}

constexpr Range UNLOCK_AND_ADJUST_SEAT = {20 * MS_PER_SECOND, 90 * MS_PER_SECOND};
constexpr Range RIDE_DISTANCE_M = {600, 4500};
constexpr unsigned E_BIKE_PERCENT = 40;
constexpr unsigned PEDAL_BIKE_METRES_PER_MINUTE = 250;
constexpr unsigned E_BIKE_METRES_PER_MINUTE = 330;
constexpr Range TRAFFIC_LIGHTS_AND_HILLS = {0, 4 * MS_PER_MINUTE};
constexpr unsigned ONE_FULL_END_STATION_EVERY = 6;
constexpr Range SEARCH_FOR_A_FREE_DOCK = {1 * MS_PER_MINUTE, 6 * MS_PER_MINUTE};
constexpr unsigned LEISURE_RIDE_PERCENT = 3;
constexpr Range LEISURE_DETOUR = {10 * MS_PER_MINUTE, 40 * MS_PER_MINUTE};
constexpr uint64_t SYNTHETIC_SEED = 42;

static unsigned ridingMs(unsigned distanceM, unsigned metresPerMinute) {
  return distanceM * MS_PER_MINUTE / metresPerMinute;
}

static unsigned syntheticTrip(SplitMix64& rng) {
  const unsigned unlock = randomIn(rng, UNLOCK_AND_ADJUST_SEAT);
  const unsigned distanceM = randomIn(rng, RIDE_DISTANCE_M);
  const bool eBike = rng.percentChance(E_BIKE_PERCENT);
  const unsigned speed = eBike ? E_BIKE_METRES_PER_MINUTE : PEDAL_BIKE_METRES_PER_MINUTE;

  unsigned ride = ridingMs(distanceM, speed);
  ride += randomIn(rng, TRAFFIC_LIGHTS_AND_HILLS);

  unsigned dockSearch = 0;
  if (rng.oneIn(ONE_FULL_END_STATION_EVERY)) {
    dockSearch = randomIn(rng, SEARCH_FOR_A_FREE_DOCK);
  }

  unsigned detour = 0;
  if (rng.percentChance(LEISURE_RIDE_PERCENT)) {
    detour = randomIn(rng, LEISURE_DETOUR);
  }

  return unlock + ride + dockSearch + detour;
}

static std::vector<unsigned> syntheticTrips(unsigned count, uint64_t seed) {
  SplitMix64 rng(seed);
  std::vector<unsigned> tripMs(count);
  for (unsigned& ms : tripMs) {
    ms = syntheticTrip(rng);
  }
  return tripMs;
}

using Histogram = std::vector<unsigned>;

static Histogram sequentialHistogram(const std::vector<unsigned>& tripMs, double& elapsedMs) {
  Histogram histo(NUM_BUCKETS, 0);
  Timer timer;
  for (unsigned ms : tripMs) {
    histo[bucketOf(ms)]++;
  }
  elapsedMs = timer.ms();
  return histo;
}

static unsigned atLeastOne(unsigned count) {
  return count > 0 ? count : 1;
}

using TripKernel = void (*)(const unsigned*, unsigned, unsigned*);

struct KernelVariant {
  const char* name;
  TripKernel kernel;
  unsigned blocks;
};

static unsigned blocksFor(unsigned count, unsigned tripsPerThread) {
  if (count == 0) {
    return 1;
  }
  return cdiv(count, THREADS_PER_BLOCK * tripsPerThread);
}

static std::vector<KernelVariant> kernelVariants(unsigned count) {
  const unsigned oneTripPerThread = 1;
  return {
      {"basic, global atomics", tripHistoBasic, blocksFor(count, oneTripPerThread)},
      {"shared private + coarsened", tripHistoPrivate, blocksFor(count, CFACTOR)},
  };
}

struct DeviceBuffers {
  unsigned* tripMs;
  unsigned* histo;
  unsigned count;
};

static DeviceBuffers copyTripsToDevice(const std::vector<unsigned>& tripMs) {
  DeviceBuffers device;
  device.count = static_cast<unsigned>(tripMs.size());
  CHECK(cudaMalloc(&device.tripMs, atLeastOne(device.count) * sizeof(unsigned)));
  CHECK(cudaMalloc(&device.histo, NUM_BUCKETS * sizeof(unsigned)));
  if (device.count > 0) {
    CHECK(cudaMemcpy(
        device.tripMs, tripMs.data(), device.count * sizeof(unsigned), cudaMemcpyHostToDevice));
  }
  return device;
}

static void freeDeviceBuffers(const DeviceBuffers& device) {
  CHECK(cudaFree(device.tripMs));
  CHECK(cudaFree(device.histo));
}

static double timeKernel(const KernelVariant& variant, const DeviceBuffers& device) {
  const unsigned* tripMs = device.tripMs;
  auto clearAndLaunch = [&] {
    CHECK(cudaMemset(device.histo, 0, NUM_BUCKETS * sizeof(unsigned)));
    launch(variant.kernel,
           dim3(variant.blocks),
           dim3(THREADS_PER_BLOCK),
           tripMs,
           device.count,
           device.histo);
  };

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

static Histogram copyHistogramToHost(const DeviceBuffers& device) {
  Histogram histo(NUM_BUCKETS);
  CHECK(cudaMemcpy(
      histo.data(), device.histo, NUM_BUCKETS * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return histo;
}

static bool runKernels(const std::vector<unsigned>& tripMs,
                       const Histogram& reference,
                       double cpuMs,
                       Histogram& gpuHistogram) {
  const DeviceBuffers device = copyTripsToDevice(tripMs);

  std::printf("\n%-28s %8s %10s  %s\n", "kernel", "blocks", "ms", "check");
  std::printf("%-28s %8s %10.3f  %s\n", "CPU sequential", "-", cpuMs, "reference");
  bool allCorrect = true;
  for (const KernelVariant& variant : kernelVariants(device.count)) {
    const double ms = timeKernel(variant, device);
    gpuHistogram = copyHistogramToHost(device);
    const bool correct = gpuHistogram == reference;
    allCorrect = allCorrect && correct;
    std::printf(
        "%-28s %8u %10.3f  %s\n", variant.name, variant.blocks, ms, correct ? "PASS" : "FAIL");
  }

  freeDeviceBuffers(device);
  return allCorrect;
}

constexpr int BAR_WIDTH = 60;
constexpr unsigned INCLUDED_MINUTES = 45;
constexpr double MEDIAN_FRACTION = 0.50;
constexpr double P90_FRACTION = 0.90;
constexpr double P99_FRACTION = 0.99;

static unsigned long long totalCount(const Histogram& histo) {
  unsigned long long total = 0;
  for (unsigned count : histo) {
    total += count;
  }
  return total;
}

static unsigned long long countBelowMinute(const Histogram& histo, unsigned minute) {
  unsigned long long count = 0;
  for (unsigned b = 0; b < minute; ++b) {
    count += histo[b];
  }
  return count;
}

static unsigned fullestBucketCount(const Histogram& histo) {
  unsigned peak = 0;
  for (unsigned count : histo) {
    if (count > peak) {
      peak = count;
    }
  }
  return peak;
}

static unsigned percentileMinute(const Histogram& histo,
                                 unsigned long long total,
                                 double fraction) {
  const unsigned long long needed = static_cast<unsigned long long>(fraction * total + 0.5);
  unsigned long long runningSum = 0;
  for (unsigned b = 0; b < NUM_BUCKETS; ++b) {
    runningSum += histo[b];
    if (runningSum >= needed) {
      return b;
    }
  }
  return OVERFLOW_BUCKET;
}

static void printBar(unsigned count, unsigned peak) {
  const int width = static_cast<int>(BAR_WIDTH * static_cast<double>(count) / peak + 0.5);
  for (int j = 0; j < width; ++j) {
    std::putchar('#');
  }
  std::putchar('\n');
}

static void printMinuteHistogram(const Histogram& histo) {
  const unsigned peak = fullestBucketCount(histo);
  std::printf("\nminutes     trips\n");
  for (unsigned b = 0; b < NUM_BUCKETS; ++b) {
    if (histo[b] == 0) {
      continue;
    }
    const char* prefix = b == OVERFLOW_BUCKET ? ">=" : "  ";
    std::printf("%s%2u  %9u  ", prefix, b, histo[b]);
    printBar(histo[b], peak);
  }
}

static void printPercentiles(const Histogram& histo) {
  const unsigned long long total = totalCount(histo);
  if (total == 0) {
    return;
  }
  const unsigned long long withinIncludedTime = countBelowMinute(histo, INCLUDED_MINUTES);
  std::printf("\nmedian %u min, p90 %u min, p99 %u min\n",
              percentileMinute(histo, total, MEDIAN_FRACTION),
              percentileMinute(histo, total, P90_FRACTION),
              percentileMinute(histo, total, P99_FRACTION));
  std::printf("within the included %u min: %.1f%%, late (%u min or more): %llu trips\n",
              INCLUDED_MINUTES,
              100.0 * withinIncludedTime / total,
              INCLUDED_MINUTES,
              total - withinIncludedTime);
}

int main() {
  std::printf("Bike-share trip durations (chapter 9 use case), backend: %s\n", CUDA_BACKEND);
  const std::vector<unsigned> tripMs = syntheticTrips(NUM_SYNTHETIC_TRIPS, SYNTHETIC_SEED);
  std::printf("%zu synthetic trips (seed %llu)\n",
              tripMs.size(),
              static_cast<unsigned long long>(SYNTHETIC_SEED));

  double cpuMs;
  const Histogram reference = sequentialHistogram(tripMs, cpuMs);

  Histogram gpuHistogram;
  const bool allCorrect = runKernels(tripMs, reference, cpuMs, gpuHistogram);

  printMinuteHistogram(gpuHistogram);
  printPercentiles(gpuHistogram);

  std::printf("%s\n", allCorrect ? "ALL PASS" : "SOME TESTS FAILED");
  return allCorrect ? 0 : 1;
}
