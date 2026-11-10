#include "cuda_compat.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

constexpr unsigned BLOCK_DIM = 256;
constexpr unsigned WARP_SIZE = 32;
constexpr unsigned WARPS_PER_BLOCK = BLOCK_DIM / WARP_SIZE;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;

constexpr int MINUTE_MS = 60000;
constexpr int SECOND_MS = 1000;
constexpr int SECONDS_PER_MINUTE = 60;
constexpr int INCLUDED_RIDE_MS = 45 * MINUTE_MS;
constexpr int MS_PER_METRE_AT_15_KMH = 240;
constexpr int CENTS_PER_BOLIVIANO = 100;

enum class TripStatus : int { Completed = 0, StillRiding = 1, FalseStart = 2 };

#ifndef CPU_TRIPS
#define CPU_TRIPS (1u << 17)
#endif
const unsigned TRIP_COUNT = BY_BACKEND(1u << 24, CPU_TRIPS);
const int TIMED_REPETITIONS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);
const bool IS_CPU_EMULATION = BY_BACKEND(false, true);

constexpr unsigned PARTIALS_COARSE_FACTOR = 64;
constexpr unsigned SMALL_ATOMICS_COARSE_FACTOR = 128;
constexpr unsigned LARGE_ATOMICS_COARSE_FACTOR = 256;
constexpr double FLOAT_REVENUE_TOLERANCE = 1e-4;

struct Kpi {
  unsigned trips;
  unsigned late;
  unsigned long long fareCents;
  unsigned long long distanceM;
  int minMs;
  int maxMs;
  float fareBs;
};

__host__ __device__ inline Kpi kpiIdentity() {
  return Kpi{0u, 0u, 0ull, 0ull, INT_MAX, INT_MIN, 0.0f};
}

__host__ __device__ inline Kpi combine(const Kpi& a, const Kpi& b) {
  Kpi r;
  r.trips = a.trips + b.trips;
  r.late = a.late + b.late;
  r.fareCents = a.fareCents + b.fareCents;
  r.distanceM = a.distanceM + b.distanceM;
  r.minMs = imin(a.minMs, b.minMs);
  r.maxMs = imax(a.maxMs, b.maxMs);
  r.fareBs = a.fareBs + b.fareBs;
  return r;
}

__host__ __device__ inline Kpi kpiOfTrip(TripStatus status, int ms, int distM, int fareCents) {
  if (status != TripStatus::Completed) {
    return kpiIdentity();
  }
  const unsigned late = ms > INCLUDED_RIDE_MS ? 1u : 0u;
  const float fareBs = static_cast<float>(fareCents) / static_cast<float>(CENTS_PER_BOLIVIANO);
  return Kpi{1u,
             late,
             static_cast<unsigned long long>(fareCents),
             static_cast<unsigned long long>(distM),
             ms,
             ms,
             fareBs};
}

__device__ inline Kpi shuffleDownKpi(const Kpi& v, unsigned delta) {
  Kpi r;
  r.trips = __shfl_down_sync(FULL_WARP_MASK, v.trips, delta);
  r.late = __shfl_down_sync(FULL_WARP_MASK, v.late, delta);
  r.fareCents = __shfl_down_sync(FULL_WARP_MASK, v.fareCents, delta);
  r.distanceM = __shfl_down_sync(FULL_WARP_MASK, v.distanceM, delta);
  r.minMs = __shfl_down_sync(FULL_WARP_MASK, v.minMs, delta);
  r.maxMs = __shfl_down_sync(FULL_WARP_MASK, v.maxMs, delta);
  r.fareBs = __shfl_down_sync(FULL_WARP_MASK, v.fareBs, delta);
  return r;
}

__device__ inline Kpi warpReduceKpi(Kpi v) {
  for (unsigned stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    v = combine(v, shuffleDownKpi(v, stride));
  }
  return v;
}

__device__ inline Kpi blockReduceKpi(Kpi v) {
  __shared__ Kpi warpKpi_s[WARPS_PER_BLOCK];
  const unsigned warp = threadIdx.x / WARP_SIZE;
  const unsigned lane = threadIdx.x % WARP_SIZE;

  v = warpReduceKpi(v);
  if (lane == 0) {
    warpKpi_s[warp] = v;
  }
  __syncthreads();

  Kpi blockKpi = kpiIdentity();
  if (warp == 0) {
    const bool laneHasWarpResult = lane < WARPS_PER_BLOCK;
    if (laneHasWarpResult) {
      blockKpi = warpKpi_s[lane];
    }
    blockKpi = warpReduceKpi(blockKpi);
  }
  return blockKpi;
}

__device__ inline void combineIntoTotalWithAtomics(Kpi* total, const Kpi& blockKpi) {
  atomicAdd(&total->trips, blockKpi.trips);
  atomicAdd(&total->late, blockKpi.late);
  atomicAdd(&total->fareCents, blockKpi.fareCents);
  atomicAdd(&total->distanceM, blockKpi.distanceM);
  atomicMin(&total->minMs, blockKpi.minMs);
  atomicMax(&total->maxMs, blockKpi.maxMs);
  atomicAdd(&total->fareBs, blockKpi.fareBs);
}

template <unsigned COARSE_FACTOR>
__global__ void KpiKernel(const TripStatus* status,
                          const int* rideMs,
                          const int* distanceM,
                          const int* fareCents,
                          unsigned n,
                          Kpi* total,
                          Kpi* partials) {
  Kpi partial = kpiIdentity();
  unsigned segment = COARSE_FACTOR * BLOCK_DIM * blockIdx.x;
  for (unsigned tile = 0; tile < COARSE_FACTOR; ++tile) {
    unsigned i = segment + threadIdx.x + tile * BLOCK_DIM;
    if (i < n) {
      partial = combine(partial, kpiOfTrip(status[i], rideMs[i], distanceM[i], fareCents[i]));
    }
  }

  const Kpi blockKpi = blockReduceKpi(partial);

  const bool combineWithAtomics = partials == nullptr;
  if (threadIdx.x == 0) {
    if (combineWithAtomics) {
      combineIntoTotalWithAtomics(total, blockKpi);
    } else {
      partials[blockIdx.x] = blockKpi;
    }
  }
}

__global__ void KpiCombineKernel(const Kpi* partials, unsigned count, Kpi* out) {
  Kpi partial = kpiIdentity();
  for (unsigned j = threadIdx.x; j < count; j += BLOCK_DIM) {
    partial = combine(partial, partials[j]);
  }
  const Kpi total = blockReduceKpi(partial);
  if (threadIdx.x == 0) {
    *out = total;
  }
}

class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {
  }

  uint64_t next() {
    state_ += GOLDEN_GAMMA;
    uint64_t mixed = state_;
    mixed = (mixed ^ (mixed >> 30)) * FIRST_MULTIPLIER;
    mixed = (mixed ^ (mixed >> 27)) * SECOND_MULTIPLIER;
    return mixed ^ (mixed >> 31);
  }

  int range(int lowest, int highest) {
    const uint64_t choices = static_cast<uint64_t>(highest - lowest + 1);
    return lowest + static_cast<int>(next() % choices);
  }

 private:
  static constexpr uint64_t GOLDEN_GAMMA = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

constexpr uint64_t DATA_SEED = 2025;
constexpr uint64_t REPLAY_SEED = 7;
constexpr int REPLAYED_BLOCK_SEQUENCES = 1000;

constexpr int PER_MILLE = 1000;
constexpr int WHOLE_PERCENT = 100;
constexpr int STILL_RIDING_PER_MILLE = 30;
constexpr int FALSE_START_PER_MILLE = 50;
constexpr int SHORTEST_TRIP_M = 300;
constexpr int LONGEST_TRIP_M = 8000;
constexpr int LEISURE_RIDE_PERCENT = 4;
constexpr int MEMBER_PERCENT = 60;
constexpr int MEMBER_DISCOUNT_PERCENT = 15;
constexpr int FARE_STEP_MS = 10 * MINUTE_MS;
constexpr int BASE_FARE_CENTS = 300;
constexpr int CENTS_PER_FARE_STEP = 100;
constexpr int METRES_PER_KM = 1000;

struct Trips {
  std::vector<TripStatus> status;
  std::vector<int> rideMs;
  std::vector<int> distanceM;
  std::vector<int> fareCents;
};

struct DeviceTrips {
  TripStatus* status;
  int* rideMs;
  int* distanceM;
  int* fareCents;
};

struct CpuReference {
  Kpi kpi;
  double fareBsInDouble;
  double ms;
};

struct PartialsRun {
  Kpi total;
  std::vector<Kpi> partials;
  unsigned blocks;
  double ms;
};

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

static int fareCentsForRide(int rideMs, bool isMember) {
  const int startedFareSteps = (rideMs + FARE_STEP_MS - 1) / FARE_STEP_MS;
  const int fullFareCents = BASE_FARE_CENTS + startedFareSteps * CENTS_PER_FARE_STEP;
  if (!isMember) {
    return fullFareCents;
  }
  return fullFareCents * (WHOLE_PERCENT - MEMBER_DISCOUNT_PERCENT) / WHOLE_PERCENT;
}

static int syntheticRideMs(SplitMix64& random, int distanceM) {
  const int pedallingMs = distanceM * MS_PER_METRE_AT_15_KMH;
  const int trafficLightsMs = random.range(1 * MINUTE_MS, 12 * MINUTE_MS);
  int rideMs = pedallingMs + trafficLightsMs;

  const bool isLeisureRide = random.range(0, 99) < LEISURE_RIDE_PERCENT;
  if (isLeisureRide) {
    rideMs += random.range(20 * MINUTE_MS, 60 * MINUTE_MS);
  }
  return rideMs;
}

static void appendTrip(Trips& trips, TripStatus status, int rideMs, int distanceM, int fareCents) {
  trips.status.push_back(status);
  trips.rideMs.push_back(rideMs);
  trips.distanceM.push_back(distanceM);
  trips.fareCents.push_back(fareCents);
}

static Trips syntheticDay(unsigned count, uint64_t seed) {
  SplitMix64 random(seed);
  Trips trips;
  for (unsigned i = 0; i < count; ++i) {
    const int outcome = random.range(0, PER_MILLE - 1);
    if (outcome < STILL_RIDING_PER_MILLE) {
      appendTrip(trips, TripStatus::StillRiding, 0, 0, 0);
      continue;
    }
    if (outcome < STILL_RIDING_PER_MILLE + FALSE_START_PER_MILLE) {
      appendTrip(trips, TripStatus::FalseStart, 0, 0, 0);
      continue;
    }

    const int distanceM = random.range(SHORTEST_TRIP_M, LONGEST_TRIP_M);
    const int rideMs = syntheticRideMs(random, distanceM);
    const bool isMember = random.range(0, 99) < MEMBER_PERCENT;
    appendTrip(trips, TripStatus::Completed, rideMs, distanceM, fareCentsForRide(rideMs, isMember));
  }
  return trips;
}

static bool sameIntegers(const Kpi& a, const Kpi& b) {
  return a.trips == b.trips && a.late == b.late && a.fareCents == b.fareCents &&
         a.distanceM == b.distanceM && a.minMs == b.minMs && a.maxMs == b.maxMs;
}

static bool sameFloat(float a, float b) {
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static double exactRevenueBs(const Kpi& kpi) {
  return static_cast<double>(kpi.fareCents) / CENTS_PER_BOLIVIANO;
}

static CpuReference computeCpuReference(const Trips& trips) {
  Timer timer;
  CpuReference reference;
  reference.kpi = kpiIdentity();
  reference.fareBsInDouble = 0;
  for (size_t i = 0; i < trips.status.size(); ++i) {
    const Kpi trip =
        kpiOfTrip(trips.status[i], trips.rideMs[i], trips.distanceM[i], trips.fareCents[i]);
    reference.kpi = combine(reference.kpi, trip);
    reference.fareBsInDouble += static_cast<double>(trip.fareBs);
  }
  reference.ms = timer.ms();
  return reference;
}

template <class T>
static T* deviceCopyOf(const std::vector<T>& values) {
  T* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, values.size() * sizeof(T)));
  CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return pointer;
}

static DeviceTrips copyTripsToDevice(const Trips& trips) {
  DeviceTrips device;
  device.status = deviceCopyOf(trips.status);
  device.rideMs = deviceCopyOf(trips.rideMs);
  device.distanceM = deviceCopyOf(trips.distanceM);
  device.fareCents = deviceCopyOf(trips.fareCents);
  return device;
}

static void freeDeviceTrips(const DeviceTrips& device) {
  CHECK(cudaFree(device.status));
  CHECK(cudaFree(device.rideMs));
  CHECK(cudaFree(device.distanceM));
  CHECK(cudaFree(device.fareCents));
}

static Kpi readKpi(const Kpi* deviceKpi) {
  Kpi kpi;
  CHECK(cudaMemcpy(&kpi, deviceKpi, sizeof(Kpi), cudaMemcpyDeviceToHost));
  return kpi;
}

static void printTableHeader(const CpuReference& reference) {
  const double exactBs = exactRevenueBs(reference.kpi);
  const double sequentialFloatBs = static_cast<double>(reference.kpi.fareBs);
  std::printf("\n%-40s %6s %9s %18s %11s  %s\n",
              "version",
              "blocks",
              "ms",
              "revenue float (Bs)",
              "- exact",
              "integer KPIs");
  std::printf("%-40s %6s %9.3f %18.4f %11.4f  %s\n",
              "CPU sequential loop, float",
              "-",
              reference.ms,
              sequentialFloatBs,
              sequentialFloatBs - exactBs,
              "reference");
  std::printf("%-40s %6s %9s %18.4f %11.4f  %s\n",
              "CPU sequential loop, double",
              "-",
              "-",
              reference.fareBsInDouble,
              reference.fareBsInDouble - exactBs,
              "-");
}

static bool reportRow(const char* name,
                      unsigned blocks,
                      double ms,
                      const Kpi& result,
                      const CpuReference& reference,
                      const char* note) {
  const double exactBs = exactRevenueBs(reference.kpi);
  const double floatBs = static_cast<double>(result.fareBs);
  const bool integersOk = sameIntegers(result, reference.kpi);
  const double relativeError = std::fabs(floatBs - exactBs) / exactBs;
  const bool floatOk = relativeError <= FLOAT_REVENUE_TOLERANCE;
  const bool ok = integersOk && floatOk;
  std::printf("%-40s %6u %9.3f %18.4f %11.4f  %s%s\n",
              name,
              blocks,
              ms,
              floatBs,
              floatBs - exactBs,
              passOrFail(ok),
              note);
  return ok;
}

template <class Work>
static double millisecondsPerRun(Work runOnce) {
  if (WARM_UP_BEFORE_TIMING) {
    runOnce();
  }
  Timer timer;
  for (int repetition = 0; repetition < TIMED_REPETITIONS; ++repetition) {
    runOnce();
  }
  const double milliseconds = timer.ms() / TIMED_REPETITIONS;
  CHECK(cudaGetLastError());
  return milliseconds;
}

static const char* identicalOrDiffer(bool identical) {
  return identical ? "identical" : "DIFFER";
}

template <unsigned COARSE_FACTOR>
static bool checkAtomicsVersion(const char* name,
                                const DeviceTrips& device,
                                unsigned n,
                                const CpuReference& reference) {
  const unsigned blocks = cdiv(n, COARSE_FACTOR * BLOCK_DIM);
  Kpi* total = nullptr;
  CHECK(cudaMalloc(&total, sizeof(Kpi)));
  Kpi* const combineWithAtomicsInsteadOfPartials = nullptr;
  const Kpi identity = kpiIdentity();

  std::vector<Kpi> runs;
  const double ms = millisecondsPerRun([&] {
    CHECK(cudaMemcpy(total, &identity, sizeof(Kpi), cudaMemcpyHostToDevice));
    launch(KpiKernel<COARSE_FACTOR>,
           dim3(blocks),
           dim3(BLOCK_DIM),
           device.status,
           device.rideMs,
           device.distanceM,
           device.fareCents,
           n,
           total,
           combineWithAtomicsInsteadOfPartials);
    runs.push_back(readKpi(total));
  });
  CHECK(cudaFree(total));

  bool floatStable = true;
  bool integersStable = true;
  for (const Kpi& run : runs) {
    floatStable = floatStable && sameFloat(run.fareBs, runs[0].fareBs);
    integersStable = integersStable && sameIntegers(run, runs[0]);
  }
  char note[96];
  std::snprintf(note,
                sizeof note,
                "  (%zu run%s: float %s, integers %s)",
                runs.size(),
                runs.size() == 1 ? "" : "s",
                identicalOrDiffer(floatStable),
                identicalOrDiffer(integersStable));
  const bool rowOk = reportRow(name, blocks, ms, runs.back(), reference, note);
  return rowOk && integersStable;
}

static PartialsRun runPartialsVersion(const DeviceTrips& device, unsigned n) {
  PartialsRun run;
  run.blocks = cdiv(n, PARTIALS_COARSE_FACTOR * BLOCK_DIM);
  Kpi* partials = nullptr;
  Kpi* total = nullptr;
  CHECK(cudaMalloc(&partials, run.blocks * sizeof(Kpi)));
  CHECK(cudaMalloc(&total, sizeof(Kpi)));
  Kpi* const unusedAtomicTotal = nullptr;

  Timer timer;
  launch(KpiKernel<PARTIALS_COARSE_FACTOR>,
         dim3(run.blocks),
         dim3(BLOCK_DIM),
         device.status,
         device.rideMs,
         device.distanceM,
         device.fareCents,
         n,
         unusedAtomicTotal,
         partials);
  launch(KpiCombineKernel, dim3(1), dim3(BLOCK_DIM), partials, run.blocks, total);
  run.ms = timer.ms();
  CHECK(cudaGetLastError());

  run.total = readKpi(total);
  run.partials.resize(run.blocks);
  CHECK(
      cudaMemcpy(run.partials.data(), partials, run.blocks * sizeof(Kpi), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(partials));
  CHECK(cudaFree(total));
  return run;
}

static void shuffleInPlace(std::vector<unsigned>& sequence, SplitMix64& random) {
  for (size_t j = sequence.size() - 1; j > 0; --j) {
    const size_t other = static_cast<size_t>(random.next() % (j + 1));
    std::swap(sequence[j], sequence[other]);
  }
}

template <class T>
static size_t sortAndCountDistinct(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  return static_cast<size_t>(std::unique(values.begin(), values.end()) - values.begin());
}

static bool checkBlockSequenceReplay(const std::vector<Kpi>& partials,
                                     const CpuReference& reference) {
  std::vector<unsigned> finishSequence(partials.size());
  for (unsigned j = 0; j < finishSequence.size(); ++j) {
    finishSequence[j] = j;
  }
  std::vector<float> floatTotals;
  std::vector<unsigned long long> centTotals;
  SplitMix64 random(REPLAY_SEED);
  for (int replay = 0; replay < REPLAYED_BLOCK_SEQUENCES; ++replay) {
    shuffleInPlace(finishSequence, random);
    float floatTotal = 0.0f;
    unsigned long long centTotal = 0;
    for (unsigned block : finishSequence) {
      floatTotal += partials[block].fareBs;
      centTotal += partials[block].fareCents;
    }
    floatTotals.push_back(floatTotal);
    centTotals.push_back(centTotal);
  }

  const size_t distinctFloatTotals = sortAndCountDistinct(floatTotals);
  const size_t distinctCentTotals = sortAndCountDistinct(centTotals);
  std::printf(
      "\n%zu block results added in 1000 random sequences (what atomicAdd sees on a GPU):\n",
      partials.size());
  std::printf("  float bolivianos: %zu distinct totals, from %.4f to %.4f\n",
              distinctFloatTotals,
              static_cast<double>(floatTotals.front()),
              static_cast<double>(floatTotals[distinctFloatTotals - 1]));
  std::printf("  integer centavos: %zu distinct total%s, %llu.%02llu\n",
              distinctCentTotals,
              distinctCentTotals == 1 ? "" : "s",
              centTotals.front() / CENTS_PER_BOLIVIANO,
              centTotals.front() % CENTS_PER_BOLIVIANO);
  return distinctCentTotals == 1 && centTotals.front() == reference.kpi.fareCents;
}

static double percentOf(unsigned part, unsigned whole) {
  return whole > 0 ? 100.0 * part / whole : 0.0;
}

static double averageOf(unsigned long long total, unsigned count) {
  return count > 0 ? static_cast<double>(total) / count : 0.0;
}

static double floatStepAt(double value) {
  const float asFloat = static_cast<float>(value);
  const float next = std::nextafter(asFloat, std::numeric_limits<float>::infinity());
  return static_cast<double>(next) - static_cast<double>(asFloat);
}

static void printKpis(const Kpi& kpi) {
  std::printf("\nday KPIs: %u completed trips, revenue Bs %llu.%02llu, %u late returns (%.2f%%)\n",
              kpi.trips,
              kpi.fareCents / CENTS_PER_BOLIVIANO,
              kpi.fareCents % CENTS_PER_BOLIVIANO,
              kpi.late,
              percentOf(kpi.late, kpi.trips));
  std::printf(
      "          shortest ride %d min %02d s, longest %d min %02d s, %.1f km ridden, %.0f m "
      "per trip\n",
      kpi.minMs / MINUTE_MS,
      kpi.minMs / SECOND_MS % SECONDS_PER_MINUTE,
      kpi.maxMs / MINUTE_MS,
      kpi.maxMs / SECOND_MS % SECONDS_PER_MINUTE,
      static_cast<double>(kpi.distanceM) / METRES_PER_KM,
      averageOf(kpi.distanceM, kpi.trips));
  const double exactBs = exactRevenueBs(kpi);
  std::printf(
      "float revenue: one rounding per addition to 24 bits; at %.0f Bs one float step is %.4f "
      "Bs.\n",
      exactBs,
      floatStepAt(exactBs));
}

int main() {
  std::printf(
      "Bike-share KPIs in one pass, a struct reduction (chapter 10 use case), backend: %s\n",
      CUDA_BACKEND);
  const unsigned n = TRIP_COUNT;
  const Trips trips = syntheticDay(n, DATA_SEED);
  std::printf("source: synthetic, %u trips (seed 2025), %u threads per block\n", n, BLOCK_DIM);

  const CpuReference reference = computeCpuReference(trips);
  const DeviceTrips device = copyTripsToDevice(trips);
  printTableHeader(reference);

  const bool smallFactorOk = checkAtomicsVersion<SMALL_ATOMICS_COARSE_FACTOR>(
      "KpiKernel CF=128, atomics per field", device, n, reference);
  const bool largeFactorOk = checkAtomicsVersion<LARGE_ATOMICS_COARSE_FACTOR>(
      "KpiKernel CF=256, atomics per field", device, n, reference);

  const PartialsRun partialsRun = runPartialsVersion(device, n);
  const bool partialsOk = reportRow("KpiKernel CF=64, partials + 2nd pass",
                                    partialsRun.blocks + 1,
                                    partialsRun.ms,
                                    partialsRun.total,
                                    reference,
                                    "  (fixed sequence)");
  std::printf("%-40s %6s %9s %18.4f %11.4f  %s\n",
              "exact: integer centavos / 100",
              "-",
              "-",
              exactRevenueBs(reference.kpi),
              0.0,
              "the right answer");

  const bool replayOk = checkBlockSequenceReplay(partialsRun.partials, reference);
  printKpis(partialsRun.total);
  freeDeviceTrips(device);

  const bool ok = smallFactorOk && largeFactorOk && partialsOk && replayOk;
  std::printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  if (IS_CPU_EMULATION) {
    std::printf(
        "(CPU emulation: blocks run one after another, so the atomics always see the same "
        "sequence.)\n");
  }
  return ok ? 0 : 1;
}
