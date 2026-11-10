#include "cuda_compat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr unsigned BLOCK_DIM = 256;
constexpr unsigned COARSE_FACTOR = 16;
constexpr unsigned TRIPS_PER_SUMMARY_BLOCK = 2 * COARSE_FACTOR * BLOCK_DIM;
constexpr int MAX_IDENTITY_FOR_DURATIONS = 0;

constexpr int MINUTE_MS = 60000;
constexpr int SECOND_MS = 1000;
constexpr int SECONDS_PER_MINUTE = 60;
constexpr int INCLUDED_RIDE_MS = 45 * MINUTE_MS;
constexpr int FARE_STEP_MS = 10 * MINUTE_MS;
constexpr int BASE_FARE_CENTS = 300;
constexpr int CENTS_PER_FARE_STEP = 100;
constexpr unsigned long long CENTS_PER_BOLIVIANO = 100;

enum class TripStatus : int { Returned = 0, StillRiding = 1, SentToRepair = 2 };

#ifndef CPU_TRIPS
#define CPU_TRIPS (1u << 18)
#endif
#ifndef CPU_BIKES
#define CPU_BIKES 240u
#endif
const unsigned SYNTHETIC_TRIPS = BY_BACKEND(1u << 24, CPU_TRIPS);
const unsigned SYNTHETIC_BIKES = BY_BACKEND(480u, CPU_BIKES);
constexpr unsigned STATION_COUNT = 48;
constexpr uint64_t SYNTHETIC_SEED = 2025;
const int TIMED_REPETITIONS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

constexpr int PER_MILLE = 1000;
constexpr int STILL_RIDING_PER_MILLE = 30;
constexpr int SENT_TO_REPAIR_PER_MILLE = 15;
constexpr int LEISURE_RIDE_PERCENT = 6;
constexpr int EBIKE_EVERY = 4;
constexpr int EBIKE_EXTRA_RIDE_PERCENT = 25;

struct Totals {
  unsigned returned;
  unsigned stillRiding;
  unsigned sentToRepair;
  unsigned late;
  unsigned long long fareCents;
  unsigned long long rideMs;
  int maxRideMs;
};

struct BikeTotals {
  unsigned rides;
  unsigned repairs;
  unsigned long long rideMs;
  int maxRideMs;
};

__host__ __device__ inline Totals emptyTotals() {
  return Totals{0u, 0u, 0u, 0u, 0ull, 0ull, MAX_IDENTITY_FOR_DURATIONS};
}

__host__ __device__ inline BikeTotals emptyBikeTotals() {
  return BikeTotals{0u, 0u, 0ull, MAX_IDENTITY_FOR_DURATIONS};
}

__host__ __device__ inline bool isLate(int rideMs) {
  return rideMs > INCLUDED_RIDE_MS;
}

__host__ __device__ inline void addTrip(Totals& totals,
                                        TripStatus status,
                                        int rideMs,
                                        int fareCents) {
  if (status == TripStatus::StillRiding) {
    ++totals.stillRiding;
    return;
  }

  if (status == TripStatus::Returned) {
    ++totals.returned;
  } else {
    ++totals.sentToRepair;
  }
  if (isLate(rideMs)) {
    ++totals.late;
  }
  totals.fareCents += static_cast<unsigned long long>(fareCents);
  totals.rideMs += static_cast<unsigned long long>(rideMs);
  totals.maxRideMs = imax(totals.maxRideMs, rideMs);
}

__host__ __device__ inline void addBikeTrip(BikeTotals& totals, TripStatus status, int rideMs) {
  if (status == TripStatus::StillRiding) {
    return;
  }

  ++totals.rides;
  if (status == TripStatus::SentToRepair) {
    ++totals.repairs;
  }
  totals.rideMs += static_cast<unsigned long long>(rideMs);
  totals.maxRideMs = imax(totals.maxRideMs, rideMs);
}

__global__ void SummaryKernel(
    const TripStatus* status, const int* rideMs, const int* fareCents, unsigned n, Totals* totals) {
  __shared__ unsigned returned_s[BLOCK_DIM];
  __shared__ unsigned stillRiding_s[BLOCK_DIM];
  __shared__ unsigned sentToRepair_s[BLOCK_DIM];
  __shared__ unsigned late_s[BLOCK_DIM];
  __shared__ unsigned long long fareCents_s[BLOCK_DIM];
  __shared__ unsigned long long rideMs_s[BLOCK_DIM];
  __shared__ int maxRideMs_s[BLOCK_DIM];

  unsigned t = threadIdx.x;
  unsigned segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  Totals partial = emptyTotals();
  for (unsigned tile = 0; tile < 2 * COARSE_FACTOR; ++tile) {
    unsigned i = segment + t + tile * BLOCK_DIM;
    if (i < n) {
      addTrip(partial, status[i], rideMs[i], fareCents[i]);
    }
  }
  returned_s[t] = partial.returned;
  stillRiding_s[t] = partial.stillRiding;
  sentToRepair_s[t] = partial.sentToRepair;
  late_s[t] = partial.late;
  fareCents_s[t] = partial.fareCents;
  rideMs_s[t] = partial.rideMs;
  maxRideMs_s[t] = partial.maxRideMs;

  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      returned_s[t] += returned_s[t + stride];
      stillRiding_s[t] += stillRiding_s[t + stride];
      sentToRepair_s[t] += sentToRepair_s[t + stride];
      late_s[t] += late_s[t + stride];
      fareCents_s[t] += fareCents_s[t + stride];
      rideMs_s[t] += rideMs_s[t + stride];
      maxRideMs_s[t] = imax(maxRideMs_s[t], maxRideMs_s[t + stride]);
    }
  }

  if (t == 0) {
    atomicAdd(&totals->returned, returned_s[0]);
    atomicAdd(&totals->stillRiding, stillRiding_s[0]);
    atomicAdd(&totals->sentToRepair, sentToRepair_s[0]);
    atomicAdd(&totals->late, late_s[0]);
    atomicAdd(&totals->fareCents, fareCents_s[0]);
    atomicAdd(&totals->rideMs, rideMs_s[0]);
    atomicMax(&totals->maxRideMs, maxRideMs_s[0]);
  }
}

__global__ void PerBikeKernel(const TripStatus* status,
                              const int* rideMs,
                              const unsigned* offset,
                              unsigned* rides,
                              unsigned* repairs,
                              unsigned long long* totalRideMs,
                              int* maxRideMs) {
  __shared__ unsigned rides_s[BLOCK_DIM];
  __shared__ unsigned repairs_s[BLOCK_DIM];
  __shared__ unsigned long long rideMs_s[BLOCK_DIM];
  __shared__ int maxRideMs_s[BLOCK_DIM];

  unsigned t = threadIdx.x;
  unsigned bike = blockIdx.x;
  BikeTotals partial = emptyBikeTotals();
  for (unsigned i = offset[bike] + t; i < offset[bike + 1]; i += blockDim.x) {
    addBikeTrip(partial, status[i], rideMs[i]);
  }
  rides_s[t] = partial.rides;
  repairs_s[t] = partial.repairs;
  rideMs_s[t] = partial.rideMs;
  maxRideMs_s[t] = partial.maxRideMs;

  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      rides_s[t] += rides_s[t + stride];
      repairs_s[t] += repairs_s[t + stride];
      rideMs_s[t] += rideMs_s[t + stride];
      maxRideMs_s[t] = imax(maxRideMs_s[t], maxRideMs_s[t + stride]);
    }
  }

  if (t == 0) {
    rides[bike] = rides_s[0];
    repairs[bike] = repairs_s[0];
    totalRideMs[bike] = rideMs_s[0];
    maxRideMs[bike] = maxRideMs_s[0];
  }
}

struct Trips {
  std::vector<TripStatus> status;
  std::vector<unsigned> bike;
  std::vector<int> rideMs;
  std::vector<int> fareCents;
};

struct TripsByBike {
  std::vector<unsigned> offset;
  std::vector<TripStatus> status;
  std::vector<int> rideMs;
};

struct SummaryRun {
  Totals totals;
  unsigned blocks;
  double ms;
};

struct PerBikeRun {
  std::vector<BikeTotals> bikes;
  double ms;
};

struct CpuReference {
  Totals totals;
  std::vector<BikeTotals> bikes;
  double ms;
};

static bool operator==(const Totals& a, const Totals& b) {
  return a.returned == b.returned && a.stillRiding == b.stillRiding &&
         a.sentToRepair == b.sentToRepair && a.late == b.late && a.fareCents == b.fareCents &&
         a.rideMs == b.rideMs && a.maxRideMs == b.maxRideMs;
}

static bool operator==(const BikeTotals& a, const BikeTotals& b) {
  return a.rides == b.rides && a.repairs == b.repairs && a.rideMs == b.rideMs &&
         a.maxRideMs == b.maxRideMs;
}

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
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

static int fareCentsForRide(int rideMs) {
  const int startedFareSteps = (rideMs + FARE_STEP_MS - 1) / FARE_STEP_MS;
  return BASE_FARE_CENTS + startedFareSteps * CENTS_PER_FARE_STEP;
}

static bool isEBike(unsigned bike) {
  return bike % EBIKE_EVERY == 0;
}

static int syntheticRideMs(SplitMix64& random, unsigned bike) {
  int rideMs = random.range(4 * MINUTE_MS, 25 * MINUTE_MS);
  const bool isLeisureRide = random.range(0, 99) < LEISURE_RIDE_PERCENT;
  if (isLeisureRide) {
    rideMs = random.range(30 * MINUTE_MS, 110 * MINUTE_MS);
  }
  if (isEBike(bike)) {
    rideMs += rideMs / 100 * EBIKE_EXTRA_RIDE_PERCENT;
  }
  return rideMs;
}

static TripStatus syntheticStatus(SplitMix64& random) {
  const int outcome = random.range(0, PER_MILLE - 1);
  if (outcome < STILL_RIDING_PER_MILLE) {
    return TripStatus::StillRiding;
  }
  if (outcome < STILL_RIDING_PER_MILLE + SENT_TO_REPAIR_PER_MILLE) {
    return TripStatus::SentToRepair;
  }
  return TripStatus::Returned;
}

static Trips syntheticTrips(unsigned count, unsigned bikes, uint64_t seed) {
  SplitMix64 random(seed);
  Trips trips;
  for (unsigned i = 0; i < count; ++i) {
    const TripStatus status = syntheticStatus(random);
    const unsigned bike = static_cast<unsigned>(random.range(0, static_cast<int>(bikes) - 1));
    const bool hasEnded = status != TripStatus::StillRiding;
    const int rideMs = hasEnded ? syntheticRideMs(random, bike) : 0;
    const int fareCents = hasEnded ? fareCentsForRide(rideMs) : 0;

    trips.status.push_back(status);
    trips.bike.push_back(bike);
    trips.rideMs.push_back(rideMs);
    trips.fareCents.push_back(fareCents);
  }
  return trips;
}

static std::vector<unsigned> bikeOffsets(const Trips& trips, unsigned bikes) {
  std::vector<unsigned> offset(bikes + 1, 0);
  for (unsigned bike : trips.bike) {
    ++offset[bike + 1];
  }
  for (unsigned b = 0; b < bikes; ++b) {
    offset[b + 1] += offset[b];
  }
  return offset;
}

static TripsByBike groupByBike(const Trips& trips, unsigned bikes) {
  TripsByBike grouped;
  grouped.offset = bikeOffsets(trips, bikes);
  const unsigned groupedCount = grouped.offset[bikes];
  grouped.status.resize(groupedCount);
  grouped.rideMs.resize(groupedCount);

  std::vector<unsigned> nextSlot(grouped.offset.begin(), grouped.offset.end() - 1);
  for (size_t i = 0; i < trips.status.size(); ++i) {
    const unsigned slot = nextSlot[trips.bike[i]]++;
    grouped.status[slot] = trips.status[i];
    grouped.rideMs[slot] = trips.rideMs[i];
  }
  return grouped;
}

static CpuReference computeCpuReference(const Trips& trips, unsigned bikes) {
  Timer timer;
  CpuReference reference;
  reference.totals = emptyTotals();
  reference.bikes.assign(bikes, emptyBikeTotals());
  for (size_t i = 0; i < trips.status.size(); ++i) {
    addTrip(reference.totals, trips.status[i], trips.rideMs[i], trips.fareCents[i]);
    addBikeTrip(reference.bikes[trips.bike[i]], trips.status[i], trips.rideMs[i]);
  }
  reference.ms = timer.ms();
  return reference;
}

template <class T>
static T* allocateOnDevice(size_t count) {
  const size_t nonEmptyCount = std::max<size_t>(count, 1);
  T* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, nonEmptyCount * sizeof(T)));
  return pointer;
}

template <class T>
static T* deviceCopyOf(const std::vector<T>& values) {
  T* pointer = allocateOnDevice<T>(values.size());
  if (!values.empty()) {
    CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  return pointer;
}

template <class T>
static std::vector<T> hostCopyOf(const T* values, size_t count) {
  std::vector<T> copy(count);
  if (count > 0) {
    CHECK(cudaMemcpy(copy.data(), values, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  return copy;
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

static SummaryRun runSummaryKernel(const Trips& trips) {
  const unsigned n = static_cast<unsigned>(trips.status.size());
  TripStatus* status = deviceCopyOf(trips.status);
  int* rideMs = deviceCopyOf(trips.rideMs);
  int* fareCents = deviceCopyOf(trips.fareCents);
  Totals* totals = allocateOnDevice<Totals>(1);
  const Totals zero = emptyTotals();

  SummaryRun run;
  run.blocks = n > 0 ? cdiv(n, TRIPS_PER_SUMMARY_BLOCK) : 1;
  run.ms = millisecondsPerRun([&] {
    CHECK(cudaMemcpy(totals, &zero, sizeof(Totals), cudaMemcpyHostToDevice));
    launch(SummaryKernel, dim3(run.blocks), dim3(BLOCK_DIM), status, rideMs, fareCents, n, totals);
  });
  CHECK(cudaMemcpy(&run.totals, totals, sizeof(Totals), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(status));
  CHECK(cudaFree(rideMs));
  CHECK(cudaFree(fareCents));
  CHECK(cudaFree(totals));
  return run;
}

static PerBikeRun runPerBikeKernel(const TripsByBike& grouped, unsigned bikes) {
  TripStatus* status = deviceCopyOf(grouped.status);
  int* rideMs = deviceCopyOf(grouped.rideMs);
  unsigned* offset = deviceCopyOf(grouped.offset);
  unsigned* rides = allocateOnDevice<unsigned>(bikes);
  unsigned* repairs = allocateOnDevice<unsigned>(bikes);
  unsigned long long* totalRideMs = allocateOnDevice<unsigned long long>(bikes);
  int* maxRideMs = allocateOnDevice<int>(bikes);

  PerBikeRun run;
  run.ms = millisecondsPerRun([&] {
    if (bikes > 0) {
      launch(PerBikeKernel,
             dim3(bikes),
             dim3(BLOCK_DIM),
             status,
             rideMs,
             offset,
             rides,
             repairs,
             totalRideMs,
             maxRideMs);
    }
  });

  const std::vector<unsigned> ridesCopy = hostCopyOf(rides, bikes);
  const std::vector<unsigned> repairsCopy = hostCopyOf(repairs, bikes);
  const std::vector<unsigned long long> rideMsCopy = hostCopyOf(totalRideMs, bikes);
  const std::vector<int> maxRideMsCopy = hostCopyOf(maxRideMs, bikes);
  for (unsigned b = 0; b < bikes; ++b) {
    run.bikes.push_back(BikeTotals{ridesCopy[b], repairsCopy[b], rideMsCopy[b], maxRideMsCopy[b]});
  }

  CHECK(cudaFree(status));
  CHECK(cudaFree(rideMs));
  CHECK(cudaFree(offset));
  CHECK(cudaFree(rides));
  CHECK(cudaFree(repairs));
  CHECK(cudaFree(totalRideMs));
  CHECK(cudaFree(maxRideMs));
  return run;
}

static unsigned long long ridesOfAllBikes(const std::vector<BikeTotals>& bikes) {
  unsigned long long total = 0;
  for (const BikeTotals& bike : bikes) {
    total += bike.rides;
  }
  return total;
}

static unsigned long long repairsOfAllBikes(const std::vector<BikeTotals>& bikes) {
  unsigned long long total = 0;
  for (const BikeTotals& bike : bikes) {
    total += bike.repairs;
  }
  return total;
}

static unsigned finishedTrips(const Totals& totals) {
  return totals.returned + totals.sentToRepair;
}

static const char* yesOrNo(bool holds) {
  return holds ? "yes" : "NO";
}

static void printTimingTable(const CpuReference& reference,
                             const SummaryRun& summary,
                             bool summaryOk,
                             const PerBikeRun& perBike,
                             bool perBikeOk) {
  std::printf("\n%-34s %7s %10s  %s\n", "kernel", "blocks", "ms", "check");
  std::printf("%-34s %7s %10.3f  %s\n", "CPU sequential (both)", "-", reference.ms, "reference");
  std::printf("%-34s %7u %10.3f  %s\n",
              "SummaryKernel (7 fused reductions)",
              summary.blocks,
              summary.ms,
              passOrFail(summaryOk));
  std::printf("%-34s %7zu %10.3f  %s\n",
              "PerBikeKernel (segmented)",
              perBike.bikes.size(),
              perBike.ms,
              passOrFail(perBikeOk));
}

static unsigned long long roundedAverageMs(const Totals& totals) {
  const unsigned finished = finishedTrips(totals);
  if (finished == 0) {
    return 0;
  }
  return (totals.rideMs + finished / 2) / finished;
}

static double percentOf(unsigned part, unsigned whole) {
  return whole > 0 ? 100.0 * part / whole : 0.0;
}

static void printDuration(const char* label, unsigned long long ms) {
  std::printf(
      "  %-26s %llu min %02llu s\n", label, ms / MINUTE_MS, ms / SECOND_MS % SECONDS_PER_MINUTE);
}

static void printBusiestAndMostRepairedBike(const std::vector<BikeTotals>& bikes) {
  if (bikes.empty()) {
    return;
  }
  size_t busiest = 0;
  size_t mostRepaired = 0;
  for (size_t b = 1; b < bikes.size(); ++b) {
    if (bikes[b].rides > bikes[busiest].rides) {
      busiest = b;
    }
    if (bikes[b].repairs > bikes[mostRepaired].repairs) {
      mostRepaired = b;
    }
  }
  std::printf("  %-26s b%zu, %u rides\n", "busiest bike:", busiest, bikes[busiest].rides);
  std::printf("  %-26s b%zu, %u trips ended in the workshop\n",
              "most repairs:",
              mostRepaired,
              bikes[mostRepaired].repairs);
}

static void printDayReport(size_t tripCount,
                           const Totals& totals,
                           const std::vector<BikeTotals>& bikes) {
  std::printf("\nBike-share day report: %u stations, %zu bikes\n", STATION_COUNT, bikes.size());
  std::printf("  %-26s %zu\n", "trips started:", tripCount);
  std::printf("  %-26s %u\n", "returned to a dock:", totals.returned);
  std::printf("  %-26s %u\n", "bikes sent to repair:", totals.sentToRepair);
  std::printf("  %-26s %u\n", "still riding at closing:", totals.stillRiding);
  std::printf("  %-26s %u (%.2f%% of finished trips)\n",
              "late returns (> 45 min):",
              totals.late,
              percentOf(totals.late, finishedTrips(totals)));
  std::printf("  %-26s Bs %llu.%02llu\n",
              "fare collected:",
              totals.fareCents / CENTS_PER_BOLIVIANO,
              totals.fareCents % CENTS_PER_BOLIVIANO);
  printDuration("average ride:", roundedAverageMs(totals));
  printDuration("longest ride:", static_cast<unsigned long long>(totals.maxRideMs));
  printBusiestAndMostRepairedBike(bikes);
}

int main() {
  std::printf("Bike-share day report as reductions (chapter 10 use case), backend: %s\n",
              CUDA_BACKEND);
  const unsigned bikes = SYNTHETIC_BIKES;
  const Trips trips = syntheticTrips(SYNTHETIC_TRIPS, bikes, SYNTHETIC_SEED);
  std::printf("source: synthetic, %zu trips, %u bikes (seed 2025)\n", trips.status.size(), bikes);

  const TripsByBike grouped = groupByBike(trips, bikes);
  const CpuReference reference = computeCpuReference(trips, bikes);
  const SummaryRun summary = runSummaryKernel(trips);
  const PerBikeRun perBike = runPerBikeKernel(grouped, bikes);

  const bool summaryOk = summary.totals == reference.totals;
  const bool perBikeOk = perBike.bikes == reference.bikes;
  const Totals& totals = summary.totals;
  const bool everyTripAccountedFor =
      finishedTrips(totals) + totals.stillRiding == trips.status.size();
  const bool bikeRidesAddUp = ridesOfAllBikes(perBike.bikes) == finishedTrips(totals);
  const bool bikeRepairsAddUp = repairsOfAllBikes(perBike.bikes) == totals.sentToRepair;

  printTimingTable(reference, summary, summaryOk, perBike, perBikeOk);
  std::printf("started == returned + repair + riding: %s\n", yesOrNo(everyTripAccountedFor));
  std::printf("sum of rides per bike == finished:     %s\n", yesOrNo(bikeRidesAddUp));
  std::printf("sum of repairs per bike == repair:     %s\n", yesOrNo(bikeRepairsAddUp));
  printDayReport(trips.status.size(), totals, perBike.bikes);

  const bool ok =
      summaryOk && perBikeOk && everyTripAccountedFor && bikeRidesAddUp && bikeRepairsAddUp;
  std::printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
