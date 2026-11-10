#include "cuda_compat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

constexpr unsigned BLOCK_DIM = 256;
constexpr unsigned WARP_SIZE = 32;
constexpr unsigned WARPS_PER_BLOCK = BLOCK_DIM / WARP_SIZE;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;
constexpr unsigned NO_INDEX = 0xffffffffu;
constexpr unsigned TOP_EARNER_SLOT = 0;
constexpr unsigned MOST_USED_SLOT = 1;

enum class TripStatus : int { Completed = 0, StillRiding = 1, FalseStart = 2 };

#ifndef CPU_TRIPS
#define CPU_TRIPS (1u << 15)
#endif
#ifndef CPU_BIKES
#define CPU_BIKES 16u
#endif
const unsigned TRIP_COUNT = BY_BACKEND(1u << 24, CPU_TRIPS);
const unsigned BIKE_COUNT = BY_BACKEND(4096u, CPU_BIKES);
const int TIMED_REPETITIONS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

struct BikeSum {
  unsigned long long distanceM;
  unsigned long long revenueCents;
  unsigned trips;
};

__host__ __device__ inline BikeSum sumIdentity() {
  return BikeSum{0ull, 0ull, 0u};
}

__host__ __device__ inline BikeSum combine(const BikeSum& a, const BikeSum& b) {
  return BikeSum{a.distanceM + b.distanceM, a.revenueCents + b.revenueCents, a.trips + b.trips};
}

__device__ inline BikeSum shuffleDownSum(const BikeSum& v, unsigned delta) {
  BikeSum r;
  r.distanceM = __shfl_down_sync(FULL_WARP_MASK, v.distanceM, delta);
  r.revenueCents = __shfl_down_sync(FULL_WARP_MASK, v.revenueCents, delta);
  r.trips = __shfl_down_sync(FULL_WARP_MASK, v.trips, delta);
  return r;
}

__device__ inline BikeSum warpReduceSum(BikeSum v) {
  for (unsigned stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    v = combine(v, shuffleDownSum(v, stride));
  }
  return v;
}

__global__ void BikeTotalsKernel(const TripStatus* status,
                                 const int* distanceM,
                                 const int* fareCents,
                                 const unsigned* offset,
                                 unsigned long long* distance,
                                 unsigned long long* revenue,
                                 unsigned* trips) {
  const unsigned bike = blockIdx.x;
  const unsigned warp = threadIdx.x / WARP_SIZE;
  const unsigned lane = threadIdx.x % WARP_SIZE;

  BikeSum partial = sumIdentity();
  for (unsigned i = offset[bike] + threadIdx.x; i < offset[bike + 1]; i += BLOCK_DIM) {
    if (status[i] == TripStatus::Completed) {
      const BikeSum trip{static_cast<unsigned long long>(distanceM[i]),
                         static_cast<unsigned long long>(fareCents[i]),
                         1u};
      partial = combine(partial, trip);
    }
  }

  __shared__ BikeSum warp_s[WARPS_PER_BLOCK];
  partial = warpReduceSum(partial);
  if (lane == 0) {
    warp_s[warp] = partial;
  }
  __syncthreads();

  if (warp == 0) {
    const bool laneHasWarpResult = lane < WARPS_PER_BLOCK;
    BikeSum total = laneHasWarpResult ? warp_s[lane] : sumIdentity();
    total = warpReduceSum(total);
    if (threadIdx.x == 0) {
      distance[bike] = total.distanceM;
      revenue[bike] = total.revenueCents;
      trips[bike] = total.trips;
    }
  }
}

struct Best {
  unsigned long long value;
  unsigned index;
};

__host__ __device__ inline Best bestIdentity() {
  return Best{0ull, NO_INDEX};
}

__host__ __device__ inline bool beats(const Best& a, const Best& b) {
  return a.value > b.value || (a.value == b.value && a.index < b.index);
}

__host__ __device__ inline Best better(const Best& a, const Best& b) {
  return beats(a, b) ? a : b;
}

__device__ inline Best shuffleDownBest(const Best& v, unsigned delta) {
  Best r;
  r.value = __shfl_down_sync(FULL_WARP_MASK, v.value, delta);
  r.index = __shfl_down_sync(FULL_WARP_MASK, v.index, delta);
  return r;
}

__device__ inline Best warpReduceBest(Best v) {
  for (unsigned stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    v = better(v, shuffleDownBest(v, stride));
  }
  return v;
}

__global__ void ArgMaxKernel(const unsigned long long* revenue,
                             const unsigned long long* distance,
                             unsigned bikes,
                             Best* out) {
  const unsigned warp = threadIdx.x / WARP_SIZE;
  const unsigned lane = threadIdx.x % WARP_SIZE;

  Best topEarner = bestIdentity();
  Best mostUsed = bestIdentity();
  for (unsigned b = threadIdx.x; b < bikes; b += BLOCK_DIM) {
    topEarner = better(topEarner, Best{revenue[b], b});
    mostUsed = better(mostUsed, Best{distance[b], b});
  }

  __shared__ Best topEarner_s[WARPS_PER_BLOCK];
  __shared__ Best mostUsed_s[WARPS_PER_BLOCK];
  topEarner = warpReduceBest(topEarner);
  mostUsed = warpReduceBest(mostUsed);
  if (lane == 0) {
    topEarner_s[warp] = topEarner;
    mostUsed_s[warp] = mostUsed;
  }
  __syncthreads();

  if (warp == 0) {
    const bool laneHasWarpResult = lane < WARPS_PER_BLOCK;
    Best blockEarner = laneHasWarpResult ? topEarner_s[lane] : bestIdentity();
    Best blockMostUsed = laneHasWarpResult ? mostUsed_s[lane] : bestIdentity();
    blockEarner = warpReduceBest(blockEarner);
    blockMostUsed = warpReduceBest(blockMostUsed);
    if (threadIdx.x == 0) {
      out[TOP_EARNER_SLOT] = blockEarner;
      out[MOST_USED_SLOT] = blockMostUsed;
    }
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
constexpr int PER_MILLE = 1000;
constexpr int STILL_RIDING_PER_MILLE = 30;
constexpr int FALSE_START_PER_MILLE = 50;
constexpr int DOWNTOWN_CHANCES_IN_TEN = 3;
constexpr int SHORTEST_TRIP_M = 300;
constexpr int LONGEST_TRIP_M = 8000;
constexpr int MINUTE_MS = 60000;
constexpr int MS_PER_METRE_AT_15_KMH = 240;
constexpr int FARE_STEP_MS = 10 * MINUTE_MS;
constexpr int BASE_FARE_CENTS = 300;
constexpr int CENTS_PER_FARE_STEP = 100;
constexpr int METRES_PER_KM = 1000;
constexpr unsigned CENTS_PER_BOLIVIANO = 100;
constexpr unsigned TOP_EARNERS_SHOWN = 5;

constexpr uint64_t TIE_TEST_SEED = 99;
constexpr unsigned TIE_TEST_BIKES = 3000;
constexpr unsigned long long RANDOM_TOTAL_LIMIT = 1000000;
constexpr unsigned long long TIED_MAXIMUM = 2000000;
constexpr unsigned TIED_EARNERS[] = {2999, 513, 259};
constexpr unsigned TIED_MOST_USED[] = {2050, 1795, 1000};

struct Trips {
  std::vector<TripStatus> status;
  std::vector<unsigned> bike;
  std::vector<int> distanceM;
  std::vector<int> fareCents;
};

struct TripsByBike {
  std::vector<unsigned> offset;
  std::vector<TripStatus> status;
  std::vector<int> distanceM;
  std::vector<int> fareCents;
};

struct BikeTotals {
  std::vector<unsigned long long> distanceM;
  std::vector<unsigned long long> revenueCents;
  std::vector<unsigned> trips;
};

struct ArgMaxResult {
  Best topEarner;
  Best mostUsed;
  double ms;
};

struct CpuReference {
  BikeTotals totals;
  Best topEarner;
  Best mostUsed;
  double ms;
};

static bool operator==(const BikeTotals& a, const BikeTotals& b) {
  return a.distanceM == b.distanceM && a.revenueCents == b.revenueCents && a.trips == b.trips;
}

static bool operator==(const Best& a, const Best& b) {
  return a.value == b.value && a.index == b.index;
}

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

static int fareCentsForRide(int rideMs) {
  const int startedFareSteps = (rideMs + FARE_STEP_MS - 1) / FARE_STEP_MS;
  return BASE_FARE_CENTS + startedFareSteps * CENTS_PER_FARE_STEP;
}

static int syntheticRideMs(SplitMix64& random, int distanceM) {
  const int pedallingMs = distanceM * MS_PER_METRE_AT_15_KMH;
  const int trafficLightsMs = random.range(1 * MINUTE_MS, 12 * MINUTE_MS);
  return pedallingMs + trafficLightsMs;
}

static TripStatus syntheticStatus(SplitMix64& random) {
  const int outcome = random.range(0, PER_MILLE - 1);
  if (outcome < STILL_RIDING_PER_MILLE) {
    return TripStatus::StillRiding;
  }
  if (outcome < STILL_RIDING_PER_MILLE + FALSE_START_PER_MILLE) {
    return TripStatus::FalseStart;
  }
  return TripStatus::Completed;
}

static unsigned syntheticBike(SplitMix64& random, unsigned bikes) {
  unsigned bike = static_cast<unsigned>(random.range(0, static_cast<int>(bikes) - 1));
  const bool docksDowntown = random.range(0, 9) < DOWNTOWN_CHANCES_IN_TEN;
  if (docksDowntown) {
    bike /= 2;
  }
  return bike;
}

static Trips syntheticDay(unsigned count, unsigned bikes, uint64_t seed) {
  SplitMix64 random(seed);
  Trips trips;
  for (unsigned i = 0; i < count; ++i) {
    const TripStatus status = syntheticStatus(random);
    const unsigned bike = syntheticBike(random, bikes);
    const bool isCompleted = status == TripStatus::Completed;
    const int distanceM = isCompleted ? random.range(SHORTEST_TRIP_M, LONGEST_TRIP_M) : 0;
    const int fareCents = isCompleted ? fareCentsForRide(syntheticRideMs(random, distanceM)) : 0;

    trips.status.push_back(status);
    trips.bike.push_back(bike);
    trips.distanceM.push_back(distanceM);
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
  grouped.distanceM.resize(groupedCount);
  grouped.fareCents.resize(groupedCount);

  std::vector<unsigned> nextSlot(grouped.offset.begin(), grouped.offset.end() - 1);
  for (size_t i = 0; i < trips.status.size(); ++i) {
    const unsigned slot = nextSlot[trips.bike[i]]++;
    grouped.status[slot] = trips.status[i];
    grouped.distanceM[slot] = trips.distanceM[i];
    grouped.fareCents[slot] = trips.fareCents[i];
  }
  return grouped;
}

static void printSliceLengths(const TripsByBike& grouped, unsigned bikes) {
  unsigned shortest = grouped.offset[bikes];
  unsigned longest = 0;
  for (unsigned b = 0; b < bikes; ++b) {
    const unsigned length = grouped.offset[b + 1] - grouped.offset[b];
    shortest = std::min(shortest, length);
    longest = std::max(longest, length);
  }
  std::printf("slices per bike: %u to %u trips (each block walks its slice %u at a time)\n",
              shortest,
              longest,
              BLOCK_DIM);
}

static Best cpuArgMax(const std::vector<unsigned long long>& values) {
  Best best = bestIdentity();
  for (unsigned b = 0; b < values.size(); ++b) {
    best = better(best, Best{values[b], b});
  }
  return best;
}

static CpuReference computeCpuReference(const Trips& trips, unsigned bikes) {
  Timer timer;
  CpuReference reference;
  reference.totals.distanceM.assign(bikes, 0);
  reference.totals.revenueCents.assign(bikes, 0);
  reference.totals.trips.assign(bikes, 0);
  for (size_t i = 0; i < trips.status.size(); ++i) {
    if (trips.status[i] != TripStatus::Completed) {
      continue;
    }
    const unsigned bike = trips.bike[i];
    reference.totals.distanceM[bike] += static_cast<unsigned long long>(trips.distanceM[i]);
    reference.totals.revenueCents[bike] += static_cast<unsigned long long>(trips.fareCents[i]);
    ++reference.totals.trips[bike];
  }
  reference.topEarner = cpuArgMax(reference.totals.revenueCents);
  reference.mostUsed = cpuArgMax(reference.totals.distanceM);
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
  CHECK(cudaMemcpy(copy.data(), values, count * sizeof(T), cudaMemcpyDeviceToHost));
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

static BikeTotals runBikeTotalsKernel(const TripsByBike& grouped, unsigned bikes, double& ms) {
  TripStatus* status = deviceCopyOf(grouped.status);
  int* distanceM = deviceCopyOf(grouped.distanceM);
  int* fareCents = deviceCopyOf(grouped.fareCents);
  unsigned* offset = deviceCopyOf(grouped.offset);
  unsigned long long* distance = allocateOnDevice<unsigned long long>(bikes);
  unsigned long long* revenue = allocateOnDevice<unsigned long long>(bikes);
  unsigned* trips = allocateOnDevice<unsigned>(bikes);

  ms = millisecondsPerRun([&] {
    launch(BikeTotalsKernel,
           dim3(bikes),
           dim3(BLOCK_DIM),
           status,
           distanceM,
           fareCents,
           offset,
           distance,
           revenue,
           trips);
  });

  BikeTotals totals;
  totals.distanceM = hostCopyOf(distance, bikes);
  totals.revenueCents = hostCopyOf(revenue, bikes);
  totals.trips = hostCopyOf(trips, bikes);

  CHECK(cudaFree(status));
  CHECK(cudaFree(distanceM));
  CHECK(cudaFree(fareCents));
  CHECK(cudaFree(offset));
  CHECK(cudaFree(distance));
  CHECK(cudaFree(revenue));
  CHECK(cudaFree(trips));
  return totals;
}

static ArgMaxResult runArgMaxKernel(const std::vector<unsigned long long>& revenue,
                                    const std::vector<unsigned long long>& distance) {
  const unsigned bikes = static_cast<unsigned>(revenue.size());
  unsigned long long* deviceRevenue = deviceCopyOf(revenue);
  unsigned long long* deviceDistance = deviceCopyOf(distance);
  Best* out = allocateOnDevice<Best>(2);

  ArgMaxResult result;
  result.ms = millisecondsPerRun([&] {
    launch(ArgMaxKernel, dim3(1), dim3(BLOCK_DIM), deviceRevenue, deviceDistance, bikes, out);
  });
  const std::vector<Best> winners = hostCopyOf(out, 2);
  result.topEarner = winners[TOP_EARNER_SLOT];
  result.mostUsed = winners[MOST_USED_SLOT];

  CHECK(cudaFree(deviceRevenue));
  CHECK(cudaFree(deviceDistance));
  CHECK(cudaFree(out));
  return result;
}

static void printTimingTable(
    double cpuMs, unsigned bikes, double totalsMs, bool totalsOk, double argMaxMs, bool argMaxOk) {
  std::printf("\n%-36s %7s %10s  %s\n", "kernel", "blocks", "ms", "check");
  std::printf(
      "%-36s %7s %10.3f  %s\n", "CPU sequential (totals + argmax)", "-", cpuMs, "reference");
  std::printf("%-36s %7u %10.3f  %s\n",
              "BikeTotalsKernel (segmented)",
              bikes,
              totalsMs,
              passOrFail(totalsOk));
  std::printf("%-36s %7u %10.3f  %s\n",
              "ArgMaxKernel (value, index) pairs",
              1u,
              argMaxMs,
              passOrFail(argMaxOk));
}

static std::vector<unsigned> bikesByRevenue(const std::vector<unsigned long long>& revenue) {
  std::vector<unsigned> ranking(revenue.size());
  for (unsigned b = 0; b < ranking.size(); ++b) {
    ranking[b] = b;
  }
  std::sort(ranking.begin(), ranking.end(), [&](unsigned a, unsigned b) {
    return beats(Best{revenue[a], a}, Best{revenue[b], b});
  });
  return ranking;
}

static void printTopEarners(const BikeTotals& totals, const ArgMaxResult& winners) {
  const std::vector<unsigned> ranking = bikesByRevenue(totals.revenueCents);
  const size_t shown = std::min<size_t>(TOP_EARNERS_SHOWN, ranking.size());
  std::printf("\n%-8s %8s %14s %12s\n", "bike", "trips", "revenue (Bs)", "ridden (km)");
  for (size_t k = 0; k < shown; ++k) {
    const unsigned b = ranking[k];
    const std::string name = "b" + std::to_string(b);
    std::printf("%-8s %8u %10llu.%02llu %12.1f\n",
                name.c_str(),
                totals.trips[b],
                totals.revenueCents[b] / CENTS_PER_BOLIVIANO,
                totals.revenueCents[b] % CENTS_PER_BOLIVIANO,
                static_cast<double>(totals.distanceM[b]) / METRES_PER_KM);
  }
  std::printf("top earner: b%u, Bs %llu.%02llu\n",
              winners.topEarner.index,
              winners.topEarner.value / CENTS_PER_BOLIVIANO,
              winners.topEarner.value % CENTS_PER_BOLIVIANO);
  std::printf("most used:  b%u, %.1f km\n",
              winners.mostUsed.index,
              static_cast<double>(winners.mostUsed.value) / METRES_PER_KM);
}

template <size_t COUNT>
static unsigned lowestOf(const unsigned (&indices)[COUNT]) {
  return *std::min_element(indices, indices + COUNT);
}

static bool checkInjectedTies() {
  std::vector<unsigned long long> revenue(TIE_TEST_BIKES);
  std::vector<unsigned long long> distance(TIE_TEST_BIKES);
  SplitMix64 random(TIE_TEST_SEED);
  for (unsigned b = 0; b < TIE_TEST_BIKES; ++b) {
    revenue[b] = random.next() % RANDOM_TOTAL_LIMIT;
    distance[b] = random.next() % RANDOM_TOTAL_LIMIT;
  }
  for (unsigned bike : TIED_EARNERS) {
    revenue[bike] = TIED_MAXIMUM;
  }
  for (unsigned bike : TIED_MOST_USED) {
    distance[bike] = TIED_MAXIMUM;
  }

  const ArgMaxResult tie = runArgMaxKernel(revenue, distance);
  const bool ok = tie.topEarner == cpuArgMax(revenue) && tie.mostUsed == cpuArgMax(distance) &&
                  tie.topEarner.index == lowestOf(TIED_EARNERS) &&
                  tie.mostUsed.index == lowestOf(TIED_MOST_USED);
  std::printf(
      "\ntie test: revenue tied at b%u, b%u, b%u -> b%u; distance tied at b%u, b%u, b%u -> b%u  "
      "%s\n",
      TIED_EARNERS[0],
      TIED_EARNERS[1],
      TIED_EARNERS[2],
      tie.topEarner.index,
      TIED_MOST_USED[0],
      TIED_MOST_USED[1],
      TIED_MOST_USED[2],
      tie.mostUsed.index,
      passOrFail(ok));
  return ok;
}

static bool checkAllZeroTotals(unsigned bikes) {
  const std::vector<unsigned long long> zeros(bikes, 0);
  const ArgMaxResult zero = runArgMaxKernel(zeros, zeros);
  const bool ok =
      zero.topEarner.index == 0 && zero.mostUsed.index == 0 && zero.topEarner.value == 0;
  std::printf("tie test: all bikes at 0 -> b%u and b%u  %s\n",
              zero.topEarner.index,
              zero.mostUsed.index,
              passOrFail(ok));
  return ok;
}

int main() {
  std::printf(
      "Per-bike totals, the most used bike and the top earner (chapter 10 use case), "
      "backend: %s\n",
      CUDA_BACKEND);
  const unsigned bikes = BIKE_COUNT;
  const Trips trips = syntheticDay(TRIP_COUNT, bikes, DATA_SEED);
  std::printf("source: synthetic, %u trips, %u bikes (seed 2025), %u threads per block\n",
              TRIP_COUNT,
              bikes,
              BLOCK_DIM);

  const TripsByBike grouped = groupByBike(trips, bikes);
  printSliceLengths(grouped, bikes);
  const CpuReference reference = computeCpuReference(trips, bikes);

  double totalsMs = 0;
  const BikeTotals totals = runBikeTotalsKernel(grouped, bikes, totalsMs);
  const bool totalsOk = totals == reference.totals;
  const ArgMaxResult winners = runArgMaxKernel(totals.revenueCents, totals.distanceM);
  const bool argMaxOk =
      winners.topEarner == reference.topEarner && winners.mostUsed == reference.mostUsed;

  printTimingTable(reference.ms, bikes, totalsMs, totalsOk, winners.ms, argMaxOk);
  printTopEarners(totals, winners);
  const bool injectedTiesOk = checkInjectedTies();
  const bool allZeroOk = checkAllZeroTotals(bikes);

  const bool ok = totalsOk && argMaxOk && injectedTiesOk && allZeroOk;
  std::printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
