#include "cuda_compat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr unsigned BLOCK_DIM = 256;
constexpr unsigned COARSE_FACTOR = 8;

constexpr int MS_PER_METRE_AT_25_KMH = 144;
constexpr int STREET_DETOUR_PERCENT = 125;
constexpr int MS_PER_METRE = MS_PER_METRE_AT_25_KMH * STREET_DETOUR_PERCENT / 100;
constexpr int TRUCK_CAPACITY_BIKES = 20;

constexpr unsigned NO_ETA = 0xffffffffu;
constexpr unsigned long long NO_KEY = ~0ull;
constexpr int TRUCK_INDEX_BITS = 32;
constexpr unsigned long long TRUCK_INDEX_MASK = (1ull << TRUCK_INDEX_BITS) - 1;

enum class TruckState : int { Parked = 0, Driving = 1, Loading = 2, OffDuty = 3 };

struct EmptyStation {
  int id;
  int x;
  int y;
  int docks;
  int curbOpensInMs;
};

__host__ __device__ inline int iabs(int x) {
  return x < 0 ? -x : x;
}

__host__ __device__ inline unsigned long long umin64(unsigned long long a, unsigned long long b) {
  return a < b ? a : b;
}

__host__ __device__ inline int manhattanMetres(int fromX, int fromY, int toX, int toY) {
  return iabs(fromX - toX) + iabs(fromY - toY);
}

__host__ __device__ inline bool canRestock(TruckState state, int bikesLoaded) {
  return state != TruckState::OffDuty && bikesLoaded > 0;
}

__host__ __device__ inline unsigned etaMs(int truckX,
                                          int truckY,
                                          TruckState state,
                                          int bikesLoaded,
                                          int busyMs,
                                          const EmptyStation& station) {
  if (!canRestock(state, bikesLoaded)) {
    return NO_ETA;
  }
  const int driveMs = manhattanMetres(truckX, truckY, station.x, station.y) * MS_PER_METRE;
  const int arrivalMs = busyMs + driveMs;
  const int stopMs = arrivalMs > station.curbOpensInMs ? arrivalMs : station.curbOpensInMs;
  return static_cast<unsigned>(stopMs);
}

__host__ __device__ inline unsigned long long packKey(unsigned eta, unsigned truck) {
  return (static_cast<unsigned long long>(eta) << TRUCK_INDEX_BITS) | truck;
}

__host__ __device__ inline unsigned long long candidateKey(unsigned eta, unsigned truck) {
  return eta == NO_ETA ? NO_KEY : packKey(eta, truck);
}

static unsigned etaOfKey(unsigned long long key) {
  return static_cast<unsigned>(key >> TRUCK_INDEX_BITS);
}

static unsigned truckOfKey(unsigned long long key) {
  return static_cast<unsigned>(key & TRUCK_INDEX_MASK);
}

__global__ void NearestTruckKernel(const int* truckX,
                                   const int* truckY,
                                   const TruckState* state,
                                   const int* bikesLoaded,
                                   const int* busyMs,
                                   unsigned numTrucks,
                                   const EmptyStation* stations,
                                   unsigned long long* best) {
  __shared__ unsigned long long key_s[BLOCK_DIM];
  const EmptyStation station = stations[blockIdx.y];
  unsigned t = threadIdx.x;
  unsigned segment = COARSE_FACTOR * blockDim.x * blockIdx.x;

  unsigned long long key = NO_KEY;
  for (unsigned tile = 0; tile < COARSE_FACTOR; ++tile) {
    unsigned truck = segment + t + tile * BLOCK_DIM;
    if (truck < numTrucks) {
      unsigned eta = etaMs(
          truckX[truck], truckY[truck], state[truck], bikesLoaded[truck], busyMs[truck], station);
      key = umin64(key, candidateKey(eta, truck));
    }
  }
  key_s[t] = key;

  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      key_s[t] = umin64(key_s[t], key_s[t + stride]);
    }
  }

  if (t == 0 && key_s[0] != NO_KEY) {
    atomicMin(&best[blockIdx.y], key_s[0]);
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

constexpr uint64_t DATA_SEED = 99;
constexpr unsigned DEFAULT_TRUCKS = BY_BACKEND(1u << 18, 1u << 14);
const int TIMED_REPETITIONS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

constexpr int MINUTE_MS = 60000;
constexpr int SECOND_MS = 1000;
constexpr int SECONDS_PER_MINUTE = 60;
constexpr int CITY_SIZE_M = 6000;
constexpr int STATION_COUNT = 48;
constexpr int FEWEST_DOCKS = 12;
constexpr int MOST_DOCKS = 30;
constexpr int EMPTY_STATION_PERCENT = 25;
constexpr int CURB_CLOSED_PERCENT = 50;
constexpr int LONGEST_CURB_CLOSURE_MS = 6 * MINUTE_MS;
constexpr int PARKED_PERCENT = 35;
constexpr int DRIVING_PERCENT = 30;
constexpr int LOADING_PERCENT = 20;

struct Station {
  int id;
  int x;
  int y;
  int docks;
  int bikesDocked;
};

struct Trucks {
  std::vector<int> x;
  std::vector<int> y;
  std::vector<TruckState> state;
  std::vector<int> bikesLoaded;
  std::vector<int> busyMs;
};

struct DeviceTrucks {
  int* x;
  int* y;
  TruckState* state;
  int* bikesLoaded;
  int* busyMs;
};

struct CpuReference {
  std::vector<unsigned long long> best;
  double ms;
};

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

static void appendTruck(Trucks& trucks, TruckState state, int bikesLoaded, int busyMs) {
  trucks.state.push_back(state);
  trucks.bikesLoaded.push_back(bikesLoaded);
  trucks.busyMs.push_back(busyMs);
}

static void addRandomTruck(Trucks& trucks, SplitMix64& random) {
  trucks.x.push_back(random.range(0, CITY_SIZE_M));
  trucks.y.push_back(random.range(0, CITY_SIZE_M));
  const int roll = random.range(0, 99);
  if (roll < PARKED_PERCENT) {
    appendTruck(trucks, TruckState::Parked, random.range(0, TRUCK_CAPACITY_BIKES), 0);
  } else if (roll < PARKED_PERCENT + DRIVING_PERCENT) {
    const int bikesLoaded = random.range(1, TRUCK_CAPACITY_BIKES);
    appendTruck(trucks, TruckState::Driving, bikesLoaded, random.range(0, 10 * MINUTE_MS));
  } else if (roll < PARKED_PERCENT + DRIVING_PERCENT + LOADING_PERCENT) {
    const int bikesLoaded = random.range(0, TRUCK_CAPACITY_BIKES - 1);
    appendTruck(trucks, TruckState::Loading, bikesLoaded, random.range(0, 15 * MINUTE_MS));
  } else {
    appendTruck(trucks, TruckState::OffDuty, 0, 0);
  }
}

static Trucks makeTrucks(unsigned count, SplitMix64& random) {
  Trucks trucks;
  for (unsigned k = 0; k < count; ++k) {
    addRandomTruck(trucks, random);
  }
  return trucks;
}

static std::vector<Station> makeStations(SplitMix64& random) {
  std::vector<Station> stations(STATION_COUNT);
  for (int s = 0; s < STATION_COUNT; ++s) {
    Station& station = stations[s];
    station.id = s;
    station.x = random.range(0, CITY_SIZE_M);
    station.y = random.range(0, CITY_SIZE_M);
    station.docks = random.range(FEWEST_DOCKS, MOST_DOCKS);
    const bool emptiedByRushHour = random.range(0, 99) < EMPTY_STATION_PERCENT;
    station.bikesDocked = emptiedByRushHour ? 0 : random.range(1, station.docks);
  }
  return stations;
}

static std::vector<EmptyStation> emptyStationsOf(const std::vector<Station>& stations,
                                                 SplitMix64& random) {
  std::vector<EmptyStation> empty;
  for (const Station& station : stations) {
    if (station.bikesDocked > 0) {
      continue;
    }
    const bool curbClosedForRushHour = random.range(0, 99) < CURB_CLOSED_PERCENT;
    const int curbOpensInMs = curbClosedForRushHour ? random.range(1, LONGEST_CURB_CLOSURE_MS) : 0;
    empty.push_back(EmptyStation{station.id, station.x, station.y, station.docks, curbOpensInMs});
  }
  return empty;
}

static unsigned truckEta(const Trucks& trucks, unsigned truck, const EmptyStation& station) {
  return etaMs(trucks.x[truck],
               trucks.y[truck],
               trucks.state[truck],
               trucks.bikesLoaded[truck],
               trucks.busyMs[truck],
               station);
}

static CpuReference computeCpuReference(const Trucks& trucks,
                                        const std::vector<EmptyStation>& empty) {
  const unsigned truckCount = static_cast<unsigned>(trucks.x.size());
  CpuReference reference;
  reference.best.assign(empty.size(), NO_KEY);
  Timer timer;
  for (size_t s = 0; s < empty.size(); ++s) {
    for (unsigned k = 0; k < truckCount; ++k) {
      reference.best[s] = umin64(reference.best[s], candidateKey(truckEta(trucks, k, empty[s]), k));
    }
  }
  reference.ms = timer.ms();
  return reference;
}

static std::vector<unsigned> trucksSharingBestEta(const Trucks& trucks,
                                                  const std::vector<EmptyStation>& empty,
                                                  const std::vector<unsigned long long>& best) {
  const unsigned truckCount = static_cast<unsigned>(trucks.x.size());
  std::vector<unsigned> ties(empty.size(), 0);
  for (size_t s = 0; s < empty.size(); ++s) {
    if (best[s] == NO_KEY) {
      continue;
    }
    for (unsigned k = 0; k < truckCount; ++k) {
      if (truckEta(trucks, k, empty[s]) == etaOfKey(best[s])) {
        ++ties[s];
      }
    }
  }
  return ties;
}

template <class T>
static T* deviceCopyOf(const std::vector<T>& values) {
  T* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, values.size() * sizeof(T)));
  CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return pointer;
}

static DeviceTrucks copyTrucksToDevice(const Trucks& trucks) {
  DeviceTrucks device;
  device.x = deviceCopyOf(trucks.x);
  device.y = deviceCopyOf(trucks.y);
  device.state = deviceCopyOf(trucks.state);
  device.bikesLoaded = deviceCopyOf(trucks.bikesLoaded);
  device.busyMs = deviceCopyOf(trucks.busyMs);
  return device;
}

static void freeDeviceTrucks(const DeviceTrucks& device) {
  CHECK(cudaFree(device.x));
  CHECK(cudaFree(device.y));
  CHECK(cudaFree(device.state));
  CHECK(cudaFree(device.bikesLoaded));
  CHECK(cudaFree(device.busyMs));
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

static double runNearestTruckKernel(const Trucks& trucks,
                                    const std::vector<EmptyStation>& empty,
                                    const dim3& grid,
                                    std::vector<unsigned long long>& best) {
  const unsigned truckCount = static_cast<unsigned>(trucks.x.size());
  const DeviceTrucks deviceTrucks = copyTrucksToDevice(trucks);
  EmptyStation* deviceStations = deviceCopyOf(empty);
  const std::vector<unsigned long long> noKeys(empty.size(), NO_KEY);
  unsigned long long* deviceBest = deviceCopyOf(noKeys);
  const size_t bestBytes = empty.size() * sizeof(unsigned long long);

  const double ms = millisecondsPerRun([&] {
    CHECK(cudaMemcpy(deviceBest, noKeys.data(), bestBytes, cudaMemcpyHostToDevice));
    launch(NearestTruckKernel,
           grid,
           dim3(BLOCK_DIM),
           deviceTrucks.x,
           deviceTrucks.y,
           deviceTrucks.state,
           deviceTrucks.bikesLoaded,
           deviceTrucks.busyMs,
           truckCount,
           deviceStations,
           deviceBest);
  });
  best.resize(empty.size());
  CHECK(cudaMemcpy(best.data(), deviceBest, bestBytes, cudaMemcpyDeviceToHost));

  freeDeviceTrucks(deviceTrucks);
  CHECK(cudaFree(deviceStations));
  CHECK(cudaFree(deviceBest));
  return ms;
}

static bool parseArguments(int argc, char** argv, unsigned& trucks) {
  trucks = DEFAULT_TRUCKS;
  if (argc > 1) {
    trucks = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
  }
  return trucks > 0;
}

static void printTimingTable(double cpuMs, const dim3& grid, double gpuMs, bool ok) {
  std::printf("\n%-36s %13s %10s  %s\n", "version", "grid", "ms", "check");
  std::printf("%-36s %13s %10.3f  %s\n", "CPU sequential argmin", "-", cpuMs, "reference");
  char gridText[32];
  std::snprintf(gridText, sizeof gridText, "%u x %u", grid.x, grid.y);
  std::printf("%-36s %13s %10.3f  %s\n",
              "NearestTruckKernel (packed atomicMin)",
              gridText,
              gpuMs,
              passOrFail(ok));
}

static unsigned countSharedChoices(std::vector<unsigned> chosen) {
  std::sort(chosen.begin(), chosen.end());
  unsigned conflicts = 0;
  for (size_t k = 1; k < chosen.size(); ++k) {
    if (chosen[k] == chosen[k - 1]) {
      ++conflicts;
    }
  }
  return conflicts;
}

static void printPlan(const Trucks& trucks,
                      const std::vector<EmptyStation>& empty,
                      const std::vector<unsigned long long>& best,
                      const std::vector<unsigned>& ties) {
  std::printf("\nempty station  docks  curb opens  nearest truck  bikes  ETA (min:s)  same ETA\n");
  unsigned withoutTruck = 0;
  std::vector<unsigned> chosen;
  for (size_t s = 0; s < empty.size(); ++s) {
    const EmptyStation& station = empty[s];
    if (best[s] == NO_KEY) {
      ++withoutTruck;
      continue;
    }
    const unsigned eta = etaOfKey(best[s]);
    const unsigned truck = truckOfKey(best[s]);
    chosen.push_back(truck);
    std::printf("s%-12d  %5d  %7d:%02d  t%-12u  %5d  %8u:%02u  %8u\n",
                station.id,
                station.docks,
                station.curbOpensInMs / MINUTE_MS,
                station.curbOpensInMs / SECOND_MS % SECONDS_PER_MINUTE,
                truck,
                trucks.bikesLoaded[truck],
                eta / MINUTE_MS,
                eta / SECOND_MS % SECONDS_PER_MINUTE,
                ties[s]);
  }

  std::printf("empty stations without any truck able to restock them: %u\n", withoutTruck);
  std::printf("empty stations whose nearest truck is also the nearest truck of another: %u\n",
              countSharedChoices(chosen));
  std::printf(
      "(each argmin is independent; a real rebalancing plan sends one truck, updates its load\n"
      " and position, and recomputes, or resolves the conflicts afterwards)\n");
}

int main(int argc, char** argv) {
  unsigned truckCount = 0;
  if (!parseArguments(argc, argv, truckCount)) {
    std::fprintf(stderr, "usage: %s [trucks >= 1]\n", argv[0]);
    return 1;
  }
  std::printf("Nearest rebalancing truck per empty station (chapter 10 use case), backend: %s\n",
              CUDA_BACKEND);

  SplitMix64 random(DATA_SEED);
  const std::vector<Station> stations = makeStations(random);
  const std::vector<EmptyStation> empty = emptyStationsOf(stations, random);
  const Trucks trucks = makeTrucks(truckCount, random);
  std::printf("%u trucks x %zu empty stations (of %d) = %llu ETA estimates (seed 99)\n",
              truckCount,
              empty.size(),
              STATION_COUNT,
              static_cast<unsigned long long>(truckCount) * empty.size());
  if (empty.empty()) {
    std::printf("\nno empty station to restock\n\nALL PASS\n");
    return 0;
  }

  const CpuReference reference = computeCpuReference(trucks, empty);
  const std::vector<unsigned> ties = trucksSharingBestEta(trucks, empty, reference.best);

  const dim3 grid(cdiv(truckCount, COARSE_FACTOR * BLOCK_DIM), static_cast<unsigned>(empty.size()));
  std::vector<unsigned long long> best;
  const double gpuMs = runNearestTruckKernel(trucks, empty, grid, best);
  const bool ok = best == reference.best;

  printTimingTable(reference.ms, grid, gpuMs, ok);
  printPlan(trucks, empty, best, ties);

  std::printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
