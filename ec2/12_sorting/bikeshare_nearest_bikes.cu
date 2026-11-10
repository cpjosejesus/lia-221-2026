#include "cuda_compat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
constexpr int UNSIGNED_BITS = 32;
constexpr unsigned PADDING_KEY = 0xFFFFFFFFu;
constexpr int RADIX_BITS = 4;
constexpr int BUCKETS = 1 << RADIX_BITS;

__device__ inline int coalesced_position(int item) {
  return item * BLOCK_THREADS + threadIdx.x;
}

__device__ inline int thread_owned_position(int item) {
  return threadIdx.x * ITEMS_PER_THREAD + item;
}

__device__ inline unsigned block_inclusive_scan(unsigned value, unsigned* buf, unsigned* total) {
  const int t = threadIdx.x;
  buf[t] = value;
  __syncthreads();

  int readHalf = 0;
  for (int stride = 1; stride < BLOCK_THREADS; stride *= 2) {
    const int writeHalf = 1 - readHalf;
    unsigned sum = buf[readHalf * BLOCK_THREADS + t];
    if (t >= stride) {
      sum += buf[readHalf * BLOCK_THREADS + t - stride];
    }
    buf[writeHalf * BLOCK_THREADS + t] = sum;
    readHalf = writeHalf;
    __syncthreads();
  }

  *total = buf[readHalf * BLOCK_THREADS + BLOCK_THREADS - 1];
  const unsigned inclusivePrefix = buf[readHalf * BLOCK_THREADS + t];
  __syncthreads();
  return inclusivePrefix;
}

__global__ void scan_block_kernel(const unsigned* in, unsigned* out, int n, unsigned* blockSums) {
  __shared__ unsigned tile[TILE];
  __shared__ unsigned buf[2 * BLOCK_THREADS];
  const int base = blockIdx.x * TILE;

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    tile[pos] = base + pos < n ? in[base + pos] : 0u;
  }
  __syncthreads();

  unsigned threadSum = 0;
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    threadSum += tile[thread_owned_position(item)];
  }

  unsigned blockTotal;
  unsigned running = block_inclusive_scan(threadSum, buf, &blockTotal) - threadSum;
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = thread_owned_position(item);
    const unsigned value = tile[pos];
    tile[pos] = running;
    running += value;
  }
  __syncthreads();

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    if (base + pos < n) {
      out[base + pos] = tile[pos];
    }
  }
  if (threadIdx.x == 0) {
    blockSums[blockIdx.x] = blockTotal;
  }
}

__global__ void add_block_offsets_kernel(unsigned* out, int n, const unsigned* blockOffsets) {
  const unsigned offset = blockOffsets[blockIdx.x];
  const int base = blockIdx.x * TILE;
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = base + coalesced_position(item);
    if (index < n) {
      out[index] += offset;
    }
  }
}

constexpr size_t SCAN_WORKSPACE_SLACK = 2;

static size_t scanWorkspace(int n) {
  size_t words = SCAN_WORKSPACE_SLACK;
  for (int blocks = cdiv(n, TILE); blocks > 1; blocks = cdiv(blocks, TILE)) {
    const size_t sumsAndScannedSums = 2 * blocks;
    words += sumsAndScannedSums;
  }
  return words + SCAN_WORKSPACE_SLACK;
}

static void exclusiveScan(const unsigned* in, unsigned* out, int n, unsigned* workspace) {
  if (n <= 0) {
    return;
  }

  const int blocks = cdiv(n, TILE);
  unsigned* sums = workspace;
  unsigned* sumsScanned = sums + blocks;
  unsigned* deeperWorkspace = sumsScanned + blocks;
  launch(scan_block_kernel, dim3(blocks), dim3(BLOCK_THREADS), in, out, n, sums);
  if (blocks > 1) {
    exclusiveScan(sums, sumsScanned, blocks, deeperWorkspace);
    launch(add_block_offsets_kernel, dim3(blocks), dim3(BLOCK_THREADS), out, n, sumsScanned);
  }
}

__host__ __device__ inline unsigned digit_of(unsigned key, int shift) {
  return shift < UNSIGNED_BITS ? (key >> shift) & (BUCKETS - 1u) : 0u;
}

__host__ __device__ inline unsigned bit_of(unsigned key, int bit) {
  return bit < UNSIGNED_BITS ? (key >> bit) & 1u : 0u;
}

__host__ __device__ inline int bucket_major_index(int bucket, int block, int numBlocks) {
  return bucket * numBlocks + block;
}

__host__ __device__ inline int block_major_index(int block, int bucket) {
  return block * BUCKETS + bucket;
}

__global__ void radix_local_sort_kernel(const unsigned* keysIn,
                                        const unsigned* valsIn,
                                        int n,
                                        int shift,
                                        unsigned* keysTmp,
                                        unsigned* valsTmp,
                                        unsigned* counts,
                                        unsigned* localStart) {
  __shared__ unsigned keys_S[2][TILE];
  __shared__ unsigned vals_S[2][TILE];
  __shared__ unsigned buf[2 * BLOCK_THREADS];
  __shared__ unsigned bucketStart[BUCKETS];
  __shared__ unsigned bucketEnd[BUCKETS];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    const bool isRealElement = pos < tileCount;
    keys_S[0][pos] = isRealElement ? keysIn[tileBase + pos] : PADDING_KEY;
    vals_S[0][pos] = isRealElement ? valsIn[tileBase + pos] : 0u;
  }
  for (int d = t; d < BUCKETS; d += BLOCK_THREADS) {
    bucketStart[d] = 0;
    bucketEnd[d] = 0;
  }
  __syncthreads();

  int current = 0;
  for (int bitInDigit = 0; bitInDigit < RADIX_BITS; ++bitInDigit) {
    const int bit = shift + bitInDigit;
    const int next = 1 - current;
    unsigned myBits[ITEMS_PER_THREAD];
    unsigned myOnes = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      myBits[item] = bit_of(keys_S[current][thread_owned_position(item)], bit);
      myOnes += myBits[item];
    }

    unsigned totalOnes;
    unsigned onesBefore = block_inclusive_scan(myOnes, buf, &totalOnes) - myOnes;
    const unsigned zeros = TILE - totalOnes;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const unsigned pos = thread_owned_position(item);
      const unsigned dst = myBits[item] ? zeros + onesBefore : pos - onesBefore;
      onesBefore += myBits[item];
      keys_S[next][dst] = keys_S[current][pos];
      vals_S[next][dst] = vals_S[current][pos];
    }
    __syncthreads();
    current = next;
  }

  const unsigned* sortedKeys = keys_S[current];
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = thread_owned_position(item);
    if (pos < tileCount) {
      const unsigned d = digit_of(sortedKeys[pos], shift);
      const bool startsBucket = pos == 0 || digit_of(sortedKeys[pos - 1], shift) != d;
      const bool endsBucket = pos == tileCount - 1 || digit_of(sortedKeys[pos + 1], shift) != d;
      if (startsBucket) {
        bucketStart[d] = pos;
      }
      if (endsBucket) {
        bucketEnd[d] = pos + 1;
      }
    }
  }
  __syncthreads();

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    if (pos < tileCount) {
      keysTmp[tileBase + pos] = keys_S[current][pos];
      valsTmp[tileBase + pos] = vals_S[current][pos];
    }
  }
  for (int d = t; d < BUCKETS; d += BLOCK_THREADS) {
    counts[bucket_major_index(d, blockIdx.x, gridDim.x)] = bucketEnd[d] - bucketStart[d];
    localStart[block_major_index(blockIdx.x, d)] = bucketStart[d];
  }
}

__global__ void radix_scatter_coalesced_kernel(const unsigned* keysTmp,
                                               const unsigned* valsTmp,
                                               int n,
                                               int shift,
                                               const unsigned* offsets,
                                               const unsigned* localStart,
                                               unsigned* keysOut,
                                               unsigned* valsOut) {
  __shared__ int localToGlobal[BUCKETS];
  const int t = threadIdx.x;
  const int tileBase = blockIdx.x * TILE;
  const int tileCount = imin(TILE, n - tileBase);

  for (int d = t; d < BUCKETS; d += BLOCK_THREADS) {
    const int globalStart = offsets[bucket_major_index(d, blockIdx.x, gridDim.x)];
    const int bucketLocalStart = localStart[block_major_index(blockIdx.x, d)];
    localToGlobal[d] = globalStart - bucketLocalStart;
  }
  __syncthreads();

  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int pos = coalesced_position(item);
    if (pos < tileCount) {
      const unsigned key = keysTmp[tileBase + pos];
      const int dst = localToGlobal[digit_of(key, shift)] + pos;
      keysOut[dst] = key;
      valsOut[dst] = valsTmp[tileBase + pos];
    }
  }
}

struct RadixSortScratch {
  unsigned* otherKeys;
  unsigned* otherVals;
  unsigned* tileSortedKeys;
  unsigned* tileSortedVals;
  unsigned* counts;
  unsigned* offsets;
  unsigned* localStart;
  unsigned* workspace;
};

static unsigned* allocateDeviceWords(size_t count) {
  unsigned* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(unsigned)));
  return pointer;
}

static RadixSortScratch allocateScratch(int n) {
  const int tableEntries = BUCKETS * cdiv(n, TILE);
  RadixSortScratch scratch;
  scratch.otherKeys = allocateDeviceWords(n);
  scratch.otherVals = allocateDeviceWords(n);
  scratch.tileSortedKeys = allocateDeviceWords(n);
  scratch.tileSortedVals = allocateDeviceWords(n);
  scratch.counts = allocateDeviceWords(tableEntries);
  scratch.offsets = allocateDeviceWords(tableEntries);
  scratch.localStart = allocateDeviceWords(tableEntries);
  scratch.workspace = allocateDeviceWords(scanWorkspace(tableEntries));
  return scratch;
}

static void freeScratch(RadixSortScratch& scratch) {
  CHECK(cudaFree(scratch.otherKeys));
  CHECK(cudaFree(scratch.otherVals));
  CHECK(cudaFree(scratch.tileSortedKeys));
  CHECK(cudaFree(scratch.tileSortedVals));
  CHECK(cudaFree(scratch.counts));
  CHECK(cudaFree(scratch.offsets));
  CHECK(cudaFree(scratch.localStart));
  CHECK(cudaFree(scratch.workspace));
}

static void radixSortPass(const unsigned* keysIn,
                          const unsigned* valsIn,
                          unsigned* keysOut,
                          unsigned* valsOut,
                          int n,
                          int shift,
                          RadixSortScratch& scratch) {
  const unsigned blocks = cdiv(n, TILE);
  launch(radix_local_sort_kernel,
         dim3(blocks),
         dim3(BLOCK_THREADS),
         keysIn,
         valsIn,
         n,
         shift,
         scratch.tileSortedKeys,
         scratch.tileSortedVals,
         scratch.counts,
         scratch.localStart);
  exclusiveScan(scratch.counts, scratch.offsets, BUCKETS * blocks, scratch.workspace);
  launch(radix_scatter_coalesced_kernel,
         dim3(blocks),
         dim3(BLOCK_THREADS),
         scratch.tileSortedKeys,
         scratch.tileSortedVals,
         n,
         shift,
         scratch.offsets,
         scratch.localStart,
         keysOut,
         valsOut);
}

static int radixPasses(int keyBits) {
  return (keyBits + RADIX_BITS - 1) / RADIX_BITS;
}

static void radixSort(unsigned* keys, unsigned* vals, int n, int keyBits) {
  RadixSortScratch scratch = allocateScratch(n);
  unsigned* keysIn = keys;
  unsigned* valsIn = vals;
  unsigned* keysOut = scratch.otherKeys;
  unsigned* valsOut = scratch.otherVals;
  for (int shift = 0; shift < keyBits; shift += RADIX_BITS) {
    radixSortPass(keysIn, valsIn, keysOut, valsOut, n, shift, scratch);
    std::swap(keysIn, keysOut);
    std::swap(valsIn, valsOut);
  }

  const bool resultIsInScratch = keysIn != keys;
  if (resultIsInScratch) {
    CHECK(cudaMemcpy(keys, keysIn, n * sizeof(unsigned), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(vals, valsIn, n * sizeof(unsigned), cudaMemcpyDeviceToDevice));
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  freeScratch(scratch);
}

constexpr int DISTANCE_BITS = 14;
constexpr unsigned DISTANCE_MASK = (1u << DISTANCE_BITS) - 1;
constexpr unsigned DISTANCE_UNAVAILABLE = DISTANCE_MASK;
constexpr int LONGEST_OFFERED_DISTANCE = DISTANCE_UNAVAILABLE - 1;
constexpr int MIN_BATTERY_PERCENT = 30;
constexpr int NO_STATION = -1;

enum class BikeState { Docked, Riding, InRepair };

struct Position {
  int x;
  int y;
};

struct Bike {
  int id;
  BikeState state;
  Position position;
  int batteryPercent;
  int station;
};

struct RideRequest {
  int id;
  Position rider;
};

__host__ __device__ inline int walking_distance_metres(Position from, Position to) {
  const int dx = from.x - to.x;
  const int dy = from.y - to.y;
  return (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
}

__host__ __device__ inline bool can_be_offered(const Bike& bike) {
  return bike.state == BikeState::Docked && bike.batteryPercent >= MIN_BATTERY_PERCENT;
}

__host__ __device__ inline unsigned offered_distance(const Bike& bike, const RideRequest& request) {
  if (!can_be_offered(bike)) {
    return DISTANCE_UNAVAILABLE;
  }
  const int metres = walking_distance_metres(request.rider, bike.position);
  return imin(metres, LONGEST_OFFERED_DISTANCE);
}

__host__ __device__ inline unsigned request_distance_key(int request, unsigned distance) {
  return (static_cast<unsigned>(request) << DISTANCE_BITS) | distance;
}

__global__ void distance_keys_kernel(const Bike* bikes,
                                     int numBikes,
                                     const RideRequest* requests,
                                     int numRequests,
                                     unsigned* keys,
                                     unsigned* vals) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < numRequests * numBikes) {
    const int request = pair / numBikes;
    const int bike = pair % numBikes;
    keys[pair] = request_distance_key(request, offered_distance(bikes[bike], requests[request]));
    vals[pair] = bikes[bike].id;
  }
}

#ifndef CPU_REQUESTS
#define CPU_REQUESTS 64
#endif

#ifndef CPU_BIKES
#define CPU_BIKES 256
#endif

constexpr int RUSH_REQUESTS = BY_BACKEND(1024, CPU_REQUESTS);
constexpr int FLEET_BIKES = BY_BACKEND(4096, CPU_BIKES);
constexpr int CANDIDATE_PAIRS = RUSH_REQUESTS * FLEET_BIKES;
constexpr int BIKES_OFFERED = 3;
constexpr int REQUESTS_SHOWN = 4;
constexpr int DISTANCE_THREADS_PER_BLOCK = 256;

constexpr int MAP_SIDE_METRES = 6000;
constexpr int STATION_COUNT = 48;
constexpr uint64_t STATION_SEED = 48;
constexpr int STATION_MARGIN_METRES = 300;
constexpr uint64_t SCENARIO_SEED = 7;
constexpr int HUNDRED_PERCENT = 100;
constexpr int DOCKED_PERCENT = 70;
constexpr int RIDING_PERCENT = 22;
constexpr int FULL_BATTERY_PERCENT = 100;

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

  int below(int n) {
    return static_cast<int>(next() % static_cast<uint64_t>(n));
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

static int bitsNeeded(unsigned maxValue) {
  int bits = 0;
  while (bits < UNSIGNED_BITS && (maxValue >> bits) != 0) {
    ++bits;
  }
  return bits;
}

static Position randomPosition(SplitMix64& rng, int marginMetres) {
  const int span = MAP_SIDE_METRES - 2 * marginMetres;
  Position position;
  position.x = marginMetres + rng.below(span);
  position.y = marginMetres + rng.below(span);
  return position;
}

static std::vector<Position> makeStationPositions() {
  SplitMix64 rng(STATION_SEED);
  std::vector<Position> stations(STATION_COUNT);
  for (Position& station : stations) {
    station = randomPosition(rng, STATION_MARGIN_METRES);
  }
  return stations;
}

static BikeState pickState(SplitMix64& rng) {
  const int draw = rng.below(HUNDRED_PERCENT);
  if (draw < DOCKED_PERCENT) {
    return BikeState::Docked;
  }
  if (draw < DOCKED_PERCENT + RIDING_PERCENT) {
    return BikeState::Riding;
  }
  return BikeState::InRepair;
}

static Bike makeBike(SplitMix64& rng, int id, const std::vector<Position>& stations) {
  Bike bike;
  bike.id = id;
  bike.state = pickState(rng);
  bike.batteryPercent = rng.below(FULL_BATTERY_PERCENT + 1);
  if (bike.state == BikeState::Riding) {
    bike.station = NO_STATION;
    bike.position = randomPosition(rng, 0);
  } else {
    bike.station = rng.below(STATION_COUNT);
    bike.position = stations[bike.station];
  }
  return bike;
}

struct Scenario {
  std::vector<Bike> fleet;
  std::vector<RideRequest> rush;
};

static Scenario makeScenario() {
  const std::vector<Position> stations = makeStationPositions();
  SplitMix64 rng(SCENARIO_SEED);

  Scenario scenario;
  scenario.fleet.resize(FLEET_BIKES);
  for (int id = 0; id < FLEET_BIKES; ++id) {
    scenario.fleet[id] = makeBike(rng, id, stations);
  }
  scenario.rush.resize(RUSH_REQUESTS);
  for (int id = 0; id < RUSH_REQUESTS; ++id) {
    scenario.rush[id] = {id, randomPosition(rng, 0)};
  }
  return scenario;
}

struct RankedCandidates {
  std::vector<unsigned> keys;
  std::vector<unsigned> bikes;
  double ms;
};

static RankedCandidates rankOnDevice(const Scenario& scenario, int keyBits) {
  Bike* deviceFleet;
  RideRequest* deviceRush;
  CHECK(cudaMalloc(&deviceFleet, FLEET_BIKES * sizeof(Bike)));
  CHECK(cudaMalloc(&deviceRush, RUSH_REQUESTS * sizeof(RideRequest)));
  unsigned* deviceKeys = allocateDeviceWords(CANDIDATE_PAIRS);
  unsigned* deviceVals = allocateDeviceWords(CANDIDATE_PAIRS);
  CHECK(cudaMemcpy(
      deviceFleet, scenario.fleet.data(), FLEET_BIKES * sizeof(Bike), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceRush,
                   scenario.rush.data(),
                   RUSH_REQUESTS * sizeof(RideRequest),
                   cudaMemcpyHostToDevice));

  Timer timer;
  launch(distance_keys_kernel,
         dim3(cdiv(CANDIDATE_PAIRS, DISTANCE_THREADS_PER_BLOCK)),
         dim3(DISTANCE_THREADS_PER_BLOCK),
         deviceFleet,
         FLEET_BIKES,
         deviceRush,
         RUSH_REQUESTS,
         deviceKeys,
         deviceVals);
  radixSort(deviceKeys, deviceVals, CANDIDATE_PAIRS, keyBits);
  const double ms = timer.ms();

  RankedCandidates ranked{
      std::vector<unsigned>(CANDIDATE_PAIRS), std::vector<unsigned>(CANDIDATE_PAIRS), ms};
  const size_t pairBytes = CANDIDATE_PAIRS * sizeof(unsigned);
  CHECK(cudaMemcpy(ranked.keys.data(), deviceKeys, pairBytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ranked.bikes.data(), deviceVals, pairBytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(deviceFleet));
  CHECK(cudaFree(deviceRush));
  CHECK(cudaFree(deviceKeys));
  CHECK(cudaFree(deviceVals));
  return ranked;
}

struct DistanceBike {
  unsigned distance;
  unsigned bike;
};

static std::vector<DistanceBike> rankOnHost(const Scenario& scenario) {
  std::vector<DistanceBike> ranking(CANDIDATE_PAIRS);
  for (int request = 0; request < RUSH_REQUESTS; ++request) {
    const auto first = ranking.begin() + static_cast<size_t>(request) * FLEET_BIKES;
    for (int bike = 0; bike < FLEET_BIKES; ++bike) {
      first[bike] = {offered_distance(scenario.fleet[bike], scenario.rush[request]),
                     static_cast<unsigned>(scenario.fleet[bike].id)};
    }
    std::stable_sort(first, first + FLEET_BIKES, [](const DistanceBike& a, const DistanceBike& b) {
      return a.distance < b.distance;
    });
  }
  return ranking;
}

static int requestOfKey(unsigned key) {
  return key >> DISTANCE_BITS;
}

static unsigned distanceOfKey(unsigned key) {
  return key & DISTANCE_MASK;
}

struct RankingChecks {
  bool offeredBikesOk;
  bool wholeRankingOk;
};

static RankingChecks compareRankings(const RankedCandidates& device,
                                     const std::vector<DistanceBike>& host) {
  RankingChecks checks{true, true};
  for (int pair = 0; pair < CANDIDATE_PAIRS; ++pair) {
    const int request = pair / FLEET_BIKES;
    const int rank = pair % FLEET_BIKES;
    const unsigned key = device.keys[pair];
    const bool same = requestOfKey(key) == request && distanceOfKey(key) == host[pair].distance &&
                      device.bikes[pair] == host[pair].bike;
    checks.wholeRankingOk = checks.wholeRankingOk && same;
    if (rank < BIKES_OFFERED) {
      checks.offeredBikesOk = checks.offeredBikesOk && same;
    }
  }
  return checks;
}

static void printOffer(const RankedCandidates& ranked, const Scenario& scenario, int request) {
  const Position rider = scenario.rush[request].rider;
  std::printf("  request %d at (%d m, %d m):", request, rider.x, rider.y);
  for (int rank = 0; rank < BIKES_OFFERED; ++rank) {
    const int pair = request * FLEET_BIKES + rank;
    const unsigned distance = distanceOfKey(ranked.keys[pair]);
    if (distance == DISTANCE_UNAVAILABLE) {
      break;
    }
    const Bike& bike = scenario.fleet[ranked.bikes[pair]];
    const char* separator = rank > 0 ? "," : "";
    std::printf("%s b%d at s%d %u m", separator, bike.id, bike.station, distance);
  }
  std::printf("\n");
}

static void printOffers(const RankedCandidates& ranked, const Scenario& scenario) {
  std::printf("nearest charged bikes offered to the first riders:\n");
  for (int request = 0; request < REQUESTS_SHOWN; ++request) {
    printOffer(ranked, scenario, request);
  }
}

static int countBikes(const std::vector<Bike>& fleet, BikeState state) {
  return std::count_if(fleet.begin(), fleet.end(), [state](const Bike& bike) {
    return bike.state == state;
  });
}

static int countDockedButLowBattery(const std::vector<Bike>& fleet) {
  return std::count_if(fleet.begin(), fleet.end(), [](const Bike& bike) {
    return bike.state == BikeState::Docked && !can_be_offered(bike);
  });
}

static void printUnavailableBikes(const std::vector<Bike>& fleet) {
  const int riding = countBikes(fleet, BikeState::Riding);
  const int inRepair = countBikes(fleet, BikeState::InRepair);
  const int lowBattery = countDockedButLowBattery(fleet);
  std::printf(
      "bikes that cannot be offered: %d of %d (riding %d, in repair %d, docked below %d %% "
      "battery %d)\n\n",
      riding + inRepair + lowBattery,
      FLEET_BIKES,
      riding,
      inRepair,
      MIN_BATTERY_PERCENT,
      lowBattery);
}

static void printTimings(double cpuMs, double deviceMs) {
  std::printf("%-44s %10s\n", "step", "time (ms)");
  std::printf("%-44s %10.2f\n", "distance + stable_sort per request (CPU)", cpuMs);
  std::printf("%-44s %10.2f\n", "distance map + one radix sort (device)", deviceMs);
}

static const char* passOrFail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

int main() {
  const int keyBits = DISTANCE_BITS + bitsNeeded(RUSH_REQUESTS - 1);
  std::printf("Nearest bikes: k charged available bikes per rider request on %s\n", CUDA_BACKEND);
  std::printf("%d requests x %d bikes = %d candidate pairs, key = %d bits -> %d radix passes\n\n",
              RUSH_REQUESTS,
              FLEET_BIKES,
              CANDIDATE_PAIRS,
              keyBits,
              radixPasses(keyBits));

  const Scenario scenario = makeScenario();
  const RankedCandidates device = rankOnDevice(scenario, keyBits);

  Timer cpuTimer;
  const std::vector<DistanceBike> host = rankOnHost(scenario);
  const double cpuMs = cpuTimer.ms();

  const RankingChecks checks = compareRankings(device, host);

  printOffers(device, scenario);
  printUnavailableBikes(scenario.fleet);
  printTimings(cpuMs, device.ms);
  std::printf("\nk = %d nearest bikes equal reference:  %s\n",
              BIKES_OFFERED,
              passOrFail(checks.offeredBikesOk));
  std::printf("whole ranking equal reference:        %s\n", passOrFail(checks.wholeRankingOk));
  const bool ok = checks.offeredBikesOk && checks.wholeRankingOk;
  std::printf("%s\n", ok ? "ALL PASS" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
