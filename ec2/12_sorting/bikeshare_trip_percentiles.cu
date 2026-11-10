#include "cuda_compat.h"

#include <algorithm>
#include <cmath>
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

#ifndef CPU_TRIPS
#define CPU_TRIPS (1 << 15)
#endif

constexpr int DEFAULT_TRIPS = BY_BACKEND(1 << 22, CPU_TRIPS);
constexpr uint64_t TRIPS_SEED = 42;
constexpr int STATION_COUNT = 48;
constexpr int BIKE_COUNT = 1500;
constexpr int MS_PER_SECOND = 1000;
constexpr int MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr int MS_PER_HOUR = 60 * MS_PER_MINUTE;
constexpr int SERVICE_OPENS_MS = 6 * MS_PER_HOUR;
constexpr int SERVICE_CLOSES_MS = 23 * MS_PER_HOUR;
constexpr int MORNING_RUSH_MS = 8 * MS_PER_HOUR;
constexpr int EVENING_RUSH_MS = 18 * MS_PER_HOUR;
constexpr int RUSH_HALF_WIDTH_MS = MS_PER_HOUR;
constexpr int COMMUTE_MS_MIN = 4 * MS_PER_MINUTE;
constexpr int COMMUTE_MS_MAX = 25 * MS_PER_MINUTE;
constexpr int LEISURE_MS_MIN = 10 * MS_PER_MINUTE;
constexpr int LEISURE_MS_MAX = 70 * MS_PER_MINUTE;
constexpr int LEISURE_PERCENT_OFF_PEAK = 50;
constexpr int ONE_FULL_STATION_IN = 25;
constexpr int DOCK_SEARCH_MS_MIN = 5 * MS_PER_MINUTE;
constexpr int DOCK_SEARCH_MS_MAX = 40 * MS_PER_MINUTE;
constexpr int INCLUDED_MS = 45 * MS_PER_MINUTE;
constexpr int BASE_FARE_CENTS = 300;
constexpr int CENTS_PER_STARTED_BLOCK = 100;
constexpr int FARE_BLOCK_MS = 10 * MS_PER_MINUTE;
constexpr double CENTS_PER_BOLIVIANO = 100.0;
constexpr int LONGEST_TRIPS_SHOWN = 5;
constexpr double HUNDRED_PERCENT = 100.0;

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

  int between(int lowest, int highest) {
    const uint64_t span = static_cast<uint64_t>(highest - lowest) + 1;
    return lowest + static_cast<int>(next() % span);
  }

  bool oneIn(int n) {
    return next() % static_cast<uint64_t>(n) == 0;
  }

  bool percentChance(int percent) {
    return between(0, 99) < percent;
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

enum class TripKind { Commute, Leisure };

struct Trip {
  int tripId;
  int bikeId;
  int startStation;
  int endStation;
  int startMs;
  int endMs;
  int fareCents;
};

struct TripDurations {
  std::vector<unsigned> durationMs;
  std::vector<unsigned> tripId;
};

static int durationOf(const Trip& trip) {
  return trip.endMs - trip.startMs;
}

static int fareCents(int durationMs) {
  const int startedBlocks = (durationMs + FARE_BLOCK_MS - 1) / FARE_BLOCK_MS;
  return BASE_FARE_CENTS + CENTS_PER_STARTED_BLOCK * startedBlocks;
}

static bool isLate(unsigned durationMs) {
  return durationMs > static_cast<unsigned>(INCLUDED_MS);
}

static bool isRushHour(int timeMs) {
  const bool morning = std::abs(timeMs - MORNING_RUSH_MS) < RUSH_HALF_WIDTH_MS;
  const bool evening = std::abs(timeMs - EVENING_RUSH_MS) < RUSH_HALF_WIDTH_MS;
  return morning || evening;
}

static TripKind pickKind(SplitMix64& rng, int startMs) {
  if (isRushHour(startMs)) {
    return TripKind::Commute;
  }
  return rng.percentChance(LEISURE_PERCENT_OFF_PEAK) ? TripKind::Leisure : TripKind::Commute;
}

static int roundDownToSecond(int ms) {
  return ms / MS_PER_SECOND * MS_PER_SECOND;
}

static int rideMs(SplitMix64& rng, TripKind kind) {
  int ms = kind == TripKind::Commute ? rng.between(COMMUTE_MS_MIN, COMMUTE_MS_MAX)
                                     : rng.between(LEISURE_MS_MIN, LEISURE_MS_MAX);
  const bool endStationWasFull = rng.oneIn(ONE_FULL_STATION_IN);
  if (endStationWasFull) {
    ms += rng.between(DOCK_SEARCH_MS_MIN, DOCK_SEARCH_MS_MAX);
  }
  return roundDownToSecond(ms);
}

static Trip makeTrip(SplitMix64& rng, int tripId) {
  Trip trip;
  trip.tripId = tripId;
  trip.bikeId = rng.between(0, BIKE_COUNT - 1);
  trip.startStation = rng.between(0, STATION_COUNT - 1);
  trip.endStation = rng.between(0, STATION_COUNT - 1);
  trip.startMs = roundDownToSecond(rng.between(SERVICE_OPENS_MS, SERVICE_CLOSES_MS));
  trip.endMs = trip.startMs + rideMs(rng, pickKind(rng, trip.startMs));
  trip.fareCents = fareCents(durationOf(trip));
  return trip;
}

static std::vector<Trip> makeTrips(int n) {
  SplitMix64 rng(TRIPS_SEED);
  std::vector<Trip> trips(n);
  for (int i = 0; i < n; ++i) {
    trips[i] = makeTrip(rng, i);
  }
  return trips;
}

static TripDurations durationsOf(const std::vector<Trip>& trips) {
  TripDurations durations;
  durations.durationMs.resize(trips.size());
  durations.tripId.resize(trips.size());
  for (size_t i = 0; i < trips.size(); ++i) {
    durations.durationMs[i] = durationOf(trips[i]);
    durations.tripId[i] = trips[i].tripId;
  }
  return durations;
}

static int bitsNeeded(unsigned maxKey) {
  int bits = 0;
  while (bits < UNSIGNED_BITS && (maxKey >> bits) != 0) {
    ++bits;
  }
  return imax(bits, 1);
}

struct DeviceSortResult {
  TripDurations sorted;
  double ms;
};

static DeviceSortResult sortOnDevice(const TripDurations& durations, int keyBits) {
  const int n = static_cast<int>(durations.durationMs.size());
  const size_t bytes = n * sizeof(unsigned);
  unsigned* deviceKeys = allocateDeviceWords(n);
  unsigned* deviceVals = allocateDeviceWords(n);
  CHECK(cudaMemcpy(deviceKeys, durations.durationMs.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceVals, durations.tripId.data(), bytes, cudaMemcpyHostToDevice));

  Timer timer;
  radixSort(deviceKeys, deviceVals, n, keyBits);
  const double ms = timer.ms();

  DeviceSortResult result{{std::vector<unsigned>(n), std::vector<unsigned>(n)}, ms};
  CHECK(cudaMemcpy(result.sorted.durationMs.data(), deviceKeys, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(result.sorted.tripId.data(), deviceVals, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(deviceKeys));
  CHECK(cudaFree(deviceVals));
  return result;
}

static unsigned percentile(const std::vector<unsigned>& sorted, double p) {
  const size_t rank = static_cast<size_t>(std::ceil(p / HUNDRED_PERCENT * sorted.size()));
  return sorted[rank == 0 ? 0 : rank - 1];
}

static size_t lateTripsInSorted(const std::vector<unsigned>& sorted) {
  const auto firstLate =
      std::upper_bound(sorted.begin(), sorted.end(), static_cast<unsigned>(INCLUDED_MS));
  return static_cast<size_t>(sorted.end() - firstLate);
}

static size_t lateTripsCountedOneByOne(const std::vector<Trip>& trips) {
  size_t late = 0;
  for (const Trip& trip : trips) {
    if (isLate(durationOf(trip))) {
      ++late;
    }
  }
  return late;
}

struct Percentile {
  double p;
  const char* name;
};

constexpr Percentile REPORTED_PERCENTILES[] = {
    {50, "median"}, {90, "p90"}, {95, "p95"}, {99, "p99"}};

static bool everyIdKeptItsKey(const TripDurations& input, const TripDurations& sorted) {
  for (size_t i = 0; i < sorted.durationMs.size(); ++i) {
    if (input.durationMs[sorted.tripId[i]] != sorted.durationMs[i]) {
      return false;
    }
  }
  return true;
}

static bool tiesKeepAscendingTripIds(const TripDurations& sorted) {
  for (size_t i = 1; i < sorted.durationMs.size(); ++i) {
    const bool tie = sorted.durationMs[i] == sorted.durationMs[i - 1];
    if (tie && sorted.tripId[i - 1] >= sorted.tripId[i]) {
      return false;
    }
  }
  return true;
}

static bool percentilesMatch(const std::vector<unsigned>& sorted,
                             const std::vector<unsigned>& reference) {
  for (const Percentile& row : REPORTED_PERCENTILES) {
    if (percentile(sorted, row.p) != percentile(reference, row.p)) {
      return false;
    }
  }
  return true;
}

static double bolivianos(int cents) {
  return cents / CENTS_PER_BOLIVIANO;
}

static void printStatistic(const char* name, unsigned ms) {
  std::printf("%-10s %12u %10.1f %10.2f\n",
              name,
              ms,
              ms / static_cast<double>(MS_PER_MINUTE),
              bolivianos(fareCents(ms)));
}

static void printStatistics(const TripDurations& sorted) {
  const std::vector<unsigned>& durationMs = sorted.durationMs;
  const int n = static_cast<int>(durationMs.size());
  double sum = 0;
  for (unsigned ms : durationMs) {
    sum += ms;
  }
  const double average = sum / n;

  std::printf("%-10s %12s %10s %10s\n", "statistic", "ms", "minutes", "fare (Bs)");
  for (const Percentile& row : REPORTED_PERCENTILES) {
    printStatistic(row.name, percentile(durationMs, row.p));
  }
  std::printf("%-10s %12.0f %10.1f\n", "average", average, average / MS_PER_MINUTE);
  printStatistic("max", durationMs[n - 1]);
}

static void printLateTrips(size_t late, size_t n) {
  std::printf("\ntrips over the %d included minutes: %zu of %zu (%.1f %%)\n",
              INCLUDED_MS / MS_PER_MINUTE,
              late,
              n,
              HUNDRED_PERCENT * late / n);
}

static void printLongestTrips(const TripDurations& sorted, const std::vector<Trip>& trips) {
  const int n = static_cast<int>(sorted.durationMs.size());
  const int lastShown = imax(0, n - LONGEST_TRIPS_SHOWN);
  std::printf("longest trips:\n");
  for (int i = n - 1; i >= lastShown; --i) {
    const Trip& trip = trips[sorted.tripId[i]];
    std::printf("  trip t%d on bike b%d, s%d -> s%d: %u ms (%.1f min, Bs %.2f)\n",
                trip.tripId,
                trip.bikeId,
                trip.startStation,
                trip.endStation,
                sorted.durationMs[i],
                sorted.durationMs[i] / static_cast<double>(MS_PER_MINUTE),
                bolivianos(trip.fareCents));
  }
}

static void printTimings(double cpuMs, double deviceMs) {
  std::printf("\n%-34s %10s\n", "step", "time (ms)");
  std::printf("%-34s %10.2f\n", "std::sort of the keys (CPU)", cpuMs);
  std::printf("%-34s %10.2f\n", "radix sort of the pairs (device)", deviceMs);
  std::printf("\n");
}

static bool reportCheck(const char* label, bool ok) {
  std::printf("%-37s%s\n", label, ok ? "PASS" : "FAIL");
  return ok;
}

struct CheckInputs {
  const std::vector<Trip>& trips;
  const TripDurations& input;
  const TripDurations& sorted;
  const std::vector<unsigned>& reference;
};

static bool runChecks(const CheckInputs& in) {
  const bool lateCountsMatch =
      lateTripsInSorted(in.sorted.durationMs) == lateTripsCountedOneByOne(in.trips);

  bool ok = true;
  ok = reportCheck("sorted keys equal std::sort:", in.sorted.durationMs == in.reference) && ok;
  ok = reportCheck("every trip id kept its key:", everyIdKeptItsKey(in.input, in.sorted)) && ok;
  ok =
      reportCheck("equal durations keep ascending ids:", tiesKeepAscendingTripIds(in.sorted)) && ok;
  ok = reportCheck("percentiles equal reference:",
                   percentilesMatch(in.sorted.durationMs, in.reference)) &&
       ok;
  ok = reportCheck("late trips equal a direct count:", lateCountsMatch) && ok;
  return ok;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : DEFAULT_TRIPS;
  std::printf("Bike-share trip-duration percentiles with a radix sort on %s\n", CUDA_BACKEND);

  const std::vector<Trip> trips = makeTrips(n);
  const TripDurations durations = durationsOf(trips);
  const unsigned longestMs =
      *std::max_element(durations.durationMs.begin(), durations.durationMs.end());
  const int keyBits = bitsNeeded(longestMs);
  std::printf("%d trips, longest %u ms -> %d key bits -> %d radix passes of %d bits\n\n",
              n,
              longestMs,
              keyBits,
              radixPasses(keyBits),
              RADIX_BITS);

  std::vector<unsigned> reference = durations.durationMs;
  Timer cpuTimer;
  std::sort(reference.begin(), reference.end());
  const double cpuMs = cpuTimer.ms();

  const DeviceSortResult device = sortOnDevice(durations, keyBits);
  const TripDurations& sorted = device.sorted;

  printStatistics(sorted);
  printLateTrips(lateTripsInSorted(sorted.durationMs), sorted.durationMs.size());
  printLongestTrips(sorted, trips);
  printTimings(cpuMs, device.ms);

  const bool ok = runChecks({trips, durations, sorted, reference});
  std::printf("%s\n", ok ? "ALL PASS" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
