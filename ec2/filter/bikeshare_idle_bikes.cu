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

#ifndef CPU_BIKES
#define CPU_BIKES 12000
#endif
#ifndef CPU_REQUESTS
#define CPU_REQUESTS 16
#endif
constexpr int DEFAULT_BIKES = BY_BACKEND(200000, CPU_BIKES);
constexpr int DEFAULT_REQUESTS = BY_BACKEND(1024, CPU_REQUESTS);
constexpr int TIMED_RUNS = BY_BACKEND(5, 1);
constexpr int WARMUP_RUNS = BY_BACKEND(1, 0);
constexpr uint64_t SNAPSHOT_SEED = 99;

constexpr int CITY_SIDE_M = 6000;
constexpr int STATIONS = 48;
constexpr int SEARCH_RADIUS_M = 600;
constexpr int MIN_BATTERY_PERCENT = 30;
constexpr int FULL_BATTERY_PERCENT = 100;
constexpr int EXPECTED_CANDIDATES_PER_REQUEST = 128;
constexpr int REQUESTS_SHOWN = 6;

constexpr int DOCKED_WEIGHT = 45;
constexpr int RIDING_WEIGHT = 45;
constexpr int IN_REPAIR_WEIGHT = 10;
constexpr int NOT_DOCKED = -1;

enum class BikeState : int { Docked, Riding, InRepair };

__host__ __device__ inline bool isReadyBike(
    BikeState state, int battery, int x, int y, int qx, int qy, int radius) {
  const int dx = x - qx;
  const int dy = y - qy;
  return state == BikeState::Docked && battery >= MIN_BATTERY_PERCENT &&
         dx * dx + dy * dy <= radius * radius;
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

__global__ void count_candidates(const BikeState* state,
                                 const int* battery,
                                 const int* x,
                                 const int* y,
                                 const int* qx,
                                 const int* qy,
                                 int* counts,
                                 int nBikes,
                                 int radius) {
  __shared__ int blockCount;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  const int q = blockIdx.y;
  const int rx = qx[q];
  const int ry = qy[q];
  int c = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    const int i = blockIdx.x * F_SECTION + k * F_THREADS + threadIdx.x;
    if (i < nBikes && isReadyBike(state[i], battery[i], x[i], y[i], rx, ry, radius)) {
      ++c;
    }
  }
  atomicAdd(&blockCount, c);
  __syncthreads();

  if (threadIdx.x == 0) {
    counts[q * gridDim.x + blockIdx.x] = blockCount;
  }
}

__global__ void compact_candidates(const BikeState* state,
                                   const int* battery,
                                   const int* x,
                                   const int* y,
                                   const int* qx,
                                   const int* qy,
                                   const int* scanned,
                                   int* candidates,
                                   int nBikes,
                                   int radius) {
  __shared__ int flag[F_SECTION];
  __shared__ int stage[F_SECTION];
  __shared__ int part[F_THREADS];
  const int t = threadIdx.x;
  const int q = blockIdx.y;
  const int rx = qx[q];
  const int ry = qy[q];
  const int base = blockIdx.x * F_SECTION;
  const int first = t * F_COARSEN;

  for (int k = 0; k < F_COARSEN; ++k) {
    const int j = k * F_THREADS + t;
    const int i = base + j;
    flag[j] = (i < nBikes && isReadyBike(state[i], battery[i], x[i], y[i], rx, ry, radius)) ? 1 : 0;
  }
  __syncthreads();

  part[t] = count_own_flags(flag, first);
  kogge_stone_in_shared(part);
  __syncthreads();

  pack_own_flagged_indices(flag, first, base, exclusive_from_inclusive(part, t), stage);
  __syncthreads();

  const int cell = q * gridDim.x + blockIdx.x;
  const int start = exclusive_from_inclusive(scanned, cell);
  const int blockCount = part[F_THREADS - 1];
  copy_out_coalesced(stage, blockCount, candidates + start);
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

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

struct Station {
  int x;
  int y;
};

struct FleetSnapshot {
  std::vector<BikeState> state;
  std::vector<int> battery;
  std::vector<int> x;
  std::vector<int> y;
  std::vector<int> dockedAt;
};

struct RiderRequests {
  std::vector<int> x;
  std::vector<int> y;
};

struct CandidateLists {
  std::vector<int> start;
  std::vector<int> bike;
};

struct DeviceData {
  BikeState* state;
  int* battery;
  int* x;
  int* y;
  int* requestX;
  int* requestY;
  int* counts;
  int* scannedCounts;
  int* scratch;
  int* candidates;
};

int blocksFor(int n, int blockSize) {
  return static_cast<int>(cdiv(n, blockSize));
}

int* allocateDeviceInts(int count) {
  int* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, count * sizeof(int)));
  return pointer;
}

template <class T>
T* upload(const std::vector<T>& values) {
  T* pointer = nullptr;
  CHECK(cudaMalloc(&pointer, values.size() * sizeof(T)));
  CHECK(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
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

struct SearchGrid {
  int bikes;
  int requests;
  int fleetSections;
  int cells;
};

SearchGrid makeSearchGrid(int bikes, int requests) {
  const int fleetSections = blocksFor(bikes, F_SECTION);
  return {bikes, requests, fleetSections, fleetSections * requests};
}

dim3 sectionsByRequestsGrid(const SearchGrid& grid) {
  return dim3(grid.fleetSections, grid.requests);
}

std::vector<Station> makeStations(SplitMix64& rng) {
  std::vector<Station> stations(STATIONS);
  for (Station& station : stations) {
    station.x = rng.below(CITY_SIDE_M);
    station.y = rng.below(CITY_SIDE_M);
  }
  return stations;
}

BikeState pickState(SplitMix64& rng) {
  const int ticket = rng.below(DOCKED_WEIGHT + RIDING_WEIGHT + IN_REPAIR_WEIGHT);
  if (ticket < DOCKED_WEIGHT) {
    return BikeState::Docked;
  }
  if (ticket < DOCKED_WEIGHT + RIDING_WEIGHT) {
    return BikeState::Riding;
  }
  return BikeState::InRepair;
}

FleetSnapshot makeFleet(SplitMix64& rng, const std::vector<Station>& stations, int bikes) {
  FleetSnapshot fleet;
  fleet.state.resize(bikes);
  fleet.battery.resize(bikes);
  fleet.x.resize(bikes);
  fleet.y.resize(bikes);
  fleet.dockedAt.assign(bikes, NOT_DOCKED);
  for (int i = 0; i < bikes; ++i) {
    fleet.state[i] = pickState(rng);
    fleet.battery[i] = rng.below(FULL_BATTERY_PERCENT + 1);
    fleet.x[i] = rng.below(CITY_SIDE_M);
    fleet.y[i] = rng.below(CITY_SIDE_M);
    if (fleet.state[i] == BikeState::Docked) {
      const int station = rng.below(STATIONS);
      fleet.dockedAt[i] = station;
      fleet.x[i] = stations[station].x;
      fleet.y[i] = stations[station].y;
    }
  }
  return fleet;
}

RiderRequests makeRiderRequests(SplitMix64& rng, int requests) {
  RiderRequests riders;
  riders.x.resize(requests);
  riders.y.resize(requests);
  for (int q = 0; q < requests; ++q) {
    riders.x[q] = rng.below(CITY_SIDE_M);
    riders.y[q] = rng.below(CITY_SIDE_M);
  }
  return riders;
}

CandidateLists cpuCandidateLists(const FleetSnapshot& fleet, const RiderRequests& riders) {
  const int bikes = static_cast<int>(fleet.state.size());
  const int requests = static_cast<int>(riders.x.size());
  CandidateLists lists;
  lists.start.resize(requests + 1);
  lists.bike.reserve(static_cast<size_t>(requests) * EXPECTED_CANDIDATES_PER_REQUEST);
  for (int q = 0; q < requests; ++q) {
    lists.start[q] = static_cast<int>(lists.bike.size());
    for (int i = 0; i < bikes; ++i) {
      if (isReadyBike(fleet.state[i],
                      fleet.battery[i],
                      fleet.x[i],
                      fleet.y[i],
                      riders.x[q],
                      riders.y[q],
                      SEARCH_RADIUS_M)) {
        lists.bike.push_back(i);
      }
    }
  }
  lists.start[requests] = static_cast<int>(lists.bike.size());
  return lists;
}

DeviceData uploadData(const FleetSnapshot& fleet,
                      const RiderRequests& riders,
                      const SearchGrid& grid) {
  DeviceData data;
  data.state = upload(fleet.state);
  data.battery = upload(fleet.battery);
  data.x = upload(fleet.x);
  data.y = upload(fleet.y);
  data.requestX = upload(riders.x);
  data.requestY = upload(riders.y);
  data.counts = allocateDeviceInts(grid.cells);
  data.scannedCounts = allocateDeviceInts(grid.cells);
  data.scratch = allocateDeviceInts(scratchIntsForHierarchicalScan(grid.cells));
  data.candidates = nullptr;
  return data;
}

void freeDeviceData(const DeviceData& data) {
  CHECK(cudaFree(data.state));
  int* const intBuffers[] = {data.battery,
                             data.x,
                             data.y,
                             data.requestX,
                             data.requestY,
                             data.counts,
                             data.scannedCounts,
                             data.scratch,
                             data.candidates};
  for (int* buffer : intBuffers) {
    CHECK(cudaFree(buffer));
  }
}

int countAndScanCandidates(const DeviceData& data, const SearchGrid& grid) {
  launch(count_candidates,
         sectionsByRequestsGrid(grid),
         dim3(F_THREADS),
         data.state,
         data.battery,
         data.x,
         data.y,
         data.requestX,
         data.requestY,
         data.counts,
         grid.bikes,
         SEARCH_RADIUS_M);
  scanInclusive(data.counts, data.scannedCounts, grid.cells, data.scratch);
  return copyIntToHost(data.scannedCounts + (grid.cells - 1));
}

void compactCandidates(const DeviceData& data, const SearchGrid& grid) {
  launch(compact_candidates,
         sectionsByRequestsGrid(grid),
         dim3(F_THREADS),
         data.state,
         data.battery,
         data.x,
         data.y,
         data.requestX,
         data.requestY,
         data.scannedCounts,
         data.candidates,
         grid.bikes,
         SEARCH_RADIUS_M);
}

CandidateLists downloadCandidateLists(const DeviceData& data, const SearchGrid& grid, int total) {
  CandidateLists lists;
  lists.bike = downloadInts(data.candidates, total);
  const std::vector<int> scanned = downloadInts(data.scannedCounts, grid.cells);
  lists.start.resize(grid.requests + 1);
  for (int q = 0; q < grid.requests; ++q) {
    lists.start[q] = exclusive_from_inclusive(scanned.data(), q * grid.fleetSections);
  }
  lists.start[grid.requests] = total;
  return lists;
}

template <class Work>
double averageMs(int runs, Work work) {
  Timer timer;
  for (int run = 0; run < runs; ++run) {
    work();
  }
  return timer.ms() / runs;
}

long long squaredDistance(int x, int y, int qx, int qy) {
  const long long dx = x - qx;
  const long long dy = y - qy;
  return dx * dx + dy * dy;
}

int integerSqrt(long long value) {
  int root = 0;
  while (static_cast<long long>(root + 1) * (root + 1) <= value) {
    ++root;
  }
  return root;
}

struct BestBike {
  int bike;
  long long squaredDistance;
};

bool closerOrMoreCharged(const BestBike& candidate,
                         const BestBike& best,
                         const FleetSnapshot& fleet) {
  if (best.bike < 0 || candidate.squaredDistance < best.squaredDistance) {
    return true;
  }
  return candidate.squaredDistance == best.squaredDistance &&
         fleet.battery[candidate.bike] > fleet.battery[best.bike];
}

BestBike bestCandidate(const CandidateLists& lists,
                       int q,
                       const FleetSnapshot& fleet,
                       const RiderRequests& riders) {
  BestBike best = {-1, -1};
  for (int k = lists.start[q]; k < lists.start[q + 1]; ++k) {
    const int i = lists.bike[k];
    const BestBike candidate = {i,
                                squaredDistance(fleet.x[i], fleet.y[i], riders.x[q], riders.y[q])};
    if (closerOrMoreCharged(candidate, best, fleet)) {
      best = candidate;
    }
  }
  return best;
}

void printSuggestions(const CandidateLists& lists,
                      const FleetSnapshot& fleet,
                      const RiderRequests& riders) {
  const int requests = static_cast<int>(riders.x.size());
  std::printf("\nFirst rider requests of the batch (q<n> at the rider position):\n");
  for (int q = 0; q < imin(requests, REQUESTS_SHOWN); ++q) {
    const BestBike best = bestCandidate(lists, q, fleet, riders);
    std::printf("  q%-3d rider at (%4d, %4d) m: %3d bikes ready",
                q,
                riders.x[q],
                riders.y[q],
                lists.start[q + 1] - lists.start[q]);
    if (best.bike >= 0) {
      std::printf(", best b%d at station s%d, %d m away, battery %d %%\n",
                  best.bike,
                  fleet.dockedAt[best.bike],
                  integerSqrt(best.squaredDistance),
                  fleet.battery[best.bike]);
    } else {
      std::printf(" -> no charged bike within %d m (suggest the next station)\n", SEARCH_RADIUS_M);
    }
  }
}

void printEmulationNote() {
#ifndef __CUDACC__
  std::printf("\nNote: CPU emulation times measure thread barriers, not a GPU.\n");
#endif
}

int argumentOr(int argc, char** argv, int position, int fallback) {
  return (argc > position) ? std::atoi(argv[position]) : fallback;
}

void printResultTable(double cpuMs, int expectedTotal, double gpuMs, int total, bool pass) {
  std::printf(
      "  %-36s %10s %8s %11s  %s\n", "version", "time [ms]", "vs CPU", "candidates", "check");
  std::printf("  %-36s %10.3f %8s %11d  %s\n",
              "CPU nested loops (reference)",
              cpuMs,
              "1.00x",
              expectedTotal,
              "-");
  std::printf("  %-36s %10.3f %7.2fx %11d  %s\n",
              "GPU count + scan + staged compact",
              gpuMs,
              cpuMs / gpuMs,
              total,
              pass ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
  const int bikes = argumentOr(argc, argv, 1, DEFAULT_BIKES);
  const int requests = argumentOr(argc, argv, 2, DEFAULT_REQUESTS);
  if (bikes < 1 || requests < 1) {
    std::fprintf(stderr, "need at least one bike and one rider request\n");
    return 1;
  }
  std::printf("Bikes ready for each rider: docked, battery >= %d %%, within %d m\n",
              MIN_BATTERY_PERCENT,
              SEARCH_RADIUS_M);
  std::printf("backend: %s, %d bikes, %d rider requests -> %lld predicate evaluations\n\n",
              CUDA_BACKEND,
              bikes,
              requests,
              static_cast<long long>(bikes) * requests);

  SplitMix64 rng(SNAPSHOT_SEED);
  const std::vector<Station> stations = makeStations(rng);
  const FleetSnapshot fleet = makeFleet(rng, stations, bikes);
  const RiderRequests riders = makeRiderRequests(rng, requests);

  CandidateLists expected;
  const double cpuMs = averageMs(TIMED_RUNS, [&] {
    expected = cpuCandidateLists(fleet, riders);
  });
  const int expectedTotal = expected.start[requests];

  const SearchGrid grid = makeSearchGrid(bikes, requests);
  DeviceData data = uploadData(fleet, riders, grid);

  const int capacity = imax(countAndScanCandidates(data, grid), 1);
  data.candidates = allocateDeviceInts(capacity);

  int total = 0;
  bool overflow = false;
  auto run = [&] {
    total = countAndScanCandidates(data, grid);
    if (total > capacity) {
      overflow = true;
      return;
    }
    compactCandidates(data, grid);
  };
  for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
    run();
  }
  CHECK(cudaDeviceSynchronize());
  const double gpuMs = averageMs(TIMED_RUNS, run);
  CHECK(cudaGetLastError());

  bool pass = !overflow && total == expectedTotal;
  CandidateLists lists;
  if (pass) {
    lists = downloadCandidateLists(data, grid, total);
    pass = (lists.start == expected.start) && (lists.bike == expected.bike);
  }

  printResultTable(cpuMs, expectedTotal, gpuMs, total, pass);
  if (pass) {
    printSuggestions(lists, fleet, riders);
  }
  printEmulationNote();
  std::printf("\n%s\n", pass ? "ALL PASS" : "SOME CHECKS FAILED");

  freeDeviceData(data);
  return pass ? 0 : 1;
}
