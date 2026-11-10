#include "cuda_compat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

enum class EventKind : int { Undock, Dock };

struct Event {
  int timeMs;
  EventKind kind;
  int stationId;
  int bikeId;
};

__host__ __device__ inline int co_rank_events(int k, const Event* A, int m, const Event* B, int n) {
  int i = imin(k, m);
  int j = k - i;
  int i_low = imax(0, k - n);
  int j_low = imax(0, k - m);
  bool searching = true;
  while (searching) {
    const bool i_too_large = i > 0 && j < n && A[i - 1].timeMs > B[j].timeMs;
    const bool i_too_small = j > 0 && i < m && B[j - 1].timeMs >= A[i].timeMs;
    if (i_too_large) {
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      i -= delta;
      j += delta;
    } else if (i_too_small) {
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      searching = false;
    }
  }
  return i;
}

__host__ __device__ inline void merge_events(
    const Event* A, int m, const Event* B, int n, Event* C) {
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < m && j < n) {
    if (A[i].timeMs <= B[j].timeMs) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }
  while (i < m) {
    C[k++] = A[i++];
  }
  while (j < n) {
    C[k++] = B[j++];
  }
}

struct ListPair {
  int leftBegin;
  int rightBegin;
  int end;
};

__host__ __device__ inline int last_pair_starting_at_or_before(int k,
                                                               const int* listBegin,
                                                               int numPairs) {
  int low = 0;
  int high = numPairs - 1;
  while (low < high) {
    const int middle = (low + high + 1) / 2;
    if (listBegin[2 * middle] <= k) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return low;
}

__host__ __device__ inline ListPair list_pair(int pair, const int* listBegin, int numLists) {
  const int left = 2 * pair;
  const int rightOrEmpty = imin(left + 1, numLists);
  const int afterRight = imin(left + 2, numLists);
  return {listBegin[left], listBegin[rightOrEmpty], listBegin[afterRight]};
}

__global__ void merge_round_kernel(const Event* in,
                                   Event* out,
                                   const int* listBegin,
                                   int numLists,
                                   int total,
                                   int outputsPerThread) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int numPairs = (numLists + 1) / 2;
  const int sliceBegin = imin(tid * outputsPerThread, total);
  const int sliceEnd = imin(sliceBegin + outputsPerThread, total);

  int k = sliceBegin;
  while (k < sliceEnd) {
    const int pairIndex = last_pair_starting_at_or_before(k, listBegin, numPairs);
    const ListPair pair = list_pair(pairIndex, listBegin, numLists);
    const int pieceEnd = imin(sliceEnd, pair.end);

    const Event* A = in + pair.leftBegin;
    const Event* B = in + pair.rightBegin;
    const int m = pair.rightBegin - pair.leftBegin;
    const int n = pair.end - pair.rightBegin;
    const int k_curr = k - pair.leftBegin;
    const int k_next = pieceEnd - pair.leftBegin;
    const int i_curr = co_rank_events(k_curr, A, m, B, n);
    const int i_next = co_rank_events(k_next, A, m, B, n);
    const int j_curr = k_curr - i_curr;
    const int j_next = k_next - i_next;
    merge_events(A + i_curr, i_next - i_curr, B + j_curr, j_next - j_curr, out + k);

    k = pieceEnd;
  }
}

#ifndef CPU_BIKES
#define CPU_BIKES 240
#endif
#ifndef CPU_TRIPS_PER_BIKE
#define CPU_TRIPS_PER_BIKE 150
#endif
constexpr int STATIONS = 48;
constexpr int BIKES = BY_BACKEND(10000, CPU_BIKES);
constexpr int TRIPS_PER_BIKE = BY_BACKEND(400, CPU_TRIPS_PER_BIKE);
constexpr uint64_t LOG_SEED = 17;

constexpr int STATION_CLOCK_TICK_MS = 1000;
constexpr int SECONDS_PER_MINUTE = 60;
constexpr int PARKED_TICKS_MAX = 45 * SECONDS_PER_MINUTE;
constexpr int RIDE_TICKS_MIN = 3 * SECONDS_PER_MINUTE;
constexpr int RIDE_TICKS_MAX = 25 * SECONDS_PER_MINUTE;

constexpr int OUTPUTS_PER_THREAD = 32;
constexpr int THREADS_PER_BLOCK = 256;

constexpr int FIRST_EVENTS_SHOWN = 6;
constexpr int TIE_GROUP_MIN_SIZE = 3;
constexpr int TIE_GROUP_MAX_SHOWN = 4;

constexpr int MS_PER_SECOND = 1000;
constexpr int SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE;
constexpr int HOURS_PER_DAY = 24;

using StationLog = std::vector<Event>;

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
    const uint64_t choices = static_cast<uint64_t>(highest - lowest) + 1;
    return lowest + static_cast<int>(next() % choices);
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

static const char* eventKindName(EventKind kind) {
  switch (kind) {
    case EventKind::Undock:
      return "undock";
    case EventKind::Dock:
      return "dock";
  }
  return "unknown";
}

static bool earlier(const Event& a, const Event& b) {
  return a.timeMs < b.timeMs;
}

class SyntheticCityWeek {
 public:
  explicit SyntheticCityWeek(uint64_t seed) : rng_(seed) {
  }

  std::vector<StationLog> makeLogs(int bikes, int tripsPerBike) {
    std::vector<StationLog> logs(STATIONS);
    for (int bike = 0; bike < bikes; ++bike) {
      recordTripsOfOneBike(bike, tripsPerBike, logs);
    }
    for (StationLog& log : logs) {
      std::stable_sort(log.begin(), log.end(), earlier);
    }
    return logs;
  }

 private:
  void recordTripsOfOneBike(int bike, int tripsPerBike, std::vector<StationLog>& logs) {
    int station = randomStation();
    int timeMs = ticks(0, PARKED_TICKS_MAX);
    for (int trip = 0; trip < tripsPerBike; ++trip) {
      const int endStation = randomStation();
      logs[station].push_back({timeMs, EventKind::Undock, station, bike});
      timeMs += ticks(RIDE_TICKS_MIN, RIDE_TICKS_MAX);
      logs[endStation].push_back({timeMs, EventKind::Dock, endStation, bike});
      timeMs += ticks(0, PARKED_TICKS_MAX);
      station = endStation;
    }
  }

  int randomStation() {
    return rng_.between(0, STATIONS - 1);
  }

  int ticks(int fewest, int most) {
    return STATION_CLOCK_TICK_MS * rng_.between(fewest, most);
  }

  SplitMix64 rng_;
};

struct ConcatenatedLogs {
  std::vector<Event> events;
  std::vector<int> listBegin;
};

static ConcatenatedLogs concatenate(const std::vector<StationLog>& logs) {
  ConcatenatedLogs result;
  result.listBegin.push_back(0);
  for (const StationLog& log : logs) {
    result.events.insert(result.events.end(), log.begin(), log.end());
    result.listBegin.push_back(static_cast<int>(result.events.size()));
  }
  return result;
}

static std::vector<int> listBeginAfterRound(const std::vector<int>& listBegin, int numLists) {
  std::vector<int> merged;
  for (int left = 0; left < numLists; left += 2) {
    merged.push_back(listBegin[left]);
  }
  merged.push_back(listBegin[numLists]);
  return merged;
}

static int otherBuffer(int buffer) {
  return 1 - buffer;
}

struct TreeMergeResult {
  std::vector<Event> merged;
  int rounds;
  double ms;
};

static TreeMergeResult mergeTreeOnDevice(const ConcatenatedLogs& logs) {
  const int total = static_cast<int>(logs.events.size());
  const size_t eventBytes = total * sizeof(Event);
  Event* deviceBuffers[2];
  int* deviceListBegin;
  CHECK(cudaMalloc(&deviceBuffers[0], eventBytes));
  CHECK(cudaMalloc(&deviceBuffers[1], eventBytes));
  CHECK(cudaMalloc(&deviceListBegin, logs.listBegin.size() * sizeof(int)));
  CHECK(cudaMemcpy(deviceBuffers[0], logs.events.data(), eventBytes, cudaMemcpyHostToDevice));

  const unsigned threadsNeeded = cdiv(total, OUTPUTS_PER_THREAD);
  const unsigned blocks = cdiv(threadsNeeded, THREADS_PER_BLOCK);
  std::vector<int> listBegin = logs.listBegin;
  int numLists = static_cast<int>(logs.listBegin.size()) - 1;
  int source = 0;
  int rounds = 0;

  Timer timer;
  while (numLists > 1) {
    CHECK(cudaMemcpy(
        deviceListBegin, listBegin.data(), (numLists + 1) * sizeof(int), cudaMemcpyHostToDevice));
    launch(merge_round_kernel,
           dim3(blocks),
           dim3(THREADS_PER_BLOCK),
           deviceBuffers[source],
           deviceBuffers[otherBuffer(source)],
           deviceListBegin,
           numLists,
           total,
           OUTPUTS_PER_THREAD);
    CHECK(cudaGetLastError());

    listBegin = listBeginAfterRound(listBegin, numLists);
    numLists = (numLists + 1) / 2;
    source = otherBuffer(source);
    ++rounds;
  }
  const double ms = timer.ms();

  TreeMergeResult result{std::vector<Event>(total), rounds, ms};
  CHECK(
      cudaMemcpy(result.merged.data(), deviceBuffers[source], eventBytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(deviceBuffers[0]));
  CHECK(cudaFree(deviceBuffers[1]));
  CHECK(cudaFree(deviceListBegin));
  return result;
}

static bool sameEvent(const Event& a, const Event& b) {
  return a.timeMs == b.timeMs && a.kind == b.kind && a.stationId == b.stationId &&
         a.bikeId == b.bikeId;
}

static bool sameEvents(const std::vector<Event>& a, const std::vector<Event>& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(), sameEvent);
}

static bool sortedByTime(const std::vector<Event>& events) {
  return std::is_sorted(events.begin(), events.end(), earlier);
}

static bool equalTimesListLowerStationFirst(const std::vector<Event>& events) {
  for (size_t i = 1; i < events.size(); ++i) {
    const Event& previous = events[i - 1];
    const Event& current = events[i];
    if (previous.timeMs == current.timeMs && previous.stationId > current.stationId) {
      return false;
    }
  }
  return true;
}

static bool eachBikeAlternatesAndStaysPut(const std::vector<Event>& events, int numBikes) {
  const int noStationYet = -1;
  std::vector<int> lastTimeMs(numBikes, 0);
  std::vector<EventKind> lastKind(numBikes, EventKind::Dock);
  std::vector<int> dockedAt(numBikes, noStationYet);
  bool ok = true;
  for (const Event& event : events) {
    const int bike = event.bikeId;
    const bool timeAdvances = event.timeMs >= lastTimeMs[bike];
    const bool kindAlternates = event.kind != lastKind[bike];
    const bool leavesWhereItWasDocked = event.kind == EventKind::Dock ||
                                        dockedAt[bike] == noStationYet ||
                                        dockedAt[bike] == event.stationId;
    ok = ok && timeAdvances && kindAlternates && leavesWhereItWasDocked;
    lastTimeMs[bike] = event.timeMs;
    lastKind[bike] = event.kind;
    if (event.kind == EventKind::Dock) {
      dockedAt[bike] = event.stationId;
    }
  }
  return ok;
}

static void printEvent(const Event& event) {
  const int totalSeconds = event.timeMs / MS_PER_SECOND;
  const int day = totalSeconds / (SECONDS_PER_HOUR * HOURS_PER_DAY);
  const int hour = totalSeconds / SECONDS_PER_HOUR % HOURS_PER_DAY;
  const int minute = totalSeconds / SECONDS_PER_MINUTE % SECONDS_PER_MINUTE;
  const int second = totalSeconds % SECONDS_PER_MINUTE;
  std::printf("  day %d %02d:%02d:%02d  station s%02d  %-6s  bike b%04d\n",
              day,
              hour,
              minute,
              second,
              event.stationId,
              eventKindName(event.kind),
              event.bikeId);
}

static void printFirstEvents(const std::vector<Event>& events) {
  std::printf("first lines of the city log:\n");
  const int shown = std::min<int>(FIRST_EVENTS_SHOWN, events.size());
  for (int i = 0; i < shown; ++i) {
    printEvent(events[i]);
  }
}

static bool startsTieGroup(const std::vector<Event>& events, size_t i) {
  if (i == 0 || i + TIE_GROUP_MIN_SIZE > events.size()) {
    return false;
  }
  const int timeMs = events[i].timeMs;
  return events[i - 1].timeMs != timeMs && events[i + 1].timeMs == timeMs &&
         events[i + 2].timeMs == timeMs;
}

static void printFirstTieGroup(const std::vector<Event>& events) {
  std::printf("a group of events in the same second (lower station ids first):\n");
  for (size_t first = 0; first < events.size(); ++first) {
    if (startsTieGroup(events, first)) {
      const size_t last = std::min(events.size(), first + TIE_GROUP_MAX_SHOWN);
      for (size_t i = first; i < last && events[i].timeMs == events[first].timeMs; ++i) {
        printEvent(events[i]);
      }
      return;
    }
  }
}

static void printTimings(double cpuMs, const TreeMergeResult& tree) {
  std::printf("\n%-40s %10s\n", "step", "time (ms)");
  std::printf("%-40s %10.2f\n", "std::stable_sort by t (CPU, reference)", cpuMs);
  std::printf("%-40s %10.2f\n", "tree of co-rank merges (device)", tree.ms);
  std::printf("rounds: %d (ceil(log2 %d))\n\n", tree.rounds, STATIONS);
}

static bool reportCheck(const char* label, bool ok) {
  std::printf("%-43s%s\n", label, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::printf("Merging per-station dock logs on %s\n", CUDA_BACKEND);

  SyntheticCityWeek simulation(LOG_SEED);
  const ConcatenatedLogs logs = concatenate(simulation.makeLogs(BIKES, TRIPS_PER_BIKE));
  std::printf("%d stations, %d bikes, %d events in all\n\n",
              STATIONS,
              BIKES,
              static_cast<int>(logs.events.size()));

  std::vector<Event> reference = logs.events;
  Timer cpuTimer;
  std::stable_sort(reference.begin(), reference.end(), earlier);
  const double cpuMs = cpuTimer.ms();

  const TreeMergeResult tree = mergeTreeOnDevice(logs);
  const std::vector<Event>& merged = tree.merged;

  printFirstEvents(merged);
  printFirstTieGroup(merged);
  printTimings(cpuMs, tree);

  bool ok = true;
  ok = reportCheck("same as stable sort:", sameEvents(merged, reference)) && ok;
  ok = reportCheck("sorted by t:", sortedByTime(merged)) && ok;
  ok = reportCheck("equal t, lower station first:", equalTimesListLowerStationFirst(merged)) && ok;
  ok = reportCheck("per bike, undock/dock alternate in place:",
                   eachBikeAlternatesAndStaysPut(merged, BIKES)) &&
       ok;
  std::printf("%s\n", ok ? "ALL PASS" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
