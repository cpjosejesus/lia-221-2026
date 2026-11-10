#include "cuda_compat.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef CPU_RIDE_REQUESTS
#define CPU_RIDE_REQUESTS 40000
#endif

const int RIDE_REQUEST_COUNT = BY_BACKEND(2000000, CPU_RIDE_REQUESTS);

const int MAP_SIDE_METRES = 6000;
const int GRID_SIDE = 125;
const int CELL_METRES = MAP_SIDE_METRES / GRID_SIDE;

#define FILTER_RADIUS 6
#define FILTER_DIM (2 * FILTER_RADIUS + 1)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

const double GAUSSIAN_SIGMA_CELLS = 3.0;

__constant__ float G_c[FILTER_DIM][FILTER_DIM];

__global__ void smooth_tiled_kernel(const float* H, float* S, int width, int height) {
  int col = (int)(blockIdx.x * OUT_TILE_DIM) + (int)threadIdx.x - FILTER_RADIUS;
  int row = (int)(blockIdx.y * OUT_TILE_DIM) + (int)threadIdx.y - FILTER_RADIUS;

  __shared__ float H_s[IN_TILE_DIM][IN_TILE_DIM];
  if (row >= 0 && row < height && col >= 0 && col < width) {
    H_s[threadIdx.y][threadIdx.x] = H[row * width + col];
  } else {
    H_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int tileCol = (int)threadIdx.x - FILTER_RADIUS;
  int tileRow = (int)threadIdx.y - FILTER_RADIUS;
  bool insideImage = col >= 0 && col < width && row >= 0 && row < height;
  bool computesOutput =
      tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM;

  if (insideImage && computesOutput) {
    float s = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
      for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
        s += G_c[fRow][fCol] * H_s[tileRow + fRow][tileCol + fCol];
      }
    }
    S[row * width + col] = s;
  }
}

using Grid = std::vector<float>;

static void smooth_cpu(const Grid& H, const std::vector<float>& G, Grid& S, int width, int height) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      float s = 0.0f;
      for (int fRow = 0; fRow < FILTER_DIM; ++fRow) {
        for (int fCol = 0; fCol < FILTER_DIM; ++fCol) {
          int inRow = row - FILTER_RADIUS + fRow;
          int inCol = col - FILTER_RADIUS + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            s += G[fRow * FILTER_DIM + fCol] * H[inRow * width + inCol];
          }
        }
      }
      S[row * width + col] = s;
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

  int below(int n) {
    return static_cast<int>(next() % static_cast<uint64_t>(n));
  }

  double uniform() {
    return static_cast<double>(next() >> UNIFORM_SHIFT) * UNIFORM_SCALE;
  }

  double standardNormal() {
    const double awayFromZero = 1.0 - uniform();
    const double angle = TWO_PI * uniform();
    return std::sqrt(-2.0 * std::log(awayFromZero)) * std::cos(angle);
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;
  static constexpr int UNIFORM_SHIFT = 11;
  static constexpr double UNIFORM_SCALE = 1.0 / 9007199254740992.0;
  static constexpr double TWO_PI = 6.283185307179586;

  uint64_t state_;
};

struct Position {
  int x;
  int y;
};

struct Station {
  int id;
  Position position;
  int docks;
  int bikesDocked;
};

const int STATION_COUNT = 48;
const uint64_t STATION_SEED = 48;
const int STATION_MARGIN_METRES = 300;
const int MIN_DOCKS = 12;
const int EXTRA_DOCKS_RANGE = 13;

enum class Rush { Morning, Evening };

struct DemandCluster {
  const char* kind;
  Position centre;
  double spreadMetres;
  double morningWeight;
  double eveningWeight;
};

const DemandCluster CLUSTERS[] = {
    {"homes (north-west)", {1200, 4800}, 260, 5.0, 0.8},
    {"homes (south)", {2500, 900}, 300, 4.0, 0.6},
    {"condos (east)", {5000, 3900}, 200, 3.0, 1.0},
    {"offices (centre)", {3100, 3000}, 200, 0.8, 5.0},
    {"university (west)", {900, 2500}, 220, 1.5, 3.5},
    {"market (south-east)", {4700, 1500}, 250, 1.0, 2.5},
};
const int CLUSTER_COUNT = sizeof(CLUSTERS) / sizeof(CLUSTERS[0]);

const uint64_t RIDE_REQUEST_SEED = 2024;
const double MORNING_REQUEST_SHARE = 0.45;
const double BACKGROUND_REQUEST_SHARE = 0.15;
const double MORNING_RUSH_HOUR = 8.0;
const double EVENING_RUSH_HOUR = 18.0;
const double RUSH_SPREAD_HOURS = 0.5;
const int SECONDS_PER_MINUTE = 60;
const int MINUTES_PER_HOUR = 60;
const int SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
const int MS_PER_SECOND = 1000;
const double MS_PER_HOUR = static_cast<double>(SECONDS_PER_HOUR) * MS_PER_SECOND;

const int REQUESTS_TO_SHOW = 3;
const int HOT_ZONE_COUNT = 5;
const int HOT_ZONE_MIN_SEPARATION_CELLS = 15;
const double NEARBY_REQUEST_RADIUS_METRES = 250.0;
const double NEW_STATION_DISTANCE_METRES = 500.0;
const float SUPPRESSED_CELL = -1.0f;
const double RELATIVE_ERROR_FLOOR = 1e-3;
const double MAX_RELATIVE_ERROR = 1e-4;

struct RideRequest {
  int id;
  int timeMs;
  Position origin;
};

struct TimeWindow {
  const char* name;
  double fromHour;
  double toHour;
};

const TimeWindow WINDOWS[] = {
    {"morning rush 07:00-09:00", 7.0, 9.0},
    {"evening rush 17:00-19:00", 17.0, 19.0},
};

struct HotZone {
  int row;
  int col;
  float value;
};

static int cellOf(int metres) {
  return imin(GRID_SIDE - 1, metres / CELL_METRES);
}

static int cellCentreMetres(int cell) {
  return cell * CELL_METRES + CELL_METRES / 2;
}

static double distanceMetres(Position a, Position b) {
  return std::hypot(static_cast<double>(a.x - b.x), static_cast<double>(a.y - b.y));
}

static bool insideMap(Position position) {
  return position.x >= 0 && position.x < MAP_SIDE_METRES && position.y >= 0 &&
         position.y < MAP_SIDE_METRES;
}

static double requestHour(const RideRequest& request) {
  return request.timeMs / MS_PER_HOUR;
}

static bool inWindow(const RideRequest& request, const TimeWindow& window) {
  const double hour = requestHour(request);
  return hour >= window.fromHour && hour < window.toHour;
}

static double demandWeight(const DemandCluster& cluster, Rush rush) {
  return rush == Rush::Morning ? cluster.morningWeight : cluster.eveningWeight;
}

static Position randomPosition(SplitMix64& rng, int marginMetres) {
  const int span = MAP_SIDE_METRES - 2 * marginMetres;
  Position position;
  position.x = marginMetres + rng.below(span);
  position.y = marginMetres + rng.below(span);
  return position;
}

static std::vector<Station> makeStations() {
  SplitMix64 rng(STATION_SEED);
  std::vector<Station> stations(STATION_COUNT);

  for (int id = 0; id < STATION_COUNT; ++id) {
    stations[id].id = id;
    stations[id].position = randomPosition(rng, STATION_MARGIN_METRES);
  }

  for (Station& station : stations) {
    station.docks = MIN_DOCKS + rng.below(EXTRA_DOCKS_RANGE);
    station.bikesDocked = rng.below(station.docks + 1);
  }

  return stations;
}

class RideRequestGenerator {
 public:
  explicit RideRequestGenerator(uint64_t seed) : rng_(seed) {
  }

  std::vector<RideRequest> makeRequests(int count) {
    std::vector<RideRequest> requests;
    requests.reserve(count);

    while (static_cast<int>(requests.size()) < count) {
      const Rush rush = rng_.uniform() < MORNING_REQUEST_SHARE ? Rush::Morning : Rush::Evening;
      const double hour = requestTimeHour(rush);
      const Position origin = requestOrigin(rush);
      if (!insideMap(origin)) {
        continue;
      }

      RideRequest request;
      request.id = static_cast<int>(requests.size());
      request.timeMs = static_cast<int>(hour * MS_PER_HOUR);
      request.origin = origin;
      requests.push_back(request);
    }

    return requests;
  }

 private:
  double requestTimeHour(Rush rush) {
    const double peak = rush == Rush::Morning ? MORNING_RUSH_HOUR : EVENING_RUSH_HOUR;
    return peak + RUSH_SPREAD_HOURS * rng_.standardNormal();
  }

  Position requestOrigin(Rush rush) {
    if (rng_.uniform() < BACKGROUND_REQUEST_SHARE) {
      return randomPosition(rng_, 0);
    }
    return nearCluster(CLUSTERS[pickCluster(rush)]);
  }

  int pickCluster(Rush rush) {
    double totalWeight = 0.0;
    for (const DemandCluster& cluster : CLUSTERS) {
      totalWeight += demandWeight(cluster, rush);
    }

    double pick = totalWeight * rng_.uniform();
    int k = 0;
    while (k < CLUSTER_COUNT - 1 && pick > demandWeight(CLUSTERS[k], rush)) {
      pick -= demandWeight(CLUSTERS[k], rush);
      ++k;
    }
    return k;
  }

  Position nearCluster(const DemandCluster& cluster) {
    Position position;
    position.x = cluster.centre.x + static_cast<int>(rng_.standardNormal() * cluster.spreadMetres);
    position.y = cluster.centre.y + static_cast<int>(rng_.standardNormal() * cluster.spreadMetres);
    return position;
  }

  SplitMix64 rng_;
};

static int indexOfMaximum(const Grid& grid) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(grid.size()); ++i) {
    if (grid[i] > grid[best]) {
      best = i;
    }
  }
  return best;
}

static void suppressAround(Grid& grid, int centreRow, int centreCol, int radius) {
  const int firstRow = imax(0, centreRow - radius);
  const int lastRow = imin(GRID_SIDE - 1, centreRow + radius);
  const int firstCol = imax(0, centreCol - radius);
  const int lastCol = imin(GRID_SIDE - 1, centreCol + radius);

  for (int row = firstRow; row <= lastRow; ++row) {
    for (int col = firstCol; col <= lastCol; ++col) {
      grid[row * GRID_SIDE + col] = SUPPRESSED_CELL;
    }
  }
}

static std::vector<HotZone> hotZones(const Grid& smoothed, int count, int minSeparation) {
  Grid candidates = smoothed;
  std::vector<HotZone> zones;

  for (int n = 0; n < count; ++n) {
    const int best = indexOfMaximum(candidates);
    const int row = best / GRID_SIDE;
    const int col = best % GRID_SIDE;
    zones.push_back({row, col, smoothed[best]});
    suppressAround(candidates, row, col, minSeparation);
  }

  return zones;
}

static std::vector<float> gaussianWeights() {
  std::vector<float> weights(FILTER_DIM * FILTER_DIM);
  const double twoSigmaSquared = 2 * GAUSSIAN_SIGMA_CELLS * GAUSSIAN_SIGMA_CELLS;
  double sum = 0.0;

  for (int i = 0; i < FILTER_DIM; ++i) {
    for (int j = 0; j < FILTER_DIM; ++j) {
      double dy = i - FILTER_RADIUS;
      double dx = j - FILTER_RADIUS;
      weights[i * FILTER_DIM + j] = (float)std::exp(-(dx * dx + dy * dy) / twoSigmaSquared);
      sum += weights[i * FILTER_DIM + j];
    }
  }

  for (float& weight : weights) {
    weight = (float)(weight / sum);
  }
  return weights;
}

static Grid binRequests(const std::vector<RideRequest>& requests,
                        const TimeWindow& window,
                        int& binnedRequests) {
  Grid histogram(static_cast<size_t>(GRID_SIDE) * GRID_SIDE, 0.0f);
  binnedRequests = 0;

  for (const RideRequest& request : requests) {
    if (!inWindow(request, window)) {
      continue;
    }
    const int cell = cellOf(request.origin.y) * GRID_SIDE + cellOf(request.origin.x);
    histogram[cell] += 1.0f;
    ++binnedRequests;
  }

  return histogram;
}

static int requestsNear(const std::vector<RideRequest>& requests,
                        const TimeWindow& window,
                        Position centre) {
  int count = 0;
  for (const RideRequest& request : requests) {
    if (inWindow(request, window) &&
        distanceMetres(centre, request.origin) <= NEARBY_REQUEST_RADIUS_METRES) {
      ++count;
    }
  }
  return count;
}

static const Station& nearestStation(const std::vector<Station>& stations, Position position) {
  const Station* nearest = &stations[0];
  for (const Station& station : stations) {
    if (distanceMetres(position, station.position) < distanceMetres(position, nearest->position)) {
      nearest = &station;
    }
  }
  return *nearest;
}

static double maxRelativeError(const Grid& expected, const Grid& actual) {
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double difference = std::fabs((double)expected[i] - actual[i]);
    worst = std::fmax(worst, difference / (RELATIVE_ERROR_FLOOR + std::fabs((double)expected[i])));
  }
  return worst;
}

class DeviceSmoother {
 public:
  DeviceSmoother()
      : bytes_(static_cast<size_t>(GRID_SIDE) * GRID_SIDE * sizeof(float)),
        block_(IN_TILE_DIM, IN_TILE_DIM),
        grid_(cdiv(GRID_SIDE, OUT_TILE_DIM), cdiv(GRID_SIDE, OUT_TILE_DIM)) {
    CHECK(cudaMalloc(&histogram_, bytes_));
    CHECK(cudaMalloc(&smoothed_, bytes_));
  }

  ~DeviceSmoother() {
    CHECK(cudaFree(histogram_));
    CHECK(cudaFree(smoothed_));
  }

  void uploadWeights(const std::vector<float>& weights) {
    CHECK(cudaMemcpyToSymbol(G_c, weights.data(), weights.size() * sizeof(float)));
  }

  Grid smooth(const Grid& histogram, double& kernelMs) {
    CHECK(cudaMemcpy(histogram_, histogram.data(), bytes_, cudaMemcpyHostToDevice));

    Timer timer;
    launch(smooth_tiled_kernel,
           grid_,
           block_,
           (const float*)histogram_,
           smoothed_,
           GRID_SIDE,
           GRID_SIDE);
    CHECK(cudaGetLastError());
    kernelMs = timer.ms();

    Grid smoothed(static_cast<size_t>(GRID_SIDE) * GRID_SIDE);
    CHECK(cudaMemcpy(smoothed.data(), smoothed_, bytes_, cudaMemcpyDeviceToHost));
    return smoothed;
  }

 private:
  size_t bytes_;
  dim3 block_;
  dim3 grid_;
  float* histogram_;
  float* smoothed_;
};

static void printHeader() {
  std::printf("Bike-share demand heatmap, %d x %d cells of %d m, %dx%d Gaussian, backend: %s\n",
              GRID_SIDE,
              GRID_SIDE,
              CELL_METRES,
              FILTER_DIM,
              FILTER_DIM,
              CUDA_BACKEND);
  std::printf("map: %d m x %d m, x grows to the east and y to the north, %d stations\n",
              MAP_SIDE_METRES,
              MAP_SIDE_METRES,
              STATION_COUNT);
}

static void printClock(int timeMs) {
  const int totalSeconds = timeMs / MS_PER_SECOND;
  const int hours = totalSeconds / SECONDS_PER_HOUR;
  const int minutes = totalSeconds / SECONDS_PER_MINUTE % MINUTES_PER_HOUR;
  const int seconds = totalSeconds % SECONDS_PER_MINUTE;
  std::printf("%02d:%02d:%02d", hours, minutes, seconds);
}

static void printFirstRequests(const std::vector<RideRequest>& requests) {
  std::printf("\n%zu synthetic ride requests (where a rider opened the app). First ones:\n",
              requests.size());
  for (int i = 0; i < REQUESTS_TO_SHOW; ++i) {
    std::printf("  request %d at ", requests[i].id);
    printClock(requests[i].timeMs);
    std::printf(" from (%d m, %d m)\n", requests[i].origin.x, requests[i].origin.y);
  }
}

static bool isLessThanHalfFull(const Station& station) {
  return 2 * station.bikesDocked < station.docks;
}

static void printSuggestion(const Station& station, double distance) {
  if (distance > NEW_STATION_DISTANCE_METRES) {
    std::printf("new station here (s%d is %.0f m away)\n", station.id, distance);
    return;
  }
  std::printf("s%d at %.0f m holds %d of %d bikes: %s\n",
              station.id,
              distance,
              station.bikesDocked,
              station.docks,
              isLessThanHalfFull(station) ? "bring bikes" : "keep it stocked");
}

static void printHotZone(int rank,
                         const HotZone& zone,
                         Position centre,
                         int nearbyRequests,
                         const std::vector<Station>& stations) {
  const Station& nearest = nearestStation(stations, centre);
  std::printf("   %4d  (%3d,%3d)    %5d  %5d   %9.3f   %8d       ",
              rank,
              zone.row,
              zone.col,
              centre.x,
              centre.y,
              zone.value,
              nearbyRequests);
  printSuggestion(nearest, distanceMetres(centre, nearest.position));
}

static void printHotZones(const std::vector<RideRequest>& requests,
                          const TimeWindow& window,
                          const Grid& smoothed,
                          const std::vector<Station>& stations) {
  std::printf("   rank  cell(row,col)  x (m)  y (m)  requests/cell  within 250 m  suggestion\n");

  const std::vector<HotZone> zones =
      hotZones(smoothed, HOT_ZONE_COUNT, HOT_ZONE_MIN_SEPARATION_CELLS);
  for (size_t k = 0; k < zones.size(); ++k) {
    const Position centre = {cellCentreMetres(zones[k].col), cellCentreMetres(zones[k].row)};
    printHotZone(static_cast<int>(k) + 1,
                 zones[k],
                 centre,
                 requestsNear(requests, window, centre),
                 stations);
  }
}

struct WindowInputs {
  const std::vector<RideRequest>& requests;
  const std::vector<Station>& stations;
  const std::vector<float>& weights;
};

static bool analyseWindow(const TimeWindow& window,
                          const WindowInputs& inputs,
                          DeviceSmoother& smoother) {
  int windowRequests = 0;
  const Grid histogram = binRequests(inputs.requests, window, windowRequests);

  double kernelMs = 0.0;
  const Grid smoothed = smoother.smooth(histogram, kernelMs);

  Grid reference(static_cast<size_t>(GRID_SIDE) * GRID_SIDE);
  Timer cpuTimer;
  smooth_cpu(histogram, inputs.weights, reference, GRID_SIDE, GRID_SIDE);
  const double cpuMs = cpuTimer.ms();

  const double error = maxRelativeError(reference, smoothed);
  const bool passed = error < MAX_RELATIVE_ERROR;

  const int busiestRawCell = indexOfMaximum(histogram);
  std::printf("\n== %s: %d requests  (kernel %.3f ms, CPU %.3f ms, %s, max rel err %.1e)\n",
              window.name,
              windowRequests,
              kernelMs,
              cpuMs,
              passed ? "PASS" : "FAIL",
              error);
  std::printf(
      "   busiest raw cell: (%d,%d) with %.0f requests (single cells are noisy; the ranking "
      "below uses the smoothed map)\n",
      busiestRawCell / GRID_SIDE,
      busiestRawCell % GRID_SIDE,
      histogram[busiestRawCell]);
  printHotZones(inputs.requests, window, smoothed, inputs.stations);

  return passed;
}

static bool analyseEveryWindow(const WindowInputs& inputs) {
  DeviceSmoother smoother;
  smoother.uploadWeights(inputs.weights);

  bool allPassed = true;
  for (const TimeWindow& window : WINDOWS) {
    allPassed = analyseWindow(window, inputs, smoother) && allPassed;
  }
  return allPassed;
}

int main() {
  printHeader();

  const std::vector<Station> stations = makeStations();
  RideRequestGenerator generator(RIDE_REQUEST_SEED);
  const std::vector<RideRequest> requests = generator.makeRequests(RIDE_REQUEST_COUNT);
  printFirstRequests(requests);

  const std::vector<float> weights = gaussianWeights();
  const WindowInputs inputs = {requests, stations, weights};
  const bool allPassed = analyseEveryWindow(inputs);

  std::printf("\nBefore each rush, trucks should fill the stations next to its rank-1 zones.\n");
  std::printf("%s\n", allPassed ? "ALL PASS" : "SOME TESTS FAILED");
  return allPassed ? 0 : 1;
}
