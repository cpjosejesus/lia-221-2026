#include "cuda_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#ifndef CPU_GRID_SIDE
#define CPU_GRID_SIDE 40
#endif

const int GRID_SIDE = BY_BACKEND(160, CPU_GRID_SIDE);
const int MAP_SIDE_METRES = 6000;

#define IN_TILE 16
#define OUT_TILE (IN_TILE - 2)

__global__ void jacobi5_tiled_kernel(const float* in,
                                     float* out,
                                     const unsigned char* fixed,
                                     float cC,
                                     float cN,
                                     int width,
                                     int height) {
  int col = (int)(blockIdx.x * OUT_TILE) + (int)threadIdx.x - 1;
  int row = (int)(blockIdx.y * OUT_TILE) + (int)threadIdx.y - 1;
  int tx = (int)threadIdx.x;
  int ty = (int)threadIdx.y;

  __shared__ float u_s[IN_TILE][IN_TILE];
  bool inside = row >= 0 && row < height && col >= 0 && col < width;
  u_s[ty][tx] = inside ? in[row * width + col] : 0.0f;
  __syncthreads();

  if (inside && tx >= 1 && tx < IN_TILE - 1 && ty >= 1 && ty < IN_TILE - 1) {
    int p = row * width + col;
    if (fixed[p]) {
      out[p] = u_s[ty][tx];
    } else {
      out[p] = cC * u_s[ty][tx] +
               cN * (u_s[ty][tx - 1] + u_s[ty][tx + 1] + u_s[ty - 1][tx] + u_s[ty + 1][tx]);
    }
  }
}

__global__ void inject_kernel(float* u, const int* cells, int n) {
  int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < n) {
    atomicAdd(&u[cells[t]], 1.0f);
  }
}

using Field = std::vector<float>;
using FixedMask = std::vector<unsigned char>;

static void jacobi5_cpu(
    const Field& in, Field& out, const FixedMask& fixed, float cC, float cN, int w, int h) {
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      int p = r * w + c;
      if (fixed[p]) {
        out[p] = in[p];
        continue;
      }
      float left = c > 0 ? in[p - 1] : 0.0f;
      float right = c < w - 1 ? in[p + 1] : 0.0f;
      float up = r > 0 ? in[p - w] : 0.0f;
      float down = r < h - 1 ? in[p + w] : 0.0f;
      out[p] = cC * in[p] + cN * (left + right + up + down);
    }
  }
}

static double maxRelativeDifference(const Field& expected, const Field& actual) {
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double difference = std::fabs((double)expected[i] - actual[i]);
    worst = std::fmax(worst, difference / (1.0 + std::fabs((double)expected[i])));
  }
  return worst;
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
};

const int STATION_COUNT = 48;
const uint64_t STATION_SEED = 48;
const int STATION_MARGIN_METRES = 300;

const int NEIGHBOURS_PER_CELL = 4;
const int ZONES_PER_SIDE = 4;
const int ZONE_SIDE_METRES = MAP_SIDE_METRES / ZONES_PER_SIDE;
const int MAP_COLUMNS = 40;
const int MAP_ROWS = 20;
const char* const MAP_SHADES = " .:-=+*#%@";
const int DARKEST_SHADE = 9;
const float SHADE_SCALE = 9.99f;
const char STATION_MARK = 'S';
const double MAX_RELATIVE_ERROR = 1e-4;

const double DECAY_LENGTH_METRES = 250.0;
const float COVERAGE_AT_STATION = 1.0f;
const double CONVERGENCE_TOLERANCE = 1e-5;
const int SWEEPS_BETWEEN_CONVERGENCE_CHECKS = BY_BACKEND(200, 25);
const int MAX_SWEEPS = 200000;
const float COVERAGE_MAP_SHADE_MAX = 0.5f;

const int FIRST_SIMULATED_MINUTE_OF_DAY = 17 * 60 + 30;
const int SIMULATED_MINUTES = 30;
const int SNAPSHOT_EVERY_MINUTES = 10;
const int MINUTES_PER_HOUR = 60;
const int BASE_GRID_SIDE = 40;
const int STEPS_PER_MINUTE_ON_BASE_GRID = 3;
constexpr float DIFFUSION_ALPHA = 0.20f;
constexpr float MAX_STABLE_DIFFUSION_ALPHA = 0.25f;
const double DEMAND_KEPT_PER_MINUTE = 0.913;
const uint64_t DEMAND_SEED = 19;
const int MIN_REQUESTS_PER_MINUTE = 6;
const int EXTRA_REQUESTS_RANGE = 8;
constexpr int INJECT_BLOCK_DIM = 64;
const int REQUESTS_TO_SHOW = 2;
const int SECONDS_BETWEEN_REQUESTS_OF_A_MINUTE = 1;

static_assert(DIFFUSION_ALPHA <= MAX_STABLE_DIFFUSION_ALPHA,
              "explicit diffusion steps are stable only for alpha <= 1/4");
static_assert(GRID_SIDE >= BASE_GRID_SIDE && GRID_SIDE % BASE_GRID_SIDE == 0,
              "the grid refines the base grid by a whole factor");
static_assert(MIN_REQUESTS_PER_MINUTE + EXTRA_REQUESTS_RANGE - 1 <= INJECT_BLOCK_DIM,
              "one block of inject_kernel covers the busiest simulated minute");

enum class StationMarks { Show, Hide };

struct Zone {
  int row;
  int col;
};

struct ZoneMeans {
  double mean[ZONES_PER_SIDE][ZONES_PER_SIDE];

  double at(Zone zone) const {
    return mean[zone.row][zone.col];
  }
};

struct StencilCoefficients {
  float cC;
  float cN;
};

static size_t cellCount() {
  return static_cast<size_t>(GRID_SIDE) * GRID_SIDE;
}

static int cellCol(int x) {
  return imin(GRID_SIDE - 1, x * GRID_SIDE / MAP_SIDE_METRES);
}

static int cellRow(int y) {
  return imin(GRID_SIDE - 1, (MAP_SIDE_METRES - 1 - y) * GRID_SIDE / MAP_SIDE_METRES);
}

static int cellOf(Position position) {
  return cellRow(position.y) * GRID_SIDE + cellCol(position.x);
}

static double cellSizeMetres() {
  return static_cast<double>(MAP_SIDE_METRES) / GRID_SIDE;
}

static bool onMapEdge(int row, int col) {
  return row == 0 || col == 0 || row == GRID_SIDE - 1 || col == GRID_SIDE - 1;
}

static bool insideMap(Position position) {
  return position.x >= 0 && position.x < MAP_SIDE_METRES && position.y >= 0 &&
         position.y < MAP_SIDE_METRES;
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

  return stations;
}

static std::string nameOf(Zone zone) {
  std::string name;
  name += (char)('A' + zone.row);
  name += (char)('1' + zone.col);
  return name;
}

static ZoneMeans zoneMeans(const Field& u) {
  double sum[ZONES_PER_SIDE][ZONES_PER_SIDE] = {};
  double count[ZONES_PER_SIDE][ZONES_PER_SIDE] = {};
  for (int r = 0; r < GRID_SIDE; ++r) {
    for (int c = 0; c < GRID_SIDE; ++c) {
      const int zoneRow = r * ZONES_PER_SIDE / GRID_SIDE;
      const int zoneCol = c * ZONES_PER_SIDE / GRID_SIDE;
      sum[zoneRow][zoneCol] += u[r * GRID_SIDE + c];
      count[zoneRow][zoneCol] += 1;
    }
  }

  ZoneMeans means;
  for (int i = 0; i < ZONES_PER_SIDE; ++i) {
    for (int j = 0; j < ZONES_PER_SIDE; ++j) {
      means.mean[i][j] = sum[i][j] / count[i][j];
    }
  }
  return means;
}

static bool isInnerZone(Zone zone) {
  const int firstInner = 1;
  const int lastInner = ZONES_PER_SIDE - 2;
  return zone.row >= firstInner && zone.row <= lastInner && zone.col >= firstInner &&
         zone.col <= lastInner;
}

static Position zoneCentre(Zone zone) {
  Position centre;
  centre.x = zone.col * ZONE_SIDE_METRES + ZONE_SIDE_METRES / 2;
  centre.y = MAP_SIDE_METRES - zone.row * ZONE_SIDE_METRES - ZONE_SIDE_METRES / 2;
  return centre;
}

static bool stationIsDrawnAt(const std::vector<Station>& stations, int mapRow, int mapCol) {
  for (const Station& station : stations) {
    if (cellRow(station.position.y) * MAP_ROWS / GRID_SIDE == mapRow &&
        cellCol(station.position.x) * MAP_COLUMNS / GRID_SIDE == mapCol) {
      return true;
    }
  }
  return false;
}

static char shadeOf(float value, float shadeMax) {
  return MAP_SHADES[imin(DARKEST_SHADE, (int)(SHADE_SCALE * value / shadeMax))];
}

static void printAsciiMap(const Field& u,
                          float shadeMax,
                          const std::vector<Station>& stations,
                          StationMarks marks) {
  for (int y = 0; y < MAP_ROWS; ++y) {
    std::printf("      |");
    for (int x = 0; x < MAP_COLUMNS; ++x) {
      const int r = (y * GRID_SIDE) / MAP_ROWS + GRID_SIDE / (2 * MAP_ROWS);
      const int c = (x * GRID_SIDE) / MAP_COLUMNS + GRID_SIDE / (2 * MAP_COLUMNS);
      char symbol = shadeOf(u[r * GRID_SIDE + c], shadeMax);
      if (marks == StationMarks::Show && stationIsDrawnAt(stations, y, x)) {
        symbol = STATION_MARK;
      }
      std::printf("%c", symbol);
    }
    std::printf("|\n");
  }
}

static FixedMask mapEdgeMask() {
  FixedMask fixed(cellCount(), 0);
  for (int i = 0; i < GRID_SIDE; ++i) {
    fixed[i] = 1;
    fixed[(GRID_SIDE - 1) * GRID_SIDE + i] = 1;
    fixed[i * GRID_SIDE] = 1;
    fixed[i * GRID_SIDE + GRID_SIDE - 1] = 1;
  }
  return fixed;
}

class DeviceField {
 public:
  DeviceField()
      : bytes_(cellCount() * sizeof(float)),
        block_(IN_TILE, IN_TILE),
        grid_(cdiv(GRID_SIDE, OUT_TILE), cdiv(GRID_SIDE, OUT_TILE)) {
    CHECK(cudaMalloc(&bufferA_, bytes_));
    CHECK(cudaMalloc(&bufferB_, bytes_));
    CHECK(cudaMalloc(&fixed_, cellCount()));
  }

  ~DeviceField() {
    CHECK(cudaFree(bufferA_));
    CHECK(cudaFree(bufferB_));
    CHECK(cudaFree(fixed_));
  }

  void upload(const Field& initial, const FixedMask& fixed) {
    CHECK(cudaMemcpy(bufferA_, initial.data(), bytes_, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bufferB_, initial.data(), bytes_, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fixed_, fixed.data(), cellCount(), cudaMemcpyHostToDevice));
    latest_ = bufferA_;
    older_ = bufferB_;
  }

  void sweep(StencilCoefficients coefficients) {
    launch(jacobi5_tiled_kernel,
           grid_,
           block_,
           (const float*)latest_,
           older_,
           (const unsigned char*)fixed_,
           coefficients.cC,
           coefficients.cN,
           GRID_SIDE,
           GRID_SIDE);
    std::swap(latest_, older_);
  }

  float* latest() {
    return latest_;
  }

  Field downloadLatest() const {
    return download(latest_);
  }

  Field downloadOneSweepBefore() const {
    return download(older_);
  }

 private:
  Field download(const float* buffer) const {
    Field field(cellCount());
    CHECK(cudaMemcpy(field.data(), buffer, bytes_, cudaMemcpyDeviceToHost));
    return field;
  }

  size_t bytes_;
  dim3 block_;
  dim3 grid_;
  float* bufferA_;
  float* bufferB_;
  unsigned char* fixed_;
  float* latest_ = nullptr;
  float* older_ = nullptr;
};

static void sweepOnCpu(Field& latest,
                       Field& older,
                       const FixedMask& fixed,
                       StencilCoefficients coefficients) {
  jacobi5_cpu(latest, older, fixed, coefficients.cC, coefficients.cN, GRID_SIDE, GRID_SIDE);
  std::swap(latest, older);
}

static double maxChange(const Field& latest, const Field& before) {
  double change = 0.0;
  for (size_t i = 0; i < latest.size(); ++i) {
    change = std::fmax(change, std::fabs(latest[i] - before[i]));
  }
  return change;
}

struct CoverageProblem {
  Field initial;
  FixedMask fixed;
};

static CoverageProblem coverageProblem(const std::vector<Station>& stations) {
  CoverageProblem problem;
  problem.initial.assign(cellCount(), 0.0f);
  problem.fixed = mapEdgeMask();

  for (const Station& station : stations) {
    const int cell = cellOf(station.position);
    problem.fixed[cell] = 1;
    problem.initial[cell] = COVERAGE_AT_STATION;
  }
  return problem;
}

static float screenedLaplaceLambda() {
  const double kappa = cellSizeMetres() / DECAY_LENGTH_METRES;
  return (float)(NEIGHBOURS_PER_CELL / (NEIGHBOURS_PER_CELL + kappa * kappa));
}

struct Convergence {
  Field coverage;
  int sweeps;
  double lastChange;
  double ms;
};

static Convergence sweepUntilConverged(DeviceField& field, StencilCoefficients coefficients) {
  Convergence result;
  result.sweeps = 0;
  result.lastChange = 1.0;

  Timer timer;
  while (result.lastChange > CONVERGENCE_TOLERANCE && result.sweeps < MAX_SWEEPS) {
    for (int s = 0; s < SWEEPS_BETWEEN_CONVERGENCE_CHECKS; ++s) {
      field.sweep(coefficients);
      ++result.sweeps;
    }
    CHECK(cudaGetLastError());
    result.coverage = field.downloadLatest();
    result.lastChange = maxChange(result.coverage, field.downloadOneSweepBefore());
  }
  result.ms = timer.ms();
  return result;
}

static Field coverageOnCpu(const CoverageProblem& problem,
                           StencilCoefficients coefficients,
                           int sweeps) {
  Field latest = problem.initial;
  Field older = problem.initial;
  for (int s = 0; s < sweeps; ++s) {
    sweepOnCpu(latest, older, problem.fixed, coefficients);
  }
  return latest;
}

static void printZoneMeans(const ZoneMeans& means) {
  std::printf("   mean coverage per zone (rows A..D north to south, columns 1..4 west to east):\n");
  for (int i = 0; i < ZONES_PER_SIDE; ++i) {
    std::printf("      %c ", 'A' + i);
    for (int j = 0; j < ZONES_PER_SIDE; ++j) {
      std::printf("  %6.4f", means.mean[i][j]);
    }
    std::printf("\n");
  }
}

static void printWorstZones(const ZoneMeans& means) {
  Zone worst = {0, 0};
  Zone worstInner = {1, 1};
  for (int i = 0; i < ZONES_PER_SIDE; ++i) {
    for (int j = 0; j < ZONES_PER_SIDE; ++j) {
      const Zone zone = {i, j};
      if (means.at(zone) < means.at(worst)) {
        worst = zone;
      }
      if (isInnerZone(zone) && means.at(zone) < means.at(worstInner)) {
        worstInner = zone;
      }
    }
  }

  const Position suggestion = zoneCentre(worstInner);
  std::printf("   worst covered zone: %s (%.4f); worst inner zone: %s (%.4f)\n",
              nameOf(worst).c_str(),
              means.at(worst),
              nameOf(worstInner).c_str(),
              means.at(worstInner));
  std::printf(
      "   suggestion: a new docking station near (%d m, %d m)\n", suggestion.x, suggestion.y);
}

static bool runStationCoverage(DeviceField& field, const std::vector<Station>& stations) {
  const float lambda = screenedLaplaceLambda();
  const float centreWeight = 0.0f;
  const float neighbourWeight = lambda / NEIGHBOURS_PER_CELL;
  const StencilCoefficients jacobi = {centreWeight, neighbourWeight};

  const CoverageProblem problem = coverageProblem(stations);
  field.upload(problem.initial, problem.fixed);
  const Convergence converged = sweepUntilConverged(field, jacobi);

  const Field expected = coverageOnCpu(problem, jacobi, converged.sweeps);
  const double error = maxRelativeDifference(expected, converged.coverage);
  const bool passed = error < MAX_RELATIVE_ERROR;

  std::printf("\n== Part A: station coverage (decay length %.0f m, lambda = %.5f)\n",
              DECAY_LENGTH_METRES,
              lambda);
  std::printf(
      "   converged after %d sweeps (last change %.1e) in %.1f ms: %s (max rel err vs CPU %.1e)\n",
      converged.sweeps,
      converged.lastChange,
      converged.ms,
      passed ? "PASS" : "FAIL",
      error);
  std::printf("   coverage map (S = docking station, north at the top):\n");
  printAsciiMap(converged.coverage, COVERAGE_MAP_SHADE_MAX, stations, StationMarks::Show);

  const ZoneMeans means = zoneMeans(converged.coverage);
  printZoneMeans(means);
  printWorstZones(means);
  return passed;
}

struct DemandSpot {
  Position centre;
  double spreadMetres;
};

const DemandSpot UNIVERSITY = {{900, 2500}, 220};
const DemandSpot OFFICES = {{3100, 3000}, 200};

using RequestsPerMinute = std::vector<std::vector<int>>;

static void printClock(int minuteOfDay, int second) {
  std::printf(
      "%02d:%02d:%02d", minuteOfDay / MINUTES_PER_HOUR, minuteOfDay % MINUTES_PER_HOUR, second);
}

static void printRequest(int minute, int requestOfMinute, int requestId, Position origin) {
  std::printf("   request %d at ", requestId);
  printClock(FIRST_SIMULATED_MINUTE_OF_DAY + minute,
             requestOfMinute * SECONDS_BETWEEN_REQUESTS_OF_A_MINUTE);
  std::printf(" from (%d m, %d m)\n", origin.x, origin.y);
}

static Position nearSpot(SplitMix64& rng, const DemandSpot& spot) {
  Position position;
  position.x = spot.centre.x + static_cast<int>(rng.standardNormal() * spot.spreadMetres);
  position.y = spot.centre.y + static_cast<int>(rng.standardNormal() * spot.spreadMetres);
  return position;
}

static RequestsPerMinute makeEveningRequests(int& requestCount) {
  SplitMix64 rng(DEMAND_SEED);
  RequestsPerMinute cells(SIMULATED_MINUTES);
  requestCount = 0;

  for (int minute = 0; minute < SIMULATED_MINUTES; ++minute) {
    const int requestsThisMinute = MIN_REQUESTS_PER_MINUTE + rng.below(EXTRA_REQUESTS_RANGE);
    const double shareStillAtUniversity = 1.0 - minute / (double)SIMULATED_MINUTES;

    for (int q = 0; q < requestsThisMinute; ++q) {
      const DemandSpot& spot = rng.uniform() < shareStillAtUniversity ? UNIVERSITY : OFFICES;
      const Position origin = nearSpot(rng, spot);
      if (!insideMap(origin)) {
        continue;
      }
      if (onMapEdge(cellRow(origin.y), cellCol(origin.x))) {
        continue;
      }

      cells[minute].push_back(cellOf(origin));
      if (requestCount < REQUESTS_TO_SHOW) {
        printRequest(minute, q, requestCount, origin);
      }
      ++requestCount;
    }
  }
  return cells;
}

static int diffusionStepsPerMinute() {
  const int refinement = GRID_SIDE / BASE_GRID_SIDE;
  return STEPS_PER_MINUTE_ON_BASE_GRID * refinement * refinement;
}

static StencilCoefficients diffusionCoefficients(int stepsPerMinute) {
  const float keptPerStep = (float)std::pow(DEMAND_KEPT_PER_MINUTE, 1.0 / stepsPerMinute);
  const float centreWeight = keptPerStep * (1.0f - NEIGHBOURS_PER_CELL * DIFFUSION_ALPHA);
  const float neighbourWeight = keptPerStep * DIFFUSION_ALPHA;
  return {centreWeight, neighbourWeight};
}

static Zone hottestZone(const ZoneMeans& means) {
  Zone hottest = {0, 0};
  for (int i = 0; i < ZONES_PER_SIDE; ++i) {
    for (int j = 0; j < ZONES_PER_SIDE; ++j) {
      const Zone zone = {i, j};
      if (means.at(zone) > means.at(hottest)) {
        hottest = zone;
      }
    }
  }
  return hottest;
}

static Position cellCentre(int cell) {
  const double size = cellSizeMetres();
  Position centre;
  centre.x = static_cast<int>((cell % GRID_SIDE + 0.5) * size);
  centre.y = static_cast<int>(MAP_SIDE_METRES - (cell / GRID_SIDE + 0.5) * size);
  return centre;
}

static void printSnapshot(int minutesElapsed, const Field& demand) {
  const int hottestCell = (int)(std::max_element(demand.begin(), demand.end()) - demand.begin());
  const Position centre = cellCentre(hottestCell);
  const Zone zone = hottestZone(zoneMeans(demand));

  std::printf("   ");
  printClock(FIRST_SIMULATED_MINUTE_OF_DAY + minutesElapsed, 0);
  std::printf("  hottest zone %s, hottest cell around (%d m, %d m)\n",
              nameOf(zone).c_str(),
              centre.x,
              centre.y);
}

class RequestInjector {
 public:
  RequestInjector() {
    CHECK(cudaMalloc(&cells_, INJECT_BLOCK_DIM * sizeof(int)));
  }

  ~RequestInjector() {
    CHECK(cudaFree(cells_));
  }

  void inject(float* demand, const std::vector<int>& cells) {
    const int n = (int)cells.size();
    if (n == 0) {
      return;
    }
    CHECK(cudaMemcpy(cells_, cells.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    launch(inject_kernel, dim3(1), dim3(INJECT_BLOCK_DIM), demand, (const int*)cells_, n);
  }

 private:
  int* cells_;
};

static bool runDemandDiffusion(DeviceField& field, const std::vector<Station>& stations) {
  const int stepsPerMinute = diffusionStepsPerMinute();
  const StencilCoefficients diffusion = diffusionCoefficients(stepsPerMinute);
  const FixedMask fixed = mapEdgeMask();
  const Field empty(cellCount(), 0.0f);
  field.upload(empty, fixed);

  std::printf("\n== Part B: rental demand diffusion from ");
  printClock(FIRST_SIMULATED_MINUTE_OF_DAY, 0);
  std::printf(" for %d minutes, first requests:\n", SIMULATED_MINUTES);
  int requestCount = 0;
  const RequestsPerMinute requests = makeEveningRequests(requestCount);
  std::printf("   %d ride requests, %d diffusion steps per simulated minute\n",
              requestCount,
              stepsPerMinute);

  RequestInjector injector;
  Field latest = empty;
  Field older = empty;
  Field snapshot(cellCount());
  Timer timer;
  for (int minute = 0; minute < SIMULATED_MINUTES; ++minute) {
    injector.inject(field.latest(), requests[minute]);
    for (int cell : requests[minute]) {
      latest[cell] += 1.0f;
    }

    for (int s = 0; s < stepsPerMinute; ++s) {
      field.sweep(diffusion);
      sweepOnCpu(latest, older, fixed, diffusion);
    }

    const int minutesElapsed = minute + 1;
    if (minutesElapsed % SNAPSHOT_EVERY_MINUTES == 0) {
      snapshot = field.downloadLatest();
      printSnapshot(minutesElapsed, snapshot);
    }
  }
  CHECK(cudaGetLastError());
  const double ms = timer.ms();

  const double error = maxRelativeDifference(latest, snapshot);
  const bool passed = error < MAX_RELATIVE_ERROR;
  std::printf(
      "   %d sweeps + %d injections, %.1f ms (GPU and CPU reference together): %s (max rel err "
      "%.1e)\n",
      SIMULATED_MINUTES * stepsPerMinute,
      SIMULATED_MINUTES,
      ms,
      passed ? "PASS" : "FAIL",
      error);
  std::printf("   demand at ");
  printClock(FIRST_SIMULATED_MINUTE_OF_DAY + SIMULATED_MINUTES, 0);
  std::printf(" (darker = more recent requests nearby):\n");
  printAsciiMap(
      snapshot, *std::max_element(snapshot.begin(), snapshot.end()), stations, StationMarks::Hide);
  return passed;
}

static bool runBothParts(const std::vector<Station>& stations) {
  DeviceField field;
  bool allPassed = true;
  allPassed = runStationCoverage(field, stations) && allPassed;
  allPassed = runDemandDiffusion(field, stations) && allPassed;
  return allPassed;
}

int main() {
  std::printf(
      "Bike-share station coverage and demand diffusion, %d x %d cells (%.0f m), "
      "backend: %s\n",
      GRID_SIDE,
      GRID_SIDE,
      cellSizeMetres(),
      CUDA_BACKEND);

  const std::vector<Station> stations = makeStations();
  const bool allPassed = runBothParts(stations);

  std::printf("\n%s\n", allPassed ? "ALL PASS" : "SOME TESTS FAILED");
  return allPassed ? 0 : 1;
}
