#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

constexpr int STENCIL_POINTS = 7;
constexpr float COEFFICIENTS[STENCIL_POINTS] = {0.40f, 0.11f, 0.09f, 0.12f, 0.08f, 0.13f, 0.07f};

constexpr bool allDifferent(const float (&values)[STENCIL_POINTS]) {
  for (int a = 0; a < STENCIL_POINTS; ++a) {
    for (int b = a + 1; b < STENCIL_POINTS; ++b) {
      if (values[a] == values[b]) {
        return false;
      }
    }
  }
  return true;
}

constexpr bool addUpToOne(const float (&values)[STENCIL_POINTS]) {
  float sum = 0.0f;
  for (float value : values) {
    sum += value;
  }
  return sum > 0.9999f && sum < 1.0001f;
}

static_assert(allDifferent(COEFFICIENTS),
              "every neighbour has its own coefficient, so a kernel that mixes up two "
              "neighbours fails the check");
static_assert(addUpToOne(COEFFICIENTS),
              "the coefficients form a weighted average, so repeated sweeps stay bounded");

__constant__ float c_c[STENCIL_POINTS];

#define IN_TILE_3D 8
#define OUT_TILE_3D (IN_TILE_3D - 2)

#define IN_TILE_2D 32
#define OUT_TILE_2D (IN_TILE_2D - 2)

__global__ void stencil_basic_kernel(const float* in, float* out, int N) {
  int i = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * N * N + j * N + k] =
        c_c[0] * in[i * N * N + j * N + k] + c_c[1] * in[i * N * N + j * N + (k - 1)] +
        c_c[2] * in[i * N * N + j * N + (k + 1)] + c_c[3] * in[i * N * N + (j - 1) * N + k] +
        c_c[4] * in[i * N * N + (j + 1) * N + k] + c_c[5] * in[(i - 1) * N * N + j * N + k] +
        c_c[6] * in[(i + 1) * N * N + j * N + k];
  }
}

__global__ void stencil_tiled_kernel(const float* in, float* out, int N) {
  int i = (int)(blockIdx.z * OUT_TILE_3D) + (int)threadIdx.z - 1;
  int j = (int)(blockIdx.y * OUT_TILE_3D) + (int)threadIdx.y - 1;
  int k = (int)(blockIdx.x * OUT_TILE_3D) + (int)threadIdx.x - 1;

  __shared__ float in_s[IN_TILE_3D][IN_TILE_3D][IN_TILE_3D];
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  }
  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    int tz = (int)threadIdx.z;
    int ty = (int)threadIdx.y;
    int tx = (int)threadIdx.x;
    if (tz >= 1 && tz < IN_TILE_3D - 1 && ty >= 1 && ty < IN_TILE_3D - 1 && tx >= 1 &&
        tx < IN_TILE_3D - 1) {
      out[i * N * N + j * N + k] = c_c[0] * in_s[tz][ty][tx] + c_c[1] * in_s[tz][ty][tx - 1] +
                                   c_c[2] * in_s[tz][ty][tx + 1] + c_c[3] * in_s[tz][ty - 1][tx] +
                                   c_c[4] * in_s[tz][ty + 1][tx] + c_c[5] * in_s[tz - 1][ty][tx] +
                                   c_c[6] * in_s[tz + 1][ty][tx];
    }
  }
}

__global__ void stencil_coarsened_kernel(const float* in, float* out, int N) {
  int iStart = (int)(blockIdx.z * OUT_TILE_2D);
  int j = (int)(blockIdx.y * OUT_TILE_2D) + (int)threadIdx.y - 1;
  int k = (int)(blockIdx.x * OUT_TILE_2D) + (int)threadIdx.x - 1;
  int ty = (int)threadIdx.y;
  int tx = (int)threadIdx.x;

  __shared__ float inPrev_s[IN_TILE_2D][IN_TILE_2D];
  __shared__ float inCurr_s[IN_TILE_2D][IN_TILE_2D];
  __shared__ float inNext_s[IN_TILE_2D][IN_TILE_2D];

  bool inPlane = j >= 0 && j < N && k >= 0 && k < N;
  if (inPlane && iStart - 1 >= 0 && iStart - 1 < N) {
    inPrev_s[ty][tx] = in[(iStart - 1) * N * N + j * N + k];
  }
  if (inPlane && iStart < N) {
    inCurr_s[ty][tx] = in[iStart * N * N + j * N + k];
  }

  for (int i = iStart; i < iStart + OUT_TILE_2D; ++i) {
    if (inPlane && i + 1 < N) {
      inNext_s[ty][tx] = in[(i + 1) * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (ty >= 1 && ty < IN_TILE_2D - 1 && tx >= 1 && tx < IN_TILE_2D - 1) {
        out[i * N * N + j * N + k] = c_c[0] * inCurr_s[ty][tx] + c_c[1] * inCurr_s[ty][tx - 1] +
                                     c_c[2] * inCurr_s[ty][tx + 1] + c_c[3] * inCurr_s[ty - 1][tx] +
                                     c_c[4] * inCurr_s[ty + 1][tx] + c_c[5] * inPrev_s[ty][tx] +
                                     c_c[6] * inNext_s[ty][tx];
      }
    }
    __syncthreads();

    inPrev_s[ty][tx] = inCurr_s[ty][tx];
    inCurr_s[ty][tx] = inNext_s[ty][tx];
  }
}

__global__ void stencil_register_tiled_kernel(const float* in, float* out, int N) {
  int iStart = (int)(blockIdx.z * OUT_TILE_2D);
  int j = (int)(blockIdx.y * OUT_TILE_2D) + (int)threadIdx.y - 1;
  int k = (int)(blockIdx.x * OUT_TILE_2D) + (int)threadIdx.x - 1;
  int ty = (int)threadIdx.y;
  int tx = (int)threadIdx.x;

  float inPrev = 0.0f;
  float inCurr = 0.0f;
  float inNext = 0.0f;
  __shared__ float inCurr_s[IN_TILE_2D][IN_TILE_2D];

  bool inPlane = j >= 0 && j < N && k >= 0 && k < N;
  if (inPlane && iStart - 1 >= 0 && iStart - 1 < N) {
    inPrev = in[(iStart - 1) * N * N + j * N + k];
  }
  if (inPlane && iStart < N) {
    inCurr = in[iStart * N * N + j * N + k];
    inCurr_s[ty][tx] = inCurr;
  }

  for (int i = iStart; i < iStart + OUT_TILE_2D; ++i) {
    if (inPlane && i + 1 < N) {
      inNext = in[(i + 1) * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (ty >= 1 && ty < IN_TILE_2D - 1 && tx >= 1 && tx < IN_TILE_2D - 1) {
        out[i * N * N + j * N + k] = c_c[0] * inCurr + c_c[1] * inCurr_s[ty][tx - 1] +
                                     c_c[2] * inCurr_s[ty][tx + 1] + c_c[3] * inCurr_s[ty - 1][tx] +
                                     c_c[4] * inCurr_s[ty + 1][tx] + c_c[5] * inPrev +
                                     c_c[6] * inNext;
      }
    }
    __syncthreads();

    inPrev = inCurr;
    inCurr = inNext;
    inCurr_s[ty][tx] = inNext;
  }
}

using Grid = std::vector<float>;

static void stencil_cpu(const Grid& in, Grid& out, int N) {
  const float* c = COEFFICIENTS;
  for (int i = 1; i < N - 1; ++i) {
    for (int j = 1; j < N - 1; ++j) {
      for (int k = 1; k < N - 1; ++k) {
        size_t p = (size_t)i * N * N + (size_t)j * N + k;
        out[p] = c[0] * in[p] + c[1] * in[p - 1] + c[2] * in[p + 1] + c[3] * in[p - N] +
                 c[4] * in[p + N] + c[5] * in[p - (size_t)N * N] + c[6] * in[p + (size_t)N * N];
      }
    }
  }
}

const int GRID_SIDE = BY_BACKEND(256, 40);
const int TIMED_LAUNCHES_AFTER_CHECK = BY_BACKEND(20, 0);
const int ITERATIVE_SWEEPS = BY_BACKEND(100, 3);
const unsigned RANDOM_SEED = 7;
const double MAX_RELATIVE_ERROR = 1e-5;

const int BASIC_BLOCK_DIM = 8;
const int BYTES_PER_FLOAT = sizeof(float);
const int STENCIL_MULTIPLICATIONS = 7;
const int STENCIL_ADDITIONS = 6;
const double FLOP_PER_POINT = STENCIL_MULTIPLICATIONS + STENCIL_ADDITIONS;
const int MARCHED_OUTPUT_PLANES = OUT_TILE_2D;
const int MARCHED_INPUT_PLANES = OUT_TILE_2D + 2;
const int COARSENED_SHARED_PLANES = 3;
const int REGISTER_TILED_SHARED_PLANES = 1;

struct DeviceGrids {
  float* in;
  float* out;
  size_t bytes;
};

struct LaunchShape {
  dim3 grid;
  dim3 block;
};

struct LaunchShapes {
  LaunchShape basic;
  LaunchShape tiled3D;
  LaunchShape marching2D;
};

static size_t pointCount() {
  return (size_t)GRID_SIDE * GRID_SIDE * GRID_SIDE;
}

static Grid randomGrid(unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> value(0.0f, 1.0f);
  Grid grid(pointCount());
  for (float& point : grid) {
    point = value(rng);
  }
  return grid;
}

static Grid sweepOnCpu(const Grid& input, int sweeps) {
  Grid current = input;
  Grid next = input;
  for (int s = 0; s < sweeps; ++s) {
    stencil_cpu(current, next, GRID_SIDE);
    std::swap(current, next);
  }
  return current;
}

struct Reference {
  Grid output;
  double ms;
};

static Reference oneSweepReference(const Grid& input) {
  Reference reference;
  reference.output = input;

  Timer timer;
  stencil_cpu(input, reference.output, GRID_SIDE);
  reference.ms = timer.ms();
  return reference;
}

static double maxRelativeError(const Grid& expected, const Grid& actual) {
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double difference = std::fabs((double)expected[i] - actual[i]);
    worst = std::fmax(worst, difference / (1.0 + std::fabs((double)expected[i])));
  }
  return worst;
}

static LaunchShape cubeOfBlocks(int blockDim, int outputsPerBlock) {
  const unsigned blocksPerSide = cdiv(GRID_SIDE, outputsPerBlock);
  return {dim3(blocksPerSide, blocksPerSide, blocksPerSide), dim3(blockDim, blockDim, blockDim)};
}

static LaunchShapes launchShapes() {
  LaunchShapes shapes;
  shapes.basic = cubeOfBlocks(BASIC_BLOCK_DIM, BASIC_BLOCK_DIM);
  shapes.tiled3D = cubeOfBlocks(IN_TILE_3D, OUT_TILE_3D);

  const unsigned blocksPerSide = cdiv(GRID_SIDE, OUT_TILE_2D);
  shapes.marching2D.grid = dim3(blocksPerSide, blocksPerSide, blocksPerSide);
  shapes.marching2D.block = dim3(IN_TILE_2D, IN_TILE_2D, 1);
  return shapes;
}

static DeviceGrids uploadToDevice(const Grid& input) {
  DeviceGrids device;
  device.bytes = input.size() * sizeof(float);
  CHECK(cudaMemcpyToSymbol(c_c, COEFFICIENTS, sizeof(COEFFICIENTS)));
  CHECK(cudaMalloc(&device.in, device.bytes));
  CHECK(cudaMalloc(&device.out, device.bytes));
  CHECK(cudaMemcpy(device.in, input.data(), device.bytes, cudaMemcpyHostToDevice));
  return device;
}

static void freeDevice(DeviceGrids& device) {
  CHECK(cudaFree(device.in));
  CHECK(cudaFree(device.out));
}

class KernelTable {
 public:
  KernelTable(const DeviceGrids& device, const Grid& expected)
      : device_(device), expected_(expected), actual_(expected.size()) {
    const double interiorSide = GRID_SIDE - 2;
    flop_ = FLOP_PER_POINT * interiorSide * interiorSide * interiorSide;
    bytesToReadAndWriteOnce_ = 2.0 * device.bytes;
  }

  void printHeader(double cpuMs) const {
    std::printf(
        "\n%-32s %10s %10s %10s  %s\n", "kernel (one sweep)", "ms", "GFLOP/s", "GB/s*", "check");
    std::printf("%-32s %10.3f %10.2f %10s  %s\n",
                "CPU reference (1 thread)",
                cpuMs,
                gflopPerSecond(cpuMs),
                "-",
                "-");
  }

  template <class LaunchKernel>
  void run(const char* name, LaunchKernel launchKernel) {
    startOutputAsCopyOfInput();
    Timer timer;
    launchKernel();
    CHECK(cudaGetLastError());
    double ms = timer.ms();

    CHECK(cudaMemcpy(actual_.data(), device_.out, device_.bytes, cudaMemcpyDeviceToHost));
    const double error = maxRelativeError(expected_, actual_);
    const bool passed = error < MAX_RELATIVE_ERROR;
    allPassed_ = allPassed_ && passed;

    if (TIMED_LAUNCHES_AFTER_CHECK > 0) {
      ms = averageMsOfRepeatedLaunches(launchKernel);
    }
    printRow(name, ms, passed, error);
  }

  bool allPassed() const {
    return allPassed_;
  }

 private:
  void startOutputAsCopyOfInput() {
    CHECK(cudaMemcpy(device_.out, device_.in, device_.bytes, cudaMemcpyDeviceToDevice));
  }

  template <class LaunchKernel>
  static double averageMsOfRepeatedLaunches(LaunchKernel launchKernel) {
    Timer timer;
    for (int r = 0; r < TIMED_LAUNCHES_AFTER_CHECK; ++r) {
      launchKernel();
    }
    return timer.ms() / TIMED_LAUNCHES_AFTER_CHECK;
  }

  double gflopPerSecond(double ms) const {
    return flop_ / (ms * 1e6);
  }

  double gigabytesPerSecond(double ms) const {
    return bytesToReadAndWriteOnce_ / (ms * 1e6);
  }

  void printRow(const char* name, double ms, bool passed, double error) const {
    std::printf("%-32s %10.3f %10.2f %10.2f  %s (max rel err %.1e)\n",
                name,
                ms,
                gflopPerSecond(ms),
                gigabytesPerSecond(ms),
                passed ? "PASS" : "FAIL",
                error);
  }

  const DeviceGrids& device_;
  const Grid& expected_;
  Grid actual_;
  double flop_;
  double bytesToReadAndWriteOnce_;
  bool allPassed_ = true;
};

static void runEveryKernel(KernelTable& table,
                           const DeviceGrids& device,
                           const LaunchShapes& shapes) {
  const float* in = device.in;
  float* out = device.out;
  const int N = GRID_SIDE;

  table.run("(a) basic", [&] {
    launch(stencil_basic_kernel, shapes.basic.grid, shapes.basic.block, in, out, N);
  });
  table.run("(b) shared-memory tiles 8^3", [&] {
    launch(stencil_tiled_kernel, shapes.tiled3D.grid, shapes.tiled3D.block, in, out, N);
  });
  table.run("(c) coarsened in z, 32^2", [&] {
    launch(stencil_coarsened_kernel, shapes.marching2D.grid, shapes.marching2D.block, in, out, N);
  });
  table.run("(d) register tiling, 32^2", [&] {
    launch(
        stencil_register_tiled_kernel, shapes.marching2D.grid, shapes.marching2D.block, in, out, N);
  });

  std::printf("* GB/s counts only the minimum traffic (read the grid once, write it once).\n");
  std::printf("  In CPU emulation the times measure thread scheduling, not memory behaviour.\n");
}

static bool runIterativeSweeps(const Grid& input,
                               const DeviceGrids& device,
                               const LaunchShapes& shapes) {
  float* src = device.in;
  float* dst = device.out;
  CHECK(cudaMemcpy(src, input.data(), device.bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dst, src, device.bytes, cudaMemcpyDeviceToDevice));

  Timer timer;
  for (int s = 0; s < ITERATIVE_SWEEPS; ++s) {
    launch(stencil_register_tiled_kernel,
           shapes.marching2D.grid,
           shapes.marching2D.block,
           (const float*)src,
           dst,
           GRID_SIDE);
    std::swap(src, dst);
  }
  CHECK(cudaGetLastError());
  const double ms = timer.ms();

  Grid result(input.size());
  CHECK(cudaMemcpy(result.data(), src, device.bytes, cudaMemcpyDeviceToHost));
  const Grid expected = sweepOnCpu(input, ITERATIVE_SWEEPS);
  const double error = maxRelativeError(expected, result);
  const bool passed = error < MAX_RELATIVE_ERROR;

  std::printf(
      "\n%d sweeps of (d) with double buffering: %.3f ms (%.3f ms/sweep)  %s (max rel err %.1e)\n",
      ITERATIVE_SWEEPS,
      ms,
      ms / ITERATIVE_SWEEPS,
      passed ? "PASS" : "FAIL",
      error);
  return passed;
}

static double cube(int side) {
  return (double)side * side * side;
}

static double square(int side) {
  return (double)side * side;
}

static void printResourceRow(
    const char* name, int threads, int sharedBytes, double outputs, double loads) {
  const double intensity = FLOP_PER_POINT * outputs / (BYTES_PER_FLOAT * (loads + outputs));
  std::printf("  %-34s %8d %10d %12.0f %10.2f\n", name, threads, sharedBytes, outputs, intensity);
}

static void printResourceTable() {
  std::printf(
      "\nPer-block resources and arithmetic intensity (13 FLOP per output, 4 B per load and per "
      "store):\n");
  std::printf("  %-34s %8s %10s %12s %10s\n", "kernel", "threads", "shared B", "outputs", "FLOP/B");

  const int basicThreads = BASIC_BLOCK_DIM * BASIC_BLOCK_DIM * BASIC_BLOCK_DIM;
  printResourceRow("(a) basic (7 loads per output)", basicThreads, 0, 1, STENCIL_POINTS);

  for (int T : {8, 10}) {
    char name[64];
    std::snprintf(name, sizeof name, "(b) 3D tile T=%d", T);
    printResourceRow(name, (int)cube(T), BYTES_PER_FLOAT * (int)cube(T), cube(T - 2), cube(T));
  }

  const int T = IN_TILE_2D;
  const int planeThreads = T * T;
  const int planeBytes = BYTES_PER_FLOAT * T * T;
  const double outputsPerPlane = square(T - 2);
  const double loadsPerPlane = square(T);
  printResourceRow("(c) coarsened T=32, 30 planes",
                   planeThreads,
                   COARSENED_SHARED_PLANES * planeBytes,
                   MARCHED_OUTPUT_PLANES * outputsPerPlane,
                   MARCHED_INPUT_PLANES * loadsPerPlane);
  printResourceRow("(c) same, one plane (steady state)",
                   planeThreads,
                   COARSENED_SHARED_PLANES * planeBytes,
                   outputsPerPlane,
                   loadsPerPlane);
  printResourceRow("(d) register tiled T=32, 30 planes",
                   planeThreads,
                   REGISTER_TILED_SHARED_PLANES * planeBytes,
                   MARCHED_OUTPUT_PLANES * outputsPerPlane,
                   MARCHED_INPUT_PLANES * loadsPerPlane);
  std::printf("  limit for very large tiles: 13/8 = 1.625 FLOP/B (H100 ridge: 20.0)\n");
}

int main() {
  const Grid input = randomGrid(RANDOM_SEED);
  const double megabytesPerArray = input.size() * sizeof(float) / 1e6;
  std::printf("3D seven-point stencil, %d^3 grid (%.1f MB per array), backend: %s\n",
              GRID_SIDE,
              megabytesPerArray,
              CUDA_BACKEND);

  const Reference reference = oneSweepReference(input);

  DeviceGrids device = uploadToDevice(input);
  const LaunchShapes shapes = launchShapes();

  KernelTable table(device, reference.output);
  table.printHeader(reference.ms);
  runEveryKernel(table, device, shapes);

  const bool iterativePassed = runIterativeSweeps(input, device, shapes);
  printResourceTable();

  freeDevice(device);
  const bool allPassed = table.allPassed() && iterativePassed;
  std::printf("\n%s\n", allPassed ? "ALL PASS" : "SOME KERNEL FAILED");
  return allPassed ? 0 : 1;
}
