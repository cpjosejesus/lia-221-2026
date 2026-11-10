#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define FILTER_RADIUS 2
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

#define TILE_DIM 32

__constant__ float F_c[FILTER_DIM][FILTER_DIM];

struct Filter {
  float w[FILTER_DIM][FILTER_DIM];
};

__global__ void convolution_2D_basic_kernel(
    const float* N, const float* F, float* P, int r, int width, int height) {
  int outCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int outRow = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (outRow < height && outCol < width) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[outRow * width + outCol] = Pvalue;
  }
}

__global__ void convolution_2D_struct_kernel(
    const float* N, Filter f, float* P, int width, int height) {
  int outCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int outRow = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (outRow < height && outCol < width) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
      for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
        int inRow = outRow - FILTER_RADIUS + fRow;
        int inCol = outCol - FILTER_RADIUS + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += f.w[fRow][fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[outRow * width + outCol] = Pvalue;
  }
}

__global__ void convolution_2D_const_mem_kernel(
    const float* N, float* P, int r, int width, int height) {
  int outCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int outRow = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (outRow < height && outCol < width) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[outRow * width + outCol] = Pvalue;
  }
}

__global__ void convolution_tiled_2D_const_mem_kernel(const float* N,
                                                      float* P,
                                                      int width,
                                                      int height) {
  int col = (int)(blockIdx.x * OUT_TILE_DIM) + (int)threadIdx.x - FILTER_RADIUS;
  int row = (int)(blockIdx.y * OUT_TILE_DIM) + (int)threadIdx.y - FILTER_RADIUS;

  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int tileCol = (int)threadIdx.x - FILTER_RADIUS;
  int tileRow = (int)threadIdx.y - FILTER_RADIUS;
  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
      float Pvalue = 0.0f;
      for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
          Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
        }
      }
      P[row * width + col] = Pvalue;
    }
  }
}

__global__ void convolution_cached_tiled_2D_const_mem_kernel(const float* N,
                                                             float* P,
                                                             int width,
                                                             int height) {
  int col = (int)(blockIdx.x * TILE_DIM + threadIdx.x);
  int row = (int)(blockIdx.y * TILE_DIM + threadIdx.y);

  __shared__ float N_s[TILE_DIM][TILE_DIM];
  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  if (col < width && row < height) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
      for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
        int sharedCol = (int)threadIdx.x - FILTER_RADIUS + fCol;
        int sharedRow = (int)threadIdx.y - FILTER_RADIUS + fRow;
        bool insideTile =
            sharedCol >= 0 && sharedCol < TILE_DIM && sharedRow >= 0 && sharedRow < TILE_DIM;

        if (insideTile) {
          Pvalue += F_c[fRow][fCol] * N_s[sharedRow][sharedCol];
        } else {
          int inRow = row - FILTER_RADIUS + fRow;
          int inCol = col - FILTER_RADIUS + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
          }
        }
      }
    }
    P[row * width + col] = Pvalue;
  }
}

const int IMAGE_WIDTH = BY_BACKEND(4096, 500);
const int IMAGE_HEIGHT = BY_BACKEND(4096, 500);
const int TIMED_LAUNCHES_AFTER_CHECK = BY_BACKEND(20, 0);
const int BASIC_BLOCK_DIM = 16;
const unsigned RANDOM_SEED = 42;
const double MAX_RELATIVE_ERROR = 1e-4;

const int FLOP_PER_TAP = 2;
const int BYTES_PER_FLOAT = sizeof(float);
const double H100_FP32_FLOP_PER_SECOND = 66.9e12;
const double H100_DRAM_BYTES_PER_SECOND = 3.35e12;
const double H100_RIDGE_FLOP_PER_BYTE = H100_FP32_FLOP_PER_SECOND / H100_DRAM_BYTES_PER_SECOND;

const int TABLE_OUTPUT_TILES[] = {8, 16, 28, 32, 64};
const int TABLE_FILTER_DIMS[] = {3, 5, 7, 9, 11, 13};

using Image = std::vector<float>;
using FilterWeights = std::vector<float>;

struct Problem {
  int width;
  int height;
  Image image;
  FilterWeights weights;
  Filter filter;
};

struct DeviceBuffers {
  float* image;
  float* weights;
  float* output;
  size_t imageBytes;
};

struct Reference {
  Image output;
  double ms;
};

static size_t pixelCount(const Problem& problem) {
  return (size_t)problem.width * problem.height;
}

static Image randomImage(int width, int height, std::mt19937& rng) {
  std::uniform_real_distribution<float> pixelValue(0.0f, 1.0f);
  Image image((size_t)width * height);
  for (float& pixel : image) {
    pixel = pixelValue(rng);
  }
  return image;
}

static FilterWeights randomWeights(std::mt19937& rng) {
  std::uniform_real_distribution<float> weightValue(-1.0f, 1.0f);
  FilterWeights weights(FILTER_DIM * FILTER_DIM);
  for (float& weight : weights) {
    weight = weightValue(rng);
  }
  return weights;
}

static Filter asFilterStruct(const FilterWeights& weights) {
  Filter filter;
  for (int row = 0; row < FILTER_DIM; ++row) {
    for (int col = 0; col < FILTER_DIM; ++col) {
      filter.w[row][col] = weights[row * FILTER_DIM + col];
    }
  }
  return filter;
}

static Problem makeProblem() {
  std::mt19937 rng(RANDOM_SEED);

  Problem problem;
  problem.width = IMAGE_WIDTH;
  problem.height = IMAGE_HEIGHT;
  problem.image = randomImage(problem.width, problem.height, rng);
  problem.weights = randomWeights(rng);
  problem.filter = asFilterStruct(problem.weights);
  return problem;
}

static void convolution_cpu(const std::vector<float>& N,
                            const std::vector<float>& F,
                            int r,
                            std::vector<float>& P,
                            int width,
                            int height) {
  const int m = 2 * r + 1;
  for (int outRow = 0; outRow < height; ++outRow) {
    for (int outCol = 0; outCol < width; ++outCol) {
      float Pvalue = 0.0f;
      for (int fRow = 0; fRow < m; ++fRow) {
        for (int fCol = 0; fCol < m; ++fCol) {
          int inRow = outRow - r + fRow;
          int inCol = outCol - r + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            Pvalue += F[fRow * m + fCol] * N[inRow * width + inCol];
          }
        }
      }
      P[outRow * width + outCol] = Pvalue;
    }
  }
}

static Reference computeReference(const Problem& problem) {
  Reference reference;
  reference.output.resize(pixelCount(problem));

  Timer timer;
  convolution_cpu(problem.image,
                  problem.weights,
                  FILTER_RADIUS,
                  reference.output,
                  problem.width,
                  problem.height);
  reference.ms = timer.ms();
  return reference;
}

static double maxRelativeError(const Image& expected, const Image& actual) {
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double difference = std::fabs((double)expected[i] - (double)actual[i]);
    double error = difference / (1.0 + std::fabs((double)expected[i]));
    if (error > worst) {
      worst = error;
    }
  }
  return worst;
}

static DeviceBuffers uploadToDevice(const Problem& problem) {
  DeviceBuffers device;
  device.imageBytes = pixelCount(problem) * sizeof(float);
  const size_t weightBytes = problem.weights.size() * sizeof(float);

  CHECK(cudaMalloc(&device.image, device.imageBytes));
  CHECK(cudaMalloc(&device.weights, weightBytes));
  CHECK(cudaMalloc(&device.output, device.imageBytes));

  CHECK(cudaMemcpy(device.image, problem.image.data(), device.imageBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(device.weights, problem.weights.data(), weightBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(F_c, problem.weights.data(), weightBytes));
  return device;
}

static void freeDevice(DeviceBuffers& device) {
  CHECK(cudaFree(device.image));
  CHECK(cudaFree(device.weights));
  CHECK(cudaFree(device.output));
}

class KernelTable {
 public:
  KernelTable(const DeviceBuffers& device, const Problem& problem, const Image& expected)
      : device_(device), expected_(expected), actual_(expected.size()) {
    const double filterTaps = FILTER_DIM * FILTER_DIM;
    flop_ = FLOP_PER_TAP * filterTaps * (double)problem.width * problem.height;
    bytesToReadAndWriteOnce_ = 2.0 * device.imageBytes;
  }

  void printHeader(double cpuMs) const {
    std::printf("\n%-34s %10s %10s %10s  %s\n", "kernel", "ms", "GFLOP/s", "GB/s*", "check");
    std::printf("%-34s %10.3f %10.2f %10s  %s\n",
                "CPU reference (1 thread)",
                cpuMs,
                gflopPerSecond(cpuMs),
                "-",
                "-");
  }

  template <class LaunchKernel>
  void run(const char* name, LaunchKernel launchKernel) {
    CHECK(cudaMemset(device_.output, 0, device_.imageBytes));
    Timer timer;
    launchKernel();
    CHECK(cudaGetLastError());
    double ms = timer.ms();

    CHECK(cudaMemcpy(actual_.data(), device_.output, device_.imageBytes, cudaMemcpyDeviceToHost));
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
  template <class LaunchKernel>
  static double averageMsOfRepeatedLaunches(LaunchKernel launchKernel) {
    Timer timer;
    for (int i = 0; i < TIMED_LAUNCHES_AFTER_CHECK; ++i) {
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
    std::printf("%-34s %10.3f %10.2f %10.2f  %s (max rel err %.1e)\n",
                name,
                ms,
                gflopPerSecond(ms),
                gigabytesPerSecond(ms),
                passed ? "PASS" : "FAIL",
                error);
  }

  const DeviceBuffers& device_;
  const Image& expected_;
  Image actual_;
  double flop_;
  double bytesToReadAndWriteOnce_;
  bool allPassed_ = true;
};

static void runEveryKernel(KernelTable& table,
                           const DeviceBuffers& device,
                           const Problem& problem) {
  const float* N = device.image;
  const float* F = device.weights;
  float* P = device.output;
  const int width = problem.width;
  const int height = problem.height;

  const dim3 basicBlock(BASIC_BLOCK_DIM, BASIC_BLOCK_DIM);
  const dim3 basicGrid(cdiv(width, BASIC_BLOCK_DIM), cdiv(height, BASIC_BLOCK_DIM));
  const dim3 inputTileBlock(IN_TILE_DIM, IN_TILE_DIM);
  const dim3 outputTileGrid(cdiv(width, OUT_TILE_DIM), cdiv(height, OUT_TILE_DIM));
  const dim3 cachedTileBlock(TILE_DIM, TILE_DIM);
  const dim3 cachedTileGrid(cdiv(width, TILE_DIM), cdiv(height, TILE_DIM));

  table.run("(a)  basic, filter in global", [&] {
    launch(
        convolution_2D_basic_kernel, basicGrid, basicBlock, N, F, P, FILTER_RADIUS, width, height);
  });
  table.run("(a') basic, filter as struct arg", [&] {
    launch(
        convolution_2D_struct_kernel, basicGrid, basicBlock, N, problem.filter, P, width, height);
  });
  table.run("(b)  constant-memory filter", [&] {
    launch(
        convolution_2D_const_mem_kernel, basicGrid, basicBlock, N, P, FILTER_RADIUS, width, height);
  });
  table.run("(c)  tiled, block = input tile", [&] {
    launch(
        convolution_tiled_2D_const_mem_kernel, outputTileGrid, inputTileBlock, N, P, width, height);
  });
  table.run("(d)  tiled, cached halo", [&] {
    launch(convolution_cached_tiled_2D_const_mem_kernel,
           cachedTileGrid,
           cachedTileBlock,
           N,
           P,
           width,
           height);
  });
}

static void printTimingFootnote() {
  std::printf("* GB/s counts only the minimum traffic (read the image once, write it once).\n");
  std::printf("  In CPU emulation the times measure thread scheduling, not memory behaviour.\n");
}

static void printInputTileUsage() {
  const int blockThreads = IN_TILE_DIM * IN_TILE_DIM;
  const int outputsPerTile = OUT_TILE_DIM * OUT_TILE_DIM;
  const double computingPercent = 100.0 * outputsPerTile / blockThreads;

  std::printf("\nKernel (c): block %dx%d = %d threads, output tile %dx%d = %d (%.1f%% compute)\n",
              IN_TILE_DIM,
              IN_TILE_DIM,
              blockThreads,
              OUT_TILE_DIM,
              OUT_TILE_DIM,
              outputsPerTile,
              computingPercent);
}

static double tiledKernelIntensity(int outputTile, int filterDim) {
  const int inputTile = outputTile + filterDim - 1;
  const double flop = (double)FLOP_PER_TAP * outputTile * outputTile * filterDim * filterDim;
  const double loadedElements = (double)inputTile * inputTile;
  const double storedElements = (double)outputTile * outputTile;
  return flop / (BYTES_PER_FLOAT * (loadedElements + storedElements));
}

static double idealIntensity(int filterDim) {
  return filterDim * filterDim / 4.0;
}

static char boundOnH100(double intensity) {
  const char COMPUTE_BOUND = 'C';
  const char MEMORY_BOUND = 'M';
  return intensity >= H100_RIDGE_FLOP_PER_BYTE ? COMPUTE_BOUND : MEMORY_BOUND;
}

static void printIntensityCell(double intensity) {
  std::printf("  %6.2f%c ", intensity, boundOnH100(intensity));
}

static void printIntensityTableHeader() {
  std::printf(
      "\nArithmetic intensity of the tiled kernel (FLOP/B), output tile t x t, filter m x m\n");
  std::printf("loads of the input tile + stores of the output tile; H100 ridge = %.1f FLOP/B\n",
              H100_RIDGE_FLOP_PER_BYTE);
  std::printf("(M = memory-bound, C = compute-bound on H100)\n");

  std::printf("  t \\ m ");
  for (int filterDim : TABLE_FILTER_DIMS) {
    char label[16];
    std::snprintf(label, sizeof label, "%dx%d", filterDim, filterDim);
    std::printf("%9s ", label);
  }
  std::printf("\n");
}

static void printIntensityTable() {
  printIntensityTableHeader();

  for (int outputTile : TABLE_OUTPUT_TILES) {
    std::printf("  %5d ", outputTile);
    for (int filterDim : TABLE_FILTER_DIMS) {
      printIntensityCell(tiledKernelIntensity(outputTile, filterDim));
    }
    std::printf("\n");
  }

  std::printf("  ideal ");
  for (int filterDim : TABLE_FILTER_DIMS) {
    printIntensityCell(idealIntensity(filterDim));
  }
  std::printf("   (m^2/4: every input read once, every output written once)\n");
}

int main() {
  std::printf("2D convolution, %d x %d image, %dx%d filter, backend: %s\n",
              IMAGE_WIDTH,
              IMAGE_HEIGHT,
              FILTER_DIM,
              FILTER_DIM,
              CUDA_BACKEND);

  const Problem problem = makeProblem();
  const Reference reference = computeReference(problem);
  DeviceBuffers device = uploadToDevice(problem);

  KernelTable table(device, problem, reference.output);
  table.printHeader(reference.ms);
  runEveryKernel(table, device, problem);
  printTimingFootnote();

  printInputTileUsage();
  printIntensityTable();

  freeDevice(device);
  std::printf("\n%s\n", table.allPassed() ? "ALL PASS" : "SOME KERNEL FAILED");
  return table.allPassed() ? 0 : 1;
}
