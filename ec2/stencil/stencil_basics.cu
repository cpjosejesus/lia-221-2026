#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

__global__ void finite_difference_kernel(const double* F, double* FD, int n, double h) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= 1 && i < n - 1) {
    FD[i] = (-1.0 / (2.0 * h)) * F[i - 1] + (1.0 / (2.0 * h)) * F[i + 1];
  }
}

const int CHAPTER_GRID_POINTS = 7;
const int FINEST_GRID_POINTS = CHAPTER_GRID_POINTS * 512;
const int FINITE_DIFFERENCE_BLOCK_DIM = 128;
const double MIN_ERROR_RATIO_FOR_SECOND_ORDER = 3.5;
const double MAX_ERROR_RATIO_FOR_SECOND_ORDER = 4.5;

using Samples = std::vector<double>;

static int pointsWithHalfTheSpacing(int points) {
  return 2 * points - 1;
}

static Samples sampleSine(int n, double h) {
  Samples F(n);
  for (int i = 0; i < n; ++i) {
    F[i] = std::sin(i * h);
  }
  return F;
}

static Samples finiteDifferenceOnDevice(const Samples& F, double h) {
  const int n = (int)F.size();
  const size_t bytes = n * sizeof(double);
  Samples FD(n, 0.0);

  double* dF;
  double* dFD;
  CHECK(cudaMalloc(&dF, bytes));
  CHECK(cudaMalloc(&dFD, bytes));
  CHECK(cudaMemcpy(dF, F.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dFD, 0, bytes));

  launch(finite_difference_kernel,
         dim3(cdiv(n, FINITE_DIFFERENCE_BLOCK_DIM)),
         dim3(FINITE_DIFFERENCE_BLOCK_DIM),
         static_cast<const double*>(dF),
         dFD,
         n,
         h);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(FD.data(), dFD, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dF));
  CHECK(cudaFree(dFD));
  return FD;
}

static void printChapterGrid(const Samples& F, const Samples& FD, double h) {
  const int n = (int)F.size();

  std::printf("   F  = ");
  for (int i = 0; i < n; ++i) {
    std::printf("%5.2f ", F[i]);
  }

  std::printf("\n   FD = %5s ", "-");
  for (int i = 1; i < n - 1; ++i) {
    std::printf("%5.2f ", FD[i]);
  }
  std::printf("%5s   (interior points only)\n", "-");

  std::printf("   cos= %5s ", "-");
  for (int i = 1; i < n - 1; ++i) {
    std::printf("%5.2f ", std::cos(i * h));
  }
  std::printf("%5s\n", "-");
}

static double maxErrorAgainstCosine(const Samples& FD, double h) {
  const int n = (int)FD.size();
  double error = 0.0;
  for (int i = 1; i < n - 1; ++i) {
    error = std::fmax(error, std::fabs(FD[i] - std::cos(i * h)));
  }
  return error;
}

static bool isSecondOrderRatio(double ratio) {
  return ratio > MIN_ERROR_RATIO_FOR_SECOND_ORDER && ratio < MAX_ERROR_RATIO_FOR_SECOND_ORDER;
}

static bool runFiniteDifference() {
  const double PI = std::acos(-1.0);
  std::printf("Part 1: derivative of sin(x) on [0, pi] with the 3-point stencil\n");
  std::printf("%8s %12s %16s %12s\n", "points", "h", "max |FD - cos|", "error ratio");

  double previousError = 0.0;
  bool passed = true;
  for (int n = CHAPTER_GRID_POINTS; n <= FINEST_GRID_POINTS; n = pointsWithHalfTheSpacing(n)) {
    const double h = PI / (n - 1);
    const Samples F = sampleSine(n, h);
    const Samples FD = finiteDifferenceOnDevice(F, h);

    if (n == CHAPTER_GRID_POINTS) {
      printChapterGrid(F, FD, h);
    }

    const double error = maxErrorAgainstCosine(FD, h);
    const bool hasPreviousRow = previousError > 0.0;
    if (hasPreviousRow) {
      const double ratio = previousError / error;
      std::printf("%8d %12.6f %16.3e %12.2f\n", n, h, error, ratio);
      passed = passed && isSecondOrderRatio(ratio);
    } else {
      std::printf("%8d %12.6f %16.3e %12s\n", n, h, error, "-");
    }
    previousError = error;
  }

  std::printf("Error ratio ~4 each time h halves: the error is O(h^2).  %s\n\n",
              passed ? "PASS" : "FAIL");
  return passed;
}

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

__global__ void stencil2d_basic(const float* in, float* out, int N) {
  int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
    out[i * N + j] =
        0.25f * (in[(i - 1) * N + j] + in[(i + 1) * N + j] + in[i * N + j - 1] + in[i * N + j + 1]);
  }
}

__global__ void stencil2d_tiled(const float* in, float* out, int N) {
  int i = (int)(blockIdx.y * OUT_TILE_DIM + threadIdx.y) - 1;
  int j = (int)(blockIdx.x * OUT_TILE_DIM + threadIdx.x) - 1;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];
  if (i >= 0 && i < N && j >= 0 && j < N) {
    in_s[threadIdx.y][threadIdx.x] = in[i * N + j];
  }
  __syncthreads();

  int ty = (int)threadIdx.y;
  int tx = (int)threadIdx.x;
  bool interiorPoint = i >= 1 && i < N - 1 && j >= 1 && j < N - 1;
  bool innerThread = ty >= 1 && ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1;
  if (interiorPoint && innerThread) {
    out[i * N + j] =
        0.25f * (in_s[ty - 1][tx] + in_s[ty + 1][tx] + in_s[ty][tx - 1] + in_s[ty][tx + 1]);
  }
}

static void stencil2d_cpu(const std::vector<float>& in, std::vector<float>& out, int N) {
  for (int i = 1; i < N - 1; ++i) {
    for (int j = 1; j < N - 1; ++j) {
      out[i * N + j] = 0.25f * (in[(i - 1) * N + j] + in[(i + 1) * N + j] + in[i * N + j - 1] +
                                in[i * N + j + 1]);
    }
  }
}

const int PLATE_SIDE = BY_BACKEND(512, 40);
const int SWEEPS = BY_BACKEND(20000, 150);
const float HOT_EDGE_TEMPERATURE = 100.0f;
const int BASIC_BLOCK_DIM = 16;
const double MAX_DIFFERENCE_FROM_CPU = 1e-3;

const int BYTES_PER_FLOAT = sizeof(float);
const int FIVE_POINT_MULTIPLICATIONS = 5;
const int FIVE_POINT_ADDITIONS = 4;
const double FIVE_POINT_FLOP = FIVE_POINT_MULTIPLICATIONS + FIVE_POINT_ADDITIONS;
const int FIVE_POINT_LOADS = 5;
const int SMALLEST_CONVOLUTION_DIM = 3;

const char* const TEMPERATURE_SHADES = " .:-=+*#%@";
const int DARKEST_SHADE = 9;
const int PICTURE_ROWS = 8;
const int PICTURE_COLUMNS = 32;

using Plate = std::vector<float>;

enum class PlateKernel { Basic, Tiled };

const PlateKernel PLATE_KERNELS[] = {PlateKernel::Basic, PlateKernel::Tiled};

static const char* nameOf(PlateKernel kernel) {
  return kernel == PlateKernel::Basic ? "basic" : "tiled";
}

static Plate plateWithHotTopEdge(int N) {
  Plate plate((size_t)N * N, 0.0f);
  for (int j = 0; j < N; ++j) {
    plate[j] = HOT_EDGE_TEMPERATURE;
  }
  return plate;
}

static Plate sweepOnCpu(const Plate& plate, int N, int sweeps) {
  Plate current = plate;
  Plate next = plate;
  for (int s = 0; s < sweeps; ++s) {
    stencil2d_cpu(current, next, N);
    std::swap(current, next);
  }
  return current;
}

static void launchSweep(PlateKernel kernel, const float* in, float* out, int N) {
  if (kernel == PlateKernel::Basic) {
    launch(stencil2d_basic,
           dim3(cdiv(N, BASIC_BLOCK_DIM), cdiv(N, BASIC_BLOCK_DIM)),
           dim3(BASIC_BLOCK_DIM, BASIC_BLOCK_DIM),
           in,
           out,
           N);
  } else {
    launch(stencil2d_tiled,
           dim3(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM)),
           dim3(IN_TILE_DIM, IN_TILE_DIM),
           in,
           out,
           N);
  }
}

static double maxAbsoluteDifference(const Plate& a, const Plate& b) {
  double worst = 0.0;
  for (size_t p = 0; p < a.size(); ++p) {
    worst = std::fmax(worst, std::fabs(a[p] - b[p]));
  }
  return worst;
}

static bool checkKernelOnPlate(
    PlateKernel kernel, const Plate& plate, const Plate& expected, float* bufferA, float* bufferB) {
  const int N = PLATE_SIDE;
  const size_t bytes = plate.size() * sizeof(float);
  CHECK(cudaMemcpy(bufferA, plate.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(bufferB, plate.data(), bytes, cudaMemcpyHostToDevice));

  float* src = bufferA;
  float* dst = bufferB;
  Timer timer;
  for (int s = 0; s < SWEEPS; ++s) {
    launchSweep(kernel, src, dst, N);
    std::swap(src, dst);
  }
  const double ms = timer.ms();
  CHECK(cudaGetLastError());

  Plate result(plate.size());
  CHECK(cudaMemcpy(result.data(), src, bytes, cudaMemcpyDeviceToHost));
  const double difference = maxAbsoluteDifference(result, expected);
  const bool passed = difference < MAX_DIFFERENCE_FROM_CPU;

  std::printf("  %-6s %9.2f ms for %d sweeps, max diff vs CPU %.2e  %s\n",
              nameOf(kernel),
              ms,
              SWEEPS,
              difference,
              passed ? "PASS" : "FAIL");
  return passed;
}

static double idealIntensity(double flopPerPoint) {
  const double bytesPerPoint = 2.0 * BYTES_PER_FLOAT;
  return flopPerPoint / bytesPerPoint;
}

static void printPlateIntensities() {
  const double basicBytes = FIVE_POINT_LOADS * BYTES_PER_FLOAT + BYTES_PER_FLOAT;
  const double basic = FIVE_POINT_FLOP / basicBytes;

  const double inputTile = IN_TILE_DIM;
  const double outputTile = OUT_TILE_DIM;
  const double tileBytes =
      BYTES_PER_FLOAT * inputTile * inputTile + BYTES_PER_FLOAT * outputTile * outputTile;
  const double tiled = FIVE_POINT_FLOP * outputTile * outputTile / tileBytes;

  const double convolutionFlop = 2.0 * SMALLEST_CONVOLUTION_DIM * SMALLEST_CONVOLUTION_DIM;

  std::printf(
      "  intensity (general 5-point, 9 FLOP/point): basic %.3f, tiled %.3f, ideal %.3f FLOP/B\n",
      basic,
      tiled,
      idealIntensity(FIVE_POINT_FLOP));
  std::printf(
      "  (a 3x3 convolution: ideal %.2f FLOP/B; the 5-point stencil reuses each input far less)\n",
      idealIntensity(convolutionFlop));
}

static char shadeOf(float temperature) {
  const int shade = static_cast<int>(temperature / HOT_EDGE_TEMPERATURE * DARKEST_SHADE + 0.5f);
  if (shade < 0) {
    return TEMPERATURE_SHADES[0];
  }
  if (shade > DARKEST_SHADE) {
    return TEMPERATURE_SHADES[DARKEST_SHADE];
  }
  return TEMPERATURE_SHADES[shade];
}

static void printTemperaturePicture(const Plate& plate, int N) {
  std::printf("  temperature (top edge hot):\n");
  for (int r = 0; r < PICTURE_ROWS; ++r) {
    const size_t plateRow = static_cast<size_t>(r * (N - 1) / (PICTURE_ROWS - 1));
    std::printf("    ");
    for (int c = 0; c < PICTURE_COLUMNS; ++c) {
      const int plateCol = c * (N - 1) / (PICTURE_COLUMNS - 1);
      std::putchar(shadeOf(plate[plateRow * N + plateCol]));
    }
    std::putchar('\n');
  }
}

static bool runHeatPlate() {
  const int N = PLATE_SIDE;
  std::printf("Part 2: 2D 5-point stencil, %dx%d plate, top edge 100, other edges 0, %d sweeps\n",
              N,
              N,
              SWEEPS);

  const Plate plate = plateWithHotTopEdge(N);
  const Plate expected = sweepOnCpu(plate, N, SWEEPS);

  const size_t bytes = plate.size() * sizeof(float);
  float* bufferA;
  float* bufferB;
  CHECK(cudaMalloc(&bufferA, bytes));
  CHECK(cudaMalloc(&bufferB, bytes));

  bool passed = true;
  for (PlateKernel kernel : PLATE_KERNELS) {
    passed = checkKernelOnPlate(kernel, plate, expected, bufferA, bufferB) && passed;
  }

  CHECK(cudaFree(bufferA));
  CHECK(cudaFree(bufferB));

  printPlateIntensities();
  printTemperaturePicture(expected, N);
  return passed;
}

int main() {
  std::printf("Stencil basics (chapter 8), backend: %s\n\n", CUDA_BACKEND);

  bool passed = runFiniteDifference();
  passed = runHeatPlate() && passed;

  std::printf("\n%s\n", passed ? "ALL PASS" : "SOME FAILED");
  return passed ? 0 : 1;
}
