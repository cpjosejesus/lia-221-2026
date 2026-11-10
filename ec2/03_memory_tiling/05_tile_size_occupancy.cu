#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int WIDTH = 6;
const int LARGE_TILE_WIDTH = 32;
const int SMALL_TILE_WIDTH = 16;
const int TILES_PER_BLOCK = 2;
const size_t KIB = 1024;
const float TOLERANCE = 1e-4f;

using Matrix = std::vector<float>;

template <int TILE_WIDTH>
__global__ void matrixMulKernel(const float* M, const float* N, float* P, int Width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;
  int phases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int ph = 0; ph < phases; ++ph) {
    int mCol = ph * TILE_WIDTH + tx;
    int nRow = ph * TILE_WIDTH + ty;

    if (Row < Width && mCol < Width) {
      Mds[ty][tx] = M[Row * Width + mCol];
    } else {
      Mds[ty][tx] = 0.0f;
    }

    if (nRow < Width && Col < Width) {
      Nds[ty][tx] = N[nRow * Width + Col];
    } else {
      Nds[ty][tx] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  if (Row < Width && Col < Width) {
    P[Row * Width + Col] = Pvalue;
  }
}

static size_t sharedBytesPerBlock(int tileWidth) {
  return TILES_PER_BLOCK * tileWidth * tileWidth * sizeof(float);
}

static void printOccupancyRow(const cudaDeviceProp& device,
                              const char* kernel,
                              int threads,
                              size_t sharedBytes) {
  const int byThreads = device.maxThreadsPerMultiProcessor / threads;
  const int byShared = static_cast<int>(device.sharedMemPerMultiprocessor / sharedBytes);
  const int blocks = imin(imin(byThreads, byShared), device.maxBlocksPerMultiProcessor);
  const float occupancy = 100.0f * blocks * threads / device.maxThreadsPerMultiProcessor;

  std::printf("%-15s %7d %9.1f %10d %10d %7d %9.0f %%\n",
              kernel,
              threads,
              static_cast<float>(sharedBytes) / KIB,
              byThreads,
              byShared,
              blocks,
              occupancy);
}

static void printOccupancyTable(const cudaDeviceProp& device) {
  std::printf("%s: %zu KB shared per block; per SM %zu KB shared, %d threads, %d blocks\n",
              device.name,
              device.sharedMemPerBlock / KIB,
              device.sharedMemPerMultiprocessor / KIB,
              device.maxThreadsPerMultiProcessor,
              device.maxBlocksPerMultiProcessor);
  std::printf("%-15s %7s %9s %10s %10s %7s %11s\n",
              "block",
              "threads",
              "shared KB",
              "by threads",
              "by shared",
              "blocks",
              "occupancy");

  printOccupancyRow(device, "tiled 8 x 8", 64, sharedBytesPerBlock(8));
  printOccupancyRow(device, "tiled 16 x 16", 256, sharedBytesPerBlock(16));
  printOccupancyRow(device, "tiled 32 x 32", 1024, sharedBytesPerBlock(32));
  printOccupancyRow(device, "38 KB, 256 thr", 256, 38 * KIB);
  printOccupancyRow(device, "48 KB, 512 thr", 512, 48 * KIB);
}

static int chooseTileWidth(const cudaDeviceProp& device) {
  const bool sharedFits = device.sharedMemPerBlock >= sharedBytesPerBlock(LARGE_TILE_WIDTH);
  const bool threadsFit = device.maxThreadsPerBlock >= LARGE_TILE_WIDTH * LARGE_TILE_WIDTH;
  return sharedFits && threadsFit ? LARGE_TILE_WIDTH : SMALL_TILE_WIDTH;
}

static Matrix makeMatrix(int width, int seed) {
  Matrix matrix(width * width);
  for (int i = 0; i < width * width; ++i) {
    matrix[i] = static_cast<float>((i * seed) % 7);
  }
  return matrix;
}

static Matrix multiplyOnCpu(const Matrix& M, const Matrix& N, int width) {
  Matrix P(width * width, 0.0f);
  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int k = 0; k < width; ++k) {
        P[row * width + col] += M[row * width + k] * N[k * width + col];
      }
    }
  }
  return P;
}

static Matrix multiplyOnDevice(const Matrix& M, const Matrix& N, int width, int tileWidth) {
  const size_t bytes = M.size() * sizeof(float);
  Matrix P(M.size());

  float* M_d;
  float* N_d;
  float* P_d;
  CHECK(cudaMalloc(&M_d, bytes));
  CHECK(cudaMalloc(&N_d, bytes));
  CHECK(cudaMalloc(&P_d, bytes));
  CHECK(cudaMemcpy(M_d, M.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(N_d, N.data(), bytes, cudaMemcpyHostToDevice));

  const unsigned blocksPerSide = cdiv(width, tileWidth);
  dim3 grid(blocksPerSide, blocksPerSide);
  dim3 block(tileWidth, tileWidth);
  if (tileWidth == LARGE_TILE_WIDTH) {
    launch(matrixMulKernel<LARGE_TILE_WIDTH>, grid, block, M_d, N_d, P_d, width);
  } else {
    launch(matrixMulKernel<SMALL_TILE_WIDTH>, grid, block, M_d, N_d, P_d, width);
  }
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(P.data(), P_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(M_d));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(P_d));
  return P;
}

static void printMatrix(const char* name, const Matrix& matrix, int width) {
  std::printf("%s =\n", name);
  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      std::printf("%6.1f", matrix[row * width + col]);
    }
    std::printf("\n");
  }
}

static bool sameMatrix(const Matrix& a, const Matrix& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  cudaDeviceProp device;
  CHECK(cudaGetDeviceProperties(&device, 0));
  printOccupancyTable(device);

  const int tileWidth = chooseTileWidth(device);
  std::printf("chosen TILE_WIDTH = %d (%zu B of shared memory per block)\n",
              tileWidth,
              sharedBytesPerBlock(tileWidth));

  const Matrix M = makeMatrix(WIDTH, 3);
  const Matrix N = makeMatrix(WIDTH, 5);
  const Matrix P = multiplyOnDevice(M, N, WIDTH, tileWidth);
  const Matrix expected = multiplyOnCpu(M, N, WIDTH);

  printMatrix("M", M, WIDTH);
  printMatrix("N", N, WIDTH);
  printMatrix("P = M * N", P, WIDTH);

  const bool pass = sameMatrix(P, expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
