#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int WIDTH = 8;
const int TILE_WIDTH = 4;
const int LOADS_PER_PHASE = 2;
const float NAIVE_FLOP_PER_BYTE = 0.25f;
const float TOLERANCE = 1e-4f;

using Matrix = std::vector<float>;

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
  for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
    Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  P[Row * Width + Col] = Pvalue;
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

static Matrix multiplyOnDevice(const Matrix& M, const Matrix& N, int width) {
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

  const unsigned blocksPerSide = width / TILE_WIDTH;
  dim3 grid(blocksPerSide, blocksPerSide);
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  launch(matrixMulKernel, grid, block, M_d, N_d, P_d, width);
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

static void printGlobalLoads(int width) {
  const int phases = width / TILE_WIDTH;
  const int naiveLoadsPerThread = 2 * width;
  const int tiledLoadsPerThread = LOADS_PER_PHASE * phases;

  std::printf("%d phases of %d x %d tiles\n", phases, TILE_WIDTH, TILE_WIDTH);
  std::printf("global loads per thread: naive %d, tiled %d (%d times fewer)\n",
              naiveLoadsPerThread,
              tiledLoadsPerThread,
              naiveLoadsPerThread / tiledLoadsPerThread);
  std::printf("compute-to-global-memory-access ratio: naive %.2f, tiled %.2f FLOP/B\n",
              NAIVE_FLOP_PER_BYTE,
              NAIVE_FLOP_PER_BYTE * TILE_WIDTH);
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
  const Matrix M = makeMatrix(WIDTH, 3);
  const Matrix N = makeMatrix(WIDTH, 5);

  const Matrix P = multiplyOnDevice(M, N, WIDTH);
  const Matrix expected = multiplyOnCpu(M, N, WIDTH);

  printMatrix("M", M, WIDTH);
  printMatrix("N", N, WIDTH);
  printMatrix("P = M * N", P, WIDTH);
  printGlobalLoads(WIDTH);

  const bool pass = sameMatrix(P, expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
