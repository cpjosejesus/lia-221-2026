#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int WIDTH = 8;
const int TILE = 4;
const float TOLERANCE = 1e-4f;

using Matrix = std::vector<float>;

__host__ __device__ inline int naiveLoadOfN(int bx, int ph, int tx, int ty, int Width) {
  int Col = bx * TILE + tx;
  return Col * Width + ph * TILE + ty;
}

__host__ __device__ inline int cornerTurningLoadOfN(int bx, int ph, int tx, int ty, int Width) {
  int c = bx * TILE + ty;
  int r = ph * TILE + tx;
  return c * Width + r;
}

__global__ void matrixMulKernel(const float* M, const float* N, float* P, int Width) {
  __shared__ float Mds[TILE][TILE];
  __shared__ float Nds[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = blockIdx.y * TILE + ty;
  int Col = blockIdx.x * TILE + tx;

  float Pvalue = 0;
  for (int ph = 0; ph < Width / TILE; ++ph) {
    Mds[ty][tx] = M[Row * Width + ph * TILE + tx];
    Nds[tx][ty] = N[cornerTurningLoadOfN(blockIdx.x, ph, tx, ty, Width)];
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  P[Row * Width + Col] = Pvalue;
}

static Matrix makeMatrix(int seed) {
  Matrix matrix(WIDTH * WIDTH);
  for (int i = 0; i < WIDTH * WIDTH; ++i) {
    matrix[i] = static_cast<float>((i * seed) % 7);
  }
  return matrix;
}

static Matrix toColumnMajor(const Matrix& rowMajor) {
  Matrix columnMajor(rowMajor.size());
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      columnMajor[col * WIDTH + row] = rowMajor[row * WIDTH + col];
    }
  }
  return columnMajor;
}

static Matrix multiplyOnCpu(const Matrix& M, const Matrix& N) {
  Matrix P(WIDTH * WIDTH, 0.0f);
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      for (int k = 0; k < WIDTH; ++k) {
        P[row * WIDTH + col] += M[row * WIDTH + k] * N[k * WIDTH + col];
      }
    }
  }
  return P;
}

static Matrix multiplyOnDevice(const Matrix& M, const Matrix& NColumnMajor) {
  const size_t bytes = M.size() * sizeof(float);
  Matrix P(M.size());

  float* M_d;
  float* N_d;
  float* P_d;
  CHECK(cudaMalloc(&M_d, bytes));
  CHECK(cudaMalloc(&N_d, bytes));
  CHECK(cudaMalloc(&P_d, bytes));
  CHECK(cudaMemcpy(M_d, M.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(N_d, NColumnMajor.data(), bytes, cudaMemcpyHostToDevice));

  dim3 grid(WIDTH / TILE, WIDTH / TILE);
  dim3 block(TILE, TILE);
  launch(matrixMulKernel, grid, block, M_d, N_d, P_d, WIDTH);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(P.data(), P_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(M_d));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(P_d));
  return P;
}

static void printLoadAddresses(int ph) {
  std::printf("phase %d, block (0, 0), threads ty = 0, tx = 0..%d load N at\n", ph, TILE - 1);
  std::printf("  naive         :");
  for (int tx = 0; tx < TILE; ++tx) {
    std::printf(" %3d", naiveLoadOfN(0, ph, tx, 0, WIDTH));
  }
  std::printf("\n  corner turning:");
  for (int tx = 0; tx < TILE; ++tx) {
    std::printf(" %3d", cornerTurningLoadOfN(0, ph, tx, 0, WIDTH));
  }
  std::printf("\n");
}

static void printMatrix(const char* name, const Matrix& matrix) {
  std::printf("%s =\n", name);
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      std::printf("%6.1f", matrix[row * WIDTH + col]);
    }
    std::printf("\n");
  }
}

int main() {
  const Matrix M = makeMatrix(3);
  const Matrix N = makeMatrix(5);
  const Matrix P = multiplyOnDevice(M, toColumnMajor(N));
  const Matrix expected = multiplyOnCpu(M, N);

  printMatrix("M (row-major)", M);
  printMatrix("N (stored column-major on the device)", N);
  printLoadAddresses(0);
  printLoadAddresses(1);
  printMatrix("P = M * N", P);

  bool pass = true;
  for (size_t i = 0; i < P.size(); ++i) {
    pass = pass && std::fabs(P[i] - expected[i]) < TOLERANCE;
  }
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
