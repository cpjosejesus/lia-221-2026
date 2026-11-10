#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int TILE_WIDTH = 2;
const int WIDTHS[] = {3, 5};
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

  const unsigned blocksPerSide = cdiv(width, TILE_WIDTH);
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

static bool sameMatrix(const Matrix& a, const Matrix& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

static void printGrid(int width) {
  const unsigned blocksPerSide = cdiv(width, TILE_WIDTH);
  const unsigned threadsPerSide = blocksPerSide * TILE_WIDTH;

  std::printf(
      "Width %d, TILE_WIDTH %d: %u x %u blocks, %u phases, %u of %u threads store a P element\n",
      width,
      TILE_WIDTH,
      blocksPerSide,
      blocksPerSide,
      blocksPerSide,
      width * width,
      threadsPerSide * threadsPerSide);
}

static bool checkWidth(int width) {
  const Matrix M = makeMatrix(width, 3);
  const Matrix N = makeMatrix(width, 5);

  const Matrix P = multiplyOnDevice(M, N, width);
  const Matrix expected = multiplyOnCpu(M, N, width);

  printGrid(width);
  printMatrix("M", M, width);
  printMatrix("N", N, width);
  printMatrix("P = M * N", P, width);
  return sameMatrix(P, expected);
}

int main() {
  bool pass = true;
  for (int width : WIDTHS) {
    pass = checkWidth(width) && pass;
    std::printf("\n");
  }

  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
