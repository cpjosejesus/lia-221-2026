#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int WIDTH = 4;
const int BLOCK_WIDTH = 2;
const int FLOPS_PER_ITERATION = 2;
const int LOADS_PER_ITERATION = 2;
const float TOLERANCE = 1e-4f;

using Matrix = std::vector<float>;

__global__ void matrixMulKernel(const float* M, const float* N, float* P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Width && col < Width) {
    float Pvalue = 0.0f;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[k * Width + col];
    }
    P[row * Width + col] = Pvalue;
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

  const unsigned blocksPerSide = cdiv(width, BLOCK_WIDTH);
  dim3 grid(blocksPerSide, blocksPerSide);
  dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
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

static void printComputeToMemoryRatio(int width) {
  const int flopsPerThread = FLOPS_PER_ITERATION * width;
  const int bytesPerThread = LOADS_PER_ITERATION * width * static_cast<int>(sizeof(float));
  const float ratio = static_cast<float>(flopsPerThread) / bytesPerThread;

  std::printf("each thread: %d FLOP, %d global loads = %d B\n",
              flopsPerThread,
              LOADS_PER_ITERATION * width,
              bytesPerThread);
  std::printf("compute-to-global-memory-access ratio = %.2f FLOP/B\n", ratio);
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
  printComputeToMemoryRatio(WIDTH);

  const bool pass = sameMatrix(P, expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
