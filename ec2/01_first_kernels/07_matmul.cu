#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int WIDTH = 4;
const int ELEMENTS = WIDTH * WIDTH;
const int BLOCK_WIDTH = 2;

using Matrix = std::vector<float>;

__global__ void matMulKernel(const float* M, const float* N, float* P, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width && col < width) {
    float sum = 0;
    for (int k = 0; k < width; ++k) {
      sum += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = sum;
  }
}

static void matMul(const float* M, const float* N, float* P, int width) {
  const size_t bytes = (size_t)width * width * sizeof(float);

  float* M_d;
  float* N_d;
  float* P_d;
  cudaMalloc(&M_d, bytes);
  cudaMalloc(&N_d, bytes);
  cudaMalloc(&P_d, bytes);

  cudaMemcpy(M_d, M, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N, bytes, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
  launch(matMulKernel, grid, block, M_d, N_d, P_d, width);

  cudaMemcpy(P, P_d, bytes, cudaMemcpyDeviceToHost);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}

static Matrix matMulOnCpu(const Matrix& M, const Matrix& N) {
  Matrix P(ELEMENTS, 0.0f);
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      float sum = 0;
      for (int k = 0; k < WIDTH; ++k) {
        sum += M[row * WIDTH + k] * N[k * WIDTH + col];
      }
      P[row * WIDTH + col] = sum;
    }
  }
  return P;
}

static void printMatrix(const char* name, const Matrix& matrix) {
  std::printf("%s =\n", name);
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      std::printf(" %6.1f", matrix[row * WIDTH + col]);
    }
    std::printf("\n");
  }
}

int main() {
  Matrix M(ELEMENTS);
  Matrix N(ELEMENTS);
  for (int i = 0; i < ELEMENTS; ++i) {
    M[i] = static_cast<float>(i + 1);
    N[i] = static_cast<float>((i % WIDTH + i / WIDTH) % 3);
  }

  Matrix P(ELEMENTS, 0.0f);
  matMul(M.data(), N.data(), P.data(), WIDTH);
  const Matrix expected = matMulOnCpu(M, N);

  const int blocksPerSide = (WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  std::printf("%d x %d matrices, blocks of %d x %d threads, grid of %d x %d blocks\n",
              WIDTH,
              WIDTH,
              BLOCK_WIDTH,
              BLOCK_WIDTH,
              blocksPerSide,
              blocksPerSide);
  std::printf("thread (row, col) computes P[row][col] = row of M times column of N\n\n");
  printMatrix("M", M);
  printMatrix("N", N);
  printMatrix("P = M x N", P);

  const bool ok = P == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
