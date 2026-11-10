#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int WIDTH = 4;
const int ELEMENTS = WIDTH * WIDTH;
const int THREADS_PER_BLOCK = 256;
const int NOT_COMPUTED = -1;

using Matrix = std::vector<float>;

__global__ void matMulRowKernel(const float* M, const float* N, float* P, int width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < width) {
    for (int col = 0; col < width; ++col) {
      float sum = 0;
      for (int k = 0; k < width; ++k) {
        sum += M[row * width + k] * N[k * width + col];
      }
      P[row * width + col] = sum;
    }
  }
}

__global__ void matMulColKernel(const float* M, const float* N, float* P, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < width) {
    for (int row = 0; row < width; ++row) {
      float sum = 0;
      for (int k = 0; k < width; ++k) {
        sum += M[row * width + k] * N[k * width + col];
      }
      P[row * width + col] = sum;
    }
  }
}

using MatMulKernel = void (*)(const float*, const float*, float*, int);

static Matrix matMulOnDevice(MatMulKernel kernel, const Matrix& M, const Matrix& N) {
  const size_t bytes = ELEMENTS * sizeof(float);
  Matrix P(ELEMENTS, NOT_COMPUTED);

  float* M_d;
  float* N_d;
  float* P_d;
  CHECK(cudaMalloc(&M_d, bytes));
  CHECK(cudaMalloc(&N_d, bytes));
  CHECK(cudaMalloc(&P_d, bytes));
  CHECK(cudaMemcpy(M_d, M.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(N_d, N.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = (WIDTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  launch(kernel, dim3(blocks), dim3(THREADS_PER_BLOCK), M_d, N_d, P_d, WIDTH);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(P.data(), P_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(M_d));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(P_d));
  return P;
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

  const int blocks = (WIDTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  std::printf("%d x %d matrices, launch <<<(%d + %d) / %d, %d>>> = %d block of %d threads\n",
              WIDTH,
              WIDTH,
              WIDTH,
              THREADS_PER_BLOCK - 1,
              THREADS_PER_BLOCK,
              THREADS_PER_BLOCK,
              blocks,
              THREADS_PER_BLOCK);
  std::printf("only %d threads work: one per row (or column) of P, each computes %d elements\n",
              WIDTH,
              WIDTH);
  std::printf("for width 1000: 1000 threads, against 1000000 with one element per thread\n\n");
  printMatrix("M", M);
  printMatrix("N", N);

  const Matrix expected = matMulOnCpu(M, N);
  const Matrix byRow = matMulOnDevice(matMulRowKernel, M, N);
  const Matrix byCol = matMulOnDevice(matMulColKernel, M, N);
  printMatrix("P, one thread per row (matMulRowKernel)", byRow);
  printMatrix("P, one thread per column (matMulColKernel)", byCol);

  const bool ok = byRow == expected && byCol == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
