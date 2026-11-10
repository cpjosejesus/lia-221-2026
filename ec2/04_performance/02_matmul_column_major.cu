#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int WIDTH = 4;
const float TOLERANCE = 1e-4f;

using Matrix = std::vector<float>;

__host__ __device__ inline int rowMajorIndex(int row, int col, int Width) {
  return row * Width + col;
}

__host__ __device__ inline int columnMajorIndex(int row, int col, int Width) {
  return col * Width + row;
}

__global__ void matrixMulKernel(const float* M, const float* N, float* P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Width && col < Width) {
    float Pvalue = 0.0f;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[rowMajorIndex(k, col, Width)];
    }
    P[row * Width + col] = Pvalue;
  }
}

__global__ void matrixMulColumnMajorKernel(const float* M, const float* N, float* P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Width && col < Width) {
    float Pvalue = 0.0f;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[columnMajorIndex(k, col, Width)];
    }
    P[row * Width + col] = Pvalue;
  }
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
      columnMajor[columnMajorIndex(row, col, WIDTH)] = rowMajor[rowMajorIndex(row, col, WIDTH)];
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

template <class Kernel>
static Matrix multiplyOnDevice(Kernel kernel, const Matrix& M, const Matrix& N) {
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

  launch(kernel, dim3(1), dim3(WIDTH, WIDTH), M_d, N_d, P_d, WIDTH);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(P.data(), P_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(M_d));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(P_d));
  return P;
}

template <class IndexOf>
static void printAddressesOfN(const char* layout, IndexOf indexOf) {
  std::printf("N %s: threads col = 0..%d read N at\n", layout, WIDTH - 1);
  for (int k = 0; k < WIDTH; ++k) {
    std::printf("  k = %d:", k);
    for (int col = 0; col < WIDTH; ++col) {
      std::printf(" %3d", indexOf(k, col, WIDTH));
    }
    std::printf("\n");
  }
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

static bool sameMatrix(const Matrix& a, const Matrix& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  const Matrix M = makeMatrix(3);
  const Matrix N = makeMatrix(5);
  const Matrix expected = multiplyOnCpu(M, N);

  const Matrix fromRowMajor = multiplyOnDevice(matrixMulKernel, M, N);
  const Matrix fromColumnMajor = multiplyOnDevice(matrixMulColumnMajorKernel, M, toColumnMajor(N));

  printMatrix("M", M);
  printMatrix("N", N);
  printAddressesOfN("row-major", rowMajorIndex);
  printAddressesOfN("column-major", columnMajorIndex);
  printMatrix("P from N row-major", fromRowMajor);
  printMatrix("P from N column-major", fromColumnMajor);

  const bool pass = sameMatrix(fromRowMajor, expected) && sameMatrix(fromColumnMajor, expected);
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
