#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int SIZE = 5;
const int THREADS_PER_BLOCK = 256;

__global__ void matVecKernel(float* a, const float* B, const float* c, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float sum = 0;
    for (int j = 0; j < N; ++j) {
      sum += B[i * N + j] * c[j];
    }
    a[i] = sum;
  }
}

static void matVec(float* a, const float* B, const float* c, int N) {
  const size_t vec = N * sizeof(float);
  const size_t mat = (size_t)N * N * sizeof(float);

  float* a_d;
  float* B_d;
  float* c_d;
  CHECK(cudaMalloc(&a_d, vec));
  CHECK(cudaMalloc(&B_d, mat));
  CHECK(cudaMalloc(&c_d, vec));
  CHECK(cudaMemcpy(B_d, B, mat, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(c_d, c, vec, cudaMemcpyHostToDevice));

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  launch(matVecKernel, grid, block, a_d, B_d, c_d, N);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(a, a_d, vec, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(a_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(c_d));
}

static std::vector<float> matVecOnCpu(const std::vector<float>& B, const std::vector<float>& c) {
  std::vector<float> a(SIZE, 0.0f);
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      a[i] += B[i * SIZE + j] * c[j];
    }
  }
  return a;
}

int main() {
  std::vector<float> B(SIZE * SIZE);
  std::vector<float> c(SIZE);
  for (int i = 0; i < SIZE; ++i) {
    c[i] = static_cast<float>(i + 1);
    for (int j = 0; j < SIZE; ++j) {
      B[i * SIZE + j] = static_cast<float>(j <= i ? 1 : 0);
    }
  }

  std::vector<float> a(SIZE, 0.0f);
  matVec(a.data(), B.data(), c.data(), SIZE);
  const std::vector<float> expected = matVecOnCpu(B, c);

  std::printf("N = %d, %d block of %d threads, thread i computes a[i] = row i of B times c\n\n",
              SIZE,
              (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
              THREADS_PER_BLOCK);
  for (int i = 0; i < SIZE; ++i) {
    std::printf("%s", i == SIZE / 2 ? "B =" : "   ");
    for (int j = 0; j < SIZE; ++j) {
      std::printf(" %3.0f", B[i * SIZE + j]);
    }
    std::printf("   %s %3.0f", i == SIZE / 2 ? "c =" : "   ", c[i]);
    std::printf("   %s %5.1f\n", i == SIZE / 2 ? "a =" : "   ", a[i]);
  }

  const bool ok = a == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
