#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int TILE_DIM = 32;
const int NUM_BANKS = 32;
const int WARP_SIZE = 32;
const int ELEMENT_COUNT = TILE_DIM * TILE_DIM;
const int CORNER = 4;
const int NO_PADDING = 0;
const int ONE_COLUMN_OF_PADDING = 1;

__device__ inline int bankOf(const float* element, const float* firstElement) {
  return static_cast<int>(element - firstElement) % NUM_BANKS;
}

template <int PADDING>
__global__ void transposeKernel(const float* in, float* out, int* bankOfThread) {
  __shared__ float a[TILE_DIM][TILE_DIM + PADDING];
  int linear = threadIdx.y * TILE_DIM + threadIdx.x;

  a[threadIdx.x][threadIdx.y] = in[linear];
  bankOfThread[linear] = bankOf(&a[threadIdx.x][threadIdx.y], &a[0][0]);
  __syncthreads();

  out[linear] = a[threadIdx.y][threadIdx.x];
}

template <class Kernel>
static std::vector<int> transposeOnDevice(Kernel kernel,
                                          const float* in_d,
                                          std::vector<float>& out) {
  std::vector<int> bankOfThread(ELEMENT_COUNT);
  float* out_d;
  int* bank_d;
  CHECK(cudaMalloc(&out_d, ELEMENT_COUNT * sizeof(float)));
  CHECK(cudaMalloc(&bank_d, ELEMENT_COUNT * sizeof(int)));

  launch(kernel, dim3(1), dim3(TILE_DIM, TILE_DIM), in_d, out_d, bank_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(out.data(), out_d, ELEMENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(
      cudaMemcpy(bankOfThread.data(), bank_d, ELEMENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(out_d));
  CHECK(cudaFree(bank_d));
  return bankOfThread;
}

static int conflictWaysOfFirstWarp(const std::vector<int>& bankOfThread) {
  std::vector<int> threadsPerBank(NUM_BANKS, 0);
  int ways = 0;
  for (int lane = 0; lane < WARP_SIZE; ++lane) {
    const int bank = bankOfThread[lane];
    ++threadsPerBank[bank];
    ways = imax(ways, threadsPerBank[bank]);
  }
  return ways;
}

static void printBanks(const char* declaration, const std::vector<int>& bankOfThread) {
  std::printf("%-29s warp 0 stores to banks", declaration);
  for (int lane = 0; lane < CORNER * 2; ++lane) {
    std::printf(" %2d", bankOfThread[lane]);
  }
  const int ways = conflictWaysOfFirstWarp(bankOfThread);
  if (ways > 1) {
    std::printf(" ... -> %d-way conflict, %d passes\n", ways, ways);
  } else {
    std::printf(" ... -> no conflict, 1 pass\n");
  }
}

static void printCorner(const char* name, const std::vector<float>& matrix) {
  std::printf("%s (top-left %d x %d) =\n", name, CORNER, CORNER);
  for (int row = 0; row < CORNER; ++row) {
    for (int col = 0; col < CORNER; ++col) {
      std::printf("%6.0f", matrix[row * TILE_DIM + col]);
    }
    std::printf("\n");
  }
}

static bool isTranspose(const std::vector<float>& in, const std::vector<float>& out) {
  for (int row = 0; row < TILE_DIM; ++row) {
    for (int col = 0; col < TILE_DIM; ++col) {
      if (out[row * TILE_DIM + col] != in[col * TILE_DIM + row]) {
        return false;
      }
    }
  }
  return true;
}

int main() {
  std::vector<float> in(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    in[i] = static_cast<float>(i);
  }

  float* in_d;
  CHECK(cudaMalloc(&in_d, ELEMENT_COUNT * sizeof(float)));
  CHECK(cudaMemcpy(in_d, in.data(), ELEMENT_COUNT * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> outUnpadded(ELEMENT_COUNT);
  std::vector<float> outPadded(ELEMENT_COUNT);
  const std::vector<int> banksUnpadded =
      transposeOnDevice(transposeKernel<NO_PADDING>, in_d, outUnpadded);
  const std::vector<int> banksPadded =
      transposeOnDevice(transposeKernel<ONE_COLUMN_OF_PADDING>, in_d, outPadded);
  CHECK(cudaFree(in_d));

  printCorner("in", in);
  printCorner("out", outPadded);
  printBanks("a[TILE_DIM][TILE_DIM]", banksUnpadded);
  printBanks("a[TILE_DIM][TILE_DIM + 1]", banksPadded);

  const bool pass = isTranspose(in, outUnpadded) && isTranspose(in, outPadded) &&
                    conflictWaysOfFirstWarp(banksPadded) == 1;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
