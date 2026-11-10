#include "cuda_compat.h"

#include <cstdio>
#include <vector>

#define FILTER_RADIUS 1
#define TILE_DIM 4

const int WIDTH = 10;
const int HEIGHT = 10;
const int PIXEL_LEVELS = 5;

using Image = std::vector<float>;

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_cached_tiled_2D_const_mem_kernel(const float* N,
                                                             float* P,
                                                             int width,
                                                             int height) {
  int col = (int)(blockIdx.x * TILE_DIM + threadIdx.x);
  int row = (int)(blockIdx.y * TILE_DIM + threadIdx.y);

  __shared__ float N_s[TILE_DIM][TILE_DIM];
  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  if (col < width && row < height) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
        int sharedCol = (int)threadIdx.x - FILTER_RADIUS + fCol;
        int sharedRow = (int)threadIdx.y - FILTER_RADIUS + fRow;
        bool insideTile =
            sharedCol >= 0 && sharedCol < TILE_DIM && sharedRow >= 0 && sharedRow < TILE_DIM;

        if (insideTile) {
          Pvalue += F_c[fRow][fCol] * N_s[sharedRow][sharedCol];
        } else {
          int inRow = row - FILTER_RADIUS + fRow;
          int inCol = col - FILTER_RADIUS + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
          }
        }
      }
    }
    P[row * width + col] = Pvalue;
  }
}

static Image convolution2DOnCpu(const Image& N, const Image& F, int r, int width, int height) {
  Image P(width * height, 0.0f);
  for (int outRow = 0; outRow < height; outRow++) {
    for (int outCol = 0; outCol < width; outCol++) {
      for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
          int inRow = outRow - r + fRow;
          int inCol = outCol - r + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            P[outRow * width + outCol] += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
          }
        }
      }
    }
  }
  return P;
}

static Image smallImage(int width, int height) {
  Image N(width * height);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      N[row * width + col] = (float)((3 * row + col) % PIXEL_LEVELS);
    }
  }
  return N;
}

static float* copyToDevice(const Image& host) {
  float* device;
  CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  return device;
}

static void printImage(const char* name, const Image& image, int width, int height) {
  std::printf("%s:\n", name);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      std::printf(" %4.0f", image[row * width + col]);
    }
    std::printf("\n");
  }
}

int main() {
  const int filterDim = 2 * FILTER_RADIUS + 1;
  const Image F = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  const Image N = smallImage(WIDTH, HEIGHT);

  CHECK(cudaMemcpyToSymbol(F_c, F.data(), F.size() * sizeof(float)));
  float* N_d = copyToDevice(N);
  float* P_d = copyToDevice(Image(WIDTH * HEIGHT, 0.0f));

  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid(cdiv(WIDTH, TILE_DIM), cdiv(HEIGHT, TILE_DIM));
  launch(convolution_cached_tiled_2D_const_mem_kernel, grid, block, N_d, P_d, WIDTH, HEIGHT);
  CHECK(cudaGetLastError());

  Image P(WIDTH * HEIGHT);
  CHECK(cudaMemcpy(P.data(), P_d, P.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(P_d));

  printImage("F (filter)", F, filterDim, filterDim);
  printImage("N (input)", N, WIDTH, HEIGHT);
  printImage("P (output)", P, WIDTH, HEIGHT);

  const bool passed = P == convolution2DOnCpu(N, F, FILTER_RADIUS, WIDTH, HEIGHT);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
