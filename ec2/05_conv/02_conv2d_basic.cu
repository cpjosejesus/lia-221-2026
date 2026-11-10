#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int WIDTH = 6;
const int HEIGHT = 6;
const int FILTER_RADIUS = 1;
const int BLOCK_DIM = 4;
const int PIXEL_LEVELS = 5;

using Image = std::vector<float>;

__global__ void convolution_2D_basic_kernel(
    const float* N, const float* F, float* P, int r, int width, int height) {
  int outCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int outRow = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (outRow < height && outCol < width) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[outRow * width + outCol] = Pvalue;
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

  float* N_d = copyToDevice(N);
  float* F_d = copyToDevice(F);
  float* P_d = copyToDevice(Image(WIDTH * HEIGHT, 0.0f));

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(cdiv(WIDTH, BLOCK_DIM), cdiv(HEIGHT, BLOCK_DIM));
  launch(convolution_2D_basic_kernel, grid, block, N_d, F_d, P_d, FILTER_RADIUS, WIDTH, HEIGHT);
  CHECK(cudaGetLastError());

  Image P(WIDTH * HEIGHT);
  CHECK(cudaMemcpy(P.data(), P_d, P.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(N_d));
  CHECK(cudaFree(F_d));
  CHECK(cudaFree(P_d));

  printImage("F (filter)", F, filterDim, filterDim);
  printImage("N (input)", N, WIDTH, HEIGHT);
  printImage("P (output)", P, WIDTH, HEIGHT);

  const bool passed = P == convolution2DOnCpu(N, F, FILTER_RADIUS, WIDTH, HEIGHT);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
