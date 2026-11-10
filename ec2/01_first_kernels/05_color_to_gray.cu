#include "cuda_compat.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHANNELS 3

const int WIDTH = 4;
const int HEIGHT = 3;
const int PIXELS = WIDTH * HEIGHT;
const int BLOCK_SIDE = 16;
const int ROUNDING_TOLERANCE = 1;

using Bytes = std::vector<unsigned char>;

__global__ void colorToGray(unsigned char* gray, const unsigned char* rgb, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;

    unsigned char r = rgb[rgbOffset];
    unsigned char g = rgb[rgbOffset + 1];
    unsigned char b = rgb[rgbOffset + 2];
    gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

static Bytes toGray(const Bytes& rgb, int width, int height) {
  const size_t grayBytes = width * height;
  const size_t rgbBytes = grayBytes * CHANNELS;
  Bytes gray(grayBytes, 0);

  unsigned char* rgb_d;
  unsigned char* gray_d;
  cudaMalloc(&rgb_d, rgbBytes);
  cudaMalloc(&gray_d, grayBytes);
  cudaMemcpy(rgb_d, rgb.data(), rgbBytes, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIDE, BLOCK_SIDE);
  dim3 grid((width + BLOCK_SIDE - 1) / BLOCK_SIDE, (height + BLOCK_SIDE - 1) / BLOCK_SIDE);
  launch(colorToGray, grid, block, gray_d, rgb_d, width, height);

  cudaMemcpy(gray.data(), gray_d, grayBytes, cudaMemcpyDeviceToHost);
  cudaFree(rgb_d);
  cudaFree(gray_d);
  return gray;
}

static Bytes toGrayOnCpu(const Bytes& rgb) {
  Bytes gray(PIXELS);
  for (int pixel = 0; pixel < PIXELS; ++pixel) {
    const unsigned char r = rgb[pixel * CHANNELS];
    const unsigned char g = rgb[pixel * CHANNELS + 1];
    const unsigned char b = rgb[pixel * CHANNELS + 2];
    gray[pixel] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
  }
  return gray;
}

const unsigned char PICTURE[PIXELS][CHANNELS] = {
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 255},
    {255, 255, 0},
    {0, 255, 255},
    {255, 0, 255},
    {0, 0, 0},
    {128, 128, 128},
    {255, 165, 0},
    {128, 0, 128},
    {165, 42, 42},
};

static Bytes tinyPicture() {
  const unsigned char* first = &PICTURE[0][0];
  return Bytes(first, first + PIXELS * CHANNELS);
}

static void printPictures(const Bytes& rgb, const Bytes& gray) {
  for (int row = 0; row < HEIGHT; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      const int pixel = row * WIDTH + col;
      std::printf(" (%3d,%3d,%3d)",
                  rgb[pixel * CHANNELS],
                  rgb[pixel * CHANNELS + 1],
                  rgb[pixel * CHANNELS + 2]);
    }
    std::printf("   ->");
    for (int col = 0; col < WIDTH; ++col) {
      std::printf(" %3d", gray[row * WIDTH + col]);
    }
    std::printf("\n");
  }
}

static bool closeEnough(const Bytes& gray, const Bytes& expected) {
  for (int pixel = 0; pixel < PIXELS; ++pixel) {
    if (std::abs(gray[pixel] - expected[pixel]) > ROUNDING_TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  const Bytes rgb = tinyPicture();
  const Bytes gray = toGray(rgb, WIDTH, HEIGHT);
  const Bytes expected = toGrayOnCpu(rgb);

  std::printf("picture %d x %d, (r,g,b) on the left, gray on the right\n", WIDTH, HEIGHT);
  printPictures(rgb, gray);

  const bool ok = closeEnough(gray, expected);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
