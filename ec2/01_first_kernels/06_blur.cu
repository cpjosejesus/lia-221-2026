#include "cuda_compat.h"

#include <cstdio>
#include <vector>

#define BLUR_SIZE 1

const int WIDTH = 6;
const int HEIGHT = 6;
const int PIXELS = WIDTH * HEIGHT;
const int BLOCK_SIDE = 16;
const unsigned char DARK = 10;
const unsigned char BRIGHT = 100;

using Bytes = std::vector<unsigned char>;

__global__ void blurKernel(const unsigned char* in, unsigned char* out, int w, int h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < w && row < h) {
    int sum = 0;
    int pixels = 0;

    for (int dr = -BLUR_SIZE; dr <= BLUR_SIZE; ++dr) {
      for (int dc = -BLUR_SIZE; dc <= BLUR_SIZE; ++dc) {
        int r = row + dr;
        int c = col + dc;
        if (r >= 0 && r < h && c >= 0 && c < w) {
          sum += in[r * w + c];
          ++pixels;
        }
      }
    }

    out[row * w + col] = (unsigned char)(sum / pixels);
  }
}

static Bytes blur(const Bytes& in, int w, int h) {
  const size_t bytes = w * h;
  Bytes out(bytes, 0);

  unsigned char* in_d;
  unsigned char* out_d;
  cudaMalloc(&in_d, bytes);
  cudaMalloc(&out_d, bytes);
  cudaMemcpy(in_d, in.data(), bytes, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIDE, BLOCK_SIDE);
  dim3 grid((w + BLOCK_SIDE - 1) / BLOCK_SIDE, (h + BLOCK_SIDE - 1) / BLOCK_SIDE);
  launch(blurKernel, grid, block, in_d, out_d, w, h);

  cudaMemcpy(out.data(), out_d, bytes, cudaMemcpyDeviceToHost);
  cudaFree(in_d);
  cudaFree(out_d);
  return out;
}

static unsigned char patchAverageOnCpu(const Bytes& in, int row, int col) {
  int sum = 0;
  int pixels = 0;
  for (int r = row - BLUR_SIZE; r <= row + BLUR_SIZE; ++r) {
    for (int c = col - BLUR_SIZE; c <= col + BLUR_SIZE; ++c) {
      const bool inside = r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH;
      if (inside) {
        sum += in[r * WIDTH + c];
        ++pixels;
      }
    }
  }
  return (unsigned char)(sum / pixels);
}

static Bytes blurOnCpu(const Bytes& in) {
  Bytes out(PIXELS);
  for (int row = 0; row < HEIGHT; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      out[row * WIDTH + col] = patchAverageOnCpu(in, row, col);
    }
  }
  return out;
}

static Bytes brightSquareAndCorner() {
  Bytes picture(PIXELS, DARK);
  picture[2 * WIDTH + 2] = BRIGHT;
  picture[2 * WIDTH + 3] = BRIGHT;
  picture[3 * WIDTH + 2] = BRIGHT;
  picture[3 * WIDTH + 3] = BRIGHT;
  picture[WIDTH - 1] = BRIGHT;
  return picture;
}

static void printPicture(const char* title, const Bytes& picture) {
  std::printf("%s\n", title);
  for (int row = 0; row < HEIGHT; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      std::printf(" %3d", picture[row * WIDTH + col]);
    }
    std::printf("\n");
  }
}

int main() {
  const Bytes in = brightSquareAndCorner();
  const Bytes out = blur(in, WIDTH, HEIGHT);
  const Bytes expected = blurOnCpu(in);

  std::printf("%d x %d patch (BLUR_SIZE = %d)\n", 2 * BLUR_SIZE + 1, 2 * BLUR_SIZE + 1, BLUR_SIZE);
  printPicture("input:", in);
  printPicture("blurred:", out);

  const bool ok = out == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
