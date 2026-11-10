#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int WIDTH = 6;
const int HEIGHT = 5;
const int PIXELS = WIDTH * HEIGHT;
const int BLOCK_SIDE = 4;

__global__ void mapPixels(int* blockOf, int* rowOf, int* colOf, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int flat = row * width + col;
    blockOf[flat] = blockIdx.y * gridDim.x + blockIdx.x;
    rowOf[flat] = row;
    colOf[flat] = col;
  }
}

struct PixelOwners {
  std::vector<int> blockOf = std::vector<int>(PIXELS, -1);
  std::vector<int> rowOf = std::vector<int>(PIXELS, -1);
  std::vector<int> colOf = std::vector<int>(PIXELS, -1);
};

static int blocksToCover(int n, int blockSide) {
  return (n + blockSide - 1) / blockSide;
}

static PixelOwners mapPixelsOnDevice(dim3 grid, dim3 block) {
  const size_t bytes = PIXELS * sizeof(int);
  PixelOwners owners;

  int* blockOf_d;
  int* rowOf_d;
  int* colOf_d;
  cudaMalloc(&blockOf_d, bytes);
  cudaMalloc(&rowOf_d, bytes);
  cudaMalloc(&colOf_d, bytes);

  launch(mapPixels, grid, block, blockOf_d, rowOf_d, colOf_d, WIDTH, HEIGHT);

  cudaMemcpy(owners.blockOf.data(), blockOf_d, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(owners.rowOf.data(), rowOf_d, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(owners.colOf.data(), colOf_d, bytes, cudaMemcpyDeviceToHost);

  cudaFree(blockOf_d);
  cudaFree(rowOf_d);
  cudaFree(colOf_d);
  return owners;
}

static void printMap(const char* title, const std::vector<int>& valueOf) {
  std::printf("%s\n", title);
  for (int row = 0; row < HEIGHT; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      std::printf(" %3d", valueOf[row * WIDTH + col]);
    }
    std::printf("\n");
  }
}

static bool ownersAreCorrect(const PixelOwners& owners, int blocksAcross) {
  for (int flat = 0; flat < PIXELS; ++flat) {
    const int row = flat / WIDTH;
    const int col = flat % WIDTH;
    const int block = (row / BLOCK_SIDE) * blocksAcross + col / BLOCK_SIDE;

    if (owners.rowOf[flat] != row || owners.colOf[flat] != col || owners.blockOf[flat] != block) {
      return false;
    }
  }
  return true;
}

int main() {
  dim3 block(BLOCK_SIDE, BLOCK_SIDE);
  dim3 grid(blocksToCover(WIDTH, BLOCK_SIDE), blocksToCover(HEIGHT, BLOCK_SIDE));
  const int launched = grid.x * grid.y * block.x * block.y;

  std::printf("picture %d x %d, blocks of %d x %d, grid %u x %u blocks\n",
              WIDTH,
              HEIGHT,
              BLOCK_SIDE,
              BLOCK_SIDE,
              grid.x,
              grid.y);
  std::printf("%d threads launched for %d pixels\n\n", launched, PIXELS);

  const PixelOwners owners = mapPixelsOnDevice(grid, block);

  std::vector<int> flatIndex(PIXELS);
  for (int flat = 0; flat < PIXELS; ++flat) {
    flatIndex[flat] = owners.rowOf[flat] * WIDTH + owners.colOf[flat];
  }

  printMap("block that owns each pixel (blockIdx.y * gridDim.x + blockIdx.x):", owners.blockOf);
  printMap("\nrow-major index row * width + col:", flatIndex);

  const bool ok = ownersAreCorrect(owners, grid.x);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
