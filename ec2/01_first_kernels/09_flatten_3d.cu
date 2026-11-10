#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int WIDTH = 4;
const int HEIGHT = 3;
const int DEPTH = 2;
const int ELEMENTS = WIDTH * HEIGHT * DEPTH;
const int BLOCK_SIDE = 2;
const int NOT_WRITTEN = -1;

__host__ __device__ int flatIndex3d(int plane, int row, int col, int width, int height) {
  return (plane * height + row) * width + col;
}

__host__ __device__ int coordinateCode(int plane, int row, int col) {
  return plane * 100 + row * 10 + col;
}

__global__ void writeCoordinates(int* P, int width, int height, int depth) {
  int plane = blockIdx.z * blockDim.z + threadIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (plane < depth && row < height && col < width) {
    P[flatIndex3d(plane, row, col, width, height)] = coordinateCode(plane, row, col);
  }
}

static int blocksToCover(int n, int blockSide) {
  return (n + blockSide - 1) / blockSide;
}

static std::vector<int> writeCoordinatesOnDevice(dim3 grid, dim3 block) {
  const size_t bytes = ELEMENTS * sizeof(int);
  std::vector<int> P(ELEMENTS, NOT_WRITTEN);

  int* P_d;
  CHECK(cudaMalloc(&P_d, bytes));
  CHECK(cudaMemcpy(P_d, P.data(), bytes, cudaMemcpyHostToDevice));

  launch(writeCoordinates, grid, block, P_d, WIDTH, HEIGHT, DEPTH);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(P.data(), P_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(P_d));
  return P;
}

static std::vector<int> coordinatesInMemoryOrderOnCpu() {
  std::vector<int> P(ELEMENTS);
  for (int flat = 0; flat < ELEMENTS; ++flat) {
    const int plane = flat / (WIDTH * HEIGHT);
    const int row = flat / WIDTH % HEIGHT;
    const int col = flat % WIDTH;
    P[flat] = coordinateCode(plane, row, col);
  }
  return P;
}

static void printMemory(const std::vector<int>& P) {
  std::printf("memory, one line, as (plane row col):\n");
  for (int flat = 0; flat < ELEMENTS; ++flat) {
    if (flat > 0 && flat % (WIDTH * HEIGHT) == 0) {
      std::printf(" |");
    }
    std::printf(" %03d", P[flat]);
  }
  std::printf("\n");
}

static void printPlanes(const std::vector<int>& P) {
  for (int plane = 0; plane < DEPTH; ++plane) {
    std::printf("plane %d, flat index of each (row, col):\n", plane);
    for (int row = 0; row < HEIGHT; ++row) {
      for (int col = 0; col < WIDTH; ++col) {
        const int flat = flatIndex3d(plane, row, col, WIDTH, HEIGHT);
        std::printf(" %2d:%03d", flat, P[flat]);
      }
      std::printf("\n");
    }
  }
}

static void printExerciseIndices() {
  const int mapWidth = 300;
  const int mapHeight = 200;
  std::printf("\nexercises 11.4 and 11.5, a %d x %d map:\n", mapWidth, mapHeight);
  std::printf("  row 15, col 40, row-major:    15 * %d + 40 = %d\n",
              mapWidth,
              flatIndex3d(0, 15, 40, mapWidth, mapHeight));
  std::printf(
      "  row 15, col 40, column-major: 40 * %d + 15 = %d\n", mapHeight, 40 * mapHeight + 15);
  std::printf("  x 40, y 15, z 3, row-major:   (3 * %d + 15) * %d + 40 = %d\n",
              mapHeight,
              mapWidth,
              flatIndex3d(3, 15, 40, mapWidth, mapHeight));
}

int main() {
  const dim3 block(BLOCK_SIDE, BLOCK_SIDE, BLOCK_SIDE);
  const dim3 grid(blocksToCover(WIDTH, BLOCK_SIDE),
                  blocksToCover(HEIGHT, BLOCK_SIDE),
                  blocksToCover(DEPTH, BLOCK_SIDE));
  const int launched = grid.x * grid.y * grid.z * BLOCK_SIDE * BLOCK_SIDE * BLOCK_SIDE;

  std::printf("%d planes x %d rows x %d columns, blocks of %d x %d x %d, grid of %u x %u x %u\n",
              DEPTH,
              HEIGHT,
              WIDTH,
              BLOCK_SIDE,
              BLOCK_SIDE,
              BLOCK_SIDE,
              grid.x,
              grid.y,
              grid.z);
  std::printf("%d threads launched for %d elements, %d stop at the 3D guard\n",
              launched,
              ELEMENTS,
              launched - ELEMENTS);
  std::printf(
      "thread (plane, row, col) writes plane*100 + row*10 + col at (plane*%d + row)*%d + col\n\n",
      HEIGHT,
      WIDTH);

  const std::vector<int> P = writeCoordinatesOnDevice(grid, block);
  const std::vector<int> expected = coordinatesInMemoryOrderOnCpu();
  printMemory(P);
  printPlanes(P);
  printExerciseIndices();

  const bool ok = P == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
