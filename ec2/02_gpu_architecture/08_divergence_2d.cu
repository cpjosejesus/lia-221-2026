#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int BLOCK_SIDE = 16;
const int THREADS_PER_BLOCK = BLOCK_SIDE * BLOCK_SIDE;
const int WARP_SIZE = 32;
const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
const int OUTSIDE = 0;
const int INSIDE = 1;

using Flags = std::vector<int>;

struct Picture {
  int width;
  int height;
};

const Picture PICTURES[] = {{76, 62}, {100, 70}};

__global__ void boundaryCheck(int* inside, int width, int height) {
  int col = blockIdx.x * BLOCK_SIDE + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIDE + threadIdx.y;
  int t = threadIdx.y * blockDim.x + threadIdx.x;
  int slot = (blockIdx.y * gridDim.x + blockIdx.x) * THREADS_PER_BLOCK + t;

  if (col < width && row < height) {
    inside[slot] = INSIDE;
  } else {
    inside[slot] = OUTSIDE;
  }
}

static int blocksToCover(int n) {
  return (n + BLOCK_SIDE - 1) / BLOCK_SIDE;
}

static Flags boundaryCheckOnDevice(const Picture& picture) {
  const dim3 grid(blocksToCover(picture.width), blocksToCover(picture.height));
  const int threads = grid.x * grid.y * THREADS_PER_BLOCK;
  Flags inside(threads, OUTSIDE);

  int* inside_d;
  CHECK(cudaMalloc(&inside_d, threads * sizeof(int)));
  launch(
      boundaryCheck, grid, dim3(BLOCK_SIDE, BLOCK_SIDE), inside_d, picture.width, picture.height);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(inside.data(), inside_d, threads * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(inside_d));
  return inside;
}

static Flags boundaryCheckOnCpu(const Picture& picture) {
  const int blocksX = blocksToCover(picture.width);
  const int blocksY = blocksToCover(picture.height);
  Flags inside;
  for (int by = 0; by < blocksY; ++by) {
    for (int bx = 0; bx < blocksX; ++bx) {
      for (int t = 0; t < THREADS_PER_BLOCK; ++t) {
        const int col = bx * BLOCK_SIDE + t % BLOCK_SIDE;
        const int row = by * BLOCK_SIDE + t / BLOCK_SIDE;
        inside.push_back(col < picture.width && row < picture.height ? INSIDE : OUTSIDE);
      }
    }
  }
  return inside;
}

static int lanesInside(const Flags& inside, int warp) {
  int lanes = 0;
  for (int i = warp * WARP_SIZE; i < (warp + 1) * WARP_SIZE; ++i) {
    lanes += inside[i];
  }
  return lanes;
}

struct WarpCounts {
  int fullyInside = 0;
  int divergent = 0;
  int fullyOutside = 0;
};

static WarpCounts countWarps(const Flags& inside, int firstWarp, int warps) {
  WarpCounts counts;
  for (int warp = firstWarp; warp < firstWarp + warps; ++warp) {
    const int lanes = lanesInside(inside, warp);
    if (lanes == WARP_SIZE) {
      ++counts.fullyInside;
    } else if (lanes == 0) {
      ++counts.fullyOutside;
    } else {
      ++counts.divergent;
    }
  }
  return counts;
}

static void printDivergentWarpsPerBlock(const Flags& inside, const Picture& picture) {
  const int blocksX = blocksToCover(picture.width);
  const int blocksY = blocksToCover(picture.height);
  std::printf("divergent warps of each block (of %d):\n", WARPS_PER_BLOCK);
  for (int by = 0; by < blocksY; ++by) {
    for (int bx = 0; bx < blocksX; ++bx) {
      const int block = by * blocksX + bx;
      std::printf(" %d", countWarps(inside, block * WARPS_PER_BLOCK, WARPS_PER_BLOCK).divergent);
    }
    std::printf("\n");
  }
}

static bool runPicture(const Picture& picture) {
  const int blocksX = blocksToCover(picture.width);
  const int blocksY = blocksToCover(picture.height);
  const int warps = blocksX * blocksY * WARPS_PER_BLOCK;

  const Flags inside = boundaryCheckOnDevice(picture);
  const Flags expected = boundaryCheckOnCpu(picture);
  const WarpCounts counts = countWarps(inside, 0, warps);

  std::printf("%d x %d picture: %d x %d blocks of %d x %d, %d warps\n",
              picture.width,
              picture.height,
              blocksX,
              blocksY,
              BLOCK_SIDE,
              BLOCK_SIDE,
              warps);
  printDivergentWarpsPerBlock(inside, picture);
  std::printf("fully inside %d, divergent %d (%.1f %%), fully outside %d\n\n",
              counts.fullyInside,
              counts.divergent,
              100.0 * counts.divergent / warps,
              counts.fullyOutside);
  return inside == expected;
}

int main() {
  std::printf("each warp is two rows of a %d x %d block; it diverges when some lanes are\n",
              BLOCK_SIDE,
              BLOCK_SIDE);
  std::printf("inside the picture and some outside\n\n");

  bool ok = true;
  for (const Picture& picture : PICTURES) {
    ok = runPicture(picture) && ok;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
