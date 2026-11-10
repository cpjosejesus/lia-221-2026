#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int BLOCKS = 2;
const int BLOCK_X = 16;
const int BLOCK_Y = 16;
const int THREADS_PER_BLOCK = BLOCK_X * BLOCK_Y;
const int TOTAL_THREADS = BLOCKS * THREADS_PER_BLOCK;

struct Sample {
  int x;
  int y;
};

const Sample SAMPLES[] = {{0, 0}, {15, 1}, {0, 2}, {5, 3}, {15, 15}};

__global__ void whoAmI(int* warpOf, int* laneOf) {
  int t = threadIdx.y * blockDim.x + threadIdx.x;
  int warp = t / warpSize;
  int lane = t % warpSize;

  int g = blockIdx.x * (blockDim.x * blockDim.y) + t;
  warpOf[g] = warp;
  laneOf[g] = lane;
}

static void runWhoAmI(std::vector<int>& warpOf, std::vector<int>& laneOf) {
  const size_t bytes = TOTAL_THREADS * sizeof(int);

  int* warp_d;
  int* lane_d;
  CHECK(cudaMalloc(&warp_d, bytes));
  CHECK(cudaMalloc(&lane_d, bytes));

  launch(whoAmI, dim3(BLOCKS), dim3(BLOCK_X, BLOCK_Y), warp_d, lane_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(warpOf.data(), warp_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(laneOf.data(), lane_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(warp_d));
  CHECK(cudaFree(lane_d));
}

static void printWarpMap(const std::vector<int>& warpOf) {
  std::printf("warp of each thread of block 0 (row = threadIdx.y, column = threadIdx.x):\n");
  for (int y = 0; y < BLOCK_Y; ++y) {
    std::printf("  y=%2d:", y);
    for (int x = 0; x < BLOCK_X; ++x) {
      std::printf(" %d", warpOf[y * BLOCK_X + x]);
    }
    std::printf("\n");
  }
}

static void printSamples(const std::vector<int>& warpOf, const std::vector<int>& laneOf) {
  std::printf("\n  (x, y)      t  warp  lane\n");
  for (const Sample& s : SAMPLES) {
    const int t = s.y * BLOCK_X + s.x;
    std::printf("  (%2d, %2d)  %3d  %4d  %4d\n", s.x, s.y, t, warpOf[t], laneOf[t]);
  }
}

static bool matchesCpu(const std::vector<int>& warpOf,
                       const std::vector<int>& laneOf,
                       int warpSizeOfDevice) {
  for (int g = 0; g < TOTAL_THREADS; ++g) {
    const int t = g % THREADS_PER_BLOCK;
    if (warpOf[g] != t / warpSizeOfDevice || laneOf[g] != t % warpSizeOfDevice) {
      return false;
    }
  }
  return true;
}

int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));

  std::vector<int> warpOf(TOTAL_THREADS, -1);
  std::vector<int> laneOf(TOTAL_THREADS, -1);
  runWhoAmI(warpOf, laneOf);

  std::printf("%d blocks of %d x %d threads, warps of %d threads\n\n",
              BLOCKS,
              BLOCK_X,
              BLOCK_Y,
              p.warpSize);
  printWarpMap(warpOf);
  printSamples(warpOf, laneOf);

  const bool ok = matchesCpu(warpOf, laneOf, p.warpSize);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
