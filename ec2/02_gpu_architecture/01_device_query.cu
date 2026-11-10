#include "cuda_compat.h"

#include <cstdio>

__global__ void readWarpSize(int* deviceWarpSize) {
  *deviceWarpSize = warpSize;
}

static void printProperties(int device, const cudaDeviceProp& p) {
  std::printf("%d: %s, compute capability %d.%d\n", device, p.name, p.major, p.minor);
  std::printf("   %d SMs, %d threads/block, warp %d\n",
              p.multiProcessorCount,
              p.maxThreadsPerBlock,
              p.warpSize);
  std::printf("   block dimensions up to %d x %d x %d\n",
              p.maxThreadsDim[0],
              p.maxThreadsDim[1],
              p.maxThreadsDim[2]);
  std::printf("   grid dimensions up to %d x %d x %d\n",
              p.maxGridSize[0],
              p.maxGridSize[1],
              p.maxGridSize[2]);
  std::printf("   per SM: %d threads, %d blocks, %d registers, %zu KiB shared memory\n",
              p.maxThreadsPerMultiProcessor,
              p.maxBlocksPerMultiProcessor,
              p.regsPerMultiprocessor,
              p.sharedMemPerMultiprocessor / 1024);
  std::printf("   per block: %d registers, %zu KiB shared memory\n",
              p.regsPerBlock,
              p.sharedMemPerBlock / 1024);
}

static int warpSizeSeenByAKernel() {
  int warp = 0;
  int* warp_d;
  CHECK(cudaMalloc(&warp_d, sizeof(int)));

  launch(readWarpSize, dim3(1), dim3(1), warp_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(&warp, warp_d, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(warp_d));
  return warp;
}

static bool limitsMakeSense(const cudaDeviceProp& p) {
  const bool blockIsWholeWarps = p.maxThreadsPerBlock % p.warpSize == 0;
  const bool smHoldsABlock = p.maxThreadsPerMultiProcessor >= p.maxThreadsPerBlock;
  return p.multiProcessorCount > 0 && blockIsWholeWarps && smHoldsABlock;
}

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  std::printf("%d device(s)\n", count);

  bool ok = count > 0;
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, device));
    printProperties(device, p);
    ok = ok && limitsMakeSense(p);
  }

  if (count > 0) {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    const int warp = warpSizeSeenByAKernel();
    std::printf("warpSize inside a kernel: %d, property warpSize: %d\n", warp, p.warpSize);
    ok = ok && warp == p.warpSize;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
