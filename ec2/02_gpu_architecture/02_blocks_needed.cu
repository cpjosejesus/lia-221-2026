#include "cuda_compat.h"

#include <cstdio>
#include <vector>

struct Launch {
  int n;
  int threadsPerBlock;
};

const Launch LAUNCHES[] = {{10, 4}, {16, 4}, {5, 8}, {8, 2048}};

__global__ void ownerKernel(int* ownerBlock, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    ownerBlock[i] = blockIdx.x;
  }
}

static int blocksToCover(int n, int threadsPerBlock) {
  return (n + threadsPerBlock - 1) / threadsPerBlock;
}

static std::vector<int> ownersOnDevice(int n, int threadsPerBlock) {
  const size_t bytes = n * sizeof(int);
  std::vector<int> ownerBlock(n, -1);

  int* ownerBlock_d;
  CHECK(cudaMalloc(&ownerBlock_d, bytes));

  launch(
      ownerKernel, dim3(blocksToCover(n, threadsPerBlock)), dim3(threadsPerBlock), ownerBlock_d, n);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(ownerBlock.data(), ownerBlock_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(ownerBlock_d));
  return ownerBlock;
}

static bool ownersAreCorrect(const std::vector<int>& ownerBlock, int threadsPerBlock) {
  for (int i = 0; i < (int)ownerBlock.size(); ++i) {
    if (ownerBlock[i] != i / threadsPerBlock) {
      return false;
    }
  }
  return true;
}

static bool runLaunch(const Launch& config, const cudaDeviceProp& p) {
  const int blocks = blocksToCover(config.n, config.threadsPerBlock);
  const int launched = blocks * config.threadsPerBlock;
  std::printf("n = %2d, %4d threads/block: %d blocks, %d threads, %d idle\n",
              config.n,
              config.threadsPerBlock,
              blocks,
              launched,
              launched - config.n);

  if (config.threadsPerBlock > p.maxThreadsPerBlock) {
    std::printf("   not launched: this device allows at most %d threads per block\n",
                p.maxThreadsPerBlock);
    return true;
  }

  const std::vector<int> ownerBlock = ownersOnDevice(config.n, config.threadsPerBlock);
  std::printf("   block of each element:");
  for (int owner : ownerBlock) {
    std::printf(" %d", owner);
  }
  std::printf("\n");
  return ownersAreCorrect(ownerBlock, config.threadsPerBlock);
}

int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("blocks needed = (n + threadsPerBlock - 1) / threadsPerBlock\n\n");

  bool ok = true;
  for (const Launch& config : LAUNCHES) {
    ok = runLaunch(config, p) && ok;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
