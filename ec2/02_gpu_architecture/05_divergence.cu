#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int N = 80;
const int BLOCK_SIZE = 64;
const int BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
const int LAUNCHED = BLOCKS * BLOCK_SIZE;
const int WARP_SIZE = 32;
const int NOT_RUN = -1;
const int IF_PATH = 1;
const int ELSE_PATH = 0;

using Path = std::vector<int>;

__global__ void branches(int* returnsEarly, int* evenThread, int* evenWarp, int n) {
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;

  if (i >= n) {
    returnsEarly[i] = IF_PATH;
    return;
  }
  returnsEarly[i] = ELSE_PATH;

  if (t % 2 == 0) {
    evenThread[i] = IF_PATH;
  } else {
    evenThread[i] = ELSE_PATH;
  }

  if ((t / warpSize) % 2 == 0) {
    evenWarp[i] = IF_PATH;
  } else {
    evenWarp[i] = ELSE_PATH;
  }
}

static int* deviceCopy(const std::vector<int>& host) {
  int* device;
  CHECK(cudaMalloc(&device, host.size() * sizeof(int)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
  return device;
}

static void copyBack(std::vector<int>& host, int* device) {
  CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(device));
}

static int lanesWith(const Path& path, int warp, int value) {
  int lanes = 0;
  for (int i = warp * WARP_SIZE; i < (warp + 1) * WARP_SIZE; ++i) {
    lanes += path[i] == value ? 1 : 0;
  }
  return lanes;
}

static void printBranch(const Path& path, int warp) {
  const int taken = lanesWith(path, warp, IF_PATH);
  const int notTaken = lanesWith(path, warp, ELSE_PATH);
  const int passes = (taken > 0 ? 1 : 0) + (notTaken > 0 ? 1 : 0);
  std::printf("   %2d/%2d %d pass%s", taken, notTaken, passes, passes == 1 ? "  " : "es");
}

struct Run {
  Path returnsEarly = Path(LAUNCHED, NOT_RUN);
  Path evenThread = Path(LAUNCHED, NOT_RUN);
  Path evenWarp = Path(LAUNCHED, NOT_RUN);
};

static Run runBranches() {
  Run run;
  int* returnsEarly_d = deviceCopy(run.returnsEarly);
  int* evenThread_d = deviceCopy(run.evenThread);
  int* evenWarp_d = deviceCopy(run.evenWarp);

  launch(branches, dim3(BLOCKS), dim3(BLOCK_SIZE), returnsEarly_d, evenThread_d, evenWarp_d, N);
  CHECK(cudaGetLastError());

  copyBack(run.returnsEarly, returnsEarly_d);
  copyBack(run.evenThread, evenThread_d);
  copyBack(run.evenWarp, evenWarp_d);
  return run;
}

static void printWarps(const Run& run) {
  std::printf("warp  threads    i >= n            t %% 2 == 0         (t / warpSize) %% 2 == 0\n");
  for (int warp = 0; warp < LAUNCHED / WARP_SIZE; ++warp) {
    std::printf("%4d  %3d-%3d", warp, warp * WARP_SIZE, (warp + 1) * WARP_SIZE - 1);
    printBranch(run.returnsEarly, warp);
    printBranch(run.evenThread, warp);
    printBranch(run.evenWarp, warp);
    std::printf("\n");
  }
}

static int expectedPath(bool taken, bool runs) {
  if (!runs) {
    return NOT_RUN;
  }
  return taken ? IF_PATH : ELSE_PATH;
}

static bool matchesCpu(const Run& run) {
  for (int i = 0; i < LAUNCHED; ++i) {
    const int t = i % BLOCK_SIZE;
    const bool runs = i < N;

    if (run.returnsEarly[i] != expectedPath(!runs, true) ||
        run.evenThread[i] != expectedPath(t % 2 == 0, runs) ||
        run.evenWarp[i] != expectedPath((t / WARP_SIZE) % 2 == 0, runs)) {
      return false;
    }
  }
  return true;
}

int main() {
  const Run run = runBranches();

  std::printf("n = %d, %d blocks of %d threads\n", N, BLOCKS, BLOCK_SIZE);
  std::printf("each branch: lanes taking if / lanes taking else, passes the warp needs\n\n");
  printWarps(run);

  const bool ok = matchesCpu(run);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
