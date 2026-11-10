#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const unsigned N = 2048;
const unsigned THREADS_PER_BLOCK = 128;
const unsigned BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
const int WARP_SIZE = 32;
const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
const int WARPS = BLOCKS * WARPS_PER_BLOCK;
const int MAX_TRIPS = 4;
const int SHOWN = 8;

using Ints = std::vector<int>;

struct Paths {
  int* line4;
  int* line6;
  int* loopTrips;
};

__global__ void k(int* a, int* b, Paths paths) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < 48 || threadIdx.x >= 112) {
    b[i] = a[i] + 1;
    paths.line4[i] = 1;
  }
  if (i % 4 == 0) {
    a[i] = b[i] * 2;
    paths.line6[i] = 1;
  }
  for (unsigned int j = 0; j < 4 - (i % 4); ++j) {
    b[i] += j;
    paths.loopTrips[i] += 1;
  }
}

struct Run {
  Ints a = Ints(N);
  Ints b = Ints(N, 0);
  Ints line4 = Ints(N, 0);
  Ints line6 = Ints(N, 0);
  Ints loopTrips = Ints(N, 0);
};

static int* deviceCopy(const Ints& host) {
  int* device;
  CHECK(cudaMalloc(&device, host.size() * sizeof(int)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
  return device;
}

static void copyBack(Ints& host, int* device) {
  CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(device));
}

static void runOnDevice(Run& run) {
  int* a_d = deviceCopy(run.a);
  int* b_d = deviceCopy(run.b);
  const Paths paths_d{deviceCopy(run.line4), deviceCopy(run.line6), deviceCopy(run.loopTrips)};

  launch(k, dim3(BLOCKS), dim3(THREADS_PER_BLOCK), a_d, b_d, paths_d);
  CHECK(cudaGetLastError());

  copyBack(run.a, a_d);
  copyBack(run.b, b_d);
  copyBack(run.line4, paths_d.line4);
  copyBack(run.line6, paths_d.line6);
  copyBack(run.loopTrips, paths_d.loopTrips);
}

static void runOnCpu(Run& run) {
  for (unsigned i = 0; i < N; ++i) {
    const unsigned t = i % THREADS_PER_BLOCK;
    if (t < 48 || t >= 112) {
      run.b[i] = run.a[i] + 1;
      run.line4[i] = 1;
    }
    if (i % 4 == 0) {
      run.a[i] = run.b[i] * 2;
      run.line6[i] = 1;
    }
    for (unsigned j = 0; j < 4 - (i % 4); ++j) {
      run.b[i] += j;
      run.loopTrips[i] += 1;
    }
  }
}

static int lanesWhere(const Ints& values, int warp, int atLeast) {
  int lanes = 0;
  for (int i = warp * WARP_SIZE; i < (warp + 1) * WARP_SIZE; ++i) {
    lanes += values[i] >= atLeast ? 1 : 0;
  }
  return lanes;
}

static void printStatement(const char* name, const Ints& ran) {
  int active = 0;
  int divergent = 0;
  for (int warp = 0; warp < WARPS; ++warp) {
    const int lanes = lanesWhere(ran, warp, 1);
    active += lanes > 0 ? 1 : 0;
    divergent += lanes > 0 && lanes < WARP_SIZE ? 1 : 0;
  }
  std::printf(
      "%s: %d active warps, %d divergent; SIMD efficiency in block 0:", name, active, divergent);
  for (int warp = 0; warp < WARPS_PER_BLOCK; ++warp) {
    std::printf(" warp %d %3d %%", warp, 100 * lanesWhere(ran, warp, 1) / WARP_SIZE);
  }
  std::printf("\n");
}

static void printLoop(const Ints& loopTrips) {
  int uniform = 0;
  int diverging = 0;
  std::printf("(d) loop on line 7, lanes of warp 0 in each iteration:");
  for (int j = 0; j < MAX_TRIPS; ++j) {
    const int lanes = lanesWhere(loopTrips, 0, j + 1);
    std::printf(" j=%d: %d", j, lanes);
    uniform += lanes == WARP_SIZE ? 1 : 0;
    diverging += lanes > 0 && lanes < WARP_SIZE ? 1 : 0;
  }
  std::printf("\n    %d iteration without divergence, %d with\n", uniform, diverging);
}

static void printFirst(const char* label, const Ints& values) {
  std::printf("%-12s", label);
  for (int i = 0; i < SHOWN; ++i) {
    std::printf(" %3d", values[i]);
  }
  std::printf(" ...\n");
}

int main() {
  Run device;
  for (unsigned i = 0; i < N; ++i) {
    device.a[i] = static_cast<int>(i % 7);
  }
  Run cpu = device;
  printFirst("a before:", device.a);
  printFirst("b before:", device.b);

  runOnDevice(device);
  runOnCpu(cpu);
  printFirst("a after:", device.a);
  printFirst("b after:", device.b);

  std::printf("\n(a) N = %u, %u blocks of %u threads: %d warps per block, %d in the grid\n",
              N,
              BLOCKS,
              THREADS_PER_BLOCK,
              WARPS_PER_BLOCK,
              WARPS);
  printStatement("(b) line 4", device.line4);
  printStatement("(c) line 6", device.line6);
  printLoop(device.loopTrips);

  const bool ok = device.a == cpu.a && device.b == cpu.b && device.line4 == cpu.line4 &&
                  device.line6 == cpu.line6 && device.loopTrips == cpu.loopTrips;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
