#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <string>

struct SmLimits {
  int threads;
  int blocks;
  int registers;
  int warpSize;
};

struct Kernel {
  int threadsPerBlock;
  int registersPerThread;
  int expectedBlocks;
  const char* expectedLimit;
};

struct Occupancy {
  int blocks;
  int percent;
  std::string limit;
};

const SmLimits H100 = {2048, 32, 65536, 32};
const int FEW_REGISTERS = 16;

const Kernel SLIDE_TABLE[] = {
    {1024, FEW_REGISTERS, 2, "threads"},
    {512, FEW_REGISTERS, 4, "threads"},
    {256, FEW_REGISTERS, 8, "threads"},
    {128, FEW_REGISTERS, 16, "threads"},
    {64, FEW_REGISTERS, 32, "threads and blocks"},
    {32, FEW_REGISTERS, 32, "blocks"},
    {768, FEW_REGISTERS, 2, "threads"},
    {256, 64, 4, "registers"},
    {512, 31, 4, "threads and registers"},
    {512, 33, 3, "registers"},
};

static void addLimit(std::string& limits, const char* name) {
  if (!limits.empty()) {
    limits += " and ";
  }
  limits += name;
}

static Occupancy occupancy(const SmLimits& sm, const Kernel& kernel) {
  const int byThreads = sm.threads / kernel.threadsPerBlock;
  const int byBlocks = sm.blocks;
  const int byRegisters = sm.registers / (kernel.registersPerThread * kernel.threadsPerBlock);

  Occupancy result;
  result.blocks = std::min(byThreads, std::min(byBlocks, byRegisters));

  const int warpsPerBlock = (kernel.threadsPerBlock + sm.warpSize - 1) / sm.warpSize;
  const int maxWarps = sm.threads / sm.warpSize;
  result.percent = 100 * result.blocks * warpsPerBlock / maxWarps;

  if (byThreads == result.blocks) {
    addLimit(result.limit, "threads");
  }
  if (byBlocks == result.blocks) {
    addLimit(result.limit, "blocks");
  }
  if (byRegisters == result.blocks) {
    addLimit(result.limit, "registers");
  }
  return result;
}

static bool printTable(const SmLimits& sm) {
  std::printf(
      "SM limits: %d threads, %d blocks, %d registers\n", sm.threads, sm.blocks, sm.registers);
  std::printf("threads/block  registers/thread  blocks/SM  occupancy  limited by\n");

  bool matchesSlide = true;
  for (const Kernel& kernel : SLIDE_TABLE) {
    const Occupancy result = occupancy(sm, kernel);
    std::printf("%13d  %16d  %9d  %8d%%  %s\n",
                kernel.threadsPerBlock,
                kernel.registersPerThread,
                result.blocks,
                result.percent,
                result.limit.c_str());
    matchesSlide = matchesSlide && result.blocks == kernel.expectedBlocks &&
                   result.limit == kernel.expectedLimit;
  }
  std::printf("\n");
  return matchesSlide;
}

int main() {
  std::printf("H100, as on the slides:\n");
  const bool ok = printTable(H100);

  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  const SmLimits thisDevice = {p.maxThreadsPerMultiProcessor,
                               p.maxBlocksPerMultiProcessor,
                               p.regsPerMultiprocessor,
                               p.warpSize};
  std::printf("this device, %s:\n", p.name);
  printTable(thisDevice);

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
