#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int WARP_SIZE = 32;
const int MAX_STRIDE = 32;
const int MAX_OFFSET = 8;
const int ARRAY_SIZE = WARP_SIZE * MAX_STRIDE + MAX_OFFSET;
const int SEGMENT_BYTES = 128;
const int SECTOR_BYTES = 32;
const int THREADS_TO_PRINT = 6;

struct AccessPattern {
  const char* name;
  int offset;
  int stride;
};

const AccessPattern PATTERNS[] = {
    {"a[t]", 0, 1},
    {"a[8 + t]", 8, 1},
    {"a[2t]", 0, 2},
    {"a[4t]", 0, 4},
    {"a[32t]", 0, 32},
};

__global__ void gatherKernel(const float* a, float* out, int offset, int stride) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  out[t] = a[offset + t * stride];
}

static std::vector<float> gatherOnDevice(const float* a_d, const AccessPattern& pattern) {
  std::vector<float> out(WARP_SIZE);
  float* out_d;
  CHECK(cudaMalloc(&out_d, WARP_SIZE * sizeof(float)));

  launch(gatherKernel, dim3(1), dim3(WARP_SIZE), a_d, out_d, pattern.offset, pattern.stride);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(out.data(), out_d, WARP_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(out_d));
  return out;
}

static int countBlocksTouched(const std::vector<float>& indicesRead, int blockBytes) {
  int blocks = 0;
  int previousBlock = -1;
  for (float index : indicesRead) {
    const int block = static_cast<int>(index) * static_cast<int>(sizeof(float)) / blockBytes;
    if (block != previousBlock) {
      ++blocks;
      previousBlock = block;
    }
  }
  return blocks;
}

static bool readTheRightElements(const std::vector<float>& out, const AccessPattern& pattern) {
  for (int t = 0; t < WARP_SIZE; ++t) {
    if (out[t] != static_cast<float>(pattern.offset + t * pattern.stride)) {
      return false;
    }
  }
  return true;
}

static void printPattern(const std::vector<float>& out, const AccessPattern& pattern) {
  const int segments = countBlocksTouched(out, SEGMENT_BYTES);
  const int sectors = countBlocksTouched(out, SECTOR_BYTES);
  const int usefulBytes = WARP_SIZE * static_cast<int>(sizeof(float));
  const float usedPercent = 100.0f * usefulBytes / (sectors * SECTOR_BYTES);

  std::printf("%-9s reads", pattern.name);
  for (int t = 0; t < THREADS_TO_PRINT; ++t) {
    std::printf(" %4.0f", out[t]);
  }
  std::printf(" ... %4.0f | %2d segments of 128 B, %2d sectors of 32 B, %5.1f %% used\n",
              out[WARP_SIZE - 1],
              segments,
              sectors,
              usedPercent);
}

int main() {
  std::vector<float> a(ARRAY_SIZE);
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    a[i] = static_cast<float>(i);
  }
  std::printf("a[i] = i for i = 0 .. %d; one warp of %d threads, thread t reads:\n",
              ARRAY_SIZE - 1,
              WARP_SIZE);

  float* a_d;
  CHECK(cudaMalloc(&a_d, ARRAY_SIZE * sizeof(float)));
  CHECK(cudaMemcpy(a_d, a.data(), ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  bool pass = true;
  for (const AccessPattern& pattern : PATTERNS) {
    const std::vector<float> out = gatherOnDevice(a_d, pattern);
    printPattern(out, pattern);
    pass = readTheRightElements(out, pattern) && pass;
  }

  CHECK(cudaFree(a_d));
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
