#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

const int GRID_SIDE = 6;
const int BLOCK_DIM = 4;
const int SWEEPS = 30;
const float HOT_EDGE_TEMPERATURE = 100.0f;
const float TOLERANCE = 1e-4f;

using Plate = std::vector<float>;

__global__ void stencil2d_basic(const float* in, float* out, int N) {
  int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
    out[i * N + j] =
        0.25f * (in[(i - 1) * N + j] + in[(i + 1) * N + j] + in[i * N + j - 1] + in[i * N + j + 1]);
  }
}

static void stencil2dOnCpu(const Plate& in, Plate& out, int N) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      out[i * N + j] = 0.25f * (in[(i - 1) * N + j] + in[(i + 1) * N + j] + in[i * N + j - 1] +
                                in[i * N + j + 1]);
    }
  }
}

static Plate sweepsOnCpu(const Plate& start, int N, int sweeps) {
  Plate current = start;
  Plate next = start;
  for (int s = 0; s < sweeps; s++) {
    stencil2dOnCpu(current, next, N);
    std::swap(current, next);
  }
  return current;
}

static Plate plateWithHotTopEdge(int N) {
  Plate plate(N * N, 0.0f);
  for (int j = 0; j < N; j++) {
    plate[j] = HOT_EDGE_TEMPERATURE;
  }
  return plate;
}

static void printPlate(const char* name, const Plate& plate, int N) {
  std::printf("%s:\n", name);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::printf(" %6.2f", plate[i * N + j]);
    }
    std::printf("\n");
  }
}

static float* copyToDevice(const std::vector<float>& host) {
  float* device;
  CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  return device;
}

static bool sameValues(const Plate& expected, const Plate& actual) {
  for (size_t p = 0; p < expected.size(); p++) {
    if (std::fabs(expected[p] - actual[p]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  const Plate in = plateWithHotTopEdge(GRID_SIDE);

  float* in_d = copyToDevice(in);
  float* out_d = copyToDevice(in);

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(cdiv(GRID_SIDE, BLOCK_DIM), cdiv(GRID_SIDE, BLOCK_DIM));
  for (int s = 0; s < SWEEPS; s++) {
    launch(stencil2d_basic, grid, block, in_d, out_d, GRID_SIDE);
    CHECK(cudaGetLastError());
    std::swap(in_d, out_d);
  }

  Plate out(in.size());
  CHECK(cudaMemcpy(out.data(), in_d, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(in_d));
  CHECK(cudaFree(out_d));

  const Plate expected = sweepsOnCpu(in, GRID_SIDE, SWEEPS);

  printPlate("in (top edge held at 100, other edges at 0)", in, GRID_SIDE);
  std::printf("after %d sweeps, each one reading the previous one:\n", SWEEPS);
  printPlate("out", out, GRID_SIDE);

  const bool passed = sameValues(expected, out);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
