#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

constexpr float DIFFUSION = 0.1f;
constexpr float c0 = 1.0f - 6.0f * DIFFUSION;
constexpr float c1 = DIFFUSION;
constexpr float c2 = DIFFUSION;
constexpr float c3 = DIFFUSION;
constexpr float c4 = DIFFUSION;
constexpr float c5 = DIFFUSION;
constexpr float c6 = DIFFUSION;

const int GRID_SIDE = 5;
const int BLOCK_DIM = 4;
const float HOT_POINT_TEMPERATURE = 100.0f;
const float TOLERANCE = 1e-4f;

using Grid = std::vector<float>;

__global__ void stencil_kernel(const float* in, float* out, unsigned int N) {
  unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * N * N + j * N + k] =
        c0 * in[i * N * N + j * N + k] + c1 * in[i * N * N + j * N + (k - 1)] +
        c2 * in[i * N * N + j * N + (k + 1)] + c3 * in[i * N * N + (j - 1) * N + k] +
        c4 * in[i * N * N + (j + 1) * N + k] + c5 * in[(i - 1) * N * N + j * N + k] +
        c6 * in[(i + 1) * N * N + j * N + k];
  }
}

static Grid stencilOnCpu(const Grid& in, int N) {
  Grid out = in;
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      for (int k = 1; k < N - 1; k++) {
        int p = i * N * N + j * N + k;
        out[p] = c0 * in[p] + c1 * in[p - 1] + c2 * in[p + 1] + c3 * in[p - N] + c4 * in[p + N] +
                 c5 * in[p - N * N] + c6 * in[p + N * N];
      }
    }
  }
  return out;
}

static Grid gridWithHotCentre(int N) {
  Grid grid(N * N * N, 0.0f);
  int centre = N / 2;
  grid[centre * N * N + centre * N + centre] = HOT_POINT_TEMPERATURE;
  return grid;
}

static void printPlanes(const Grid& in, const Grid& out, int N) {
  for (int i = 0; i < N; i++) {
    std::printf("plane i = %d (in | out)\n", i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        std::printf(" %4.0f", in[i * N * N + j * N + k]);
      }
      std::printf("  |");
      for (int k = 0; k < N; k++) {
        std::printf(" %4.0f", out[i * N * N + j * N + k]);
      }
      std::printf("\n");
    }
  }
}

static float* copyToDevice(const std::vector<float>& host) {
  float* device;
  CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  return device;
}

static bool sameValues(const Grid& expected, const Grid& actual) {
  for (size_t p = 0; p < expected.size(); p++) {
    if (std::fabs(expected[p] - actual[p]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  const Grid in = gridWithHotCentre(GRID_SIDE);

  float* in_d = copyToDevice(in);
  float* out_d = copyToDevice(in);

  const unsigned blocksPerSide = cdiv(GRID_SIDE, BLOCK_DIM);
  dim3 block(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
  dim3 grid(blocksPerSide, blocksPerSide, blocksPerSide);
  launch(stencil_kernel, grid, block, in_d, out_d, GRID_SIDE);
  CHECK(cudaGetLastError());

  Grid out(in.size());
  CHECK(cudaMemcpy(out.data(), out_d, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(in_d));
  CHECK(cudaFree(out_d));

  printPlanes(in, out, GRID_SIDE);

  const bool passed = sameValues(stencilOnCpu(in, GRID_SIDE), out);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
