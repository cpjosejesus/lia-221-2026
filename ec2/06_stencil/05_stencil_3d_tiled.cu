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

#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

const int GRID_SIDE = 10;
const int VALUE_LEVELS = 10;
const float TOLERANCE = 1e-4f;

using Grid = std::vector<float>;

__global__ void stencil_tiled_kernel(const float* in, float* out, int N) {
  int i = (int)(blockIdx.z * OUT_TILE_DIM + threadIdx.z) - 1;
  int j = (int)(blockIdx.y * OUT_TILE_DIM + threadIdx.y) - 1;
  int k = (int)(blockIdx.x * OUT_TILE_DIM + threadIdx.x) - 1;
  int tz = (int)threadIdx.z;
  int ty = (int)threadIdx.y;
  int tx = (int)threadIdx.x;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    in_s[tz][ty][tx] = in[i * N * N + j * N + k];
  }
  __syncthreads();

  bool interiorPoint = i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1;
  bool innerThread = tz >= 1 && tz < IN_TILE_DIM - 1 && ty >= 1 && ty < IN_TILE_DIM - 1 &&
                     tx >= 1 && tx < IN_TILE_DIM - 1;
  if (interiorPoint && innerThread) {
    out[i * N * N + j * N + k] = c0 * in_s[tz][ty][tx] + c1 * in_s[tz][ty][tx - 1] +
                                 c2 * in_s[tz][ty][tx + 1] + c3 * in_s[tz][ty - 1][tx] +
                                 c4 * in_s[tz][ty + 1][tx] + c5 * in_s[tz - 1][ty][tx] +
                                 c6 * in_s[tz + 1][ty][tx];
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

static Grid smallGrid(int N) {
  Grid grid(N * N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        grid[i * N * N + j * N + k] = (float)((i * j * k + 3 * j * j + k * k) % VALUE_LEVELS);
      }
    }
  }
  return grid;
}

static void printPlane(const Grid& in, const Grid& out, int N, int i) {
  std::printf("plane i = %d of %d (in | out):\n", i, N);
  for (int j = 0; j < N; j++) {
    for (int k = 0; k < N; k++) {
      std::printf(" %4.1f", in[i * N * N + j * N + k]);
    }
    std::printf("  |");
    for (int k = 0; k < N; k++) {
      std::printf(" %4.1f", out[i * N * N + j * N + k]);
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

static bool sameValues(const Grid& expected, const Grid& actual) {
  for (size_t p = 0; p < expected.size(); p++) {
    if (std::fabs(expected[p] - actual[p]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main() {
  const Grid in = smallGrid(GRID_SIDE);

  float* in_d = copyToDevice(in);
  float* out_d = copyToDevice(in);

  const unsigned blocksPerSide = cdiv(GRID_SIDE, OUT_TILE_DIM);
  dim3 block(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
  dim3 grid(blocksPerSide, blocksPerSide, blocksPerSide);
  launch(stencil_tiled_kernel, grid, block, in_d, out_d, GRID_SIDE);
  CHECK(cudaGetLastError());

  Grid out(in.size());
  CHECK(cudaMemcpy(out.data(), out_d, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(in_d));
  CHECK(cudaFree(out_d));

  printPlane(in, out, GRID_SIDE, GRID_SIDE / 2);

  const bool passed = sameValues(stencilOnCpu(in, GRID_SIDE), out);
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
