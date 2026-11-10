#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int ELEMENT_COUNT = BY_BACKEND((1 << 24) + 2, 4098);
const int FLOATS_PER_CHUNK = 4;
const int BLOCK_SIZE = 256;
const int TIMED_LAUNCHES = BY_BACKEND(10, 1);
const int VALUES_SHOWN = 8;
const double H100_PEAK_GB_PER_SECOND = 3350.0;
const double ARRAYS_MOVED = 3.0;

__global__ void vecAdd4(const float* x, const float* y, float* z, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n4 = N / FLOATS_PER_CHUNK;

  if (i < n4) {
    float4 a = reinterpret_cast<const float4*>(x)[i];
    float4 b = reinterpret_cast<const float4*>(y)[i];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    reinterpret_cast<float4*>(z)[i] = c;
  }

  if (i == n4) {
    for (int j = FLOATS_PER_CHUNK * n4; j < N; ++j) {
      z[j] = x[j] + y[j];
    }
  }
}

static void printFirstValues(const char* name, const std::vector<float>& values) {
  std::printf("%s[0..%d] =", name, VALUES_SHOWN - 1);
  for (int i = 0; i < VALUES_SHOWN; ++i) {
    std::printf("%4.0f", values[i]);
  }
  std::printf("\n");
}

int main() {
  std::vector<float> x(ELEMENT_COUNT);
  std::vector<float> y(ELEMENT_COUNT);
  std::vector<float> z(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    x[i] = static_cast<float>(i % 100);
    y[i] = static_cast<float>(10 * (i % 3));
  }

  const size_t bytes = ELEMENT_COUNT * sizeof(float);
  float* x_d;
  float* y_d;
  float* z_d;
  CHECK(cudaMalloc(&x_d, bytes));
  CHECK(cudaMalloc(&y_d, bytes));
  CHECK(cudaMalloc(&z_d, bytes));
  CHECK(cudaMemcpy(x_d, x.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(y_d, y.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = ELEMENT_COUNT / FLOATS_PER_CHUNK + 1;
  const dim3 grid(cdiv(threads, BLOCK_SIZE));
  const dim3 block(BLOCK_SIZE);
  launch(vecAdd4, grid, block, x_d, y_d, z_d, ELEMENT_COUNT);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  Timer timer;
  for (int launchIndex = 0; launchIndex < TIMED_LAUNCHES; ++launchIndex) {
    launch(vecAdd4, grid, block, x_d, y_d, z_d, ELEMENT_COUNT);
  }
  const double ms = timer.ms() / TIMED_LAUNCHES;
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(z.data(), z_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(x_d));
  CHECK(cudaFree(y_d));
  CHECK(cudaFree(z_d));

  printFirstValues("x", x);
  printFirstValues("y", y);
  printFirstValues("z", z);

  const double bytesMoved = ARRAYS_MOVED * ELEMENT_COUNT * sizeof(float);
  const double gbps = bytesMoved / (ms * 1e6);
  std::printf("backend: %s\n", CUDA_BACKEND);
  std::printf("N = %d floats: 2 reads + 1 write = %.0f B per launch\n", ELEMENT_COUNT, bytesMoved);
  std::printf("%.3f ms per launch, %.3g GB/s = %.3g %% of the H100's %.0f GB/s\n",
              ms,
              gbps,
              100.0 * gbps / H100_PEAK_GB_PER_SECOND,
              H100_PEAK_GB_PER_SECOND);

  bool pass = true;
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    pass = pass && z[i] == x[i] + y[i];
  }
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
