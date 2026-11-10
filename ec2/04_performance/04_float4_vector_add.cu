#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int ELEMENT_COUNT = 18;
const int FLOATS_PER_CHUNK = 4;
const int BLOCK_SIZE = 8;

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

static void printVector(const char* name, const std::vector<float>& values) {
  std::printf("%s =", name);
  for (float value : values) {
    std::printf("%4.0f", value);
  }
  std::printf("\n");
}

int main() {
  std::vector<float> x(ELEMENT_COUNT);
  std::vector<float> y(ELEMENT_COUNT);
  std::vector<float> z(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    x[i] = static_cast<float>(i);
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
  launch(vecAdd4, dim3(cdiv(threads, BLOCK_SIZE)), dim3(BLOCK_SIZE), x_d, y_d, z_d, ELEMENT_COUNT);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(z.data(), z_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(x_d));
  CHECK(cudaFree(y_d));
  CHECK(cudaFree(z_d));

  printVector("x", x);
  printVector("y", y);
  printVector("z", z);
  std::printf("%d threads: %d add one float4 chunk each, thread %d adds the %d tail elements\n",
              threads,
              ELEMENT_COUNT / FLOATS_PER_CHUNK,
              ELEMENT_COUNT / FLOATS_PER_CHUNK,
              ELEMENT_COUNT % FLOATS_PER_CHUNK);

  bool pass = true;
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    pass = pass && z[i] == x[i] + y[i];
  }
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
