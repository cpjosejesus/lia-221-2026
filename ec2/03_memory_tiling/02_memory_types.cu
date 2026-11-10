#include "cuda_compat.h"

#include <cmath>
#include <cstdio>
#include <vector>

const int BLOCK_SIZE = 8;
const int ELEMENT_COUNT = 16;
const int COEFFICIENT_COUNT = 2;
const float TOLERANCE = 1e-4f;

__constant__ float coeff[COEFFICIENT_COUNT];
__device__ int negCount = 0;

__global__ void kernel(const float* in, float* out, int n) {
  __shared__ float tile[BLOCK_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;

  tile[threadIdx.x] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  int left = threadIdx.x > 0 ? threadIdx.x - 1 : 0;
  acc = coeff[0] * tile[threadIdx.x] + coeff[1] * tile[left];

  if (i < n) {
    out[i] = acc;
    if (acc < 0) {
      atomicAdd(&negCount, 1);
    }
  }
}

__global__ void readNegCountKernel(int* result) {
  *result = negCount;
}

static std::vector<float> makeInput() {
  std::vector<float> in(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    in[i] = static_cast<float>(i % 5 - 2);
  }
  return in;
}

static std::vector<float> applyOnCpu(const std::vector<float>& in, const float* coefficients) {
  std::vector<float> out(in.size());
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    const int left = i % BLOCK_SIZE > 0 ? i - 1 : i;
    out[i] = coefficients[0] * in[i] + coefficients[1] * in[left];
  }
  return out;
}

static int countNegatives(const std::vector<float>& values) {
  int count = 0;
  for (float value : values) {
    if (value < 0) {
      ++count;
    }
  }
  return count;
}

static void printVector(const char* name, const std::vector<float>& values) {
  std::printf("%-4s =", name);
  for (float value : values) {
    std::printf("%5.1f", value);
  }
  std::printf("\n");
}

int main() {
  const float coefficients[COEFFICIENT_COUNT] = {1.0f, -0.5f};
  const std::vector<float> in = makeInput();
  std::vector<float> out(ELEMENT_COUNT);
  const size_t bytes = ELEMENT_COUNT * sizeof(float);

  float* in_d;
  float* out_d;
  int* negCount_d;
  CHECK(cudaMalloc(&in_d, bytes));
  CHECK(cudaMalloc(&out_d, bytes));
  CHECK(cudaMalloc(&negCount_d, sizeof(int)));
  CHECK(cudaMemcpy(in_d, in.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(coeff, coefficients, sizeof coefficients));

  launch(
      kernel, dim3(cdiv(ELEMENT_COUNT, BLOCK_SIZE)), dim3(BLOCK_SIZE), in_d, out_d, ELEMENT_COUNT);
  CHECK(cudaGetLastError());
  launch(readNegCountKernel, dim3(1), dim3(1), negCount_d);
  CHECK(cudaGetLastError());

  int negatives = 0;
  CHECK(cudaMemcpy(out.data(), out_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&negatives, negCount_d, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(in_d));
  CHECK(cudaFree(out_d));
  CHECK(cudaFree(negCount_d));

  const std::vector<float> expected = applyOnCpu(in, coefficients);
  printVector("in", in);
  printVector("out", out);
  std::printf("coeff = {%.1f, %.1f} in constant memory\n", coefficients[0], coefficients[1]);
  std::printf("negCount read by a second kernel = %d\n", negatives);

  bool pass = negatives == countNegatives(expected);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    pass = pass && std::fabs(out[i] - expected[i]) < TOLERANCE;
  }
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
