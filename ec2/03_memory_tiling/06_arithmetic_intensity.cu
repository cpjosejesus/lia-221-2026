#include "cuda_compat.h"

#include <algorithm>
#include <cstdio>
#include <vector>

const int ELEMENT_COUNT = 8;
const int BLOCK_SIZE = 4;
const int FLOPS_PER_ELEMENT = 1;
const int LOADS_PER_ELEMENT = 2;
const int STORES_PER_ELEMENT = 1;
const double PEAK_TFLOPS = 66.9;
const double PEAK_TB_PER_SECOND = 3.35;
const double MEASURED_ELEMENTS = 1e9;
const double MEASURED_MILLISECONDS = 4.0;
const double MATRIX_WIDTH = 1024.0;
const double NAIVE_MATMUL_FLOP_PER_BYTE = 2.0 / 8.0;
const double TILE_WIDTH = 32.0;

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

static std::vector<float> addOnDevice(const std::vector<float>& A, const std::vector<float>& B) {
  const size_t bytes = A.size() * sizeof(float);
  std::vector<float> C(A.size());

  float* A_d;
  float* B_d;
  float* C_d;
  CHECK(cudaMalloc(&A_d, bytes));
  CHECK(cudaMalloc(&B_d, bytes));
  CHECK(cudaMalloc(&C_d, bytes));
  CHECK(cudaMemcpy(A_d, A.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B.data(), bytes, cudaMemcpyHostToDevice));

  launch(vecAddKernel,
         dim3(cdiv(ELEMENT_COUNT, BLOCK_SIZE)),
         dim3(BLOCK_SIZE),
         A_d,
         B_d,
         C_d,
         ELEMENT_COUNT);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(C.data(), C_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  return C;
}

static void printVector(const char* name, const std::vector<float>& values) {
  std::printf("%s =", name);
  for (float value : values) {
    std::printf("%4.0f", value);
  }
  std::printf("\n");
}

static void printRooflineRow(const char* kernel, double flopPerByte, double threshold) {
  const double attainableTflops = std::min(PEAK_TFLOPS, PEAK_TB_PER_SECOND * flopPerByte);
  std::printf("%-26s %7.3f FLOP/B  %-14s at most %5.2f TFLOPS = %5.1f %% of the peak\n",
              kernel,
              flopPerByte,
              flopPerByte < threshold ? "memory-bound" : "compute-bound",
              attainableTflops,
              100.0 * attainableTflops / PEAK_TFLOPS);
}

static void printSpeedOfLight(int bytesPerElement) {
  const double bytesMoved = MEASURED_ELEMENTS * bytesPerElement;
  const double achievedTbPerSecond = bytesMoved / (MEASURED_MILLISECONDS * 1e-3) / 1e12;
  std::printf(
      "speed of light: %.0e elements in %.1f ms move %.0f GB: %.2f TB/s = %.1f %% of %.2f TB/s\n",
      MEASURED_ELEMENTS,
      MEASURED_MILLISECONDS,
      bytesMoved / 1e9,
      achievedTbPerSecond,
      100.0 * achievedTbPerSecond / PEAK_TB_PER_SECOND,
      PEAK_TB_PER_SECOND);
}

int main() {
  std::vector<float> A(ELEMENT_COUNT);
  std::vector<float> B(ELEMENT_COUNT);
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    A[i] = static_cast<float>(i);
    B[i] = static_cast<float>(10 * (i % 3));
  }
  const std::vector<float> C = addOnDevice(A, B);

  printVector("A", A);
  printVector("B", B);
  printVector("C = A + B", C);

  const int bytesPerElement =
      (LOADS_PER_ELEMENT + STORES_PER_ELEMENT) * static_cast<int>(sizeof(float));
  const double vectorAddFlopPerByte = static_cast<double>(FLOPS_PER_ELEMENT) / bytesPerElement;
  const double threshold = PEAK_TFLOPS / PEAK_TB_PER_SECOND;
  std::printf("each thread: %d FLOP, %d loads + %d store = %d B\n",
              FLOPS_PER_ELEMENT,
              LOADS_PER_ELEMENT,
              STORES_PER_ELEMENT,
              bytesPerElement);
  std::printf("threshold = %.1f TFLOPS / %.2f TB/s = %.2f FLOP/B\n",
              PEAK_TFLOPS,
              PEAK_TB_PER_SECOND,
              threshold);

  printRooflineRow("vector add", vectorAddFlopPerByte, threshold);
  printRooflineRow("naive matmul", NAIVE_MATMUL_FLOP_PER_BYTE, threshold);
  printRooflineRow("tiled matmul, 32 x 32", NAIVE_MATMUL_FLOP_PER_BYTE * TILE_WIDTH, threshold);
  const double matmulFlops = 2.0 * MATRIX_WIDTH * MATRIX_WIDTH * MATRIX_WIDTH;
  const double matmulBytesLoadedOnce = 3.0 * MATRIX_WIDTH * MATRIX_WIDTH * sizeof(float);
  printRooflineRow("matmul in principle, 1024", matmulFlops / matmulBytesLoadedOnce, threshold);
  printSpeedOfLight(bytesPerElement);

  bool pass = true;
  for (int i = 0; i < ELEMENT_COUNT; ++i) {
    pass = pass && C[i] == A[i] + B[i];
  }
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
