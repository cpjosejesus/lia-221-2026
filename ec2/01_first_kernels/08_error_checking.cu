#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int N = 8;
const int TOO_MANY_THREADS_PER_BLOCK = 2048;
const int THREADS_PER_BLOCK = 256;
const float NOT_COMPUTED = -1.0f;

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

struct DeviceVectors {
  float* A_d;
  float* B_d;
  float* C_d;
};

static DeviceVectors copyInputsToDevice(const std::vector<float>& A, const std::vector<float>& B) {
  const size_t bytes = A.size() * sizeof(float);
  DeviceVectors device{};
  CHECK(cudaMalloc(&device.A_d, bytes));
  CHECK(cudaMalloc(&device.B_d, bytes));
  CHECK(cudaMalloc(&device.C_d, bytes));
  CHECK(cudaMemcpy(device.A_d, A.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(device.B_d, B.data(), bytes, cudaMemcpyHostToDevice));
  return device;
}

static void fillOutputWithMarker(const DeviceVectors& device) {
  const std::vector<float> marker(N, NOT_COMPUTED);
  CHECK(cudaMemcpy(device.C_d, marker.data(), N * sizeof(float), cudaMemcpyHostToDevice));
}

static std::vector<float> copyOutputToHost(const DeviceVectors& device) {
  std::vector<float> C(N);
  CHECK(cudaMemcpy(C.data(), device.C_d, N * sizeof(float), cudaMemcpyDeviceToHost));
  return C;
}

static void freeDevice(const DeviceVectors& device) {
  CHECK(cudaFree(device.A_d));
  CHECK(cudaFree(device.B_d));
  CHECK(cudaFree(device.C_d));
}

static void printVector(const char* label, const std::vector<float>& v) {
  std::printf("%s =", label);
  for (float x : v) {
    std::printf(" %5.1f", x);
  }
  std::printf("\n");
}

static void launchWithoutChecking(const DeviceVectors& device) {
  std::printf("1) launch with %d threads per block and do not check:\n",
              TOO_MANY_THREADS_PER_BLOCK);
  fillOutputWithMarker(device);
  launch(vecAddKernel,
         dim3(1),
         dim3(TOO_MANY_THREADS_PER_BLOCK),
         device.A_d,
         device.B_d,
         device.C_d,
         N);
  printVector("   C", copyOutputToHost(device));
  std::printf("   the kernel never ran (C still holds the marker %.1f) and nothing said so\n\n",
              NOT_COMPUTED);
}

static void launchAndAskForTheError(const DeviceVectors& device) {
  std::printf("2) the same launch, then ask cudaGetLastError():\n");
  fillOutputWithMarker(device);
  launch(vecAddKernel,
         dim3(1),
         dim3(TOO_MANY_THREADS_PER_BLOCK),
         device.A_d,
         device.B_d,
         device.C_d,
         N);
  const cudaError_t launchError = cudaGetLastError();
  std::printf("   cudaGetLastError() = %d: \"%s\"\n",
              static_cast<int>(launchError),
              cudaGetErrorString(launchError));
  std::printf(
      "   CHECK(cudaGetLastError()) would print that message with file and line, and stop\n\n");
}

static std::vector<float> launchCorrectlyWithCheck(const DeviceVectors& device) {
  const int blocks = cdiv(N, THREADS_PER_BLOCK);
  std::printf("3) a valid launch, %d block of %d threads, checked with CHECK:\n",
              blocks,
              THREADS_PER_BLOCK);
  fillOutputWithMarker(device);
  launch(
      vecAddKernel, dim3(blocks), dim3(THREADS_PER_BLOCK), device.A_d, device.B_d, device.C_d, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<float> C = copyOutputToHost(device);
  printVector("   C", C);
  return C;
}

int main() {
  std::vector<float> A(N);
  std::vector<float> B(N);
  std::vector<float> expected(N);
  for (int i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i);
    B[i] = 10.0f * (i + 1);
    expected[i] = A[i] + B[i];
  }

  printVector("A", A);
  printVector("B", B);
  std::printf("\n");

  const DeviceVectors device = copyInputsToDevice(A, B);
  launchWithoutChecking(device);
  launchAndAskForTheError(device);
  const std::vector<float> C = launchCorrectlyWithCheck(device);
  freeDevice(device);

  const bool ok = C == expected;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
