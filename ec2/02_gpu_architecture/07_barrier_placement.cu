#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int THREADS = 8;

using Values = std::vector<float>;

__global__ void step(float* x, float* y) {
  int t = threadIdx.x;
  if (t % 2 == 0) {
    x[t] *= 2.0f;
  } else {
    x[t] += 1.0f;
  }
  __syncthreads();
  y[t] = x[t] - x[t ^ 1];
}

struct StepResult {
  Values x;
  Values y;
};

static StepResult stepOnDevice(const Values& x) {
  const size_t bytes = THREADS * sizeof(float);
  StepResult result{Values(THREADS), Values(THREADS)};

  float* x_d;
  float* y_d;
  CHECK(cudaMalloc(&x_d, bytes));
  CHECK(cudaMalloc(&y_d, bytes));
  CHECK(cudaMemcpy(x_d, x.data(), bytes, cudaMemcpyHostToDevice));

  launch(step, dim3(1), dim3(THREADS), x_d, y_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(result.x.data(), x_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(result.y.data(), y_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(x_d));
  CHECK(cudaFree(y_d));
  return result;
}

static StepResult stepOnCpu(const Values& input) {
  StepResult result{input, Values(THREADS)};
  for (int t = 0; t < THREADS; ++t) {
    result.x[t] = t % 2 == 0 ? input[t] * 2.0f : input[t] + 1.0f;
  }
  for (int t = 0; t < THREADS; ++t) {
    result.y[t] = result.x[t] - result.x[t ^ 1];
  }
  return result;
}

static void printValues(const char* label, const Values& values) {
  std::printf("%-30s", label);
  for (float v : values) {
    std::printf(" %5.1f", v);
  }
  std::printf("\n");
}

int main() {
  Values x(THREADS);
  for (int t = 0; t < THREADS; ++t) {
    x[t] = static_cast<float>(10 * (t + 1));
  }

  std::printf("1 block of %d threads; even threads double x[t], odd threads add 1\n", THREADS);
  std::printf("one __syncthreads() after the if/else, reached by all %d threads\n\n", THREADS);
  std::printf("%-30s", "thread t:");
  for (int t = 0; t < THREADS; ++t) {
    std::printf(" %5d", t);
  }
  std::printf("\n");
  printValues("x before:", x);

  const StepResult result = stepOnDevice(x);
  const StepResult expected = stepOnCpu(x);
  printValues("x after the branch:", result.x);
  printValues("y[t] = x[t] - x[t ^ 1]:", result.y);
  std::printf("\nthread 0 reads x[1] = %.1f, written by thread 1 before the barrier\n",
              result.x[1]);
  std::printf(
      "the wrong version (one barrier in each branch) is not run: its behaviour is undefined\n");

  const bool ok = result.x == expected.x && result.y == expected.y;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
