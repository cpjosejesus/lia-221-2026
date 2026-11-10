#include "cuda_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

constexpr unsigned int N = 16;
const float INPUT[N] = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 8, 1, 9, 2, 4, 6};

constexpr float SUM_IDENTITY = 0.0f;
const float MAX_IDENTITY = -INFINITY;

float sum_sequential(const float* input, unsigned int n) {
  float sum = 0.0f;
  for (unsigned int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}

float add(float a, float b) {
  return a + b;
}

float maximum(float a, float b) {
  return std::fmax(a, b);
}

template <class Operator>
float reduce_sequential(const float* input, unsigned int n, float identity, Operator op) {
  float acc = identity;
  for (unsigned int i = 0; i < n; ++i) {
    acc = op(acc, input[i]);
  }
  return acc;
}

static void printValues(const char* label, const float* values, unsigned int n) {
  std::printf("%-12s", label);
  for (unsigned int i = 0; i < n; ++i) {
    std::printf(" %3.0f", values[i]);
  }
  std::printf("\n");
}

template <class Operator>
static void printRunningValues(const char* label, float identity, Operator op) {
  float running[N];
  for (unsigned int n = 1; n <= N; ++n) {
    running[n - 1] = reduce_sequential(INPUT, n, identity, op);
  }
  printValues(label, running, N);
}

int main() {
  printValues("input", INPUT, N);
  std::printf("\nthe accumulator after each element (one step per element, %u steps):\n", N);
  printRunningValues("sum (0, +)", SUM_IDENTITY, add);
  printRunningValues("max (-inf)", MAX_IDENTITY, maximum);

  const float sum = sum_sequential(INPUT, N);
  const float generalSum = reduce_sequential(INPUT, N, SUM_IDENTITY, add);
  const float generalMax = reduce_sequential(INPUT, N, MAX_IDENTITY, maximum);
  const float expectedSum = std::accumulate(INPUT, INPUT + N, 0.0f);
  const float expectedMax = *std::max_element(INPUT, INPUT + N);

  std::printf("\nsum_sequential:              %g\n", sum);
  std::printf("reduce_sequential with +:    %g (std::accumulate: %g)\n", generalSum, expectedSum);
  std::printf("reduce_sequential with max:  %g (std::max_element: %g)\n", generalMax, expectedMax);

  const bool ok = sum == expectedSum && generalSum == expectedSum && generalMax == expectedMax;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
