// Tiling on a CPU: the same idea as the tiled CUDA kernel of session 13, but the fast memory
// is the cache instead of shared memory. A naive matrix product walks a whole column of B
// for every element of C and keeps missing the cache; the tiled version works on T x T tiles
// that fit in the cache and reuses every loaded value T times. Strips of tiles then run on
// several threads, as in session 10.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

using Matrix = std::vector<float>;   // row-major: m[row * n + col]

void naive(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      float sum = 0;
      for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];   // B is read down a column
      C[i * n + j] = sum;
    }
}

// same arithmetic, loops reordered (i, k, j): B and C are read along rows, no tiling
void reordered(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  std::fill(C.begin(), C.end(), 0.0f);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      float a = A[i * n + k];
      for (int j = 0; j < n; ++j) C[i * n + j] += a * B[k * n + j];
    }
}

// rows [r0, r1) of C, computed tile by tile
void tiledRows(const Matrix& A, const Matrix& B, Matrix& C, int n, int T, int r0, int r1) {
  for (int i = r0; i < r1; ++i) std::fill(C.begin() + i * n, C.begin() + (i + 1) * n, 0.0f);
  for (int ii = r0; ii < r1; ii += T)
    for (int kk = 0; kk < n; kk += T)            // one "phase": a tile of A and a tile of B
      for (int jj = 0; jj < n; jj += T)
        for (int i = ii; i < std::min(ii + T, r1); ++i)
          for (int k = kk; k < std::min(kk + T, n); ++k) {
            float a = A[i * n + k];              // kept in a register: register tiling
            for (int j = jj; j < std::min(jj + T, n); ++j) C[i * n + j] += a * B[k * n + j];
          }
}

void tiledParallel(const Matrix& A, const Matrix& B, Matrix& C, int n, int T, int nTasks) {
  int rowsPerTask = ((n + nTasks - 1) / nTasks + T - 1) / T * T;   // whole tiles per task
  std::vector<std::future<void>> tasks;
  for (int r0 = 0; r0 < n; r0 += rowsPerTask)
    tasks.push_back(std::async(std::launch::async, tiledRows, std::cref(A), std::cref(B), std::ref(C), n, T, r0, std::min(n, r0 + rowsPerTask)));
  for (auto& t : tasks) t.get();
}

template <class F>
double ms(F f) {
  auto t0 = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  const int n = 1536, T = 64;
  Matrix A(n * n), B(n * n), C1(n * n), C2(n * n), C3(n * n), C4(n * n);
  for (int i = 0; i < n * n; ++i) { A[i] = (i % 7) * 0.5f - 1.5f; B[i] = (i % 5) * 0.25f - 0.5f; }
  int cores = std::thread::hardware_concurrency();

  double t1 = ms([&] { naive(A, B, C1, n); });
  double t4 = ms([&] { reordered(A, B, C4, n); });
  double t2 = ms([&] { tiledRows(A, B, C2, n, T, 0, n); });
  double t3 = ms([&] { tiledParallel(A, B, C3, n, T, cores); });

  float maxDiff = 0;
  for (int i = 0; i < n * n; ++i) maxDiff = std::max({maxDiff, std::abs(C1[i] - C2[i]), std::abs(C1[i] - C3[i]), std::abs(C1[i] - C4[i])});
  double flop = 2.0 * n * n * n;
  std::printf("n = %d, tile = %d x %d, %d threads\n", n, T, T, cores);
  std::printf("naive           %8.1f ms  %6.2f GFLOP/s\n", t1, flop / t1 / 1e6);
  std::printf("rows, no tiles  %8.1f ms  %6.2f GFLOP/s  (%.1fx)\n", t4, flop / t4 / 1e6, t1 / t4);
  std::printf("tiled           %8.1f ms  %6.2f GFLOP/s  (%.1fx)\n", t2, flop / t2 / 1e6, t1 / t2);
  std::printf("tiled, threads  %8.1f ms  %6.2f GFLOP/s  (%.1fx)\n", t3, flop / t3 / 1e6, t1 / t3);
  std::printf("max difference between results: %g\n", maxDiff);
  return 0;
}
