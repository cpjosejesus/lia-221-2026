// Exercise 10.1 - Order heat map
//
// The city is divided into a grid of cells; orders[y][x] is how many orders came
// from each cell today. The dashboard shows a smoothed heat map: every cell becomes
// the mean of its (2r+1)x(2r+1) neighbourhood. The program does it on one thread.
//
// TODO (STARTS here):
//   1. Implement smoothParallel() by splitting the grid into horizontal strips and
//      giving one task per strip (example_58). The output grid is pre-sized: each task
//      writes only its own rows.
//   2. Make a table: strips -> time -> speedup for 1, 2, 4, 8, hardware_concurrency().
//      The result must be identical to the sequential version (the check prints it).
//   3. Now split into VERTICAL strips instead (one task per column range) and measure
//      again. Which is faster and why? (Think about how rows are laid out in memory.)
//   4. Bonus: the kernel re-reads (2r+1)^2 cells per output. A "separable" blur does a
//      horizontal pass then a vertical pass, (2r+1)+(2r+1) reads. Implement it in
//      parallel and compare.
// TODO (ENDS here)
#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct Grid {
  int w, h;
  std::vector<float> cells;                 // row-major: cells[y * w + x]
  float& at(int x, int y) { return cells[y * w + x]; }
  float at(int x, int y) const { return cells[y * w + x]; }
};

float meanAround(const Grid& g, int x, int y, int r) {
  float sum = 0; int n = 0;
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx) { sum += g.at(std::clamp(x + dx, 0, g.w - 1), std::clamp(y + dy, 0, g.h - 1)); ++n; }
  return sum / n;
}

void smoothSequential(const Grid& in, Grid& out, int r) {
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) out.at(x, y) = meanAround(in, x, y, r);
}

void smoothParallel(const Grid& in, Grid& out, int r, int nTasks) {
  // TODO: horizontal strips, one task each. Placeholder: sequential.
  (void)nTasks;
  smoothSequential(in, out, r);
}

int main() {
  Grid orders{2000, 1500, {}};
  orders.cells.resize(orders.w * orders.h);
  for (int y = 0; y < orders.h; ++y)
    for (int x = 0; x < orders.w; ++x) orders.at(x, y) = (float)(((x * 31 + y * 17) % 97) < 5 ? 40 : (x * y) % 3);
  Grid ref{orders.w, orders.h, std::vector<float>(orders.cells.size())};
  Grid out{orders.w, orders.h, std::vector<float>(orders.cells.size())};
  const int r = 3;

  auto t0 = std::chrono::steady_clock::now();
  smoothSequential(orders, ref, r);
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "sequential: " << seqMs << " ms\n";

  for (int n : {1, 2, 4, 8, (int)std::thread::hardware_concurrency()}) {
    t0 = std::chrono::steady_clock::now();
    smoothParallel(orders, out, r, n);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n << " strip(s): " << ms << " ms\tspeedup " << seqMs / ms << "x\t" << (out.cells == ref.cells ? "identical" : "MISMATCH") << "\n";
  }
  return 0;
}
