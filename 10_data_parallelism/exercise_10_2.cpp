// Exercise 10.2 - Where to build three delivery hubs
//
// Every delivery of the last month has a coordinate. The company wants to place
// k = 3 hubs so that deliveries are close to a hub: k-means over the delivery
// positions. The sequential version below works; make it parallel.
//
// TODO (STARTS here):
//   1. Implement kmeansParallel() as in example_59: the assignment step over chunks
//      in parallel, each task accumulating its OWN per-cluster sums and counts, and a
//      cheap sequential merge before the update step.
//   2. The centroids must come out identical to the sequential version (they will, as
//      long as you merge in the same order: think about which sums are exact and which
//      are not).
//   3. Table: tasks -> time -> speedup. Where does the speedup stop growing, and what
//      part of an iteration is not parallel?
//   4. Bonus: instead of a fixed number of iterations, stop when no centroid moved more
//      than 1e-6. Which part of that test needs care when done in parallel?
// TODO (ENDS here)
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

struct Point { double x, y; };

int nearest(const Point& p, const std::vector<Point>& c) {
  int best = 0; double bestD = 1e300;
  for (size_t i = 0; i < c.size(); ++i) {
    double d = (p.x - c[i].x) * (p.x - c[i].x) + (p.y - c[i].y) * (p.y - c[i].y);
    if (d < bestD) { bestD = d; best = (int)i; }
  }
  return best;
}

std::vector<Point> kmeansSequential(const std::vector<Point>& pts, int k, int iterations) {
  std::vector<Point> c(pts.begin(), pts.begin() + k);
  for (int it = 0; it < iterations; ++it) {
    std::vector<double> sx(k, 0), sy(k, 0); std::vector<long> n(k, 0);
    for (const auto& p : pts) { int i = nearest(p, c); sx[i] += p.x; sy[i] += p.y; n[i]++; }
    for (int i = 0; i < k; ++i) if (n[i]) c[i] = {sx[i] / n[i], sy[i] / n[i]};
  }
  return c;
}

std::vector<Point> kmeansParallel(const std::vector<Point>& pts, int k, int iterations, int nTasks) {
  // TODO: parallel assignment with per-task partial sums. Placeholder: sequential.
  (void)nTasks;
  return kmeansSequential(pts, k, iterations);
}

int main() {
  std::mt19937 rng(11);
  std::vector<Point> centres{{1.5, 2.0}, {8.0, 7.5}, {3.0, 8.5}};
  std::normal_distribution<double> noise(0.0, 0.8);
  std::vector<Point> deliveries;
  for (int i = 0; i < 3'000'000; ++i) { const Point& h = centres[i % 3]; deliveries.push_back({h.x + noise(rng), h.y + noise(rng)}); }

  auto t0 = std::chrono::steady_clock::now();
  std::vector<Point> ref = kmeansSequential(deliveries, 3, 10);
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "sequential: " << seqMs << " ms, hubs:";
  for (auto& p : ref) std::cout << " (" << p.x << ", " << p.y << ")";
  std::cout << "\n";

  for (int n : {1, 2, 4, 8, (int)std::thread::hardware_concurrency()}) {
    t0 = std::chrono::steady_clock::now();
    std::vector<Point> c = kmeansParallel(deliveries, 3, 10, n);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    bool same = true; for (int i = 0; i < 3; ++i) same = same && std::abs(c[i].x - ref[i].x) < 1e-9 && std::abs(c[i].y - ref[i].y) < 1e-9;
    std::cout << n << " task(s): " << ms << " ms\tspeedup " << seqMs / ms << "x\t" << (same ? "same hubs" : "DIFFERENT hubs") << "\n";
  }
  return 0;
}
