// Parallel k-means: group vehicle positions into k clusters (e.g. where to place
// k service stations). Each iteration has a data-parallel assignment step and an
// update step done with per-task partial sums (the "combiner" pattern), so no
// shared accumulator is ever written by two tasks.
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

struct Point { double x, y; };

struct Partial {                       // per-task accumulator: sum and count per cluster
  std::vector<double> sx, sy;
  std::vector<long> n;
  Partial(int k) : sx(k, 0), sy(k, 0), n(k, 0) {}
};

int nearest(const Point& p, const std::vector<Point>& centroids) {
  int best = 0; double bestD = 1e300;
  for (size_t c = 0; c < centroids.size(); ++c) {
    double d = (p.x - centroids[c].x) * (p.x - centroids[c].x) + (p.y - centroids[c].y) * (p.y - centroids[c].y);
    if (d < bestD) { bestD = d; best = (int)c; }
  }
  return best;
}

std::vector<Point> kmeans(const std::vector<Point>& pts, int k, int iterations, int nTasks) {
  std::vector<Point> centroids(pts.begin(), pts.begin() + k);   // start from the first k points
  for (int it = 0; it < iterations; ++it) {
    std::vector<std::future<Partial>> parts;
    size_t chunk = (pts.size() + nTasks - 1) / nTasks;
    for (int t = 0; t < nTasks; ++t) {
      size_t b = t * chunk, e = std::min(pts.size(), b + chunk);
      parts.emplace_back(std::async(std::launch::async, [&, b, e] {
        Partial acc(k);                                             // owned by this task
        for (size_t i = b; i < e; ++i) {
          int c = nearest(pts[i], centroids);                       // assignment step (reads shared centroids)
          acc.sx[c] += pts[i].x; acc.sy[c] += pts[i].y; acc.n[c]++;
        }
        return acc;
      }));
    }
    Partial total(k);                                               // combine: sequential, k-sized, cheap
    for (auto& p : parts) {
      Partial acc = p.get();
      for (int c = 0; c < k; ++c) { total.sx[c] += acc.sx[c]; total.sy[c] += acc.sy[c]; total.n[c] += acc.n[c]; }
    }
    for (int c = 0; c < k; ++c)                                     // update step
      if (total.n[c]) centroids[c] = {total.sx[c] / total.n[c], total.sy[c] / total.n[c]};
  }
  return centroids;
}

int main() {
  std::mt19937 rng(7);
  std::vector<Point> hotspots{{2, 3}, {9, 8}, {4, 9}, {8, 2}};
  std::normal_distribution<double> noise(0.0, 0.7);
  std::vector<Point> vehicles;
  for (int i = 0; i < 2'000'000; ++i) { const Point& h = hotspots[i % 4]; vehicles.push_back({h.x + noise(rng), h.y + noise(rng)}); }

  for (int n : {1, 2, 4, (int)std::thread::hardware_concurrency()}) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Point> c = kmeans(vehicles, 4, 8, n);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n << " task(s): " << ms << " ms, centroids:";
    for (auto& p : c) std::cout << " (" << p.x << ", " << p.y << ")";
    std::cout << "\n";
  }
  std::cout << "\nThe centroids should sit near the four hotspots (2,3) (9,8) (4,9) (8,2), in some order.\n";
  return 0;
}
