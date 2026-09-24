// Recursive fork/join: split the fleet in halves until a chunk is small enough,
// process the small chunks, join the halves on the way back. This is the shape of
// every divide-and-conquer parallel algorithm (parallel sum, sort, tree walks).
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <vector>

struct Vehicle { int id; double weight; };

double weigh(const Vehicle& v) {
  double w = v.weight;
  for (int i = 0; i < 100; ++i) w = std::sqrt(w * w + 0.5);
  return w;
}

// Sum of weights in [begin, end). Below `cutoff` elements the work is done sequentially:
// spawning a task costs far more than weighing a few vehicles.
double parallelSum(const std::vector<Vehicle>& fleet, size_t begin, size_t end, size_t cutoff, int depth = 0) {
  if (end - begin <= cutoff) {
    double s = 0;
    for (size_t i = begin; i < end; ++i) s += weigh(fleet[i]);
    return s;
  }
  size_t mid = begin + (end - begin) / 2;
  // fork: the left half runs on a new task, the right half on the current thread
  std::future<double> left = std::async(std::launch::async, parallelSum, std::cref(fleet), begin, mid, cutoff, depth + 1);
  double right = parallelSum(fleet, mid, end, cutoff, depth + 1);
  return left.get() + right;
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 1'000'000; ++i) fleet.push_back({i, 1.0 + i % 7});

  auto measure = [&](size_t cutoff) {
    auto t0 = std::chrono::steady_clock::now();
    double s = parallelSum(fleet, 0, fleet.size(), cutoff);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    int tasks = (int)std::ceil((double)fleet.size() / cutoff);   // roughly: leaves of the recursion
    std::cout << "cutoff " << cutoff << "\t~" << tasks << " leaf tasks\t" << ms << " ms\t(sum " << s << ")\n";
  };
  std::cout << "cutoff\t\tleaf tasks\ttime\n";
  measure(fleet.size());   // 1 leaf: purely sequential
  measure(250'000);
  measure(50'000);
  measure(5'000);
  measure(500);            // thousands of tasks: overhead dominates
  return 0;
}
