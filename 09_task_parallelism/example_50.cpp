// Fan-out / fan-in with std::async: total distance driven by a fleet of vehicles,
// computed sequentially and then split into chunks across tasks. Prints the speedup.
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

struct Vehicle {
  int id;
  double x0, y0, x1, y1;
};

// simulate some per-vehicle work: a distance plus a slow-ish correction
double tripDistance(const Vehicle& v) {
  double d = std::hypot(v.x1 - v.x0, v.y1 - v.y0);
  for (int i = 0; i < 200; ++i) d = std::sqrt(d * d + 1e-9);
  return d;
}

double sumSequential(const std::vector<Vehicle>& fleet) {
  double total = 0;
  for (const auto& v : fleet) total += tripDistance(v);
  return total;
}

double sumParallel(const std::vector<Vehicle>& fleet, int nTasks) {
  std::vector<std::future<double>> parts;
  size_t chunk = (fleet.size() + nTasks - 1) / nTasks;
  for (int t = 0; t < nTasks; ++t) {
    size_t begin = t * chunk, end = std::min(fleet.size(), begin + chunk);
    parts.emplace_back(std::async(std::launch::async, [&fleet, begin, end] {   // fan-out
      double s = 0;
      for (size_t i = begin; i < end; ++i) s += tripDistance(fleet[i]);
      return s;                                                                // each task owns its partial sum
    }));
  }
  double total = 0;
  for (auto& p : parts) total += p.get();                                     // fan-in
  return total;
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 400000; ++i) fleet.push_back({i, i % 100 * 1.0, i % 37 * 1.0, i % 53 * 1.0, i % 71 * 1.0});

  auto time = [](auto&& fn) {
    auto t0 = std::chrono::steady_clock::now();
    double r = fn();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return std::make_pair(r, ms);
  };

  auto [seqResult, seqMs] = time([&] { return sumSequential(fleet); });
  unsigned n = std::thread::hardware_concurrency();
  auto [parResult, parMs] = time([&] { return sumParallel(fleet, n); });

  std::cout << "hardware_concurrency = " << n << "\n";
  std::cout << "sequential: " << seqResult << " in " << seqMs << " ms\n";
  std::cout << "parallel  : " << parResult << " in " << parMs << " ms\n";
  std::cout << "speedup = " << seqMs / parMs << "x  (ideal " << n << "x)\n";
  std::cout << "results differ by " << std::abs(seqResult - parResult) << " (floating-point sums are not associative, see example_56)\n";
  return 0;
}
