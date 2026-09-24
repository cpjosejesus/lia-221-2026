// C++17 parallel algorithms: std::sort / std::transform with an execution policy.
// Not every standard library ships them (Apple clang does not; GCC needs -ltbb),
// so the program detects support and otherwise falls back to a hand-made version
// built from the folder-09 primitives, which is what the policy does for you.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_parallel_algorithm
#include <execution>
#endif

struct Vehicle { int id; double km; double score; };

double scoreOf(const Vehicle& v) {
  double s = v.km;
  for (int i = 0; i < 200; ++i) s = std::sqrt(s * s + 1.0);
  return s;
}

template <class Fn>
double timeMs(Fn fn) {
  auto t0 = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 2'000'000; ++i) fleet.push_back({i, 1.0 + (i * 7919) % 1000, 0.0});

  double seq = timeMs([&] { std::transform(fleet.begin(), fleet.end(), fleet.begin(), [](Vehicle v) { v.score = scoreOf(v); return v; }); });
  std::cout << "sequential transform: " << seq << " ms\n";

#ifdef __cpp_lib_parallel_algorithm
  double par = timeMs([&] { std::transform(std::execution::par, fleet.begin(), fleet.end(), fleet.begin(), [](Vehicle v) { v.score = scoreOf(v); return v; }); });
  std::cout << "std::execution::par transform: " << par << " ms (speedup " << seq / par << "x)\n";
  double sortMs = timeMs([&] { std::sort(std::execution::par, fleet.begin(), fleet.end(), [](const Vehicle& a, const Vehicle& b) { return a.score < b.score; }); });
  std::cout << "std::execution::par sort: " << sortMs << " ms\n";
#else
  std::cout << "This standard library has no parallel algorithms (__cpp_lib_parallel_algorithm undefined).\n"
               "GCC: link with -ltbb. Apple clang: not available. Falling back to a hand-made parallel map:\n";
  int n = std::thread::hardware_concurrency();
  double par = timeMs([&] {
    std::vector<std::future<void>> tasks;
    size_t chunk = (fleet.size() + n - 1) / n;
    for (int t = 0; t < n; ++t) {
      size_t b = t * chunk, e = std::min(fleet.size(), b + chunk);
      tasks.emplace_back(std::async(std::launch::async, [&, b, e] { for (size_t i = b; i < e; ++i) fleet[i].score = scoreOf(fleet[i]); }));
    }
    for (auto& t : tasks) t.get();
  });
  std::cout << "hand-made parallel transform (" << n << " tasks): " << par << " ms (speedup " << seq / par << "x)\n";
  std::cout << "(a parallel sort is a fork/join merge sort: split, sort halves in parallel, merge; see the self-study list)\n";
#endif
  std::cout << "score[0] = " << fleet[0].score << ", score[last] = " << fleet.back().score << "\n";
  return 0;
}
