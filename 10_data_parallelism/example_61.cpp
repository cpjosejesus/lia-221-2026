// Shared accumulator vs per-task accumulators (combiners). A histogram of vehicle
// speeds built by many tasks: (a) one shared histogram protected by a mutex,
// (b) one shared histogram of atomics, (c) a private histogram per task, merged at the end.
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

const int BINS = 32;

template <class Fn>
double timeMs(Fn fn) {
  auto t0 = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  std::vector<int> speeds(30'000'000);
  for (size_t i = 0; i < speeds.size(); ++i) speeds[i] = (int)((i * 2654435761u) % BINS);
  int n = std::thread::hardware_concurrency();
  size_t chunk = (speeds.size() + n - 1) / n;
  auto range = [&](int t) { size_t b = t * chunk; return std::make_pair(b, std::min(speeds.size(), b + chunk)); };

  // (a) shared histogram + mutex per increment: correct, but every increment serializes
  std::vector<long> shared(BINS, 0);
  std::mutex m;
  double a = timeMs([&] {
    std::vector<std::future<void>> ts;
    for (int t = 0; t < n; ++t) ts.emplace_back(std::async(std::launch::async, [&, t] {
      auto [b, e] = range(t);
      for (size_t i = b; i < e; ++i) { std::lock_guard<std::mutex> lock(m); shared[speeds[i]]++; }
    }));
    for (auto& f : ts) f.get();
  });

  // (b) shared histogram of atomics: no lock, but all tasks hammer the same 32 counters (contention, false sharing)
  std::vector<std::atomic<long>> atomicHist(BINS);
  for (auto& x : atomicHist) x.store(0);
  double bb = timeMs([&] {
    std::vector<std::future<void>> ts;
    for (int t = 0; t < n; ++t) ts.emplace_back(std::async(std::launch::async, [&, t] {
      auto [b, e] = range(t);
      for (size_t i = b; i < e; ++i) atomicHist[speeds[i]].fetch_add(1, std::memory_order_relaxed);
    }));
    for (auto& f : ts) f.get();
  });

  // (c) combiner: each task fills its own histogram, the 32-entry results are merged once
  std::vector<long> merged(BINS, 0);
  double c = timeMs([&] {
    std::vector<std::future<std::vector<long>>> ts;
    for (int t = 0; t < n; ++t) ts.emplace_back(std::async(std::launch::async, [&, t] {
      std::vector<long> local(BINS, 0);
      auto [b, e] = range(t);
      for (size_t i = b; i < e; ++i) local[speeds[i]]++;
      return local;
    }));
    for (auto& f : ts) { auto local = f.get(); for (int k = 0; k < BINS; ++k) merged[k] += local[k]; }
  });

  bool same = true;
  for (int k = 0; k < BINS; ++k) same = same && shared[k] == merged[k] && atomicHist[k].load() == merged[k];
  std::cout << "(a) shared + mutex : " << a << " ms\n(b) shared atomics : " << bb << " ms\n(c) per-task + merge: " << c << " ms\n";
  std::cout << (same ? "all three histograms identical" : "MISMATCH") << ", bin 0 = " << merged[0] << "\n";
  std::cout << "\nThe combiner pattern turns 30M synchronized increments into " << n << " x 32 plain additions at the end.\n";
  return 0;
}
