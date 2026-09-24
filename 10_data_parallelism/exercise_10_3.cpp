// Exercise 10.3 - Uneven orders and load balancing
//
// Orders are processed in parallel. Most take a moment; catering orders (about 5%)
// take 100x longer, and they arrive clustered at the start of the day. The program
// splits the day into equal static chunks and the speedup is disappointing.
//
// TODO (STARTS here):
//   1. Run it and note the static speedup. Explain in a comment why it is far below
//      the number of cores even though every task got the same NUMBER of orders.
//   2. Implement processDynamic(): tasks claim the next `grain` orders from a shared
//      std::atomic<size_t> counter with fetch_add (example_60) until it runs past the end.
//   3. Table: grain 10000, 1000, 100, 10, 1 -> time. Explain both ends of the curve.
//   4. Per-restaurant totals are also needed. Add them with the combiner pattern of
//      example_61: a private vector per task, merged at the end. No mutex, no atomics
//      in the inner loop.
//   5. Bonus: shuffle the orders before the static split. Does static catch up? Why is
//      shuffling not a general solution?
// TODO (ENDS here)
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct Order { int id; int restaurant; int items; };
const int RESTAURANTS = 12;

double process(const Order& o) {
  double x = o.items;
  for (int i = 0; i < o.items * 40; ++i) x = std::sqrt(x + 1.0);
  return x;
}

double processStatic(const std::vector<Order>& orders, int nTasks) {
  std::vector<std::future<double>> parts;
  size_t chunk = (orders.size() + nTasks - 1) / nTasks;
  for (int t = 0; t < nTasks; ++t) {
    size_t b = t * chunk, e = std::min(orders.size(), b + chunk);
    parts.emplace_back(std::async(std::launch::async, [&, b, e] { double s = 0; for (size_t i = b; i < e; ++i) s += process(orders[i]); return s; }));
  }
  double s = 0; for (auto& p : parts) s += p.get(); return s;
}

double processDynamic(const std::vector<Order>& orders, int nTasks, size_t grain) {
  // TODO: atomic "next" counter, grain-sized claims. Placeholder: static.
  (void)grain;
  return processStatic(orders, nTasks);
}

int main() {
  std::vector<Order> orders;
  for (int i = 0; i < 300'000; ++i) orders.push_back({i, i % RESTAURANTS, i < 15'000 ? 100 : 1});   // catering first
  int n = std::thread::hardware_concurrency();

  auto t0 = std::chrono::steady_clock::now();
  double seq = 0; for (const auto& o : orders) seq += process(o);
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "sequential: " << seqMs << " ms\n";

  t0 = std::chrono::steady_clock::now();
  double st = processStatic(orders, n);
  double stMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "static, " << n << " tasks: " << stMs << " ms\tspeedup " << seqMs / stMs << "x\t" << (std::abs(st - seq) < 1e-6 * seq ? "ok" : "WRONG") << "\n";

  for (size_t grain : {10000ul, 1000ul, 100ul, 10ul, 1ul}) {
    t0 = std::chrono::steady_clock::now();
    double dy = processDynamic(orders, n, grain);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "dynamic, grain " << grain << ": " << ms << " ms\tspeedup " << seqMs / ms << "x\t" << (std::abs(dy - seq) < 1e-6 * seq ? "ok" : "WRONG") << "\n";
  }
  return 0;
}
