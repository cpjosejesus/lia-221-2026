// Exercise 8.3 - Per-courier delivery counters and false sharing
//
// Every courier has its own counter of delivered orders, so there is no data race:
// each thread writes only its own element. Yet with 8 couriers the program is
// far slower than one courier alone. Find out why, and fix it without changing
// what the program computes.
//
// TODO (STARTS here):
//   1. Run the program as it is and note the time and the "speedup" it prints.
//   2. Explain in a comment why eight threads that never touch the same variable
//      still slow each other down (example_48 has the vocabulary you need).
//   3. Change the layout of `Counter` so that each courier's counter lives on its own
//      cache line. Keep std::atomic<long>: the fix is about layout, not about atomics.
//   4. Run again and report the before/after times in a comment. Then try 2, 4 and 16
//      couriers and describe how the effect changes.
// TODO (ENDS here)
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

struct Counter {
  std::atomic<long> delivered{0};   // TODO: layout
};

long long run(int couriers) {
  std::vector<Counter> counters(couriers);
  const long ordersPerCourier = 10'000'000;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int c = 0; c < couriers; ++c)
    threads.emplace_back([&counters, c] {
      for (long i = 0; i < ordersPerCourier; ++i) counters[c].delivered.fetch_add(1, std::memory_order_relaxed);
    });
  for (auto& t : threads) t.join();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  long total = 0;
  for (auto& c : counters) total += c.delivered.load();
  std::cout << couriers << " courier(s): " << total << " deliveries in " << ms << " ms\n";
  return ms;
}

int main() {
  std::cout << "sizeof(Counter) = " << sizeof(Counter) << " bytes\n";
  long long one = run(1);
  long long eight = run(8);
  std::cout << "8 couriers did 8x the work in " << (one ? (double)eight / one : 0) << "x the time"
            << " (ideal on 8+ cores: about 1x)" << std::endl;
  return 0;
}
