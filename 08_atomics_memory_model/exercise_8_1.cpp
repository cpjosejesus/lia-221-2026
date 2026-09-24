// Exercise 8.1 - Order statistics without a mutex
//
// A delivery service counts how many orders were created, delivered and rejected.
// Sixteen courier threads report events at the same time. The current version
// protects three plain integers with one mutex, which serializes every report.
//
// TODO (STARTS here):
//   1. Replace the three `int` members and the mutex by std::atomic<int> members.
//   2. Reimplement the three report methods with a single atomic read-modify-write
//      each (no load-then-store pairs: see example_43 for why that is still a race).
//   3. Choose a memory order for these counters and justify it in a comment:
//      nobody reads one counter to decide whether another counter is visible, so
//      is std::memory_order_relaxed enough? What would you lose, what would you gain?
//   4. `consistent()` must keep returning true: created == delivered + rejected + pending.
//      Hint: is that check itself atomic across three counters? Explain in a comment
//      what a reader can observe while couriers are still reporting.
// TODO (ENDS here)
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class OrderStats {
 public:
  void reportCreated()   { std::lock_guard<std::mutex> lock(_mutex); ++_created; }
  void reportDelivered() { std::lock_guard<std::mutex> lock(_mutex); ++_delivered; }
  void reportRejected()  { std::lock_guard<std::mutex> lock(_mutex); ++_rejected; }

  int created()   { std::lock_guard<std::mutex> lock(_mutex); return _created; }
  int delivered() { std::lock_guard<std::mutex> lock(_mutex); return _delivered; }
  int rejected()  { std::lock_guard<std::mutex> lock(_mutex); return _rejected; }

 private:
  std::mutex _mutex;
  int _created = 0, _delivered = 0, _rejected = 0;
};

int main() {
  OrderStats stats;
  const int couriers = 16, ordersPerCourier = 50000;

  std::vector<std::thread> threads;
  for (int c = 0; c < couriers; ++c) {
    threads.emplace_back([&stats, c] {
      for (int i = 0; i < ordersPerCourier; ++i) {
        stats.reportCreated();
        if ((i + c) % 10 == 0) stats.reportRejected();
        else stats.reportDelivered();
      }
    });
  }
  for (auto& t : threads) t.join();

  int pending = stats.created() - stats.delivered() - stats.rejected();
  std::cout << "created=" << stats.created() << " delivered=" << stats.delivered()
            << " rejected=" << stats.rejected() << " pending=" << pending << std::endl;
  std::cout << (stats.created() == couriers * ordersPerCourier && pending == 0 ? "consistent" : "INCONSISTENT") << std::endl;
  return 0;
}
