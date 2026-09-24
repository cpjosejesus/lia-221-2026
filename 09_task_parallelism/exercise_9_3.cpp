// Exercise 9.3 - Pricing orders on a thread pool
//
// Every incoming order must be priced (an expensive call). Today the program prices
// them one after another. The service wants to use all cores WITHOUT creating one
// thread per order.
//
// TODO (STARTS here):
//   1. Bring in the ThreadPool of example_55 (copy it here or into a header).
//   2. Implement priceAllWithPool(): submit one pricing job per order, collect the
//      futures, and store every result in `prices` at the order's index.
//   3. Run it with pool sizes 1, 2, 4, 8 and hardware_concurrency(). Produce a table:
//      workers -> time -> speedup. Explain the shape of the curve.
//   4. Compare with one std::async per order (example_55 does the same comparison).
//      Which one would you ship, and what happens to std::async with 5 million orders?
//   5. Bonus: submit chunks of 1000 orders per job instead of one order per job.
//      Does it help? Why?
// TODO (ENDS here)
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

struct Order { int id; double distanceKm; int items; };

double priceOrder(const Order& o) {
  double p = 2.5 + 0.8 * o.distanceKm + 0.3 * o.items;
  for (int i = 0; i < 1500; ++i) p = std::sqrt(p * p + 0.25);    // surge model, expensive
  return p;
}

void priceAllSequential(const std::vector<Order>& orders, std::vector<double>& prices) {
  prices.resize(orders.size());
  for (size_t i = 0; i < orders.size(); ++i) prices[i] = priceOrder(orders[i]);
}

void priceAllWithPool(const std::vector<Order>& orders, std::vector<double>& prices, size_t workers) {
  // TODO: thread pool. Placeholder: sequential.
  (void)workers;
  priceAllSequential(orders, prices);
}

int main() {
  std::vector<Order> orders;
  for (int i = 0; i < 50'000; ++i) orders.push_back({i, 0.5 + i % 15, 1 + i % 6});
  std::vector<double> prices;

  auto t0 = std::chrono::steady_clock::now();
  priceAllSequential(orders, prices);
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  double check = prices[123];
  std::cout << "sequential: " << seqMs << " ms\n";

  for (size_t w : {1ul, 2ul, 4ul, 8ul, (unsigned long)std::thread::hardware_concurrency()}) {
    t0 = std::chrono::steady_clock::now();
    priceAllWithPool(orders, prices, w);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << w << " worker(s): " << ms << " ms\tspeedup " << seqMs / ms << "x\t" << (prices[123] == check ? "ok" : "WRONG RESULT") << "\n";
  }
  return 0;
}
