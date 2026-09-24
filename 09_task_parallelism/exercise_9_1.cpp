// Exercise 9.1 - Parallel revenue with fork/join
//
// A delivery service wants the total revenue of a day: the sum of price * quantity
// over one million orders, plus a per-order fee that is expensive to compute.
// The program below does it sequentially and prints the time.
//
// TODO (STARTS here):
//   1. Implement parallelRevenue() as a recursive fork/join (example_52): split the
//      range in halves with std::async until a chunk has at most `cutoff` orders.
//   2. Print a table of cutoff -> time -> speedup for cutoffs 1'000'000 (sequential),
//      100'000, 10'000, 1'000 and 100. Where is the sweet spot on your machine, and why
//      does the smallest cutoff get slower again?
//   3. The result must match the sequential one exactly. Why is that guaranteed here
//      when example_50 showed a difference? (Look at the type of `revenue`.)
//   4. Bonus: replace std::async with the ThreadPool of example_55 and compare.
// TODO (ENDS here)
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <vector>

struct Order { int id; long priceCents; int quantity; double distanceKm; };

long feeCents(const Order& o) {           // deliberately expensive
  double f = o.distanceKm;
  for (int i = 0; i < 100; ++i) f = std::sqrt(f * f + 3.0);
  return (long)f;
}

long sequentialRevenue(const std::vector<Order>& orders) {
  long revenue = 0;
  for (const auto& o : orders) revenue += o.priceCents * o.quantity + feeCents(o);
  return revenue;
}

long parallelRevenue(const std::vector<Order>& orders, size_t begin, size_t end, size_t cutoff) {
  // TODO: fork/join. Placeholder: sequential over the range.
  long revenue = 0;
  for (size_t i = begin; i < end; ++i) revenue += orders[i].priceCents * orders[i].quantity + feeCents(orders[i]);
  (void)cutoff;
  return revenue;
}

int main() {
  std::vector<Order> orders;
  for (int i = 0; i < 1'000'000; ++i) orders.push_back({i, 500 + i % 2000, 1 + i % 4, 0.5 + i % 12});

  auto t0 = std::chrono::steady_clock::now();
  long seq = sequentialRevenue(orders);
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "sequential: " << seq << " cents in " << seqMs << " ms\n";

  for (size_t cutoff : {1'000'000ul, 100'000ul, 10'000ul, 1'000ul, 100ul}) {
    t0 = std::chrono::steady_clock::now();
    long par = parallelRevenue(orders, 0, orders.size(), cutoff);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "cutoff " << cutoff << "\t" << ms << " ms\tspeedup " << seqMs / ms << "x\t" << (par == seq ? "match" : "MISMATCH") << "\n";
  }
  return 0;
}
