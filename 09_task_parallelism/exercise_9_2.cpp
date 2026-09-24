// Exercise 9.2 - Cumulative deliveries timeline with a parallel scan
//
// deliveredPerMinute[m] is how many orders were delivered during minute m of a
// very long day. The dashboard needs the cumulative count after every minute:
// a prefix sum. std::partial_sum does it sequentially.
//
// TODO (STARTS here):
//   1. Implement parallelScan() with the three-phase algorithm of example_54:
//      per-chunk local scan + chunk totals, a tiny sequential scan over the totals,
//      then add each chunk's offset in parallel.
//   2. The result must be identical to the sequential one (the check prints it).
//   3. Measure both for 1, 2, 4 and hardware_concurrency() tasks and put the times in
//      a table. Explain why the parallel version needs at least ~2 cores to break even.
//   4. Extra: the day has 20 million minutes here only to make the timing visible.
//      What is the smallest input size at which your parallel scan still wins?
// TODO (ENDS here)
#include <chrono>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

std::vector<long> parallelScan(const std::vector<int>& in, int nTasks) {
  // TODO: three phases. Placeholder: sequential.
  std::vector<long> out(in.size());
  std::partial_sum(in.begin(), in.end(), out.begin());
  (void)nTasks;
  return out;
}

int main() {
  std::vector<int> deliveredPerMinute(20'000'000);
  for (size_t m = 0; m < deliveredPerMinute.size(); ++m) deliveredPerMinute[m] = (int)((m * 7) % 6);

  auto t0 = std::chrono::steady_clock::now();
  std::vector<long> seq(deliveredPerMinute.size());
  std::partial_sum(deliveredPerMinute.begin(), deliveredPerMinute.end(), seq.begin());
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "sequential scan: " << seqMs << " ms, total delivered " << seq.back() << "\n";

  for (int n : {1, 2, 4, (int)std::thread::hardware_concurrency()}) {
    t0 = std::chrono::steady_clock::now();
    std::vector<long> par = parallelScan(deliveredPerMinute, n);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n << " task(s): " << ms << " ms\t" << (par == seq ? "identical" : "MISMATCH") << "\n";
  }
  return 0;
}
