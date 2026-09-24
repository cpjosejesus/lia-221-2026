// Associativity: a parallel reduce only equals the sequential one if the operation
// is associative (and commutative if chunks are combined in any order). Three cases:
// integer sum (exact), floating-point sum (tiny differences), subtraction (wrong),
// and how to make "average" reducible with a (sum, count) pair.
#include <future>
#include <iostream>
#include <numeric>
#include <vector>

struct Vehicle { int id; double fuelLitres; };

template <class T, class Fn>
T parallelReduce(const std::vector<T>& in, T init, Fn op, int nTasks) {
  std::vector<std::future<T>> parts;
  size_t chunk = (in.size() + nTasks - 1) / nTasks;
  for (int t = 0; t < nTasks; ++t) {
    size_t b = t * chunk, e = std::min(in.size(), b + chunk);
    parts.emplace_back(std::async(std::launch::async, [&, b, e] {
      T acc = init;
      for (size_t i = b; i < e; ++i) acc = op(acc, in[i]);
      return acc;
    }));
  }
  T acc = init;
  for (auto& p : parts) acc = op(acc, p.get());                // combine the chunks with the SAME op
  return acc;
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 1'000'000; ++i) fleet.push_back({i, 0.1 + (i % 9) * 0.01});
  std::vector<double> fuel;
  std::vector<long> ids;
  for (auto& v : fleet) { fuel.push_back(v.fuelLitres); ids.push_back(v.id); }

  // 1) integer sum: associative and exact -> identical
  long seqIds = std::accumulate(ids.begin(), ids.end(), 0L);
  long parIds = parallelReduce(ids, 0L, [](long a, long b) { return a + b; }, 8);
  std::cout << "id sum       sequential " << seqIds << "  parallel " << parIds << (seqIds == parIds ? "  (identical)\n" : "  (DIFFERENT)\n");

  // 2) floating-point sum: mathematically associative, numerically not -> last digits differ
  double seqFuel = std::accumulate(fuel.begin(), fuel.end(), 0.0);
  double parFuel = parallelReduce(fuel, 0.0, [](double a, double b) { return a + b; }, 8);
  std::cout.precision(12);
  std::cout << "fuel sum     sequential " << seqFuel << "  parallel " << parFuel << "  diff " << seqFuel - parFuel << "\n";

  // 3) subtraction: not associative -> parallel result is simply wrong
  double seqSub = std::accumulate(fuel.begin(), fuel.end(), 1000.0, [](double a, double b) { return a - b; });
  double parSub = parallelReduce(fuel, 1000.0, [](double a, double b) { return a - b; }, 8);
  std::cout << "1000 - fuel  sequential " << seqSub << "  parallel " << parSub << "  <- wrong: '-' is not associative\n";

  // 4) average: not directly reducible; reduce a (sum, count) pair instead, divide at the end
  struct SumCount { double sum; long count; };
  std::vector<SumCount> pairs;
  for (double f : fuel) pairs.push_back({f, 1});
  SumCount sc = parallelReduce(pairs, SumCount{0, 0}, [](SumCount a, SumCount b) { return SumCount{a.sum + b.sum, a.count + b.count}; }, 8);
  std::cout << "average fuel " << sc.sum / sc.count << " litres over " << sc.count << " vehicles (via (sum,count) pairs)\n";
  return 0;
}
