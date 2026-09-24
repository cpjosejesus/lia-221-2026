// Monte Carlo with independent random streams: what fraction of random vehicle
// positions inside the square around a roundabout fall inside the circle? Times 4,
// that estimates pi. Each task owns its random generator (sharing one std::mt19937
// between threads is a data race) and its own count.
#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

long insideCircle(long samples, unsigned seed) {
  std::mt19937_64 rng(seed);                              // private generator, private seed
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  long inside = 0;
  for (long i = 0; i < samples; ++i) {
    double x = u(rng), y = u(rng);
    if (x * x + y * y <= 1.0) ++inside;
  }
  return inside;
}

int main() {
  const long total = 200'000'000;
  for (int n : {1, 2, 4, (int)std::thread::hardware_concurrency()}) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::future<long>> parts;
    for (int t = 0; t < n; ++t) parts.emplace_back(std::async(std::launch::async, insideCircle, total / n, 1000u + t));
    long inside = 0; for (auto& p : parts) inside += p.get();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n << " task(s): pi ~ " << 4.0 * inside / (total / n * n) << " in " << ms << " ms\n";
  }
  std::cout << "\nEmbarrassingly parallel: no shared state at all, just a sum at the end. Near-ideal speedup.\n"
               "Different seeds per task matter: identical streams would just repeat the same samples n times.\n";
  return 0;
}
