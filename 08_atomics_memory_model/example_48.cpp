// False sharing: two threads update two DIFFERENT counters that happen to share
// one cache line. The hardware ping-pongs the line between cores and the "parallel"
// version crawls. Padding the counters to separate lines fixes it.
#include <atomic>
#include <chrono>
#include <iostream>
#include <new>
#include <thread>

struct Adjacent {                 // both counters in the same 64-byte cache line
  std::atomic<long> entered{0};
  std::atomic<long> left{0};
};

struct Padded {                   // each counter gets its own cache line
  alignas(64) std::atomic<long> entered{0};
  alignas(64) std::atomic<long> left{0};
};

template <class Counters>
long long measure(const char* label) {
  Counters c;
  const long N = 20'000'000;
  auto t0 = std::chrono::steady_clock::now();
  std::thread a([&] { for (long i = 0; i < N; ++i) c.entered.fetch_add(1, std::memory_order_relaxed); });
  std::thread b([&] { for (long i = 0; i < N; ++i) c.left.fetch_add(1, std::memory_order_relaxed); });
  a.join(); b.join();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  std::cout << label << ": entered=" << c.entered << " left=" << c.left << " in " << ms << " ms\n";
  return ms;
}

int main() {
  std::cout << "cache line size (hint): " << std::hardware_destructive_interference_size << " bytes\n";
  std::cout << "sizeof(Adjacent) = " << sizeof(Adjacent) << ", sizeof(Padded) = " << sizeof(Padded) << "\n";
  long long slow = measure<Adjacent>("adjacent counters");
  long long fast = measure<Padded>("padded counters  ");
  std::cout << "padding made it " << (fast ? (double)slow / fast : 0.0) << "x faster" << std::endl;
  return 0;
}
