// Staggered start times force a deterministic-looking order -- but that order
// comes from the sleeps, not from any guarantee. Delete the sleep and the
// output scrambles.
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  std::vector<std::thread> pool;
  pool.reserve(10);

  for (int i = 0; i < 10; ++i) {
    pool.emplace_back([i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(10 * i));
      std::cout << "worker #" << i << " reporting\n";
    });
  }

  std::cout << "main thread reporting\n";

  for (std::thread& worker : pool) {
    worker.join();
  }

  return 0;
}
