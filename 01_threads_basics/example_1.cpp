// Spawn one spinning worker per core. Each never returns, so main blocks in join()
// forever -- press Ctrl-C. The point: creating a thread is cheap, but every thread
// you create is one you must eventually account for.
#include <thread>
#include <vector>

int main() {
  const unsigned workerCount = 4;

  std::vector<std::thread> pool;
  pool.reserve(workerCount);

  for (unsigned w = 0; w < workerCount; ++w) {
    pool.emplace_back([w] {
      // A busy loop stands in for a market-data feed handler that never idles.
      for (;;) {
        (void)w;
      }
    });
  }

  for (std::thread& worker : pool) {
    worker.join();
  }
}
