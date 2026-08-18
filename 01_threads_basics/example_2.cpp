// How much *real* parallelism does this box offer? hardware_concurrency() is a
// hint, not a promise -- it may return 0 when the answer is unknown.
#include <iostream>
#include <thread>

int main() {
  const unsigned cores = std::thread::hardware_concurrency();

  if (cores == 0) {
    std::cout << "Core count unavailable on this platform; assuming 1\n";
  } else {
    std::cout << "Matching engine can run on " << cores << " hardware threads\n";
  }

  return 0;
}
