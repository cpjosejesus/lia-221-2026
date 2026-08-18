// WHY MANUAL UNLOCK IS A TRAP.
// The catch block returns early -- skipping mtx.unlock() at the bottom. The
// first thread to hit denominator == 0 leaves the mutex locked forever and
// every other thread blocks on it. The program hangs; Ctrl-C to stop it.
//
// example_35 is this same code with a lock_guard, which cannot leak.
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

std::mutex bookMutex;
double lastQuote;

void reportQuote(double denominator) {
  std::cout << "denominator " << denominator << " -> quote " << lastQuote << '\n';
}

void quote(double notional, double denominator) {
  bookMutex.lock();

  try {
    if (denominator == 0) {
      throw std::invalid_argument("worker: cannot quote against zero volume");
    }

    lastQuote = notional / denominator;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    reportQuote(denominator);
  } catch (const std::invalid_argument& e) {
    std::cout << e.what() << '\n';
    return;  // <-- BUG: returns with bookMutex still held
  }

  bookMutex.unlock();  // never reached on the exception path
}

int main() {
  std::vector<std::future<void>> tasks;

  for (double d = -5; d <= +5; ++d) {
    tasks.emplace_back(std::async(std::launch::async, quote, 50.0, d));
  }

  for (std::future<void>& task : tasks) {
    task.wait();
  }

  return 0;
}
