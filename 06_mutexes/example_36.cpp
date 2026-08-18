// lock_guard is rigid: locked for its whole scope. std::unique_lock adds the
// flexibility to unlock() and lock() again mid-scope, so you can drop the lock
// around work that does not touch shared state.
//
// Holding a mutex across a sleep -- or any I/O -- serialises every other
// thread behind you. Shrink critical sections to the shared access itself.
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
  std::unique_lock<std::mutex> lock(bookMutex);

  try {
    if (denominator == 0) {
      throw std::invalid_argument("worker: cannot quote against zero volume");
    }

    lastQuote = notional / denominator;  // shared state -- lock required
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    reportQuote(denominator);

    lock.unlock();  // release early: the next part touches nothing shared

    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // private work

    lock.lock();  // re-acquire for more shared access
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } catch (const std::invalid_argument& e) {
    std::cout << e.what() << '\n';
    return;  // still safe -- unique_lock unlocks in its destructor
  }
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
