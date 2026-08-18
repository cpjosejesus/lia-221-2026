// example_33 fixed. std::lock_guard locks on construction and unlocks in its
// destructor, so the lock is released on EVERY exit path -- normal return,
// early return, or an exception unwinding the stack.
//
// This is RAII: tie the resource's lifetime to a scope and stop hand-managing it.
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
  std::lock_guard<std::mutex> guard(bookMutex);  // unlocked when 'guard' dies

  try {
    if (denominator == 0) {
      throw std::invalid_argument("worker: cannot quote against zero volume");
    }

    lastQuote = notional / denominator;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    reportQuote(denominator);
  } catch (const std::invalid_argument& e) {
    std::cout << e.what() << '\n';
    return;  // safe now -- guard's destructor releases the mutex
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
