// std::launch::deferred creates NO thread. The task runs lazily, inside get(),
// on the calling thread -- so both ids match and nothing runs at all if you
// never call get(). Same code as example_24; only the policy changed.
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

double computeRatio(double numerator, double denominator) {
  std::cout << "task ran on thread id = " << std::this_thread::get_id() << '\n';

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  if (denominator == 0) {
    throw std::runtime_error("worker: cannot price against zero volume");
  }

  return numerator / denominator;
}

int main() {
  std::cout << "main thread id        = " << std::this_thread::get_id() << '\n';

  auto result = std::async(std::launch::deferred, computeRatio, 42.0, 2.0);

  std::cout << "(task has not started yet)\n";

  try {
    std::cout << "ratio = " << result.get() << '\n';  // runs the task right here
  } catch (const std::runtime_error& e) {
    std::cout << "caught in main: " << e.what() << '\n';
  }

  return 0;
}
