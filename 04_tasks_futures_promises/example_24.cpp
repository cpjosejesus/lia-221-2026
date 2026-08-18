// std::launch::async FORCES a new thread. Watch the two ids: they differ.
// Compare against example_25, which differs only in the launch policy.
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

  auto result = std::async(std::launch::async, computeRatio, 42.0, 2.0);

  try {
    std::cout << "ratio = " << result.get() << '\n';
  } catch (const std::runtime_error& e) {
    std::cout << "caught in main: " << e.what() << '\n';
  }

  return 0;
}
