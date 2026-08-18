// std::async wires up the promise/future pair for you: return a value and it
// lands in the future; throw and the exception is rethrown by get().
// Compare the line count with example_22 -- same semantics, far less plumbing.
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

double computeRatio(double numerator, double denominator) {
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  if (denominator == 0) {
    throw std::runtime_error("worker: cannot price against zero volume");
  }

  return numerator / denominator;
}

int main() {
  std::future<double> result = std::async(computeRatio, 42.0, 2.0);

  try {
    std::cout << "ratio = " << result.get() << '\n';
  } catch (const std::runtime_error& e) {
    std::cout << "caught in main: " << e.what() << '\n';
  }

  return 0;
}
