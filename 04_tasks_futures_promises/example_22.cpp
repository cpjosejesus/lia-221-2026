// A future carries exceptions as well as values. set_exception() on the worker
// side makes get() rethrow in the consumer -- so error handling crosses the
// thread boundary intact instead of terminating the process.
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

void computeRatio(std::promise<double>&& sink, double numerator, double denominator) {
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  try {
    if (denominator == 0) {
      throw std::runtime_error("worker: cannot price against zero volume");
    }
    sink.set_value(numerator / denominator);
  } catch (...) {
    sink.set_exception(std::current_exception());  // ship the exception home
  }
}

int main() {
  std::promise<double> sink;
  std::future<double> result = sink.get_future();

  std::thread worker(computeRatio, std::move(sink), 42.0, 0.0);

  try {
    std::cout << "ratio = " << result.get() << '\n';
  } catch (const std::runtime_error& e) {
    std::cout << "caught in main: " << e.what() << '\n';
  }

  worker.join();

  return 0;
}
