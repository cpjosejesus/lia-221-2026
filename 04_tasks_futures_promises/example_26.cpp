// Threads are not free. With a tiny workload the spawn overhead dominates and
// the "parallel" version loses to doing it inline.
//
// Try it yourself: raise workPerTask to 1'000'000 so each task actually earns
// its thread, then compare std::launch::async against std::launch::deferred by
// swapping the policy below. The crossover point is the whole lesson.
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

void revalueBook(int iterations) {
  std::cout << "task on thread id = " << std::this_thread::get_id() << '\n';

  double checksum = 0.0;
  for (int i = 0; i < iterations; ++i) {
    checksum += std::sqrt(12345.6789);
  }

  // Consume the result so the optimiser cannot delete the loop entirely.
  if (checksum < 0) {
    std::cout << "unreachable\n";
  }
}

int main() {
  std::cout << "main thread id    = " << std::this_thread::get_id() << '\n';

  const int workPerTask = 10;  // deliberately tiny -- overhead will dominate
  const int taskCount = 5;

  const auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::future<void>> tasks;
  tasks.reserve(taskCount);
  for (int i = 0; i < taskCount; ++i) {
    tasks.emplace_back(std::async(std::launch::async, revalueBook, workPerTask));
  }

  for (const std::future<void>& task : tasks) {
    task.wait();
  }

  const auto stop = std::chrono::high_resolution_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

  std::cout << "finished in " << elapsed << " microseconds\n";

  return 0;
}
