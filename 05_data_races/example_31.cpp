// THE CLASSIC LOST UPDATE -- the defect this repo's 05 -> 06 -> 07 arc fixes.
//
// 1000 tasks each run: read _filled, sleep, write back read+1. Two tasks that
// read the same value both write back the same value, so one increment simply
// vanishes. Nothing here is synchronised, so the read-modify-write is not
// atomic and the final count is wrong -- and different on every run.
//
// Expected: 1000. Actual: typically a few hundred.
//
// Note _orders is guarded while _filled deliberately is not. Leaving the
// vector unguarded too would corrupt its internals and crash the process
// before it could print anything -- a real bug, but it would hide the lesson.
// Try it: drop the lock_guard from submit() and watch it abort instead.
//
// Next stop: 06_mutexes/example_32 guards the whole thing with a std::mutex.
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class Order {
 public:
  explicit Order(int id) : _id(id) {}
  int id() const { return _id; }

 private:
  int _id;
};

class OrderBook {
 public:
  OrderBook() : _filled(0) {}

  void printFilled() const {
    std::cout << "filled = " << _filled << " (expected 1000)\n";
  }

  // Called from 1000 threads at once.
  void submit(Order&& order) {
    {
      // The vector IS protected -- otherwise concurrent reallocation is UB
      // and the program aborts before reaching the interesting part.
      std::lock_guard<std::mutex> lock(_mutex);
      _orders.push_back(std::move(order));
    }

    // The counter is NOT protected. This is the race:
    const int seen = _filled;                                   // read
    std::this_thread::sleep_for(std::chrono::microseconds(1));  // widen the window
    _filled = seen + 1;                                         // write -- update lost
  }

 private:
  std::vector<Order> _orders;
  std::mutex _mutex;
  int _filled;
};

int main() {
  auto book = std::make_shared<OrderBook>();

  std::vector<std::future<void>> tasks;
  tasks.reserve(1000);

  for (int i = 0; i < 1000; ++i) {
    Order order(i);
    tasks.emplace_back(
        std::async(std::launch::async, &OrderBook::submit, book, std::move(order)));
  }

  for (std::future<void>& task : tasks) {
    task.wait();
  }

  book->printFilled();

  return 0;
}
