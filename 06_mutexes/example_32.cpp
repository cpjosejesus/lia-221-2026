// The fix for 05/example_31: a std::mutex serialises access to the shared
// state, so the 1000 submissions no longer trample each other. Result is now
// reliably 1000, every run.
//
// Note the manual lock()/unlock() pair -- correct here, but fragile. If
// push_back throws, the mutex is never released and the program deadlocks.
// example_35 replaces this with RAII.
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
  OrderBook() = default;

  void printSize() {
    _mutex.lock();
    std::cout << "orders = " << _orders.size() << " (expected 1000)\n";
    _mutex.unlock();
  }

  void submit(Order&& order) {
    _mutex.lock();                          // enter the critical section
    _orders.emplace_back(std::move(order));
    _mutex.unlock();                        // leave it -- easy to forget
  }

 private:
  std::vector<Order> _orders;
  std::mutex _mutex;
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

  book->printSize();

  return 0;
}
