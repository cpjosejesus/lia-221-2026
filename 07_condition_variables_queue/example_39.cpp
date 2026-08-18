// The MONITOR OBJECT pattern: the mutex lives INSIDE the class alongside the
// data it protects, and every public method locks it. Callers cannot forget to
// synchronise, because there is no unsynchronised way in.
//
// This version is producer-only -- all writes, then one read at the end.
// example_40 adds a consumer running concurrently, which is where it gets hard.
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

  void printAll() {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const Order& order : _orders) {
      std::cout << "   order #" << order.id() << " is resting in the book\n";
    }
  }

  void submit(Order&& order) {
    std::lock_guard<std::mutex> lock(_mutex);
    std::cout << "   order #" << order.id() << " submitted\n";
    _orders.emplace_back(std::move(order));

    // NOTE: sleeping while holding the lock -- every other producer waits.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

 private:
  std::vector<Order> _orders;
  std::mutex _mutex;
};

int main() {
  auto book = std::make_shared<OrderBook>();

  std::cout << "submitting...\n";
  std::vector<std::future<void>> producers;
  producers.reserve(10);

  for (int i = 0; i < 10; ++i) {
    Order order(i);
    producers.emplace_back(
        std::async(std::launch::async, &OrderBook::submit, book, std::move(order)));
  }

  for (std::future<void>& producer : producers) {
    producer.wait();
  }

  std::cout << "resting book:\n";
  book->printAll();

  return 0;
}
