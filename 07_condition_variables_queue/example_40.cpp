// BUSY-WAIT (POLLING) -- runs forever and pins a core at 100%. Ctrl-C to stop.
//
// The consumer loops asking "anything yet?" as fast as it can. Two problems:
//   1. It burns CPU doing nothing between arrivals.
//   2. hasOrders() and take() are each locked, but the GAP between them is not
//      -- with a second consumer, both could pass the check and one would then
//      pop from an empty vector. A check-then-act race.
//
// example_41 replaces the polling with a condition variable, fixing both.
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

  bool hasOrders() {
    std::lock_guard<std::mutex> lock(_mutex);
    return !_orders.empty();
  }

  Order take() {
    std::lock_guard<std::mutex> lock(_mutex);
    Order order = std::move(_orders.back());
    _orders.pop_back();
    return order;  // moved out, not copied (RVO / implicit move on return)
  }

  void submit(Order&& order) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // work OUTSIDE the lock

    std::lock_guard<std::mutex> lock(_mutex);
    std::cout << "   order #" << order.id() << " submitted\n";
    _orders.emplace_back(std::move(order));
  }

 private:
  std::vector<Order> _orders;
  std::mutex _mutex;
};

int main() {
  auto book = std::make_shared<OrderBook>();

  std::cout << "starting producers...\n";
  std::vector<std::future<void>> producers;
  producers.reserve(10);

  for (int i = 0; i < 10; ++i) {
    Order order(i);
    producers.emplace_back(
        std::async(std::launch::async, &OrderBook::submit, book, std::move(order)));
  }

  std::cout << "matching...\n";
  while (true) {  // spins hot forever, even once the producers are done
    if (book->hasOrders()) {
      Order order = book->take();
      std::cout << "   order #" << order.id() << " matched\n";
    }
  }

  // Unreachable -- the loop above never exits.
  return 0;
}
