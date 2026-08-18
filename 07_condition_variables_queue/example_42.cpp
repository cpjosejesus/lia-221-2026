// The finished producer/consumer queue -- and unlike the previous examples in
// this folder, IT TERMINATES.
//
// Two additions over example_41:
//   1. A _closed flag plus notify_all() gives the consumer a way to learn that
//      no more orders are coming. The wait predicate becomes
//      "an order is available OR the book is closed".
//   2. take() returns std::optional<Order>: an empty optional is the drained
//      -and-closed signal, which is cleaner than an exception or a sentinel id.
//
// This is the shape of nearly every real work queue: blocking pop, no polling,
// no busy-wait, and a shutdown path that does not leak a blocked thread.
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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

  // Blocks until an order is available, or returns std::nullopt once the book
  // is closed and fully drained.
  std::optional<Order> take() {
    std::unique_lock<std::mutex> lock(_mutex);

    _ready.wait(lock, [this] { return !_orders.empty() || _closed; });

    if (_orders.empty()) {
      return std::nullopt;  // closed and drained -- tell the consumer to stop
    }

    Order order = std::move(_orders.back());
    _orders.pop_back();
    return order;
  }

  void submit(Order&& order) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::lock_guard<std::mutex> lock(_mutex);
    std::cout << "   order #" << order.id() << " submitted\n";
    _orders.push_back(std::move(order));

    _ready.notify_one();
  }

  // Called once every producer has finished.
  void close() {
    std::lock_guard<std::mutex> lock(_mutex);
    _closed = true;
    _ready.notify_all();  // wake EVERY waiter so none blocks forever
  }

 private:
  std::mutex _mutex;
  std::condition_variable _ready;
  std::vector<Order> _orders;
  bool _closed = false;
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

  // The consumer runs concurrently with the producers.
  std::future<void> consumer = std::async(std::launch::async, [book] {
    while (std::optional<Order> order = book->take()) {
      std::cout << "   order #" << order->id() << " matched\n";
    }
    std::cout << "   consumer drained the book and stopped\n";
  });

  for (std::future<void>& producer : producers) {
    producer.wait();
  }

  book->close();     // no more orders will arrive
  consumer.wait();   // let the consumer finish draining

  std::cout << "done\n";

  return 0;
}
