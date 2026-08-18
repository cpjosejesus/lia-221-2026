// PRODUCER / CONSUMER with a CONDITION VARIABLE. Still runs forever once the
// producers finish (the consumer waits for a 11th order that never comes) --
// Ctrl-C to stop. example_42 adds clean shutdown.
//
// The difference from example_40: take() now BLOCKS instead of spinning. The
// waiting thread consumes no CPU until notify_one() wakes it.
//
// Two details that matter:
//   * wait() needs a unique_lock, not a lock_guard -- it must unlock while
//     asleep and re-lock on wake, which lock_guard cannot do.
//   * The predicate is re-checked on every wake, which is what makes SPURIOUS
//     WAKEUPS harmless. Never wait() without one.
#include <condition_variable>
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

  Order take() {
    std::unique_lock<std::mutex> lock(_mutex);

    // Sleeps here, holding nothing, until the predicate is true on a wake.
    _ready.wait(lock, [this] { return !_orders.empty(); });

    Order order = std::move(_orders.back());
    _orders.pop_back();
    return order;
  }

  void submit(Order&& order) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::lock_guard<std::mutex> lock(_mutex);
    std::cout << "   order #" << order.id() << " submitted\n";
    _orders.push_back(std::move(order));

    _ready.notify_one();  // wake one waiting consumer
  }

 private:
  std::mutex _mutex;
  std::condition_variable _ready;
  std::vector<Order> _orders;
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
  while (true) {  // blocks in take() rather than spinning -- but never exits
    Order order = book->take();
    std::cout << "   order #" << order.id() << " matched\n";
  }

  return 0;
}
