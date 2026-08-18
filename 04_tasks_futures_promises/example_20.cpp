// promise/future is a single-use, one-way channel between two threads.
// The worker owns the promise (write end); main holds the future (read end).
// ftr.get() blocks until the value arrives -- and may be called only once.
#include <future>
#include <iostream>
#include <string>
#include <thread>

void priceOrder(std::promise<std::string>&& sink, std::string order) {
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  sink.set_value(order + " -> filled @ 101.25");
}

int main() {
  std::string order = "BUY 100 ACME";

  std::promise<std::string> sink;
  std::future<std::string> result = sink.get_future();

  // The promise is move-only: it must be moved into the worker, never copied.
  std::thread worker(priceOrder, std::move(sink), order);

  std::cout << "sent:     " << order << '\n';

  std::cout << "received: " << result.get() << '\n';  // blocks until set_value

  worker.join();

  return 0;
}
