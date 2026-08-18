// Blocking on get() means main can do nothing else. wait_for() lets you ask
// "is it ready yet?" with a deadline and stay responsive in the meantime.
#include <cmath>
#include <future>
#include <iostream>
#include <thread>

void computeVwap(std::promise<double>&& sink, double notional) {
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));  // slow pricing
  sink.set_value(std::sqrt(notional));
}

int main() {
  const double notional = 42.0;

  std::promise<double> sink;
  std::future<double> result = sink.get_future();

  std::thread worker(computeVwap, std::move(sink), notional);

  // TODO (STARTS here): poll the future instead of blocking on get().
  //   1. Call result.wait_for(...) with a timeout SHORTER than the worker's
  //      2000 ms of simulated pricing (1000 ms is a good start) and keep the
  //      returned std::future_status.
  //   2. If the status is std::future_status::ready the value has arrived --
  //      call result.get() and print it.
  //   3. If it is std::future_status::timeout (not ready yet) or
  //      std::future_status::deferred (the task has not even started), print
  //      "quote not ready".
  //   Experiment: raise the timeout to 3000 ms and watch which branch runs.
  //   Then try polling in a loop, printing a heartbeat between attempts --
  //   that is the pattern behind any responsive UI or health-checked worker.
  auto status = result.wait_for(std::chrono::milliseconds(1000));

  if (status == std::future_status::ready) {
    std::cout << "vwap = " << result.get() << '\n';
  } else if (status == std::future_status::timeout ||
             status == std::future_status::deferred) {
    std::cout << "quote not ready\n";
  }
  // TODO (ENDS here)

  worker.join();

  return 0;
}
