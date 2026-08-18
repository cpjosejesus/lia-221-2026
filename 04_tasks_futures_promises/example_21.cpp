#include <cmath>
#include <future>
#include <iostream>
#include <thread>

void computeSqrt(std::promise<double>&& prms, double input) {
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));  // simulate work
  double output = sqrt(input);
  prms.set_value(output);
}

int main() {
  // define input data
  double inputData = 42.0;

  // create promise and future
  std::promise<double> prms;
  std::future<double> ftr = prms.get_future();

  // start thread and pass promise as argument
  std::thread t(computeSqrt, std::move(prms), inputData);

  // TODO (STARTS here): Poll the future with a timeout instead of blocking on get().
  //   1. Call ftr.wait_for(...) with a timeout SHORTER than the worker's 2000 ms
  //      of simulated work (e.g. 1000 ms) and capture the returned status.
  //   2. If the status is std::future_status::ready, the value is available:
  //      call ftr.get() and print the result.
  //   3. If the status is std::future_status::timeout (result not ready yet) or
  //      std::future_status::deferred (task not started), print "Result unavailable".
  //   Experiment: change the timeout to 3000 ms and observe how the branch taken
  //   changes. Notice that wait_for lets you keep the main thread responsive
  //   instead of blocking indefinitely on ftr.get().
  auto status = ftr.wait_for(std::chrono::milliseconds(1000));
  if (status == std::future_status::ready)  // result is ready
  {
    std::cout << "Result = " << ftr.get() << std::endl;
  }

  //  timeout has expired or function has not yet been started
  else if (status == std::future_status::timeout || status == std::future_status::deferred) {
    std::cout << "Result unavailable" << std::endl;
  }
  // TODO (ENDS here)

  // thread barrier
  t.join();

  return 0;
}
