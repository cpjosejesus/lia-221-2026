// join() placed AFTER main's own work: both sides run concurrently and their
// output interleaves unpredictably. Run it a few times -- the order changes.
#include <iostream>
#include <thread>

void settleTrade() {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "worker: leg 1 settled\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "worker: leg 2 settled\n";
}

int main() {
  std::thread settler(settleTrade);

  // Main keeps working while the settler runs -- this is the concurrency.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "main: order 1 acknowledged\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "main: order 2 acknowledged\n";

  settler.join();  // barrier: main waits here until the worker is done

  return 0;
}
