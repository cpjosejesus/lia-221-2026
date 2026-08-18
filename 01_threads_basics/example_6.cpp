// UNDEFINED BEHAVIOUR ON PURPOSE (same defect as example_4).
// Note where join() *would* go: calling it here, before main's own work, would
// serialise the two and destroy the concurrency you paid a thread for.
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

  // settler.join();   // joining HERE would make the program fully sequential

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "main: order 1 acknowledged\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "main: order 2 acknowledged\n";

  return 0;
}
