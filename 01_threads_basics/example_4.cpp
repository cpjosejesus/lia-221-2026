// UNDEFINED BEHAVIOUR ON PURPOSE.
// The worker is neither joined nor detached, so ~thread() calls std::terminate
// while the worker may still be running. Uncomment the join() to fix it.
#include <iostream>
#include <thread>

void settleTrade() {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::cout << "Trade settled on worker\n";
}

int main() {
  std::thread settler(settleTrade);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "Order acknowledged on main\n";

  // settler.join();   // <-- without this, the program terminates abnormally

  return 0;
}
