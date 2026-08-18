// The std::thread constructor is variadic: extra arguments are forwarded to the
// callable. They must satisfy the signature exactly -- see the third one.
#include <iostream>
#include <string>
#include <thread>

void logOrder(int orderId) {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "order " << orderId << '\n';
}

void logOrderWithSymbol(int orderId, std::string symbol) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::cout << "order " << orderId << " on " << symbol << '\n';
}

int main() {
  int orderId = 0;

  std::thread t1(logOrder, orderId);
  std::thread t2(logOrderWithSymbol, ++orderId, "ACME");

  // COMPILE ERROR if uncommented: logOrderWithSymbol needs two arguments.
  // Thread argument mismatches are caught at compile time, not at run time.
  // std::thread t3(logOrderWithSymbol, ++orderId);

  t1.join();
  t2.join();

  return 0;
}
