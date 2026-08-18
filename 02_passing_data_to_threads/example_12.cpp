// std::thread COPIES its arguments into internal storage by default.
// Wrapping one in std::move() transfers ownership instead -- after which the
// source string is valid but unspecified (typically empty).
#include <iostream>
#include <string>
#include <thread>

void logSymbol(std::string symbol, int delayMs) {
  std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
  std::cout << "worker got symbol = " << symbol << '\n';
}

int main() {
  std::string copied = "ACME";
  std::string moved = "GLOBEX";

  std::thread t1(logSymbol, copied, 50);              // copied into the thread
  std::thread t2(logSymbol, std::move(moved), 100);   // moved into the thread

  t1.join();
  t2.join();

  std::cout << "main still holds copied = '" << copied << "'\n";
  std::cout << "main after move,  moved = '" << moved << "' (moved-from)\n";

  return 0;
}
