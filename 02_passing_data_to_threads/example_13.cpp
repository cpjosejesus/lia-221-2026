// To hand a thread a genuine reference you must say so explicitly with
// std::ref -- otherwise the copy above would silently defeat the &-parameter.
// The thread then mutates main's object, which is exactly why folder 05 exists.
#include <iostream>
#include <string>
#include <thread>

void annotate(std::string& symbol, int delayMs) {
  std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
  symbol += " [touched by worker]";
  std::cout << symbol << '\n';
}

int main() {
  std::string symbol = "ACME";

  std::thread worker(annotate, std::ref(symbol), 50);

  worker.join();  // joining FIRST is what makes the shared access safe here

  symbol += " [touched by main]";
  std::cout << symbol << '\n';

  return 0;
}
