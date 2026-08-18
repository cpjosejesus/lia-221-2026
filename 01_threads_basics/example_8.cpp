// Six detached workers and a sleep() used as a (bad) synchronisation device.
// Sleeping is a guess, not a guarantee -- shorten the final sleep and output
// starts disappearing. Folders 4 and 7 show how to wait properly.
#include <iostream>
#include <thread>

void handleBuy() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::cout << "buy side\n";
}

void handleSell() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::cout << "sell side\n";
}

int main() {
  for (int i = 0; i < 6; ++i) {
    std::thread worker(i % 2 == 0 ? handleBuy : handleSell);
    worker.detach();
  }

  // Hoping the detached workers finish in time -- not a real barrier.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  std::cout << "main returning\n";
  return 0;
}
