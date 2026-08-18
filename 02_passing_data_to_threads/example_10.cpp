// The same capture choice, now with real threads -- and the stakes are higher.
// The by-reference thread reads orderId long after main mutated it, so its
// output depends on timing. The by-value thread is immune.
#include <iostream>
#include <thread>

int main() {
  int orderId = 0;

  // Captures by reference: races with main's ++orderId below.
  std::thread byRef([&orderId] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "a) by-reference thread sees = " << orderId << '\n';
  });

  // Captures by value: snapshotted at thread creation, safe from main.
  std::thread byValue([orderId]() mutable {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "b) by-value thread sees     = " << orderId << '\n';
  });

  ++orderId;
  std::cout << "c) main mutated orderId to  = " << orderId << '\n';

  byRef.join();
  byValue.join();

  return 0;
}
