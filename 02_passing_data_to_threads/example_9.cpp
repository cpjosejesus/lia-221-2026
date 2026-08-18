// Closures before threads: what a lambda captures, and when it captures it.
// No concurrency here yet -- get the capture rules straight first.
#include <iostream>

int main() {
  int orderId = 0;

  // Capture by reference: reads orderId at CALL time, not at definition time.
  auto showByRef = [&orderId] {
    std::cout << "a) by-ref sees orderId = " << orderId << '\n';
  };

  // Capture by value + mutable: mutates the closure's own private copy.
  auto bumpCopy = [orderId]() mutable {
    std::cout << "b) inside copy      = " << ++orderId << '\n';
  };
  bumpCopy();
  std::cout << "c) caller unchanged = " << orderId << '\n';

  // Capture by reference: mutates the caller's variable for real.
  auto bumpOriginal = [&orderId] {
    std::cout << "d) inside by-ref    = " << ++orderId << '\n';
  };
  bumpOriginal();
  std::cout << "e) caller changed   = " << orderId << '\n';

  // No capture at all -- plain parameter passing.
  auto showParam = [](int id) {
    std::cout << "f) as parameter     = " << id << '\n';
  };
  showParam(++orderId);

  // showByRef was defined when orderId was 0, but prints its value NOW.
  showByRef();

  return 0;
}
