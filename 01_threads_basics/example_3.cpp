// Every program already has one thread of execution before you create any.
// This prints the id of that implicit main thread.
#include <iostream>
#include <thread>

int main() {
  std::cout << "Engine starting on main thread id = "
            << std::this_thread::get_id() << '\n';

  return 0;
}
