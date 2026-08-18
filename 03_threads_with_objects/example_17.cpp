// std::thread is move-only: it owns an OS resource, so copying it makes no
// sense. Build containers of threads with emplace_back (construct in place)
// rather than push_back of a named thread.
#include <iostream>
#include <thread>
#include <vector>

void greetFromWorker() {
  std::cout << "worker thread id = " << std::this_thread::get_id() << '\n';
}

int main() {
  std::vector<std::thread> pool;
  pool.reserve(5);

  for (int i = 0; i < 5; ++i) {
    // COMPILE ERROR if uncommented -- thread has no copy constructor:
    // std::thread t(greetFromWorker);
    // pool.push_back(t);

    pool.emplace_back(greetFromWorker);  // constructed directly in the vector
  }

  std::cout << "main thread id   = " << std::this_thread::get_id() << '\n';

  for (std::thread& worker : pool) {
    worker.join();
  }

  return 0;
}
