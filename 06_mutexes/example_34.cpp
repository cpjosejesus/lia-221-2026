// DEADLOCK ON PURPOSE -- this program hangs. Ctrl-C to stop it.
//
// Two mutexes, two threads, opposite acquisition orders. Each thread holds one
// lock and waits for the other, forever. This is the textbook circular wait,
// and no amount of care inside the critical section prevents it -- the bug is
// the ORDER, not the body.
//
// example_38 fixes it.
#include <iostream>
#include <mutex>
#include <thread>

std::mutex bookMutex, feedMutex;

void settleWorker() {
  feedMutex.lock();  // takes feed first...
  std::cout << "settle worker holds feed\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(10));  // make it reliable
  bookMutex.lock();  // ...then wants book -- which quoteWorker holds
  std::cout << "settle worker holds both\n";
  feedMutex.unlock();
  bookMutex.unlock();
}

void quoteWorker() {
  bookMutex.lock();  // takes book first...
  std::cout << "quote worker holds book\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  feedMutex.lock();  // ...then wants feed -- which settleWorker holds
  std::cout << "quote worker holds both\n";
  bookMutex.unlock();
  feedMutex.unlock();
}

int main() {
  std::thread t1(settleWorker);
  std::thread t2(quoteWorker);

  t1.join();
  t2.join();

  std::cout << "unreachable -- both threads are stuck\n";

  return 0;
}
