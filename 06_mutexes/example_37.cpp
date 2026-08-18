// DEADLOCK ON PURPOSE -- this program hangs. Ctrl-C to stop it.
//
// The lesson: RAII is NOT a deadlock cure. Every lock here is a lock_guard, so
// each will unlock correctly -- but the acquisition order is still opposite
// between the two threads, so neither reaches its second guard.
//
// Correctness of release and correctness of ORDER are separate problems.
#include <iostream>
#include <mutex>
#include <thread>

std::mutex bookMutex, feedMutex;

void settleWorker() {
  std::lock_guard<std::mutex> feedLock(feedMutex);  // feed, then book
  std::cout << "settle worker holds feed\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::lock_guard<std::mutex> bookLock(bookMutex);
  std::cout << "settle worker holds both\n";
}

void quoteWorker() {
  std::lock_guard<std::mutex> bookLock(bookMutex);  // book, then feed
  std::cout << "quote worker holds book\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::lock_guard<std::mutex> feedLock(feedMutex);
  std::cout << "quote worker holds both\n";
}

int main() {
  std::thread t1(settleWorker);
  std::thread t2(quoteWorker);

  t1.join();
  t2.join();

  std::cout << "unreachable -- both threads are stuck\n";

  return 0;
}
