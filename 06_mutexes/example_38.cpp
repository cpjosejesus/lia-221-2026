// The deadlock fix. std::lock() acquires both mutexes as one atomic step using
// a deadlock-avoidance algorithm -- so the acquisition ORDER in the source no
// longer matters. std::adopt_lock then hands the already-held locks to guards
// for automatic release.
//
// In C++17 std::scoped_lock does the same in one line:
//   std::scoped_lock guard(bookMutex, feedMutex);
// Try replacing both bodies with it -- that is the modern form.
#include <iostream>
#include <mutex>
#include <thread>

std::mutex bookMutex, feedMutex;

void settleWorker() {
  std::lock(bookMutex, feedMutex);  // both, atomically, no deadlock possible
  std::lock_guard<std::mutex> feedLock(feedMutex, std::adopt_lock);
  std::lock_guard<std::mutex> bookLock(bookMutex, std::adopt_lock);
  std::cout << "settle worker holds both\n";
}

void quoteWorker() {
  std::lock(bookMutex, feedMutex);  // opposite listing order -- still safe
  std::lock_guard<std::mutex> bookLock(bookMutex, std::adopt_lock);
  std::lock_guard<std::mutex> feedLock(feedMutex, std::adopt_lock);
  std::cout << "quote worker holds both\n";
}

int main() {
  std::thread t1(settleWorker);
  std::thread t2(quoteWorker);

  t1.join();
  t2.join();

  std::cout << "finished -- no deadlock\n";

  return 0;
}
