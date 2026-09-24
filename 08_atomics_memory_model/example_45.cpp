// A spinlock built from std::atomic_flag, used exactly like a mutex to protect
// the vehicle queue. Compare its behaviour and timing with std::mutex.
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class Vehicle {
 public:
  Vehicle(int id) : _id(id) {}
 private:
  int _id;
};

class Spinlock {
 public:
  void lock() {
    // test_and_set returns the previous value: keep spinning while it was already set (held)
    while (_flag.test_and_set(std::memory_order_acquire)) {
      // busy-wait: the thread burns CPU instead of sleeping. Fine for very short critical sections.
    }
  }
  void unlock() { _flag.clear(std::memory_order_release); }

 private:
  std::atomic_flag _flag = ATOMIC_FLAG_INIT;
};

template <class Lock>
class WaitingVehicles {
 public:
  void pushBack(Vehicle&& v) {
    std::lock_guard<Lock> guard(_lock);  // lock_guard works with anything that has lock()/unlock()
    _vehicles.push_back(std::move(v));
  }
  size_t size() {
    std::lock_guard<Lock> guard(_lock);
    return _vehicles.size();
  }

 private:
  std::vector<Vehicle> _vehicles;
  Lock _lock;
};

template <class Lock>
void benchmark(const char* label) {
  WaitingVehicles<Lock> queue;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&queue, t] {
      for (int i = 0; i < 20000; ++i) queue.pushBack(Vehicle(t * 100000 + i));
    });
  }
  for (auto& th : threads) th.join();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  std::cout << label << ": " << queue.size() << " vehicles in " << ms << " ms" << std::endl;
}

int main() {
  benchmark<std::mutex>("std::mutex ");
  benchmark<Spinlock>("Spinlock   ");
  return 0;
}
