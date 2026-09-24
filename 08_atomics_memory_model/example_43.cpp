// Three ways to count 1000 vehicles entering an intersection from 1000 tasks.
// Only one of them is correct. Compare the three printed totals.
#include <algorithm>
#include <atomic>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

class Vehicle {
 public:
  Vehicle(int id) : _id(id) {}
 private:
  int _id;
};

// a) the folder-05 version: plain int, read-modify-write with a gap -> lost updates
class CounterPlain {
 public:
  void arrive() {
    int old = _n;
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    _n = old + 1;
  }
  int value() const { return _n; }
 private:
  int _n = 0;
};

// b) an atomic used WRONG: load and store are each atomic, but the sequence load -> add -> store is not
class CounterAtomicWrong {
 public:
  void arrive() {
    int old = _n.load();
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    _n.store(old + 1);  // overwrites increments made by other threads in between
  }
  int value() const { return _n.load(); }
 private:
  std::atomic<int> _n{0};
};

// c) an atomic used RIGHT: one indivisible read-modify-write
class CounterAtomic {
 public:
  void arrive() {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    _n.fetch_add(1);
  }
  int value() const { return _n.load(); }
 private:
  std::atomic<int> _n{0};
};

template <class Counter>
int run(const char* label) {
  auto counter = std::make_shared<Counter>();
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 1000; ++i) {
    Vehicle v(i);
    futures.emplace_back(std::async(std::launch::async, [counter](Vehicle) { counter->arrive(); }, std::move(v)));
  }
  std::for_each(futures.begin(), futures.end(), [](std::future<void>& f) { f.wait(); });
  std::cout << label << " -> #vehicles = " << counter->value() << std::endl;
  return counter->value();
}

int main() {
  run<CounterPlain>("a) plain int                ");
  run<CounterAtomicWrong>("b) atomic load/store        ");
  run<CounterAtomic>("c) atomic fetch_add         ");
  return 0;
}
