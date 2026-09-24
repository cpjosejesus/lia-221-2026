// Compare-and-swap (CAS): admit at most `capacity` vehicles into an intersection
// without a mutex. 20 vehicles try at once; exactly `capacity` succeed.
#include <atomic>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

class Vehicle {
 public:
  Vehicle(int id) : _id(id) {}
  int getID() const { return _id; }
 private:
  int _id;
};

class Intersection {
 public:
  Intersection(int capacity) : _capacity(capacity) {}

  // Try to take one of the free slots. Returns true if this vehicle got in.
  bool tryEnter() {
    int current = _inside.load();
    while (current < _capacity) {
      // If _inside still equals `current`, set it to current+1 and return true.
      // Otherwise another thread changed it: `current` is refreshed and we re-check the condition.
      if (_inside.compare_exchange_weak(current, current + 1))
        return true;
    }
    return false;
  }

  void leave() { _inside.fetch_sub(1); }
  int inside() const { return _inside.load(); }

 private:
  const int _capacity;
  std::atomic<int> _inside{0};
};

int main() {
  Intersection intersection(3);
  std::atomic<int> admitted{0}, rejected{0};

  std::vector<std::future<void>> futures;
  for (int i = 0; i < 20; ++i) {
    Vehicle v(i);
    futures.emplace_back(std::async(std::launch::async, [&](Vehicle vehicle) {
      if (intersection.tryEnter()) {
        admitted.fetch_add(1);
        std::cout << "Vehicle #" << vehicle.getID() << " entered (inside now " << intersection.inside() << ")\n";
        // deliberately NOT leaving, so the slots stay taken for the demonstration
      } else {
        rejected.fetch_add(1);
      }
    }, std::move(v)));
  }
  for (auto& f : futures) f.wait();

  std::cout << "admitted = " << admitted << ", rejected = " << rejected
            << ", inside = " << intersection.inside() << " (capacity 3)" << std::endl;

  // exchange(): atomically replace and get the old value. Here: "reset and report".
  std::atomic<int> token{42};
  int old = token.exchange(0);
  std::cout << "exchange returned " << old << ", token is now " << token << std::endl;
  return 0;
}
