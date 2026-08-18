// Is this a data race? No -- and the reason matters.
// The lambda takes Order BY VALUE, so the worker mutates its own copy. Main's
// object is untouched. Passing by reference (or sharing a pointer) is what
// turns concurrent access into a race.
#include <future>
#include <iostream>
#include <thread>

class Order {
 public:
  Order() : _id(0) {
    std::cout << "Order #" << _id << ": default ctor\n";
  }

  explicit Order(int id) : _id(id) {
    std::cout << "Order #" << _id << ": init ctor\n";
  }

  void setId(int id) { _id = id; }
  int id() const { return _id; }

 private:
  int _id;
};

int main() {
  Order resting;      // default ctor
  Order marketable(1);  // init ctor
  (void)marketable;

  // 'resting' is COPIED into the task -- the worker cannot reach main's object.
  std::future<void> task = std::async([](Order o) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    o.setId(2);  // mutates the copy only
  }, resting);

  resting.setId(3);

  task.wait();
  std::cout << "main's order id = " << resting.id() << " (3 -- worker saw a copy)\n";

  return 0;
}
