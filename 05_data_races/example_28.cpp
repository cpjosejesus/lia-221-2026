// Now the class holds a RAW OWNING POINTER and has no user-defined copy
// constructor, so the implicit one copies the pointer, not the string.
// Original and copy alias the same heap object: two threads, one buffer.
//
// This is a shallow-copy bug AND a double-delete waiting to happen. Run it
// under a sanitiser: g++ -std=c++17 -pthread -fsanitize=thread example_28.cpp
#include <future>
#include <iostream>
#include <string>
#include <thread>

class Order {
 public:
  Order() : _id(0), _symbol(new std::string("UNSET")) {
    std::cout << "Order #" << _id << ": default ctor\n";
  }

  Order(int id, std::string symbol) : _id(id), _symbol(new std::string(std::move(symbol))) {
    std::cout << "Order #" << _id << ": init ctor\n";
  }

  // No copy ctor, no destructor, no rule of five -- that is the defect.

  void setSymbol(std::string symbol) { *_symbol = std::move(symbol); }
  std::string symbol() const { return *_symbol; }

 private:
  int _id;
  std::string* _symbol;  // owning raw pointer: shallow-copied
};

int main() {
  Order resting;

  // The copy shares _symbol with main's object -- both threads write through it.
  std::future<void> task = std::async([](Order o) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    o.setSymbol("GLOBEX");  // writes into MAIN's string
  }, resting);

  resting.setSymbol("ACME");  // concurrent write to the same string

  task.wait();
  std::cout << "symbol = " << resting.symbol() << " (racy: either value)\n";

  return 0;
}
