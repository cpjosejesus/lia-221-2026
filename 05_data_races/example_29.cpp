// An attempt at the rule of five -- with a deliberate landmine.
// The copy constructor initialises NOTHING: it prints _id and _symbol while
// both are uninitialised garbage. The move constructor is correct, and moving
// (rather than copying) is what keeps the task from sharing state.
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

  // BROKEN ON PURPOSE: copies nothing, then reads uninitialised members.
  // Fix it as an exercise -- deep-copy _symbol and copy _id.
  Order(Order const& src) {
    std::cout << "Order #" << _id << ": copy ctor (broken -- copies nothing)\n";
  }

  // Correct move: steal the pointer, then null out the source.
  Order(Order&& src) noexcept : _id(src._id), _symbol(src._symbol) {
    src._id = 0;
    src._symbol = nullptr;  // source must not delete what we now own
    std::cout << "Order #" << _id << ": move ctor\n";
  }

  void setSymbol(std::string symbol) { *_symbol = std::move(symbol); }
  std::string symbol() const { return *_symbol; }

 private:
  int _id;
  std::string* _symbol;
};

int main() {
  Order resting;

  // std::move selects the move ctor, so no sharing occurs -- but 'resting' is
  // left hollow (_symbol == nullptr).
  std::future<void> task = std::async([](Order o) {
    o.setSymbol("GLOBEX");
  }, std::move(resting));

  task.wait();

  // Dereferencing the moved-from order's null _symbol is UB -- see example_30
  // for the version that reports the mistake instead of crashing silently.
  // std::cout << resting.symbol() << '\n';

  std::cout << "task complete; 'resting' was moved from\n";

  return 0;
}
