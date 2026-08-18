// The fix: let a smart pointer own the resource. unique_ptr is move-only, so
// the compiler DELETES the copy constructor for you -- the shallow-copy bug of
// example_28 becomes impossible to write, and the destructor is free.
//
// The remaining hazard is the moved-from state: accessing it is still your
// responsibility, so this version reports it instead of invoking UB.
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

class Order {
 public:
  Order() : _id(0), _symbol(std::make_unique<std::string>("UNSET")) {
    std::cout << "Order #" << _id << ": default ctor\n";
  }

  Order(int id, std::string symbol)
      : _id(id), _symbol(std::make_unique<std::string>(std::move(symbol))) {
    std::cout << "Order #" << _id << ": init ctor\n";
  }

  // Copy ctor is implicitly deleted (unique_ptr is not copyable) -- good.
  Order(Order&& src) noexcept : _id(src._id), _symbol(std::move(src._symbol)) {
    src._id = 0;
    std::cout << "Order #" << _id << ": move ctor\n";
  }

  void setSymbol(std::string symbol) { *_symbol = std::move(symbol); }

  std::string symbol() const {
    if (!_symbol) {
      throw std::logic_error("read from a moved-from Order");
    }
    return *_symbol;
  }

 private:
  int _id;
  std::unique_ptr<std::string> _symbol;
};

int main() {
  Order resting;

  std::future<void> task = std::async([](Order o) {
    o.setSymbol("GLOBEX");
  }, std::move(resting));

  task.wait();

  try {
    std::cout << resting.symbol() << '\n';  // 'resting' was moved from
  } catch (const std::logic_error& e) {
    std::cout << "caught: " << e.what() << '\n';
  }

  return 0;
}
