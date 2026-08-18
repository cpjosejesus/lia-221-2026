// A shared_ptr is the safe way to give a thread access to an object: the
// pointer is copied (cheap), the pointee is shared, and the object outlives
// main's scope for as long as any thread still holds a reference to it.
#include <iostream>
#include <memory>
#include <thread>

class Account {
 public:
  Account() : _balance(0) {}

  void deposit(int amount) { _balance = amount; }

  void print() const {
    std::cout << "Account balance = " << _balance << '\n';
  }

 private:
  int _balance;
};

int main() {
  auto account = std::make_shared<Account>();

  std::thread worker(&Account::deposit, account, 150);
  worker.join();

  account->print();  // 150 -- the thread mutated the shared object

  return 0;
}
