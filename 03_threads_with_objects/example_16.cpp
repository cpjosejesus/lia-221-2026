// Two threads mutating two DIFFERENT members of one shared object. This
// happens to be fine, because distinct non-overlapping members are distinct
// memory locations. Point both threads at the SAME member and you have the
// data race that folder 05 is about.
#include <iostream>
#include <memory>
#include <string>
#include <thread>

class Account {
 public:
  Account() : _balance(0) {}

  void deposit(int amount) { _balance = amount; }
  void rename(std::string owner) { _owner = std::move(owner); }

  void print() const {
    std::cout << "Account owner=" << _owner << " balance=" << _balance << '\n';
  }

 private:
  int _balance;
  std::string _owner;
};

int main() {
  auto account = std::make_shared<Account>();

  std::thread t1(&Account::deposit, account, 500);
  std::thread t2(&Account::rename, account, "desk-a");

  t1.join();
  t2.join();

  account->print();

  return 0;
}
