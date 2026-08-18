// Running a member function on a thread. The second argument is the object the
// call binds to -- and passing it BY VALUE copies it, so the mutation lands on
// the copy and is lost. Passing an address binds to the original.
#include <iostream>
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
  Account copied, original;

  std::thread t1(&Account::deposit, copied, 100);     // operates on a COPY
  std::thread t2(&Account::deposit, &original, 200);  // operates on the original

  t1.join();
  t2.join();

  copied.print();    // still 0 -- the deposit went to a temporary copy
  original.print();  // 200 -- as intended

  return 0;
}
