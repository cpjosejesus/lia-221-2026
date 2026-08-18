// Two equivalent ways to hand data to a thread, side by side: a variadic
// argument versus a lambda capture. Both copy; pick whichever reads better.
#include <iostream>
#include <string>
#include <thread>

void report(std::string note) {
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::cout << "variadic form: " << note << '\n';
}

int main() {
  std::string note = "fill received";

  std::thread t1(report, note);  // forwarded as a constructor argument

  std::thread t2([note] {        // captured by the closure
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "lambda form:   " << note << '\n';
  });

  t1.join();
  t2.join();

  return 0;
}
