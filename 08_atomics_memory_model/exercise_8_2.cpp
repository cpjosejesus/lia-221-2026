// Exercise 8.2 - Restaurant pickup slots with compare-and-swap
//
// A restaurant hands food over at `capacity` pickup slots. Couriers arrive at random
// moments, take a slot if one is free, spend a little time collecting the order and
// release the slot. At no instant may more couriers be collecting than there are slots.
//
// TODO (STARTS here):
//   1. Implement tryTakeSlot() with a compare_exchange loop on `_busy`, as in
//      example_44: succeed only if the number of busy slots is below `_capacity`.
//   2. Implement releaseSlot() with one atomic decrement.
//   3. Implement recordPeak(): keep `_peak` equal to the highest value `_busy` ever had.
//      A plain "if (busy > peak) peak = busy" is a race between the compare and the
//      store. Write it as a CAS loop that only ever raises `_peak`.
//   4. Run it: the final line must report peak <= capacity and every courier served.
//      Then set capacity to 1 and 8 and explain how the total time changes.
// TODO (ENDS here)
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

class Restaurant {
 public:
  Restaurant(int capacity) : _capacity(capacity) {}

  bool tryTakeSlot() {
    // TODO: CAS loop. Placeholder always refuses so the program compiles and terminates.
    return _capacity < 0;
  }

  void releaseSlot() {
    // TODO
  }

  void recordPeak() {
    // TODO: raise _peak to _busy if _busy is larger, atomically
  }

  int peak() const { return _peak.load(); }
  int busy() const { return _busy.load(); }

 private:
  const int _capacity;
  std::atomic<int> _busy{0};
  std::atomic<int> _peak{0};
};

int main() {
  const int capacity = 2, couriers = 12;
  Restaurant restaurant(capacity);
  std::atomic<int> served{0};

  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int c = 0; c < couriers; ++c) {
    threads.emplace_back([&, c] {
      std::mt19937 rng(c);
      std::uniform_int_distribution<int> wait(1, 5);
      for (int attempt = 0; attempt < 200; ++attempt) {
        if (restaurant.tryTakeSlot()) {
          restaurant.recordPeak();
          std::this_thread::sleep_for(std::chrono::milliseconds(wait(rng)));
          restaurant.releaseSlot();
          served.fetch_add(1);
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }
  for (auto& t : threads) t.join();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

  std::cout << "capacity=" << capacity << " served=" << served << "/" << couriers
            << " peak busy=" << restaurant.peak() << " busy now=" << restaurant.busy()
            << " in " << ms << " ms" << std::endl;
  std::cout << (restaurant.peak() <= capacity && served == couriers ? "OK" : "NOT YET: implement the TODOs") << std::endl;
  return 0;
}
