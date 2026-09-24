// Memory ordering: publishing data from one thread to another.
// The producer fills a plain Vehicle record, then sets an atomic flag with
// memory_order_release. The consumer waits for the flag with memory_order_acquire.
// The release/acquire pair guarantees the record is fully visible after the flag.
#include <atomic>
#include <iostream>
#include <string>
#include <thread>

struct VehicleRecord {  // deliberately NOT atomic: several plain fields
  int id = 0;
  std::string name;
  double speed = 0.0;
};

VehicleRecord record;               // shared, written by the producer only before the flag
std::atomic<bool> ready{false};
std::atomic<int> polls{0};          // statistics only: relaxed is enough

void producer() {
  record.id = 7;
  record.name = "Vehicle 7";
  record.speed = 42.5;
  ready.store(true, std::memory_order_release);   // everything above happens-before a matching acquire
}

void consumer() {
  while (!ready.load(std::memory_order_acquire)) {
    polls.fetch_add(1, std::memory_order_relaxed);  // nobody synchronizes on this counter
  }
  // Safe: the writes to `record` are visible here, and only here, because of release/acquire.
  std::cout << "consumer sees id=" << record.id << " name=" << record.name << " speed=" << record.speed
            << " after " << polls.load(std::memory_order_relaxed) << " polls" << std::endl;
}

int main() {
  std::thread c(consumer);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::thread p(producer);
  p.join();
  c.join();

  std::cout << "\nOrderings from weakest to strongest:\n"
               "  relaxed  - atomicity only, no ordering (statistics counters)\n"
               "  release  - writes before this store are visible to an acquire that reads it\n"
               "  acquire  - reads after this load see the writes before the matching release\n"
               "  seq_cst  - the default: a single global order everyone agrees on\n";
  return 0;
}
