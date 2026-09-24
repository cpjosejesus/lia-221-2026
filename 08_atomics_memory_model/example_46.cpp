// Which atomics are really lock-free? std::atomic<T> works for any trivially
// copyable T, but large T is implemented with a hidden lock.
#include <atomic>
#include <iostream>

struct Position {   // 8 bytes: fits in one machine word on 64-bit
  float x, y;
};

struct VehicleSnapshot {  // 72 bytes: too big for a single atomic instruction
  int id;
  char name[64];
  float speed;
};

int main() {
  std::atomic<bool> flag{false};
  std::atomic<int> counter{0};
  std::atomic<long long> big{0};
  std::atomic<Position> pos{Position{1.0f, 2.0f}};
  std::atomic<VehicleSnapshot> snapshot{};

  std::cout << std::boolalpha;
  std::cout << "atomic<bool>            lock-free: " << flag.is_lock_free() << "\n";
  std::cout << "atomic<int>             lock-free: " << counter.is_lock_free() << "\n";
  std::cout << "atomic<long long>       lock-free: " << big.is_lock_free() << "\n";
  std::cout << "atomic<Position>  (8 B) lock-free: " << pos.is_lock_free() << "\n";
  std::cout << "atomic<VehicleSnapshot> (" << sizeof(VehicleSnapshot) << " B) lock-free: " << snapshot.is_lock_free() << "\n";

  // A whole Position is read and written atomically: no torn reads (x from one write, y from another)
  pos.store(Position{10.0f, 20.0f});
  Position p = pos.load();
  std::cout << "position = (" << p.x << ", " << p.y << ")\n";

  // ATOMIC_INT_LOCK_FREE == 2 means "always lock-free on this platform"
  std::cout << "ATOMIC_INT_LOCK_FREE = " << ATOMIC_INT_LOCK_FREE << " (2 = always)\n";
  return 0;
}
