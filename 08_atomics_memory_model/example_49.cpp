// A lock-free stack of vehicles (Treiber stack): push and pop with compare-and-swap,
// no mutex. Note the two classic problems it exposes: the ABA problem and memory
// reclamation. This demo never frees popped nodes, which is why it is safe.
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

struct Vehicle {
  int id;
};

class LockFreeStack {
  struct Node {
    Vehicle vehicle;
    Node* next;
  };

 public:
  void push(Vehicle v) {
    Node* node = new Node{v, _head.load(std::memory_order_relaxed)};
    // Try to make `node` the new head as long as node->next is still the current head.
    while (!_head.compare_exchange_weak(node->next, node, std::memory_order_release, std::memory_order_relaxed)) {
      // on failure node->next has been updated to the current head; just retry
    }
  }

  bool pop(Vehicle& out) {
    Node* old = _head.load(std::memory_order_acquire);
    while (old && !_head.compare_exchange_weak(old, old->next, std::memory_order_acquire, std::memory_order_relaxed)) {
    }
    if (!old) return false;
    out = old->vehicle;
    // NOT deleting `old`: another thread may still be reading old->next in its CAS loop.
    // Safe reclamation needs hazard pointers, epochs or a garbage collector (self-study).
    // Also: if `old` were freed and the same address reused for a new node, a CAS could
    // succeed on a stale view of the stack: the ABA problem.
    return true;
  }

 private:
  std::atomic<Node*> _head{nullptr};
};

int main() {
  LockFreeStack stack;
  const int producers = 4, perProducer = 5000;
  std::atomic<int> popped{0};
  std::atomic<long> idSum{0};

  std::vector<std::thread> threads;
  for (int p = 0; p < producers; ++p)
    threads.emplace_back([&, p] { for (int i = 0; i < perProducer; ++i) stack.push(Vehicle{p * perProducer + i}); });
  for (int c = 0; c < 4; ++c)
    threads.emplace_back([&] {
      Vehicle v;
      for (int tries = 0; tries < 200000 && popped.load() < producers * perProducer; ++tries)
        if (stack.pop(v)) { popped.fetch_add(1); idSum.fetch_add(v.id); }
    });
  for (auto& t : threads) t.join();

  Vehicle v;
  while (stack.pop(v)) { popped.fetch_add(1); idSum.fetch_add(v.id); }

  long n = producers * perProducer;
  std::cout << "pushed " << n << ", popped " << popped << ", id sum " << idSum
            << " (expected " << n * (n - 1) / 2 << ")" << std::endl;
  return 0;
}
