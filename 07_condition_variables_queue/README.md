# 07 — Condition Variables and a Thread-Safe Queue

## Theory you need before reading the code

This folder builds a **concurrent queue** step by step and ends at the classic
**producer/consumer** pattern — the foundation of most real concurrent systems.

### The monitor object pattern
A **monitor object** wraps shared data together with the mutex that protects it,
so *every* access to the data goes through methods that lock internally. Callers
can never touch the raw data unlocked. `OrderBook` here is a monitor: its
`submit`/`take` each acquire the lock before touching `_orders`. Sharing the
monitor across threads is done with a `std::shared_ptr` (folder 03/05 lifetime
rule).

### The problem with polling (busy-waiting)
A consumer needs to wait until the producer has put something in the queue. The
naive approach is to **spin**:
```cpp
while (true) {
    if (book->hasOrders())      // lock, check, unlock — repeat forever
        match(book->take());
}
```
This is a **busy-wait / polling loop** (`example_40`). It "works" but:
- It burns 100% CPU checking an empty queue.
- It repeatedly locks/unlocks the mutex, creating contention.
- It has a lurking bug: `hasOrders()` and `take()` are two separate locked
  calls, so another consumer could empty the queue *between* them (a
  **check-then-act race**).

### Condition variables — sleep until notified
A **`std::condition_variable`** lets a thread **block efficiently** until another
thread signals that a condition may now be true. No CPU spinning.

Consumer side:
```cpp
std::unique_lock<std::mutex> lk(_mutex);          // must be unique_lock, not lock_guard
_ready.wait(lk, [this]{ return !_orders.empty(); });
// woken up AND predicate true AND lock re-held -> safe to pop
```
What `wait(lk, pred)` does, atomically:
1. checks `pred()`; if true, returns immediately holding the lock;
2. if false, **releases the lock and puts the thread to sleep**;
3. when notified, **re-acquires the lock** and re-checks `pred()`.

Producer side:
```cpp
{
  std::lock_guard<std::mutex> lk(_mutex);
  _orders.push_back(std::move(order));
}
_ready.notify_one();     // wake one waiting consumer
```

### Why the predicate (and why `unique_lock`)
- The **predicate** (`[]{ return !empty(); }`) guards against **spurious
  wakeups** — a condition variable may wake a thread even when no notify
  happened, and against **lost/steal races** — another consumer may have taken
  the item first. Re-checking the predicate in a loop (which `wait(lk, pred)`
  does for you) is mandatory. Never use a bare `wait(lk)` without a predicate for
  real logic.
- A condition variable needs to **unlock while sleeping and relock on wakeup**,
  so it requires the flexibility of **`std::unique_lock`**; a `lock_guard` can't
  be handed to `wait`.
- `notify_one()` wakes a single waiter; `notify_all()` wakes them all (use when
  several waiters could each make progress, e.g. after pushing multiple items).

### Producer / consumer
The end result is the **producer/consumer** (a.k.a. bounded-buffer) pattern:
producers `pushBack` and notify; consumers `wait` then `popBack`. Work flows
between threads with no busy-waiting and no lost updates. This is how thread
pools, task queues, and pipelines are built.

### Shutdown — the part most tutorials skip
A blocking consumer creates a new problem: once the producers stop, the consumer
is still asleep in `wait()`, waiting for an item that will never arrive. The
program never ends.

`example_42` closes the loop with the standard solution:
1. a `_closed` flag, set by `close()` once every producer has finished;
2. a predicate that also accepts closure —
   `return !_orders.empty() || _closed;`
3. `notify_all()` on close, so **every** blocked consumer wakes, not just one;
4. `take()` returning `std::optional<Order>`, where `std::nullopt` means
   "drained and closed — stop looping".

> **Note on these examples:** `example_40` and `example_41` deliberately run
> forever — stop them with Ctrl-C. `40` spins hot; `41` blocks politely but has
> no way to learn that the producers are done. `example_42` is the finished
> version and **exits cleanly**.

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_39.cpp` | The **monitor object** foundation: `submit`/`printAll` protected by a `lock_guard`, the monitor shared via `shared_ptr`. Producer-only. |
| `example_40.cpp` | A **busy-wait** consumer polling `hasOrders()` — works, burns a core, and has a check-then-act gap. **Runs forever**; Ctrl-C. |
| `example_41.cpp` | **Condition variable**: the consumer `wait`s on a predicate, the producer `notify_one()`s. No polling. **Runs forever**; Ctrl-C. |
| `example_42.cpp` | The finished queue: adds `close()`, `notify_all()`, and an `optional`-returning `take()` for clean shutdown. **Terminates.** |

## Try it yourself
- Watch CPU usage while `example_40` runs (`top`/Activity Monitor) — one core
  pinned at 100%. Then watch `example_41`: near zero while it waits. Same work,
  same result, completely different machine cost.
- In `example_39.cpp`, move the `sleep_for` outside the `lock_guard` scope and
  time the difference. Holding a lock across a sleep serialises every producer.
- In `example_42.cpp`, replace `notify_all()` in `close()` with `notify_one()`,
  then start a **second** consumer. One of them will never wake — the reason
  `notify_all` is the correct call for shutdown.
- Give `example_42` a bounded capacity: a second condition variable that makes
  `submit()` wait while the book is full. That turns it into the classic
  **bounded buffer**, which is how real systems apply backpressure.

## Compiling
```sh
g++ -std=c++17 -pthread example_42.cpp -o example_42 && ./example_42
```
