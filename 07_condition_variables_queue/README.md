# 07 — Condition Variables and a Thread-Safe Queue

## Theory you need before reading the code

This folder builds a **concurrent queue** step by step and ends at the classic
**producer/consumer** pattern — the foundation of most real concurrent systems.

### The monitor object pattern
A **monitor object** wraps shared data together with the mutex that protects it,
so *every* access to the data goes through methods that lock internally. Callers
can never touch the raw data unlocked. `WaitingVehicles` here is a monitor: its
`pushBack`/`popBack` each take the lock before touching `_vehicles`. Sharing the
monitor across threads is done with a `std::shared_ptr` (folder 03/05 lifetime
rule).

### The problem with polling (busy-waiting)
A consumer needs to wait until the producer has put something in the queue. The
naive approach is to **spin**:
```cpp
while (true) {
    if (queue->dataIsAvailable())      // lock, check, unlock — repeat forever
        process(queue->popBack());
}
```
This is a **busy-wait / polling loop** (`example_40`). It "works" but:
- It burns 100% CPU checking an empty queue.
- It repeatedly locks/unlocks the mutex, creating contention.
- It has a lurking bug: `dataIsAvailable()` and `popBack()` are two separate
  locked calls, so another consumer could empty the queue *between* them (a
  **check-then-act race**).

### Condition variables — sleep until notified
A **`std::condition_variable`** lets a thread **block efficiently** until another
thread signals that a condition may now be true. No CPU spinning.

Consumer side:
```cpp
std::unique_lock<std::mutex> lk(_mutex);          // must be unique_lock, not lock_guard
_cond.wait(lk, [this]{ return !_vehicles.empty(); });
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
  _vehicles.push_back(std::move(v));
}
_cond.notify_one();     // wake one waiting consumer
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

> **Note on these examples:** the `main()` consumer loop is `while(true)`, so the
> program runs forever (it never reaches the `futures` join / "Finished!"). That's
> intentional for demonstration — stop it with Ctrl-C. A real system would add a
> shutdown signal (e.g. a sentinel value or an atomic `done` flag checked in the
> predicate).

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_39.cpp` | The **monitor object** foundation: `pushBack`/`printIDs` protected by a `lock_guard`; a shared_ptr shared across threads. |
| `example_40.cpp` | A **busy-wait** consumer polling `dataIsAvailable()` — works but wastes CPU and has a check-then-act gap. |
| `example_41.cpp` | **Condition variable**: consumer `wait`s on a predicate; producer `notify_one()`s. No polling. |
| `example_42.cpp` | Same producer/consumer with the condition variable (reinforcement / template for reuse). |

## Try it yourself
- Compare CPU usage of `example_40` (spins, high CPU) vs `example_41` (sleeps,
  near-zero CPU) while the queue is empty.
- Add a graceful shutdown to `example_41`: an atomic `bool done`, have the
  predicate become `return done || !_vehicles.empty();`, and `notify_all()` when
  done so the consumer can exit the loop and the program reaches "Finished!".

## Compiling
```sh
g++ -std=c++17 -pthread example_41.cpp -o example_41 && ./example_41
```
