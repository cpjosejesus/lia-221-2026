# 06 — Mutexes, Locks, and Deadlock

## Theory you need before reading the code

### The mutex: mutual exclusion
A **`std::mutex`** ("mutual exclusion") is a lock that guarantees only **one**
thread at a time can be inside a **critical section** — the region of code that
touches shared data. The pattern:
```cpp
mtx.lock();     // blocks until this thread owns the mutex
// ... critical section: exclusive access to shared data ...
mtx.unlock();   // release so another waiting thread can proceed
```
This directly fixes the data race from folder `05`: the read-modify-write of the
shared counter now happens *atomically with respect to other threads*, so no
update is lost (`example_32`).

### Why raw `lock()`/`unlock()` is dangerous
Manual `unlock()` is fragile: if an **exception** is thrown, or you `return`
early, between `lock()` and `unlock()`, the mutex is **never released** and every
other thread waiting on it blocks forever. `example_33` shows this bug — a
`throw` in the try-block skips `mtx.unlock()`.

### RAII lock wrappers — the correct way
**RAII** ("Resource Acquisition Is Initialization") ties a resource's lifetime to
a stack object: acquire in the constructor, release in the destructor. Because
C++ guarantees destructors run when a scope exits — *including during stack
unwinding from an exception* — the lock is **always** released.

- **`std::lock_guard<std::mutex> lk(mtx);`** — locks on construction, unlocks on
  destruction. Minimal, no extra features. Use this by default (`example_35`).
- **`std::unique_lock<std::mutex> lk(mtx);`** — same RAII guarantee but *flexible*:
  you can `lk.unlock()` and `lk.lock()` again, defer locking, transfer ownership
  (move), and — crucially — hand it to a **condition variable** (folder `07`).
  Slightly heavier than `lock_guard`. Use it when you need that flexibility
  (`example_36`).

> Rule of thumb: `lock_guard` unless you need to unlock early or use a condition
> variable — then `unique_lock`.

### Deadlock
A **deadlock** is a standstill where threads wait on each other in a cycle and
none can proceed. The classic four Coffman conditions must all hold; the one you
create by accident is **circular wait** on **multiple mutexes acquired in
different orders**:

```
Thread A: lock(mutex2); lock(mutex1);   // wants 2 then 1
Thread B: lock(mutex1); lock(mutex2);   // wants 1 then 2
```
If A holds `mutex2` and B holds `mutex1`, each waits forever for the other's
lock. `example_34` (raw locks) and `example_37` (lock_guards) both deadlock —
RAII does **not** save you here, because the problem is *lock ordering*, not
*lock release*.

### Fixing deadlock
Two standard cures:
1. **Consistent lock ordering** — every thread always acquires the mutexes in the
   same global order. Then no cycle can form.
2. **Lock them together atomically** — `std::lock(mutex1, mutex2)` acquires both
   with a deadlock-avoidance algorithm, then you adopt them into RAII guards with
   `std::adopt_lock` so they still auto-release (`example_38`). (Modern C++17 also
   offers `std::scoped_lock lk(mutex1, mutex2);` which does both in one line.)

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_32.cpp` | Mutex fixes the folder-05 race: `lock()`/`unlock()` around the shared vector → correct count. |
| `example_33.cpp` | The **bug** in manual unlocking: a `throw` skips `mtx.unlock()` and can leave the mutex locked. |
| `example_34.cpp` | **Deadlock** via two mutexes locked in opposite orders (raw lock/unlock). |
| `example_35.cpp` | `std::lock_guard` — exception-safe locking; unlock happens automatically. |
| `example_36.cpp` | `std::unique_lock` — manually `unlock()`/`lock()` around work done *outside* the critical section. |
| `example_37.cpp` | Deadlock again, this time with `lock_guard` — proves RAII alone doesn't prevent deadlock. |
| `example_38.cpp` | **Fix**: `std::lock(m1, m2)` + `std::adopt_lock` guards to acquire both without deadlock. |

## Try it yourself
- Run `example_34.cpp` / `example_37.cpp` — they hang (deadlock). Kill with Ctrl-C.
  Then run `example_38.cpp` and see it finish. Try rewriting `38` with
  `std::scoped_lock lk(mutex1, mutex2);`.
- In `example_33.cpp`, feed a zero denominator so the exception fires, then note
  how a following thread can block. Rewrite it with `lock_guard` (as in `35`).

## Compiling
```sh
g++ -std=c++17 -pthread example_35.cpp -o example_35 && ./example_35
```
