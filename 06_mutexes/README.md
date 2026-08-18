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
settleWorker: lock(feed); lock(book);   // wants feed then book
quoteWorker:  lock(book); lock(feed);   // wants book then feed
```
If the settle worker holds `feed` and the quote worker holds `book`, each waits
forever for the other's lock. (Both examples sleep briefly between the two
acquisitions, which makes the deadlock happen reliably rather than occasionally.) `example_34` (raw locks) and `example_37` (lock_guards) both deadlock —
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
| `example_32.cpp` | The mutex fixes the folder-05 race: `lock()`/`unlock()` around the shared vector → a correct count of 1000, every run. |
| `example_33.cpp` | The **bug** in manual unlocking: an early `return` from the catch block skips `unlock()` and the program **hangs**. Ctrl-C. |
| `example_34.cpp` | **Deadlock** — two mutexes, opposite acquisition orders, raw lock/unlock. **Hangs**; Ctrl-C. |
| `example_35.cpp` | `std::lock_guard` — `example_33` fixed. Unlock happens on every exit path, exceptions included. |
| `example_36.cpp` | `std::unique_lock` — `unlock()`/`lock()` mid-scope to keep the critical section as small as the shared access itself. |
| `example_37.cpp` | Deadlock again, this time with `lock_guard` — proof that RAII fixes *release*, not *ordering*. **Hangs**; Ctrl-C. |
| `example_38.cpp` | **The fix**: `std::lock(m1, m2)` + `std::adopt_lock` guards acquire both without deadlock. |

## Try it yourself
- Run `example_33.cpp`, `example_34.cpp` and `example_37.cpp` — all three hang.
  Then run `example_38.cpp` and watch it finish. Rewrite `38` using C++17's
  `std::scoped_lock guard(bookMutex, feedMutex);` — it replaces all three lines
  in each function.
- Fix `example_33.cpp` yourself by swapping the manual lock for a `lock_guard`,
  then diff your version against `example_35.cpp`.
- In `example_36.cpp`, delete the `lock.unlock()` / `lock.lock()` pair so the
  mutex is held for the whole function, and time both versions. The private work
  in the middle is now serialised across every thread.
- In `example_37.cpp`, make both functions acquire the mutexes in the **same**
  order and confirm the deadlock disappears — that is cure #1 from above,
  without needing `std::lock` at all.

## Compiling
```sh
g++ -std=c++17 -pthread example_35.cpp -o example_35 && ./example_35
```
