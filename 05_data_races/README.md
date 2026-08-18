# 05 — Data Races and Object Copy/Move Semantics

## Theory you need before reading the code

### What is a data race?
A **data race** occurs when:
1. two or more threads access the **same memory location** concurrently,
2. **at least one** access is a **write**, and
3. there is **no synchronization** ordering those accesses.

In C++ a data race is **undefined behavior** — not "you get one of two values,"
but "the standard makes *no* guarantee at all." The program may print garbage,
crash, corrupt data, or appear to work until it doesn't. Never reason about a
racy program's output; fix the race (folder `06`).

### Race condition vs. data race
- A **data race** is the low-level UB above (unsynchronized concurrent memory
  access). It is always a bug.
- A **race condition** is a higher-level *logic* bug where correctness depends on
  timing/ordering (e.g. "reader ran before writer"). You can have a race
  condition even with perfectly synchronized, race-free memory.

### Why copies quietly avoid some races
Many of these examples pass an object **by value** into `std::async`/`std::thread`.
Because the argument is *copied* into the task, the worker mutates its **own
copy** — the original in `main` is a *different* object, so there is no shared
memory and thus no data race (`example_27`, `example_28`). The lesson: **copying
is one way to avoid sharing.** The cost is duplicated data and the fact that the
worker's result never reaches `main` unless you send it back (futures, folder 04).

### The rule of five (and why it matters here)
When a class manages a resource (raw `new`/`delete`, a file, a socket), you
usually must define/consider all five:
- destructor
- copy constructor, copy assignment
- move constructor, move assignment

If you only define some, the compiler-generated ones may do the **wrong** thing —
e.g. a *shallow copy* of a raw pointer means two objects own the same memory and
both try to `delete` it (double free), or one mutates the other's data.

### Copy vs. move for owned resources
- **Copy** should perform a **deep copy**: allocate new storage and duplicate the
  contents so the two objects are independent.
- **Move** transfers the pointer/handle and **nulls out the source**, leaving it
  in a valid-but-empty state. Reading a moved-from object's resource is a bug
  (`example_30` deliberately dereferences a moved-from `unique_ptr` → exception).

### Smart pointers to express ownership
- `std::unique_ptr<T>` — **exclusive** ownership, move-only. Automatically deletes
  the resource. Using it as a member makes the move constructor almost free to
  write correctly (`example_30`).
- `std::shared_ptr<T>` — **shared** ownership, reference counted. Lets several
  threads keep an object alive (`example_31`), but does **not** protect its
  contents from concurrent writes — that still needs a mutex.

### The classic non-atomic read-modify-write race
`example_31` is the payoff. `submit()` does:
```cpp
const int seen = _filled;    // READ
sleep(...);                  // (window where another thread interleaves)
_filled = seen + 1;          // WRITE back a stale value
```
1000 threads each try to increment the counter, but because the read and write
are **not atomic and not locked**, updates are **lost** — the final count lands
somewhere in the hundreds. The `sleep` merely widens the window so the bug
reproduces on every run. Folder `06` fixes exactly this with a mutex.

Note that `example_31` *does* lock the vector while leaving the counter
unguarded. That is deliberate: an unsynchronized `push_back` corrupts the
vector's internals and aborts the process before it can print anything, which
would hide the lesson behind a crash. Drop that `lock_guard` and see for
yourself — a good reminder that UB does not politely produce a wrong number.

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_27.cpp` | Passing an `Order` **by value** to a task — the worker mutates a copy, `main` is unaffected. **No race**, and the reason is the copy. |
| `example_28.cpp` | Same shape, but the object owns a raw `std::string*`. The implicit **shallow copy** shares the pointer, so both threads write through it. |
| `example_29.cpp` | Explicit **copy** and **move** constructors — including a deliberately broken copy ctor to fix as an exercise. |
| `example_30.cpp` | The fix: `std::unique_ptr` makes the class move-only, so the shallow-copy bug becomes *impossible to write*. Reading the moved-from source throws. |
| `example_31.cpp` | **The lost update**: 1000 unsynchronized increments, final count well below 1000, different every run. |

## Try it yourself
- Run `example_31.cpp` several times — the count varies and is always wrong.
  Then run `06/example_32.cpp`, which adds a mutex and prints 1000 every time.
- In `example_31.cpp`, remove the `lock_guard` around `push_back` and re-run.
  It now usually **crashes** instead of printing a wrong answer.
- In `example_28.cpp`, change the raw `std::string*` to a plain `std::string`
  member — the shallow-copy problem disappears entirely, for free.
- In `example_29.cpp`, repair the broken copy constructor (deep-copy `_symbol`,
  copy `_id`) and confirm the copy and the original are independent.
- Build any of these with `-fsanitize=thread` and let the tool find the race for
  you:
  ```sh
  g++ -std=c++17 -pthread -fsanitize=thread example_31.cpp -o example_31 && ./example_31
  ```

## Compiling
```sh
g++ -std=c++17 -pthread example_31.cpp -o example_31 && ./example_31
```
