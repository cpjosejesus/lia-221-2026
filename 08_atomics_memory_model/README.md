# 08 — Atomics and the Memory Model

## Theory you need before reading the code

### Where folder 06 left us
A mutex makes a whole critical section exclusive. That is the right tool when several
fields must change together. But a lot of real sharing is just **one number**: a counter, a
flag, a pointer to the newest thing. For those, taking a lock is heavy, and folder 07's
condition variables are for *waiting*, not for *updating*. Atomics fill that gap.

### What `std::atomic<T>` guarantees
An **atomic** operation is *indivisible*: no other thread can observe it half done, and two
atomic operations on the same object never form a data race. `std::atomic<T>` works for any
trivially copyable `T`, but:
- small types (`bool`, `int`, pointers, an 8-byte struct) are **lock-free**: a single machine
  instruction (`example_46`);
- large types are silently implemented with a **hidden lock**. `is_lock_free()` tells you.

### Atomicity of one operation is not atomicity of a sequence
This is the mistake everyone makes once (`example_43`, case b):
```cpp
int old = counter.load();   // atomic
counter.store(old + 1);     // atomic
```
Both calls are atomic, the *pair* is not: another thread can increment between them and its
update is lost, exactly like the plain `int` of folder 05. You need a single
**read-modify-write** (RMW) operation:
- `fetch_add`, `fetch_sub`, `++`, `--` — add and return the old value;
- `exchange(x)` — set to `x`, return the old value;
- `compare_exchange_weak/strong(expected, desired)` — **compare-and-swap (CAS)**: *if* the
  value still equals `expected`, set it to `desired` and return true; *otherwise* load the
  current value into `expected` and return false.

### The CAS loop, the universal building block
Any "read, decide, write" that must be atomic is written as a loop:
```cpp
int cur = x.load();
while (cur < limit && !x.compare_exchange_weak(cur, cur + 1)) { /* cur was refreshed, retry */ }
```
`example_44` admits at most N vehicles this way; `example_49` builds a whole lock-free stack
from it. `_weak` may fail spuriously (return false even if equal) and is meant for loops;
`_strong` never does and is for a single attempt.

### Spinlocks: a mutex made of one atomic
`std::atomic_flag::test_and_set` is an RMW, so a lock is three lines (`example_45`). A
spinlock **burns CPU** while waiting instead of sleeping. It wins only for very short
critical sections with little contention. Under contention it usually **loses** to
`std::mutex`, as the example's timing shows: use it to understand what a mutex does, not to
replace one.

### The memory model: atomicity is not ordering
Atomic operations are also the way you **order** other memory accesses between threads.
Each operation takes a `std::memory_order`:
| Order | Meaning | Typical use |
|-------|---------|-------------|
| `relaxed` | atomic, but no ordering with other memory accesses | statistics counters nobody synchronizes on |
| `release` (on a store) | every write *before* it becomes visible to whoever `acquire`-loads the value | publishing data: "the record is complete" |
| `acquire` (on a load) | every read *after* it sees the writes before the matching `release` | consuming published data |
| `seq_cst` (default) | acquire + release + one global order all threads agree on | when in doubt |

`example_47` is the canonical pattern: fill a plain struct, then `store(true, release)`;
the consumer `load(acquire)`s until it sees `true` and may then read the struct. Without the
pair, the consumer could see `ready == true` and *still* read stale fields. The default
`seq_cst` is correct everywhere; the weaker orders are optimizations you reach for only with
a measurement in hand.

### False sharing: correct code that is slow
Cores move memory in **cache lines** (64 bytes on most CPUs, 128 on Apple Silicon). Two
threads writing two *different* variables that sit in the *same* line force the line to
bounce between cores on every write. `example_48` shows a 3–5x slowdown with no data race at
all; `alignas(64)` (or `std::hardware_destructive_interference_size`) puts each hot variable
on its own line.

### Lock-free structures and their two classic problems
`example_49` is a Treiber stack: `push` and `pop` are CAS loops on the head pointer.
1. **ABA**: a thread reads head = A, is interrupted; others pop A, pop B, push A back (same
   address). The first thread's CAS sees "still A" and succeeds on a stack it no longer
   understands. Cures: tagged pointers (version counters) or never reusing addresses.
2. **Memory reclamation**: you cannot `delete` a popped node while another thread may still
   be dereferencing it inside its CAS loop. The demo leaks nodes on purpose; real solutions
   are hazard pointers, epoch-based reclamation or garbage collection (self-study, folder 10's
   reading list).

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_43.cpp` | The folder-05 lost update three ways: plain `int`, atomic `load`/`store` (still wrong), atomic `fetch_add` (right). |
| `example_44.cpp` | CAS loop: admit at most 3 vehicles into an intersection without a mutex; `exchange`. |
| `example_45.cpp` | A spinlock from `std::atomic_flag`, used through `lock_guard`; timing against `std::mutex`. |
| `example_46.cpp` | `is_lock_free()` on small and large `std::atomic<T>`; no torn reads of an 8-byte position. |
| `example_47.cpp` | Publishing a record with release/acquire; relaxed for a statistics counter. |
| `example_48.cpp` | False sharing: adjacent vs padded counters, measured. |
| `example_49.cpp` | A lock-free Treiber stack of vehicles; ABA and reclamation explained in comments. |

## Exercises (change of context: orders and couriers)
Each file compiles and runs as given, and prints that the work is not done yet. Search for
`TODO`.
| File | Task |
|------|------|
| `exercise_8_1.cpp` | Replace a mutex-protected order statistics object by atomics; pick and justify a memory order; discuss what a reader can observe across three counters. |
| `exercise_8_2.cpp` | Restaurant pickup slots: `tryTakeSlot` as a CAS loop, an atomic "peak busy" tracker, then vary the capacity and explain the timing. |
| `exercise_8_3.cpp` | Per-courier counters that never race but still crawl: diagnose false sharing, fix the layout, report before/after for 2, 4, 8 and 16 couriers. |

## Try it yourself
- In `example_43`, remove the `sleep_for` from case b) and run several times: does the bug
  disappear or only hide?
- In `example_47`, change both orders to `relaxed`. On x86 you will probably never see the
  bug; on ARM (Apple Silicon) you may. Explain why a program that "works" is still wrong.
- Build `example_45` with `-O2` and with 2 threads instead of 8: when does the spinlock win?

## Compiling
```sh
g++ -std=c++17 -pthread -O2 example_48.cpp -o example_48 && ./example_48
```
Timing examples (`45`, `48`, `exercise_8_3`) only mean something with `-O2`.
