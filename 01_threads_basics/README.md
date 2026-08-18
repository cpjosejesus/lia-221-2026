# 01 — Thread Basics

## Theory you need before reading the code

### Concurrency vs. Parallelism
- **Concurrency** is a *structuring* property: the program is decomposed into
  independent tasks that can make progress in overlapping time periods. On a
  single core, the OS scheduler *interleaves* them — only one runs at any instant,
  but they appear to advance together.
- **Parallelism** is an *execution* property: two or more tasks literally run at
  the *same instant* on different physical cores.
- Concurrency is about *dealing with* many things at once (design); parallelism
  is about *doing* many things at once (hardware). You can have concurrency
  without parallelism (one core, time-slicing) but not the reverse.

### Process vs. Thread
- A **process** has its own isolated address space (memory). Communication
  between processes is expensive (IPC, pipes, sockets).
- A **thread** is a unit of execution *inside* a process. All threads of a
  process **share the same heap and global memory** but each has its **own stack**
  and registers. Sharing memory is what makes threads fast to communicate — and
  also what makes them dangerous (data races, see folder `05` and `06`).

### The `std::thread` lifecycle
1. **Construct** — `std::thread t(fn, args...)`. The new thread starts running
   `fn` *immediately*. The arguments are **copied/moved into the thread's own
   storage** by default (see folder `02` for how to pass by reference).
2. **Joinable** — a freshly constructed thread is *joinable*. You MUST decide its
   fate before the `std::thread` object is destroyed.
3. **`join()`** — the calling thread *blocks* until the worker finishes. This is
   a **barrier**: it also synchronizes memory (everything the worker wrote is
   visible after join returns).
4. **`detach()`** — cuts the thread loose to run on its own. You give up the
   ability to join it or observe when it ends.
5. If a `std::thread` object is destroyed while still *joinable* (neither joined
   nor detached), the program calls **`std::terminate()`** and crashes. This is
   deliberate — the standard refuses to guess whether you meant join or detach.

### Key API in this folder
- `std::thread::hardware_concurrency()` — a *hint* for how many threads can run
  truly in parallel (often the core/hyperthread count). May return `0` if unknown.
- `std::this_thread::get_id()` — a unique id for the current thread.
- `std::this_thread::sleep_for(duration)` — used here to *simulate work* and to
  make race conditions and ordering effects observable.

### The core lesson of these examples
The order in which `main` and worker threads print is **not guaranteed**. Sleep
durations only *nudge* the scheduler; they never *guarantee* ordering. If you
depend on ordering, you need real synchronization (folders `04`, `06`, `07`).

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_1.cpp` | Spawning N workers into a `std::vector` and joining them all. The workers spin forever on purpose — it illustrates thread creation, not a program that terminates. **Ctrl-C to stop.** |
| `example_2.cpp` | Querying `hardware_concurrency()` — and handling the `0` ("unknown") case. |
| `example_3.cpp` | Printing the id of the implicit main thread. |
| `example_4.cpp` | A worker that is **neither joined nor detached** — `~thread()` calls `std::terminate`. The `join()` is commented out. |
| `example_5.cpp` | Interleaved work in main and worker, correctly `join()`ed *after* main's own work. |
| `example_6.cpp` | Same defect as 4, but showing where a *premature* `join()` would go — and why it would destroy the concurrency. |
| `example_7.cpp` | `detach()` — the worker runs unowned; main may exit before it prints. |
| `example_8.cpp` | Six detached workers with `sleep` used as a (fragile, wrong) barrier. |

## Try it yourself
- In `example_4.cpp` / `example_6.cpp`, restore `join()` and compare behavior.
- In `example_5.cpp`, run it several times — the interleaving is not stable.
- In `example_8.cpp`, remove the final `sleep_for` and watch output disappear:
  detached threads die when `main` returns.

## Compiling
```sh
g++ -std=c++17 -pthread example_5.cpp -o example_5 && ./example_5
```
