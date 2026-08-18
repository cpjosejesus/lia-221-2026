# 04 — Tasks: Promises, Futures, and `std::async`

## Theory you need before reading the code

### The problem `std::thread` leaves unsolved
A `std::thread` runs a function but gives you **no built-in way to get a return
value back** and **no built-in way to propagate exceptions** from the worker to
the caller. `join()` only tells you *that* the thread finished, not *what it
produced*. Promises/futures and tasks fill that gap.

### Promise / Future — a one-shot communication channel
Think of it as a single-use pipe with two ends:
- **`std::promise<T>`** — the *producer* end. The worker thread calls
  `prms.set_value(x)` (or `prms.set_exception(...)`) exactly once.
- **`std::future<T>`** — the *consumer* end, obtained via `prms.get_future()`.
  The main thread calls `ftr.get()` to retrieve the value.

Rules:
- A promise is **move-only** (like a thread). You pass it into the worker with
  `std::move(prms)` and the worker takes it by `std::promise<T>&&`.
- `ftr.get()` **blocks** until the value is set, then returns it. You may call
  `get()` **only once** (it moves the result out).
- If the worker sets an *exception*, that exception is **re-thrown** inside
  `get()` in the consumer thread. This is how you move errors across threads.

### Waiting without blocking forever
Instead of blocking on `get()` you can poll:
- `ftr.wait()` — block until ready (no value returned).
- `ftr.wait_for(duration)` / `ftr.wait_until(timepoint)` — wait with a timeout,
  returning a **`std::future_status`**:
  - `ready` — the value is available; safe to call `get()`.
  - `timeout` — not ready yet within the time budget.
  - `deferred` — the task hasn't started (see deferred launch below).

This is the pattern in `example_21` (now marked as a **TODO** exercise): poll with
a short timeout so the main thread stays responsive.

### `std::async` — the high-level shortcut
`std::async(fn, args...)` packages "run `fn` and hand me its result" into one
call and returns a `std::future`. It manages the promise for you. The launch
**policy** controls *when/where* it runs:
- `std::launch::async` — run `fn` on a **new thread**, right away.
- `std::launch::deferred` — **lazy**: `fn` does not run until you call `.get()`
  or `.wait()` on the future, and then it runs **on the calling thread** (no new
  thread at all). Until then `wait_for` reports `deferred`.
- Default (`async | deferred`) — the implementation may pick either. Do not
  assume a new thread is created unless you pass `std::launch::async` explicitly.

> **Task vs. thread:** a *task* is "a piece of work whose result I care about";
> the runtime decides how to schedule it. A *thread* is a concrete OS resource you
> manage yourself. Prefer tasks (`async`/futures) when you just want a result;
> reach for raw threads when you need explicit control.

### Don't discard an `async` future carelessly
The future returned by `std::async(std::launch::async, ...)` **blocks in its
destructor** until the task finishes. If you don't keep the future in a variable,
the temporary is destroyed at the end of the full expression and effectively runs
synchronously — a common surprise.

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_20.cpp` | The raw `promise`/`future` pair: a one-shot channel, with the promise **moved** into the worker. |
| `example_21.cpp` | `wait_for` and `std::future_status` — polling with a deadline instead of blocking. **Contains the TODO exercise.** |
| `example_22.cpp` | `set_exception` — shipping an exception across the thread boundary so `get()` rethrows it. |
| `example_23.cpp` | `std::async` does all of the above plumbing for you: return a value, throw an exception, same semantics, far less code. |
| `example_24.cpp` | `std::launch::async` — **forces** a new thread. The two printed ids differ. |
| `example_25.cpp` | `std::launch::deferred` — creates **no** thread; the task runs lazily inside `get()`, so the ids match. |
| `example_26.cpp` | Measuring the cost: with a tiny workload, thread-spawn overhead dominates the "parallel" run. |

## The exercise
`example_21.cpp` marks a `TODO (STARTS here)` / `TODO (ENDS here)` block. A
working solution is already in place so the file compiles and runs — the task is
to experiment with it:

1. Set the `wait_for` timeout **shorter** than the worker's 2000 ms → `timeout`.
2. Raise it to 3000 ms → `ready`, and `get()` returns the value.
3. Wrap it in a loop that prints a heartbeat between polls. That is the pattern
   behind every responsive UI and health-checked worker.

Find it with:
```sh
grep -rn "TODO" .
```

## Try it yourself
- In `example_23.cpp`, change the denominator to `0` and watch the exception
  surface out of `get()` rather than terminating the worker.
- In `example_25.cpp`, delete the `result.get()` call entirely. With `deferred`,
  the task then **never runs at all** — no output from it whatsoever.
- In `example_26.cpp`, raise `workPerTask` to `1000000` and re-time it, then
  swap `std::launch::async` for `std::launch::deferred` and compare. Somewhere
  between the two workloads is the crossover point where threads start to pay.

## Compiling
```sh
g++ -std=c++17 -pthread example_23.cpp -o example_23 && ./example_23
```
