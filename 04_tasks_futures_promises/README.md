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
| `example_20.cpp` | Basic promise/future: worker sets a value, main retrieves it with `get()`. |
| `example_21.cpp` | **TODO exercise** — poll the future with `wait_for` and branch on `future_status` (ready / timeout / deferred). |
| `example_22.cpp` | Passing an **exception** across threads via `prms.set_exception` + try/catch around `get()`. |
| `example_23.cpp` | Same idea with `std::async` — exceptions re-thrown in `get()`. |
| `example_24.cpp` | `std::launch::deferred`: note the worker runs on the **main** thread id. |
| `example_25.cpp` | Reinforces deferred launch (compare thread ids with 24). |
| `example_26.cpp` | Launching several tasks and **timing** the whole batch with `std::chrono`. |

## Try it yourself
- Complete the TODO in `example_21.cpp`, then change the timeout from 1000 ms to
  3000 ms and watch the branch flip from "unavailable" to "Result = ...".
- In `example_24.cpp`, switch `std::launch::deferred` to `std::launch::async` and
  observe that the worker now prints a **different** thread id.

## Compiling
```sh
g++ -std=c++17 -pthread example_20.cpp -o example_20 && ./example_20
```
