# C++ Concurrency & Parallelism — Worked Examples

A hands-on tour of the C++ standard threading library, organized as a progressive
course. Each folder groups related examples and ships a `README.md` with the
**theory you need before reading the code**, a per-file table, and
"try it yourself" experiments.

## Concurrency vs. Parallelism (the two words this whole repo is about)

- **Concurrency** — *structuring* a program as independent tasks that can make
  progress in overlapping time windows. It's a design property; it can happen on a
  single core by *interleaving* (time-slicing).
- **Parallelism** — *executing* multiple tasks at the literal same instant on
  multiple physical cores. It's a hardware/execution property.

> Concurrency is about **dealing with** many things at once. Parallelism is about
> **doing** many things at once. The tools here (`std::thread`, `std::async`,
> mutexes, condition variables) let you express concurrency; the hardware
> (`hardware_concurrency()` cores) decides how much real parallelism you get.

## Learning path

| # | Folder | Concepts |
|---|--------|----------|
| 1 | [`01_threads_basics`](01_threads_basics/) | `std::thread`, `join`/`detach`, thread lifecycle, `hardware_concurrency`, why ordering isn't guaranteed |
| 2 | [`02_passing_data_to_threads`](02_passing_data_to_threads/) | lambdas & captures (value/ref/`mutable`), how threads copy args, `std::ref`, move semantics, variadic templates |
| 3 | [`03_threads_with_objects`](03_threads_with_objects/) | member functions on threads, `shared_ptr` for shared objects & lifetime, threads are move-only, `vector<std::thread>` |
| 4 | [`04_tasks_futures_promises`](04_tasks_futures_promises/) | `promise`/`future`, returning values & exceptions across threads, `std::async`, launch policies, `wait_for` (**contains a TODO exercise**) |
| 5 | [`05_data_races`](05_data_races/) | what a data race is (UB), copy/move & rule-of-five, smart pointers, the classic lost-update race |
| 6 | [`06_mutexes`](06_mutexes/) | `std::mutex`, critical sections, `lock_guard` vs `unique_lock` (RAII), deadlock & how to fix it |
| 7 | [`07_condition_variables_queue`](07_condition_variables_queue/) | monitor object, busy-wait vs `condition_variable`, predicates & spurious wakeups, producer/consumer queue |

Work through them in order — each folder assumes the previous ones.

## The through-line
Folders 5→6→7 tell one story with the same `WaitingVehicles` example:
1. **05** exposes a **data race** (lost updates on a shared counter/queue).
2. **06** fixes it with a **mutex**, then shows the new hazard: **deadlock**.
3. **07** turns the locked queue into an efficient **producer/consumer** using a
   **condition variable** instead of wasteful polling.

## Exercises (TODO)
Search the tree for `TODO` to find the hands-on tasks:
```sh
grep -rn "TODO" .
```
Currently: [`04_tasks_futures_promises/example_21.cpp`](04_tasks_futures_promises/example_21.cpp)
— poll a `std::future` with `wait_for` and branch on `future_status`.

## Building
All examples are single-file C++17 programs that need the pthreads library:
```sh
g++ -std=c++17 -pthread path/to/example_N.cpp -o example_N && ./example_N
# clang++ works too:
clang++ -std=c++17 -pthread path/to/example_N.cpp -o example_N && ./example_N
```

> ⚠️ A few examples deliberately **misbehave** to teach a point: infinite loops
> (`01/example_1`, the `while(true)` consumers in `07`), deadlocks
> (`06/example_34`, `06/example_37`), and undefined behavior from un-joined
> threads (`01/example_4`, `01/example_6`). Their READMEs say which — stop hung
> ones with Ctrl-C.
