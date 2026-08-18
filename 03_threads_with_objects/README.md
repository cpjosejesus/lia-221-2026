# 03 — Threads with Objects and Containers of Threads

## Theory you need before reading the code

### Running a member function on a thread
To run `obj.method(args)` on a new thread you pass a **pointer-to-member
function** plus the object the method should act on:

```cpp
std::thread t(&Class::method, objectOrPointer, args...);
```

The second argument is the *implicit `this`*. There are three ways to supply it,
and they have very different consequences:

| Second arg | What the thread receives | Effect of the method on the original |
|------------|--------------------------|--------------------------------------|
| `obj` (by value) | a **copy** of the object | original is **unchanged** |
| `&obj` (raw pointer) | a pointer to the original | original **is modified** — but you must guarantee `obj` outlives the thread |
| `std::shared_ptr<T>` | shared ownership of the original | original is modified **and** kept alive as long as the thread holds the pointer |

> Remember from folder `02`: `std::thread` copies its arguments. So passing `obj`
> copies the object; the worker mutates the copy and your original never changes
> (`example_14`, thread `t1`). Passing `&obj` or a `shared_ptr` shares the actual
> object.

### Why `std::shared_ptr` for shared objects
A `std::shared_ptr<T>` is a reference-counted smart pointer. Every thread that
holds a copy contributes to the count; the managed object is destroyed only when
the **last** owner goes away. This solves the *lifetime* problem: you never have
to prove the object outlives every thread by hand — shared ownership guarantees
it. (It does **not** solve the *data-race* problem — concurrent writes still need
a mutex, see folder `06`.)

### `std::thread` is move-only, not copyable
A `std::thread` **owns** an OS thread resource, so it cannot be *copied* (two
objects can't own the same thread). It can only be **moved**. This is why you
cannot `push_back(t)` a thread into a vector, but you can:
- `threads.emplace_back(std::thread(fn));` — constructs in place / moves, or
- `threads.push_back(std::move(t));`

Then join them all with a range-based loop (`for (auto& t : threads) t.join();`).

### Thread pools, the idea behind a `vector<std::thread>`
Creating a container of worker threads and joining them at the end is the
simplest form of *fan-out / fan-in* parallelism: split work across N workers,
then wait for all of them (the join loop is the *barrier*). Real thread pools
reuse threads instead of creating one per task, but the mental model starts here.

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_14.cpp` | Member function on a thread bound **by value** (mutation lost on a copy) vs. **by address** (mutation lands). Compare the two `print()` outputs. |
| `example_15.cpp` | `std::shared_ptr` as the safe way to share an object with a worker — cheap to copy, and it keeps the object alive. |
| `example_16.cpp` | Two threads mutating **different members** of one shared object — safe, because distinct members are distinct memory locations. |
| `example_17.cpp` | `std::thread` is **move-only**; build containers with `emplace_back`, never `push_back` of a named thread (compile error shown). |
| `example_18.cpp` | Ten workers with staggered sleeps — an order that *looks* deterministic but is only an artifact of the timing. |
| `example_19.cpp` | Two equivalent ways to hand data to a thread: variadic argument vs. lambda capture. |

## Try it yourself
- In `example_16.cpp`, point **both** threads at the same member (call `deposit`
  from both) and you have converted a safe program into a data race — the exact
  subject of folder `05`.
- In `example_17.cpp`, uncomment the `push_back` lines and read the error: it is
  the deleted copy constructor.
- In `example_18.cpp`, delete the `sleep_for` and re-run several times — the
  tidy ordering collapses. Nothing ever guaranteed it.

## Compiling
```sh
g++ -std=c++17 -pthread example_16.cpp -o example_16 && ./example_16
```
