# 02 — Passing Data to Threads (Lambdas, Captures, Argument Semantics)

## Theory you need before reading the code

### Lambdas and closures
A **lambda** is an anonymous function object. Its **capture list** `[...]`
decides what state from the enclosing scope it can see:
- `[x]` — **capture by value**: a *copy* of `x` is stored inside the lambda at the
  moment the lambda is *created*. Later changes to the outer `x` are not seen.
- `[&x]` — **capture by reference**: the lambda refers to the *original* `x`. If
  the original outlives... — careful: if the referenced variable is destroyed
  before the lambda runs, you get a **dangling reference** (undefined behavior).
- `[=]` / `[&]` — capture *everything* used, by value / by reference.
- `mutable` — by default a by-value capture is *const* inside the lambda;
  `mutable` lets the lambda modify its *own copy* (the outer variable is
  untouched).

> **Why this matters for threads:** the timing of *when* a captured value is read
> is decoupled from *when* the thread actually runs. Capture-by-reference into a
> thread is a classic source of races and dangling references.

### How `std::thread` forwards arguments — the copy trap
`std::thread t(fn, arg1, arg2)` **copies (or moves) each argument into internal
thread storage** and then calls `fn` with those internal copies. Consequences:
- Even if `fn` takes a parameter **by reference** (`T&`), the thread passes it a
  reference to its *internal copy*, not to your original variable. Modifications
  are lost.
- To genuinely pass a reference, wrap the argument in **`std::ref(x)`**
  (or `std::cref(x)` for const). This tells `std::thread` to store a
  reference-wrapper instead of a copy. You are now responsible for keeping `x`
  alive for the thread's whole lifetime.

### Value copy vs. move semantics
- **Copy**: duplicates the object (expensive for large data).
- **Move** (`std::move(x)`): transfers ownership of `x`'s internals to the new
  location, leaving `x` in a valid-but-unspecified ("empty") state. Cheap. Use it
  when you no longer need the source. After a move, do **not** read the source's
  value expecting the old contents.

### Variadic templates
`std::thread` uses **variadic templates** (`template<class... Args>`) so it can
accept *any number* of arguments of *any types* and forward them to the callable.
This is why `std::thread(fn, a, b, c)` "just works" — but also why an argument
count that doesn't match `fn` is a **compile-time** error (see `example_11`).

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_9.cpp` | Lambda captures with **no threads yet**: by-reference vs. by-value, `mutable`, and *when* a capture reads its variable. |
| `example_10.cpp` | The same capture choice across **real threads** — the by-reference worker races with `main`, the by-value one cannot. |
| `example_11.cpp` | The variadic `std::thread` constructor; an argument-count mismatch is a **compile-time** error (commented out). |
| `example_12.cpp` | Thread arguments are **copied** by default; `std::move` transfers instead, leaving the source moved-from. |
| `example_13.cpp` | `std::ref` to pass a genuine reference — the worker mutates `main`'s object. |

## Try it yourself
- In `example_10.cpp`, swap the two sleep durations and re-run — the by-reference
  thread's output changes, the by-value thread's never does.
- In `example_11.cpp`, uncomment the third thread and read the compiler error.
- In `example_13.cpp`, remove `std::ref` — the code still compiles only if the
  parameter is `const&` or by value; with `std::string&` it fails. That failure
  is the compiler protecting you from silently mutating a temporary copy.
- Also in `example_13.cpp`, move `join()` to the end of `main` and you have
  built a data race — which is exactly what folder `05` is about.

## Compiling
```sh
g++ -std=c++17 -pthread example_13.cpp -o example_13 && ./example_13
```
