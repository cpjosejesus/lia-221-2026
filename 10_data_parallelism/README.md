# 10 — Data Parallelism (and the road to GPUs)

## Theory you need before reading the code

### Task parallelism vs data parallelism
Folder 09 split *work* into tasks. **Data parallelism** splits *data*: the same operation
runs on every element of a large collection, and the elements are independent. It is the
most scalable kind of parallelism there is, because the amount of parallel work grows with
the data, not with your program. It is also exactly how a GPU works, which is why this
folder ends the CPU part of the course.

### The primitives, applied to grids and collections
- **Map over a grid** (`example_58`): a box blur of a traffic-density image. Every output
  pixel depends only on the *input*, so the image is cut into **horizontal strips**, one
  task each. The input is shared read-only (no lock needed: nobody writes it), the output
  is written in disjoint rows. Strips, not pixels: a task per pixel would be two million
  tasks, and task cost must be dwarfed by the work it does.
- **Map + combine** (`example_59`, k-means): the assignment step is a map, the update step
  is a reduce. Each task accumulates into **its own** sums and counts (a *combiner*),
  and the tiny per-task results are merged once. No shared accumulator is written by two
  tasks, ever.
- **Embarrassingly parallel** (`example_62`, Monte Carlo): no shared state at all, just a
  sum at the end. Near-ideal speedup. The one trap: each task must own its random number
  generator with its own seed. Sharing one `std::mt19937` is a data race, and identical
  seeds just repeat the same samples n times.

### Combiners beat shared accumulators
`example_61` builds a histogram three ways:
| Approach | 30M increments, 10 tasks |
|----------|--------------------------|
| shared vector + mutex per increment | ~750 ms |
| shared vector of atomics | ~1200 ms (contention and false sharing on 32 counters) |
| private histogram per task, merged once | ~3 ms |
Atomics are not free: all tasks hammering the same cache lines is *slower* than a mutex.
The pattern that wins is **accumulate privately, merge rarely**. Everything in this
folder follows it.

### Load balancing: static vs dynamic
Equal-size chunks assume equal work per element. When the work is uneven, and the heavy
elements are clustered, one task gets them all and the others idle (`example_60`: 1046 ms
static vs 150 ms dynamic on 10 cores). **Dynamic scheduling** hands out chunks of `grain`
elements from an atomic counter: `next.fetch_add(grain)`, whoever is free takes the next
chunk. Grain too large: imbalance again. Grain too small: the counter becomes the
bottleneck (the example prints the curve down to grain 1).

### Memory layout matters as much as cores
Rows of an image are contiguous in memory; columns are not. A task walking a row strip
streams through memory; a task walking a column strip jumps `w` elements at a time and
misses the cache. Exercise 10.1 asks you to measure the difference. This is the CPU version
of *memory coalescing* on a GPU.

### The standard library's parallel algorithms
C++17 added execution policies: `std::sort(std::execution::par, …)`,
`std::transform(std::execution::par, …)` and friends. They do for you what `example_53`
did by hand. Support varies: GCC needs `-ltbb`, MSVC has it, **Apple clang does not ship
it**. `example_57` checks `__cpp_lib_parallel_algorithm` and otherwise falls back to the
hand-made version so the program runs everywhere.

### Why this leads straight to GPUs
Everything above maps one-to-one onto CUDA:
| CPU (this folder) | GPU (next session) |
|-------------------|--------------------|
| the function applied to each element | the **kernel** |
| one task per strip of rows | a **grid** of **blocks** of **threads**, one thread per element |
| chunking by `hardware_concurrency()` | the hardware schedules tens of thousands of threads itself |
| walking rows, not columns | **coalesced** memory access |
| combiner then merge | per-block reduction in shared memory, then a second kernel |
| `std::vector` in RAM | host memory vs **device** memory, explicit copies, and their cost |
| parallel scan in three phases | the same three phases, one block per chunk |
A GPU is not a faster CPU. It is a machine for map, reduce and scan over very large data,
with a memory copy at each end. Learn those three here, then the GPU is the same program
with a different launcher.

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_57.cpp` | `std::execution::par` where available, detected with a feature-test macro; hand-made fallback otherwise. |
| `example_58.cpp` | Box blur over a traffic-density image in horizontal strips; strips-vs-pixels reasoning. |
| `example_59.cpp` | Parallel k-means with per-task partial sums (combiners) and a cheap merge. |
| `example_60.cpp` | Static vs dynamic chunking on uneven work, with the grain-size curve down to 1. |
| `example_61.cpp` | Histogram three ways: mutex, atomics, per-task combiners. |
| `example_62.cpp` | Monte Carlo pi with one random generator per task; embarrassingly parallel. |

## Exercises (change of context: orders and deliveries)
| File | Task |
|------|------|
| `exercise_10_1.cpp` | Smooth an order heat map in horizontal strips; then vertical strips; measure why rows win; bonus separable blur. |
| `exercise_10_2.cpp` | Place three delivery hubs with parallel k-means over three million deliveries; identical centroids; convergence test. |
| `exercise_10_3.cpp` | Uneven orders: diagnose the static split, implement dynamic claiming with a grain table, add per-restaurant totals with combiners. |

## Try it yourself
- In `example_61`, replace the 32 atomics by 32 *padded* atomics (`alignas(64)`). How much
  of the atomic version's cost was false sharing?
- In `example_58`, set the radius to 1 and to 12. How does the speedup change, and why?
- In `example_62`, give every task the *same* seed. Does pi still converge? Why is it a bug
  anyway?

## Compiling
```sh
g++ -std=c++17 -pthread -O2 example_61.cpp -o example_61 && ./example_61
# example_57 with GCC and parallel algorithms:
g++ -std=c++17 -pthread -O2 example_57.cpp -ltbb -o example_57
```
Every timing in this folder is meaningless without `-O2`.
