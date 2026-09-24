# 09 — Task Parallelism and Measuring Speedup

## Theory you need before reading the code

### From concurrency to parallelism
Folders 01 to 08 were about **concurrency**: structuring a program as tasks that share
things safely. This folder is about **parallelism**: making one computation *finish
sooner* by giving different parts of it to different cores. The tools are the ones you
already know (`std::async`, futures, a mutex-and-condition-variable queue); what changes is
the question. Not "is it safe?" but "**is it faster, and by how much?**"

### Fan-out / fan-in
The simplest parallel program: split the input into chunks, run one task per chunk, wait
for all of them, combine the partial results (`example_50`). Each task **owns** its chunk
and its partial result, so there is nothing to lock. The join at the end is the barrier.

### Speedup, and why you must measure it
```
speedup(n) = T(1) / T(n)        ideal: n     efficiency = speedup / n
```
Always measure the sequential version and the parallel version on the **same data**, with
`-O2`, several times. `example_50` reaches about 7.5x on 10 cores: not 10x, because
creating tasks, combining results and memory bandwidth all cost something.

### Amdahl's law
If a fraction **f** of the work is inherently serial (loading input, combining results),
then with n cores:
```
speedup(n) = 1 / (f + (1 - f) / n)      and     speedup(∞) = 1 / f
```
`example_51` measures f, prints the measured speedup next to Amdahl's prediction for 1 to
16 tasks, and shows the curve flattening at 16 tasks on a 10-core machine: more tasks than
cores gain nothing and cost switching. The lesson is to **shrink the serial part** before
adding cores.

### Fork/join and the cutoff
Divide-and-conquer parallelism: split the range in halves, run one half on a new task and
the other half on the current thread, join (`example_52`). Recursion stops at a **cutoff**
below which the chunk is processed sequentially, because spawning a task costs far more
than processing a few elements. Too large a cutoff: too few tasks, cores idle. Too small:
thousands of tasks, overhead dominates. The example prints the whole curve.

### Parallel map, reduce and scan: the three primitives
- **map** (`example_53`): apply a function to every element, writing to a *pre-sized*
  output vector. Task t writes only `out[b_t, e_t)`, so no lock. `push_back` from several
  tasks would be a data race.
- **reduce** (`example_56`): combine elements with an operation. Parallel only if the
  operation is **associative** (and commutative if chunks may combine in any order).
  Integer `+`: exact. Floating-point `+`: results differ in the last digits. `-`: simply
  wrong. Things like "average" are made reducible by reducing a `(sum, count)` pair.
- **scan** / prefix sum (`example_54`): output[i] = input[0] + … + input[i]. Looks
  inherently serial (each value depends on the previous), yet parallelizes in **three
  phases**: scan each chunk locally and record its total, scan the totals (tiny,
  sequential), add each chunk's offset in parallel. It does about twice the work of the
  sequential scan, so it wins only with enough cores and data.

### Thread pools
`std::async(std::launch::async, …)` creates an OS thread per call (50–100 µs each). For
many small jobs that is the bottleneck. A **thread pool** (`example_55`) creates N workers
once and feeds them from a queue: exactly the monitor object of folder 07 with a
`std::condition_variable`, plus `std::packaged_task` so every submitted job returns a
`std::future`. Shutdown is a `stopping` flag checked *inside* the wait predicate, then
`notify_all` and `join` every worker. Job runs happen **outside** the lock.

## Files
| File | What it demonstrates |
|------|----------------------|
| `example_50.cpp` | Fan-out/fan-in with `std::async`; measured speedup vs `hardware_concurrency()`. |
| `example_51.cpp` | Speedup table for 1…16 tasks; serial fraction and Amdahl's law next to the measurement. |
| `example_52.cpp` | Recursive fork/join with a cutoff; timing for 1 to 2000 leaf tasks. |
| `example_53.cpp` | Parallel map into a pre-sized output: disjoint ranges, no lock. |
| `example_54.cpp` | Parallel prefix sum in three phases; identical result to `std::partial_sum`. |
| `example_55.cpp` | A thread pool built from the folder-07 queue plus `std::packaged_task`; pool vs one thread per job. |
| `example_56.cpp` | Associativity: integer, floating point, subtraction, and the `(sum, count)` trick for averages. |

## Exercises (change of context: orders and pricing)
| File | Task |
|------|------|
| `exercise_9_1.cpp` | Daily revenue over a million orders as a recursive fork/join; find the cutoff sweet spot; explain why the result is exact here. |
| `exercise_9_2.cpp` | Cumulative deliveries timeline as a three-phase parallel scan; break-even analysis. |
| `exercise_9_3.cpp` | Price 50 000 orders on the thread pool; workers-vs-speedup table; pool vs one `std::async` per order; chunked jobs. |

## Try it yourself
- Run `example_50` with `nTasks` = 2 × `hardware_concurrency()` and 4 ×. Does it help?
- In `example_51`, make the serial part ten times bigger (load 30 million) and watch
  1/f collapse.
- In `example_55`, add a `std::cout` inside `inspect` and see why *I/O inside a job* kills
  a pool's throughput.

## Compiling
```sh
g++ -std=c++17 -pthread -O2 example_51.cpp -o example_51 && ./example_51
```
Every timing in this folder is meaningless without `-O2`.
