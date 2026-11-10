# Filtering (stream compaction) on the GPU: chapter 12

A **filter** keeps the elements of an array that satisfy a condition and packs them at the
front of the output:

```
in     812  45 300  17 999  60 431   8      predicate: value < 100
flag     0   1   0   1   0   1   0   1
out     45  17  60   8                       count = 4
```

On the CPU this is the loop with `push_back` from session 09. That loop is exactly the thing
you cannot share between tasks: two tasks calling `push_back` on one vector is a data race
(`example_53`). On the GPU the question is the same, for a million threads at once: **when my
element passes, which output slot is mine?** The slot depends on how many elements before
mine also passed, and that is an exclusive scan of the flags (`../09_scan`, chapter 11). This
chapter is mostly about doing that scan, and the writes that follow it, without wasting memory
bandwidth.

Filtering is also called *stream compaction*, *select* or `copy_if`. It appears whenever work
becomes irregular. Examples: removing finished rentals from the active set, keeping only the
rays that still bounce, or listing the candidates that survive a cheap test before an
expensive one.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_sequential_filter.cu`](01_sequential_filter.cu) | The sequential filter, run by a single GPU thread: keep `v < 100`, pack the kept values at the front of the output and count them. | Of the 16 inputs, 8 are below 100, and `out` holds them in input order: 45 30 17 60 8 77 91 3. Only the first `count` entries of the output are meaningful. Try: set `THRESHOLD = 50`; 60, 77 and 91 drop out and the count becomes 5. |
| 2 | [`02_global_atomic.cu`](02_global_atomic.cu) | One thread per element. A kept element takes its slot with `atomicAdd` on one global counter, so the output holds the right set in an unpredictable order. | Compare the `out` and `seq` rows: they hold the same 8 values, but in the order in which the threads reached `atomicAdd` (here 30 17 45 at the front instead of 45 30 17). Run the cell again: the order can change from run to run and may even match by chance, but the set of values never changes. |
| 3 | [`03_block_private.cu`](03_block_private.cu) | A private counter per block in shared memory. Thread 0 reserves the block's range with one global atomic and prints it. The output is still unordered. | Block 0 keeps 3 values and reserves out[0..3), block 1 reserves out[3..5), and so on: 4 global atomics in total instead of 8. Each block's values stay together in its range, but inside a range they are unordered (block 0 wrote 17 30 45), and on a GPU the blocks can also reserve their ranges in a different order. |
| 4 | [`04_flags_scan_scatter.cu`](04_flags_scan_scatter.cu) | Order kept: a map kernel writes 0/1 flags, a one-block Kogge-Stone scans them, and the scatter writes `in[i]` to `out[pos[i] - 1]`. Flags and positions are printed. | `flag` marks the kept elements and `pos` is its inclusive scan, so every kept element knows its slot: 60 at index 5 has `pos` 4 and goes to out[3]. The last `pos`, 8, is the count, and `out` now has exactly the sequential order. Try: set `THRESHOLD = 50`; the flags of 60, 77 and 91 become 0, `pos` stays flat there and ends at 5. |
| 5 | [`05_staged_compaction.cu`](05_staged_compaction.cu) | Three 16-element sections, 4 threads with 4 elements each: count per section, scan the counts, then each block packs its kept elements in shared memory and copies them out coalesced. The last section is half padding. | The counts 8 7 3 are the kept elements of each 16-element section, and their scan 8 15 18 gives each section's end in `out`, so section 1 writes out[8..15). Section 2 has only 8 real values; its 8 padding slots read as 0, which would pass `v < 100`, so `is_real_and_kept` checks the index first and the count stays 3. The 18 values come out in input order. |
| 6 | [`06_single_pass.cu`](06_single_pass.cu) | The same block compaction in one kernel: a ticket and the domino chain of `../09_scan` give each section its start, so the input is read only once. | The sections, ranges and `out` are the same as in step 5, but each line now comes from inside the single compaction kernel: ticket 1 waits until ticket 0 has published 8, writes its 7 values to out[8..15) and publishes 15 for ticket 2. There is no separate count kernel or scan kernel any more, so the input is read once. |

After the steps come the complete programs: `filter.cu` runs the five variants on 16 M elements at two selectivities with checks and timings, and the bike-sharing examples (`bikeshare_late_returns.cu`, `bikeshare_idle_bikes.cu`) use the filter in an application.

---

## 1. The pattern

- **Predicate**: `keep(x)`, a pure function of one element (or of one element's fields). It
  must not depend on what other threads keep, and it must give the same answer every time it
  is evaluated, because some variants evaluate it twice (exercise 7).
- **Output size is unknown in advance.** It lies anywhere from 0 to n. Either allocate n, or
  count first and allocate exactly (the two-pass variants know the size before they write
  anything; `bikeshare_idle_bikes.cu` uses this).
- **Order.** A sequential loop produces the kept elements in input order. Some parallel
  variants keep that order and some do not. Whether that matters depends on the consumer: a
  report read next to the log needs it, and a set of candidate bikes for a rider does not.

The work is tiny (one comparison per element), so a filter is limited entirely by memory
traffic. The minimum is to read n elements and write the k that are kept. Every variant below
is judged by how close it gets to `n + k` accesses, and by how much it serializes.

## 2. Naive: one global atomic counter

```cpp
__global__ void filter_global_atomic(const int* in, int* out, int* counter, int n, int threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int v = in[i];
    if (keep(v, threshold)) {
      out[atomicAdd(counter, 1)] = v;
    }
  }
}
```

`atomicAdd` returns the old value, which is a unique slot. The output therefore holds exactly
the kept elements, each once. Memory traffic is close to the minimum. Two problems remain:

- **The order is lost.** The slot depends on which thread's atomic reached the counter first.
  Even neighbouring elements can swap. To get log order back you would need a sort, which costs
  far more than the filter.
- **Contention.** Every kept element of the whole grid updates *the same address*. Atomics on
  one address are serialized in the L2 cache. With n = 16 M and half the elements kept, that is
  8 M back-to-back operations on a single counter. Throughput then falls as selectivity rises,
  which is the opposite of what you want.

## 3. Block-level privatization: one global atomic per block

This is the same remedy as for the histogram (chapter 9). Give each block a **private** counter
in shared memory, let the threads take their slots there, and touch the global counter once per
block to reserve the whole block's range:

```cpp
if (threadIdx.x == 0) {
  blockCount = 0;
}
__syncthreads();

const int v = (i < n) ? in[i] : 0;
const bool k = (i < n) && keep(v, threshold);
int local = 0;
if (k) {
  local = atomicAdd(&blockCount, 1);
}
__syncthreads();

if (threadIdx.x == 0) {
  blockStart = atomicAdd(counter, blockCount);
}
__syncthreads();

if (k) {
  out[blockStart + local] = v;
}
```

The `atomicAdd` on `blockCount` is a shared-memory atomic; the one on `counter` is the only
global atomic of the block, done by thread 0 alone. The second barrier makes sure the block has
finished counting before thread 0 reserves the range; the third makes sure every thread sees
`blockStart` before it writes. For n = 16 M with 256-thread blocks, global atomics drop from up to 16 M to 62 500. Shared
atomics are fast, and they contend only within one block. Each block's kept elements land
together, and those writes are mostly coalesced. **The output is still unordered**, both
between blocks (whichever block reaches the global counter first comes first) and inside a
block (whichever thread wins the shared atomic).

On a real GPU you can go one level further with **warp-aggregated atomics**. A warp computes a
bit mask of its kept lanes (`__ballot_sync`), counts the bits (`__popc`), lets one lane do the
atomic, and gives every kept lane its rank inside the mask. That is a scan of 32 flags done in
hardware. `filter.cu` keeps the chapter's block-level versions; the CPU emulation now supports
`__ballot_sync` (see `../08_reduction`), so a warp-aggregated filter is a good exercise.

## 4. Keeping the order: flags, exclusive scan, scatter

The slot of a kept element in a sequential loop is *the number of kept elements before it*.
That is the exclusive scan of the flags:

```
in      812  45 300  17 999  60 431   8
flag      0   1   0   1   0   1   0   1
excl      0   0   1   1   2   2   3   3     <- slot of each kept element
incl      0   1   1   2   2   3   3   4     <- incl[n-1] = number kept
out[excl[i]] = in[i]   for every i with flag[i] = 1   ->   45 17 60 8
```

Variant 3 in `filter.cu` writes this down as directly as possible: a map kernel (`make_flags`),
the hierarchical scan of chapter 11 on the n flags, and a scatter kernel. It uses the inclusive
scan, so the slot is `pos[i] - 1`:

```cpp
__global__ void scatter_kept(const int* in, const int* flags, const int* pos, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) {
    out[pos[i] - 1] = in[i];
  }
}
```

It is correct, ordered and easy to reason about. It is also expensive in traffic. The map reads
n and writes n flags. The hierarchical scan reads and writes n twice. The scatter reads `in`,
`flags` and `pos` (3n) and writes k. That is about **8n + k**, compared with the minimum n + k.

## 5. Fusing: count per block, scan the counts, compact inside the block

A block does not need a global array of flags. It only needs **one number from the outside**:
how many elements all earlier blocks kept. Everything else can happen in shared memory. That
gives variant 4 (`count_kept`, then a scan of the block counts, then `compact_staged`):

```
pass 1  count_kept       each block counts its kept elements         read n, write n/2048
pass 2  scan of counts   inclusive scan of n/2048 numbers            negligible
pass 3  compact_staged   each block, in shared memory:               read n, write k
          load section (coalesced) -> count per thread -> scan of the 256 counts
          -> pack kept elements into stage[] in input order -> coalesced copy to
             out[start of this block ...]
```

Traffic: about **2n + k**. The flags and positions never leave the chip.

### Coarsening: contiguous runs per thread

Each thread owns **8 contiguous elements** of a 2048-element section. It evaluates the predicate
on its run sequentially, and only the 256 per-thread counts are scanned (Kogge-Stone, 8 steps).
Because each thread's run is contiguous and the runs are in thread order, packing "thread 0's
kept elements, then thread 1's, …" keeps the input order. Coarsening also gives 8× fewer blocks,
so the count array and its scan shrink by 8×. That is the three-phase scan of chapter 11,
applied to flags.

### Staging in shared memory (packing) for coalesced writes

If each thread wrote its kept elements straight to global memory, thread t would write to
`start + pos_t`, `start + pos_t + 1`, … while thread t + 1 writes a few slots further on. In one
instruction the 32 threads of a warp would hit 32 scattered addresses, and most of them would be
inactive (their element was not kept). Instead, the block **packs** its kept elements into a
shared array first. Then all threads copy it out together, and consecutive threads write
consecutive addresses. The end of `block_compact` packs; `part` holds the inclusive scan of
the per-thread counts, so `exclusive_from_inclusive(part, t)` (that is, `part[t - 1]`, or 0 for
thread 0) is where thread t's kept elements start:

```cpp
int pos = exclusive_from_inclusive(part, t);
for (int k = 0; k < F_COARSEN; ++k) {
  if (is_real_and_kept(vals, first + k, base, n, threshold)) {
    stage[pos++] = vals[first + k];
  }
}
__syncthreads();
return part[F_THREADS - 1];
```

and `compact_staged` copies the packed elements out with consecutive threads writing
consecutive addresses:

```cpp
__device__ void copy_out_coalesced(const int* stage, int count, int* out) {
  for (int k = threadIdx.x; k < count; k += F_THREADS) {
    out[k] = stage[k];
  }
}
```

Every lane is busy and every warp writes one contiguous 128-byte run. The load phase uses the
same trick in the other direction: it is coalesced into `vals[]` and then read per thread from
shared memory.

## 6. Single pass

Variant 5 (`compact_single_pass`) removes pass 1. It does the block compaction first, then gets
its starting offset from the **domino chain** of chapter 11: wait for the previous section's
running total, publish its own, then copy the staged elements out. The section a block works on
comes from an atomic **ticket**, not from `blockIdx.x`. A block can only wait for a block that
has already started, so the chain cannot deadlock however the GPU schedules blocks. The hand-off
uses the same device functions as `domino_scan` (`next_ticket`,
`wait_for_sum_of_previous_sections`, `publish_sum_of_sections_up_to`). See
`../09_scan/README.md`, section 8. Traffic: **n + k**, the minimum. The cost is the serial hand-off
between sections, which takes about one L2 round trip per section.

## 7. Comparison

| variant | ordered | global atomics | memory traffic | kernels | serial part |
|---------|:-------:|----------------|----------------|:-------:|-------------|
| 1 global atomic | no | one per kept element | n + k | 1 | the atomic counter |
| 2 block-private | no | one per block | n + k | 1 | — |
| 3 flags + scan + scatter | yes | 0 | ≈ 8n + k | 3 + scan | — |
| 4 count + scan + staged compact | yes | 0 | ≈ 2n + k | 2 + scan | — |
| 5 single pass (domino) | yes | a few per block (ticket, chain) | n + k | 1 | one hand-off per section |

What to expect on a GPU (measure it, exercise 2):

- **Selectivity matters for variant 1** (more kept elements means more atomics on one address).
  It matters much less for 2–5, where only the k writes grow.
- **Variant 3 is the slowest ordered one**: its traffic is about 4× that of variant 4, and its
  time follows.
- **Variants 2, 4 and 5** are all close to memory speed. Variant 4 pays one extra read of the
  input. Variant 5 pays chain latency, which matters most when sections are few and small.
- **The emulation numbers** on a laptop say nothing about any of this. They measure
  `std::thread` barriers. Use them only to see that every variant runs and passes.

## 8. Performance notes

- **Memory-bound.** Arithmetic intensity is about one comparison per 4 bytes. Compare your time
  with a device-to-device `cudaMemcpy` of n ints.
- **Structure of arrays.** The bike-sharing programs evaluate predicates on separate arrays
  (`startMs[]`, `endMs[]`, `state[]`, `battery[]`, `x[]`, `y[]`), so each warp load is one coalesced
  128-byte transaction. With an array of structs, the same predicate would drag every field
  through the memory bus.
- **Compact indices, then gather.** When the records are wide, compact only the *indices* of the
  kept records (small and coalesced), and gather the fields for the k survivors afterwards
  (`bikeshare_late_returns.cu`).
- **Shared memory budget.** Variant 4/5 blocks use `vals` + `stage` + `part` = 8 + 8 + 1 KB.
  That is small enough for several resident blocks per SM. You could alias `stage` onto `vals`
  if values were kept in registers (exercise 5).
- **Size the output from the scan.** In the two-pass variants, the total is known after the scan
  and before any output is written. `bikeshare_idle_bikes.cu` allocates its candidate array
  from that number.

## 9. Pitfalls

- **Padding elements that pass.** The last section is padded with 0 in shared memory, and
  `0 < threshold` is true. Every predicate test therefore also checks `base + j < n`; in
`block_compact` that check and the predicate together form `is_real_and_kept`. Remove
  that check and the output grows by the number of padding slots (exercise 4).
- **Reading the counter too early.** The global counter or the scanned total is only valid after
  the kernel has finished. `cudaMemcpy` synchronizes, so read it with `cudaMemcpy`, never from a
  still-running kernel.
- **Writing past the output.** A bug in the offsets (an inclusive value used where an exclusive
  one is needed) shifts everything by one and writes `out[k]`. Use a buffer filled with −1, as
  `filter.cu` does, so that missing or shifted elements show up.
- **Filtering in place.** `out == in` is a race: a block may overwrite elements that another
  block has not read yet. Use two buffers and swap them between rounds.
- **A predicate on floats.** nvcc may fuse `dx*dx + dy*dy` into an FMA, so values exactly on the
  boundary can be classified differently on the GPU and on the CPU reference.
  `bikeshare_idle_bikes.cu` uses integer metres to stay exact.
- **Believing the emulation's order.** On the laptop, blocks run one after another, so the
  "unordered" variants often come out in order. That is luck. On a GPU they will not.
- **An early `return` before a barrier** in the compaction kernels: threads past `n` must still
  reach every `__syncthreads()`. Guard the work with `if`.

## 10. Ideas for your project

### The bike-sharing examples in this folder

The two programs below apply the filter to a city bike-sharing system: 48 docking stations on a
6 km × 6 km map, bikes that are docked, out on a ride or in repair, and a log of finished trips.
The data is synthetic and generated deterministically with a small SplitMix64 generator inside
each file, so every run prints the same numbers.

**Late returns report (`bikeshare_late_returns.cu`).** Each finished trip has an undock time
and a dock time, and the log lists the trips by dock time. The fare includes 45 minutes; the
report lists the trips that ran over that time **in log sequence**, so it can be read next to
the log. That needs an order-preserving filter, so the program uses variant 4: `count_late`
counts per 2048-trip section, the hierarchical Brent-Kung scan turns the counts into offsets,
and `compact_late` stages the *indices* of the late trips in shared memory and copies them out
coalesced. A gather kernel then builds the report columns (trip, bike, start station, dock
time, ride time and fare) for the survivors only. The fare is computed in integer cents, Bs 3
plus Bs 1 per started 10 minutes, so the GPU and the CPU agree exactly. For comparison the
program also runs the block-private atomic version (`late_by_atomics`). It finds the same set,
in a scrambled sequence on a GPU, which the report could not use without sorting it first. The
synthetic log covers the evening rush (16:00 to 20:00): most rides are commutes of 6 to 46
minutes and some are leisure rides of up to two and a half hours, so about 13 % of the trips
are late. The output is a table with the number of late trips per version and whether it came
out in log sequence, the first report lines, and the start station and the bike with the most
late returns, counted on the host from the small compacted list.

**Bikes ready for each rider (`bikeshare_idle_bikes.cu`).** A batch of rider requests, each a
position on the map. For each request the program lists the bikes that are docked, have at
least 30 % battery and stand within 600 m. That is one filter per request, and all of them run
in a single 2-D grid: `blockIdx.y` is the request and `blockIdx.x` a section of the fleet. The
per-block counts form a request-major matrix. Scanning it in that sequence makes each request's
candidates one contiguous slice, by ascending bike id, and the scan also gives every slice's
start offset. Because the variant has two passes, the total is known before anything is
written, and the candidate array is allocated with exactly that size. The predicate
`isReadyBike` combines state, battery and an integer distance test, and the same function
serves the kernels and the CPU reference. A docked bike stands at its station's position. The
output is the total number of candidates against the CPU, and for the first requests the number
of bikes ready and the best one (nearest, the higher battery on a tie) with its station,
distance and battery, picked on the host from the compacted slice.

### Where this pattern may appear in your project

- A report that must be read next to a log (only the lines that meet a condition, in their
  original sequence) is an order-preserving filter. Do you need that sequence, or only the set?
- Before an expensive computation per candidate, is there a cheap test that discards most of
  them? Filter first, and spend the expensive work on the survivors only.
- When the records are wide, could you compact only their indices and fetch the fields you
  need afterwards?
- Many small searches over the same large array (one per waiting request) can share one launch.
  How would you keep the results of each search together, and know where each one starts?
- A set of active items that shrinks every tick, or the items still unfinished when a run stops,
  can be produced by compaction into a second buffer.

## 11. Files and how to run

| file | what it shows |
|------|---------------|
| `01_*.cu` … `06_*.cu` | the learning path above, one idea per step on 16 or 40 values, each printing its input, its intermediate arrays and PASS/FAIL |
| `filter.cu` | the five variants (global atomic, block-private, flags + scan + scatter, count + scan + staged compact, single pass) at 10 % and 50 % selectivity, with a CPU reference, order and multiset checks, and timings |
| `bikeshare_late_returns.cu` | order-preserving filter of trips over the included time, gather into a report with fares, comparison with the unordered version |
| `bikeshare_idle_bikes.cu` | many filters in one 2-D grid: docked, charged bikes within a radius of each rider request, output grouped by request |

Laptop (no GPU, the header emulates CUDA with threads), the steps first:

```sh
cd ec2/10_filter
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_sequential_filter.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_global_atomic.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_block_private.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_flags_scan_scatter.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_staged_compaction.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_single_pass.cu -o step && ./step
```

Then the complete programs and the bike-sharing examples:

```sh
cd ec2/10_filter
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common filter.cu -o filter && ./filter
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_late_returns.cu -o late && ./late
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_idle_bikes.cu -o idle && ./idle
# race check (smaller size: TSan is slow)
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common filter.cu -o filter_tsan && ./filter_tsan 5000
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common -DCPU_TRIPS=8000 bikeshare_late_returns.cu -o late_tsan && ./late_tsan
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common -DCPU_BIKES=3000 -DCPU_REQUESTS=4 bikeshare_idle_bikes.cu -o idle_tsan && ./idle_tsan
```

The bike-sharing programs read their emulation sizes from `CPU_TRIPS` (default 60 000) and from
`CPU_BIKES` and `CPU_REQUESTS` (defaults 12 000 and 16). They also run under ThreadSanitizer at
the default sizes, in a few seconds; the smaller values only make the race check faster.

> **Easiest way on Colab:** this folder has two notebooks. [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) runs the steps of the learning path in order, and [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) runs the complete programs and the bike-sharing examples. Upload one to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

Google Colab (GPU runtime), from a cell:

```sh
!cd ec2/10_filter && nvcc -O2 -I../common 04_flags_scan_scatter.cu -o step && ./step
!cd ec2/10_filter && nvcc -O2 -I../common filter.cu -o filter && ./filter
!cd ec2/10_filter && nvcc -O2 -I../common bikeshare_late_returns.cu -o late && ./late
!cd ec2/10_filter && nvcc -O2 -I../common bikeshare_idle_bikes.cu -o idle && ./idle
```

The step files have a fixed tiny input. The complete programs take optional size arguments
(`./filter 5000`, `./late 5000`, `./idle 3000 4` for bikes and rider requests). Every program
prints PASS/FAIL against a CPU reference and exits non-zero if a check fails.

## 12. Exercises

1. **Count the atomics.** For n = 16 M, 256-thread blocks and 50 % selectivity, how many global
   atomic operations do variants 1 and 2 perform? And variant 5, with 2048-element sections?
2. **Your own table.** On Colab (`colab_2_full_programs.ipynb`), run `filter.cu` and add a column "GB/s = 4·(n + k) bytes ÷ time".
   Explain the ranking at 10 % and at 50 % using section 7.
3. **Stable partition.** Change variant 3 so it writes the kept elements first, in order,
   *followed by* the rejected ones, also in order (like `std::stable_partition`). Give the slot
   of a rejected element in terms of `i`, `excl[i]` and the total kept.
4. **The padding bug.** Remove `base + j < n &&` from `is_real_and_kept` and run
   `./filter 5000`. Predict how many extra elements appear at threshold 100, and what value they
   have.
5. **Save shared memory.** `block_compact` keeps `vals[]` and `stage[]` separate. Show that, if
   each thread first copies its 8 values into a local array `int v[F_COARSEN]`, `stage` can
   reuse `vals`. Which barrier makes it safe?
6. **A filter that looks at neighbours.** Remove consecutive duplicate bike positions from a
   GPS trace (keep `p[i]` if `i == 0 || p[i] != p[i-1]`). Is this still a filter in the sense of
   this chapter? What must the predicate read from, and what must it never read from?
7. **Evaluate once or twice?** Variant 4 evaluates the predicate in `count_kept` and again in
   `compact_staged`. Give a predicate for which that is wrong, and fix it without falling back
   to variant 3.
8. **Many queries.** In `bikeshare_idle_bikes.cu`, the count matrix is stored as
   `counts[q * gridDim.x + blockIdx.x]`. What goes wrong if it is stored as
   `counts[blockIdx.x * Q + q]` and scanned the same way?

### Short solutions

1. Variant 1: 8 M atomics on one address. Variant 2: 16 M / 256 = 62 500 (plus 8 M cheap shared
   atomics). Variant 5: one ticket plus about two chain atomics per section, so about 3 × 7813
   ≈ 23 000, spread over different addresses.
2. Variant 1 gets slower from 10 % to 50 % (5× more atomics on one counter). Variants 2, 4 and 5
   barely change except for the extra writes. Variant 3 is the slowest ordered one, at about
   8n traffic.
3. Kept: `excl[i]`. Rejected: `total + (i - excl[i])`, because `i - excl[i]` is the number of
   rejected elements before `i`.
4. The padding is `vals[j] = 0` for `base + j >= n`. With n = 5000 and 2048-element sections,
   the last section pads 6144 − 5000 = 1144 slots, and all of them pass `0 < 100`. Variant 5 adds
   1144 to its total and FAILs the count check. Variant 4 is worse. `count_kept` still checks
   `j < n`, so its total looks right, but `compact_staged` copies 1144 extra zeros past the end
   of the last slice, and so past the end of the output buffer. That is a memory error the count
   check cannot see. (Run it under AddressSanitizer.)
5. After `v[k] = vals[first + k]`, a thread no longer reads `vals`. The first barrier of the
   Kogge-Stone loop over `part[]` comes after every thread has made its copy. So, by the time
   any thread writes `stage[pos]` (= `vals[pos]`), all reads of `vals` are done.
6. Yes. The flag of element i depends only on the *input* (`p[i]` and `p[i-1]`), never on the
   output or on other flags. Reading `p[i-1]` across a section boundary is fine, because it is
   global input that nobody writes.
7. A predicate that is not repeatable, for example one that uses a random number generator, or
   reads a value that another kernel is changing. The two passes then disagree, and the counts
   no longer match the elements that get written. Fix: write the flags (as bits or bytes) in
   pass 1 and read them in pass 3, or use the single-pass variant 5.
8. The scan would then interleave the requests. The offset of (q, b) would count the candidates
   of *all* requests in blocks before b, so the candidates of one request would no longer be
   contiguous, and `start[q]` could not be read off the scan.
