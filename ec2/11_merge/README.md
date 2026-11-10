# Parallel merge (chapter 13)

Based on chapter 13 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj).
The earlier patterns of the course (map, stencil, histogram, reduction, scan) have one thing
in common: thread `t` knows from its index alone which input elements it must read. Merge
breaks that rule. Where thread `t` should start reading depends on the **values** in the
data, so each thread first has to *search* for its input. The tool for that search, the
**co-rank function**, is the main idea of the chapter, and chapter 14 uses it again to build
a parallel merge sort.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_merge_sequential.cu`](01_merge_sequential.cu) | The sequential merge of two small sorted arrays, run by a single GPU thread. Ties go to `A` (`<=`), so the merge is stable. | C has 11 values, 5 from A and 6 from B. Look at the three 5s: the two 5s of A (C[2] and C[3]) come before the 5 of B (C[4]), and the 8 of A comes before the 8 of B. With plain integers the order of equal values is invisible, but with records it is what makes the merge stable. |
| 2 | [`02_co_rank.cu`](02_co_rank.cu) | The co-rank function: one thread per output position `k` finds how many elements of `A` the first `k` outputs take. The binary search for `k = 6` is printed step by step, as in the worked example of the README. | The trace starts at i = 5, finds `A[4]=12 > B[1]=5` (too many elements from A), halves the range and stops at i = 3, j = 3: the first 6 outputs take A[0, 3) and B[0, 3). In the table i grows by 1 when output k came from A and j when it came from B, for example from k = 4 to k = 5 only j grows, because C[4] is the 5 of B. Try: set `TRACED_K = 9`; the search stops after two lines at i = 4, j = 5, the same as the table row for k = 9. |
| 3 | [`03_merge_basic.cu`](03_merge_basic.cu) | The basic parallel merge kernel: each of 4 threads owns a fixed range of outputs, finds its input ranges with two co-rank calls and merges them sequentially. The program prints the range of every thread. | Every thread owns 3 outputs (the last one 2), and its line gives the input ranges its two co-rank calls found, for example thread 1 merges A[2, 3) = 5 with B[1, 3) = 5 6. The input ranges have different lengths (thread 0 takes 2 elements of A and 1 of B), but the output ranges are fixed in advance. Try: set `THREADS = 3`; each thread owns 4 outputs and thread 0 takes A[0, 3) and B[0, 1). |
| 4 | [`04_merge_tiled.cu`](04_merge_tiled.cu) | The tiled kernel: each block finds its ranges once, then loads `TILE` elements of `A` and of `B` into `A_S` and `B_S` with coalesced loads and runs the co-rank inside shared memory (2 blocks, 2 threads each, `TILE = 4`). | Each block owns 11 of the 22 outputs: block 0 merges A[0, 6) with B[0, 5) and block 1 merges A[6, 10) with B[5, 12), so block 1 takes 7 elements of B and only 4 of A. With `TILE = 4` each block works in three rounds of 4, 4 and 3 outputs, loading up to 4 elements of each input into shared memory per round. Try: set `BLOCKS = 3`; each block owns 8 outputs and block 2 merges A[8, 10) with B[8, 12). |
| 5 | [`05_merge_circular.cu`](05_merge_circular.cu) | Circular buffers: the unused part of each tile stays in shared memory and only the consumed slots are refilled, so every input element is loaded once. Every index into a tile goes through `circular_slot`. | The output is the same C as in step 4, and nothing about the tiles is printed: the difference is in the loads. The tiled kernel reloads a full tile every round, even the elements it did not consume, while here only the consumed slots are refilled, at `circular_slot(x) = x & 3`, so each input element is read from global memory once. |

The steps use tiny inputs so the output can be checked by eye. After them, `merge.cu` runs the
three kernels on edge cases and on millions of elements with timings, and
`bikeshare_station_logs.cu` uses the merge to build one city log from the logs of the stations.

## Contents

1. [Ordered merge and stability](#1-ordered-merge-and-stability)
2. [The sequential merge](#2-the-sequential-merge)
3. [Splitting the output: the co-rank function](#3-splitting-the-output-the-co-rank-function)
4. [Finding the co-rank with a binary search (worked example)](#4-finding-the-co-rank-with-a-binary-search)
5. [The basic kernel](#5-the-basic-kernel)
6. [The tiled kernel](#6-the-tiled-kernel)
7. [Circular buffers](#7-circular-buffers)
8. [Thread coarsening](#8-thread-coarsening)
9. [Performance analysis](#9-performance-analysis)
10. [Pitfalls](#10-pitfalls)
11. [Ideas for your project](#11-ideas-for-your-project)
12. [Files and how to run](#12-files-and-how-to-run)
13. [Exercises](#13-exercises)

---

## 1. Ordered merge and stability

An **ordered merge** takes two arrays that are each sorted, `A` with `m` elements and `B`
with `n` elements, and produces one sorted array `C` with all `m + n` elements.

```
A = [3, 5, 5, 8, 12]           B = [1, 5, 6, 8, 9, 14]
C = [1, 3, 5, 5, 5, 6, 8, 8, 9, 12, 14]
```

When `A` and `B` hold the same value, the merge has to decide which copy goes first. If
the elements are plain numbers it makes no difference. If they are **records** sorted by a
key (events sorted by time, trips sorted by duration), it matters, because two
records with the same key are still different records.

A merge is **stable** when equal keys keep the order they had before the merge:

- equal keys that come from the same input keep their order within that input, and
- when an element of `A` and an element of `B` have the same key, the one from `A` comes
  first.

All the code in this folder uses that rule ("ties go to `A`"), which is also the rule of
`std::merge`. Stability is what lets you sort data in several steps without each step
undoing the work of the one before. Radix sort and merge sort in chapter 14 both rely on it.

## 2. The sequential merge

The sequential algorithm keeps one cursor in each input and repeatedly copies the smaller
front element to the output. It is linear: every element is read once and written once.

```cpp
__host__ __device__ inline void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }
  while (i < m) {
    C[k++] = A[i++];
  }
  while (j < n) {
    C[k++] = B[j++];
  }
}
```

The `<=` is the tie rule: when `A[i] == B[j]` the element of `A` is taken first. If you
write `<` instead, the merge still sorts but it is no longer stable.

This function is marked `__host__ __device__`. The kernels call it, and `main()` can also
call it on the CPU to test it.

## 3. Splitting the output: the co-rank function

To merge in parallel we need to divide the work. Dividing the **inputs** evenly does not
work: if thread 0 takes the first half of `A` and the first half of `B`, nobody knows where
in `C` its results belong, and its result may not even be a contiguous piece of `C`.

Dividing the **output** evenly does work. Give thread `t` the output positions
`C[k_t, k_{t+1})`. Every prefix of `C` is made of a prefix of `A` and a prefix of `B`:

```
the first k elements of C  =  merge( A[0, i),  B[0, j) )      with  i + j = k
```

So the piece `C[k_t, k_{t+1})` is exactly the merge of `A[i_t, i_{t+1})` and
`B[j_t, j_{t+1})`, and a thread that knows those four numbers can produce its piece with
the sequential merge, without talking to any other thread.

For an output position `k`, the value `i` is called the **co-rank** of `k` (and `j = k - i`
is its co-rank in `B`). Different threads use different `k`, so they can compute their
co-ranks independently and in parallel.

How do we recognise the right `i`? A split `(i, j)` with `i + j = k` is correct when no
element left out of the prefix should have been in it:

```
(1)  A[i-1] <= B[j]     the last A taken does not belong after the first B left out
(2)  B[j-1] <  A[i]     the last B taken really comes before the first A left out
```

The comparison in (2) is strict because of the tie rule: if `B[j-1] == A[i]`, the `A`
element must come first, so taking `B[j-1]` while leaving `A[i]` out would be wrong.
Conditions that refer to an index outside an array (for example `A[i-1]` when `i == 0`) are
treated as true. For each `k` exactly one `i` satisfies both, which is why every thread
gets a well-defined, non-overlapping piece.

## 4. Finding the co-rank with a binary search

If condition (1) fails, `i` is too large: we took an element of `A` that belongs later.
If condition (2) fails, `i` is too small. That is everything a binary search needs. The
search moves `i` and `j` together (keeping `i + j = k`) and halves the remaining range each
time:

```cpp
__host__ __device__ inline int co_rank(int k, const int* A, int m, const int* B, int n) {
  int i = imin(k, m);
  int j = k - i;
  int i_low = imax(0, k - n);
  int j_low = imax(0, k - m);
  bool searching = true;
  while (searching) {
    const bool i_too_large = i > 0 && j < n && A[i - 1] > B[j];
    const bool i_too_small = j > 0 && i < m && B[j - 1] >= A[i];
    if (i_too_large) {
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      i -= delta;
      j += delta;
    } else if (i_too_small) {
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      searching = false;
    }
  }
  return i;
}
```

The search starts by taking as many elements from `A` as possible, `i = min(k, m)`.
`i_too_large` is condition (1) failing: `A[i-1]` would land after `B[j]`, so `i` moves down
and `j` up. `i_too_small` is condition (2) failing: `A[i]` must come before `B[j-1]` (or tie
with it, and ties go to `A`), so `i` moves up and `j` down. When neither holds, both
conditions are true and `(i, j)` is the split.

`i_low` and `j_low` are the lower bounds that are already known to be safe. They start at
`max(0, k - n)` and `max(0, k - m)`: `i` can never be below `k - n`, because `B` has only `n`
elements to fill the rest of the prefix. Each step jumps halfway from the current guess to the
bound on the side it has to move towards, and raises the bound on the other side. The search
takes `O(log min(k, m + n - k))` steps.

### Worked example

`A = [3, 5, 5, 8, 12]` (`m = 5`), `B = [1, 5, 6, 8, 9, 14]` (`n = 6`), `k = 6`.
The first six outputs are `1 3 5 5 5 6`: three from `A` (`3 5 5`) and three from `B`
(`1 5 6`), so we expect `i = 3, j = 3`.

| step | i | j | i_low | j_low | test | result |
|------|---|---|-------|-------|------|--------|
| start | 5 | 1 | 0 | 1 | | |
| 1 | 5 | 1 | 0 | 1 | `A[4]=12 > B[1]=5`: i too large | delta = 3, j_low = 1, i = 2, j = 4 |
| 2 | 2 | 4 | 0 | 1 | `A[1]=5 > B[4]=9`? no. `B[3]=8 >= A[2]=5`: i too small | delta = 2, i_low = 2, i = 4, j = 2 |
| 3 | 4 | 2 | 2 | 1 | `A[3]=8 > B[2]=6`: i too large | delta = 1, j_low = 2, i = 3, j = 3 |
| 4 | 3 | 3 | 2 | 2 | `A[2]=5 > B[3]=8`? no. `B[2]=6 >= A[3]=8`? no | stop: **i = 3, j = 3** |

`merge.cu` checks the co-rank of every `k` from 0 to 11 for this example when it starts, and
`02_co_rank.cu` prints this search step by step.

## 5. The basic kernel

Each thread gets the same number of outputs, finds where its piece starts and ends with two
co-rank calls, and merges the piece sequentially:

```cpp
__global__ void merge_basic_kernel(const int* A, int m, const int* B, int n, int* C) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalThreads = gridDim.x * blockDim.x;
  const int elementsPerThread = (m + n + totalThreads - 1) / totalThreads;

  const int k_curr = imin(tid * elementsPerThread, m + n);
  const int k_next = imin(k_curr + elementsPerThread, m + n);
  const int i_curr = co_rank(k_curr, A, m, B, n);
  const int i_next = co_rank(k_next, A, m, B, n);
  const int j_curr = k_curr - i_curr;
  const int j_next = k_next - i_next;

  merge_sequential(A + i_curr, i_next - i_curr, B + j_curr, j_next - j_curr, C + k_curr);
}
```

`elementsPerThread` is the thread coarsening factor of section 8. (Thread `t`'s `k_next` is
thread `t+1`'s `k_curr`, so each split is computed twice. That is cheap compared with sharing
it through memory.)

It is short and correct, but it uses memory badly:

- **The co-rank search reads global memory at data-dependent addresses.** Each step of the
  binary search jumps somewhere else in `A` and `B`. Neighbouring threads search for nearby
  `k`, so their early steps touch the same addresses (the cache helps a little), but the
  accesses are not coalesced and each step depends on the previous load.
- **The sequential merge reads are not coalesced.** Thread `t` reads `A[i_t]`,
  `A[i_t + 1]`, ... while thread `t + 1` reads around `A[i_{t+1}]`, about `elementsPerThread`
  elements away. In one load instruction the 32 threads of a warp touch 32 different memory
  segments. Writes to `C` have the same problem: stride `elementsPerThread`.
- **Control divergence.** Whether a thread takes the `A` branch or the `B` branch depends on
  its data, so the threads of a warp disagree all the time.

```
         thread 0        thread 1        thread 2        thread 3
C:  [ 0 1 2 ... 63 | 64 65 ... 127 | 128 ... 191 | 192 ... 255 ]
         ^               ^               ^               ^
  one warp step: four addresses 64 elements apart -> four memory transactions
```

## 6. The tiled kernel

The fix is the same as for matrix multiplication: let a block bring the data it needs into
shared memory with coalesced loads, then let the threads do their irregular accesses in
shared memory.

The work is split in two levels:

1. **Block level.** Block `b` owns a large contiguous piece of `C` (16 K outputs in our
   timed run). Thread 0 computes its two co-ranks in global memory and puts them in shared
   memory; every thread then knows the block's ranges of `A`, `B` and `C`. That is two
   searches per block instead of two per thread.
2. **Tile level.** The block walks through its piece in steps of `TILE` outputs. In each
   step:
   - all threads load the **next `TILE` elements of the block's `A` range and the next
     `TILE` elements of its `B` range** into `A_S` and `B_S`. Consecutive threads read
     consecutive addresses, so the loads are coalesced;
   - barrier;
   - the `TILE` outputs of this step come from the two tiles only (at most `TILE` from
     each), so each thread takes `TILE / blockDim.x` of them, finds its sub-piece with a
     **second co-rank, run on the shared-memory tiles**, and merges it into `C`;
   - every thread computes how many elements of `A` the step used (one more co-rank, of
     `TILE`), advances the cursors `A_consumed` and `B_consumed`, and waits at a second barrier
     before the tiles are overwritten.

```
block's range of A:  [##########|..........|..........]   A_consumed ->
block's range of B:  [######|..........|..........]       B_consumed ->
                        \        /
                    A_S (TILE)  B_S (TILE)     loaded together, coalesced
                         \      /
             128 threads, 8 outputs each, co-rank inside shared memory
                            |
C:            [ step 0: TILE outputs | step 1 | step 2 | ... ]
```

The core of the loop in `merge.cu`:

```cpp
    const int A_S_length = imin(TILE, A_length - A_consumed);
    const int B_S_length = imin(TILE, B_length - B_consumed);
    for (int x = threadIdx.x; x < A_S_length; x += blockDim.x) {
      A_S[x] = A[A_curr + A_consumed + x];
    }
    for (int x = threadIdx.x; x < B_S_length; x += blockDim.x) {
      B_S[x] = B[B_curr + B_consumed + x];
    }
    __syncthreads();

    const int C_step = imin(TILE, C_length - C_completed);
    const int c_curr = imin(threadIdx.x * TILE_OUTPUTS_PER_THREAD, C_step);
    const int c_next = imin(c_curr + TILE_OUTPUTS_PER_THREAD, C_step);
    const int a_curr = co_rank(c_curr, A_S, A_S_length, B_S, B_S_length);
    const int a_next = co_rank(c_next, A_S, A_S_length, B_S, B_S_length);
    ...
    const int A_step = co_rank(C_step, A_S, A_S_length, B_S, B_S_length);
    A_consumed += A_step;
    B_consumed += C_step - A_step;
    C_completed += C_step;
    __syncthreads();
```

`TILE_OUTPUTS_PER_THREAD` is `TILE / TILED_THREADS_PER_BLOCK` (8), and a `static_assert`
checks that the division is exact, so every thread merges the same share of a tile. Every
thread computes `A_step` itself: all of them get the same value, so the cursors stay equal
across the block without any communication.

Why is it correct to search only inside the tiles? The next `C_step <= TILE` outputs can
take at most `TILE` elements from `A` and at most `TILE` from `B`. The tiles hold exactly
the next `TILE` of each (fewer only when an input has run out), so the answer is always
inside them.

The two barriers play the usual roles: the first one is read-after-write (no thread merges
before the tiles are complete), the second is write-after-read (no thread starts loading
the next tiles while another one is still merging from the current ones).

**The weakness: half of every load is wasted.** Each step loads up to `2 * TILE` elements
but uses only `TILE` of them. The unused ones (for example the part of `B_S` that was not
needed because `A`'s values were smaller) are thrown away and loaded **again** in the next
step. On average each element is read from global memory about twice.

## 7. Circular buffers

The circular-buffer version removes that waste. The idea: keep the unused part of each tile
where it is, and only load new elements into the slots that the last step consumed.

`merge_circular_kernel` gives every element a fixed slot: element `x` of the block's range of
`A` always lives at `A_S[x % TILE]`. The buffer holds the window `[A_consumed, A_loaded)`. Each
step tops the window up to `TILE` elements, which writes exactly the slots that were freed:

```
step s:    A_S = [ 8 9 10 11 | 4 5 6 7 ]      window = elements 4..11, A_consumed = 4
           the step consumes elements 4, 5, 6           -> A_consumed = 7
step s+1:  load elements 12, 13, 14 into slots 12%8=4, 13%8=5, 14%8=6
           A_S = [ 8 9 10 11 | 12 13 14 7 ]   window = elements 7..14
```

Because the window no longer starts at slot 0, the co-rank search and the sequential merge
must read the tiles through the modulo. In `merge.cu` they are separate functions,
`co_rank_circular` and `merge_sequential_circular`, that read
`A_S[circular_slot(A_S_start + i)]` instead of `A_S[i]`, where `A_S_start` is the first
element of the window (`A_consumed`). With `TILE` a power of two, `% TILE` is a cheap bit
mask: `circular_slot(x)` is `x & TILE_MASK`, with `TILE_MASK = TILE - 1`, and a
`static_assert` checks that `TILE` is a power of two.

What it buys and what it costs:

- every element of `A` and `B` is loaded from global memory **exactly once** (about half the
  global reads of the tiled kernel);
- the index arithmetic in the innermost loops is more expensive, and the code is harder to
  get right. Whether it pays off depends on how memory-bound the tiled kernel was on your GPU.

## 8. Thread coarsening

In merge, coarsening is not an optional add-on: it is built into the design. A thread that
produced a single output would pay two `O(log n)` searches to copy one element. Giving each
thread many outputs spreads the cost of the searches over all of them:

- **more outputs per thread:** fewer co-rank searches in total, longer sequential runs;
  but fewer threads, and once there are not enough warps to hide memory latency the GPU
  idles;
- **fewer outputs per thread:** more parallelism, but the searches dominate.

`merge.cu` runs the basic kernel with 8, 64 and 512 outputs per thread so you can see the
trade-off. In the tiled kernels the same choice appears twice: outputs per block (how
often the global co-rank is paid) and `TILE / blockDim.x` outputs per thread inside a tile.

## 9. Performance analysis

- **Work.** With `T` threads the basic kernel does `O(m + n)` work for the merging plus
  `O(T log(m + n))` for the searches: work-efficient when each thread has many outputs.
- **Memory.** Merge does almost no arithmetic: one comparison per output. It is
  memory-bound, so the quality of the memory accesses decides the speed. The column `GB/s`
  in the output of `merge.cu` counts one read and one write per element; compare it with
  the bandwidth of your GPU (about 300 GB/s for the T4 that `colab_2_full_programs.ipynb` runs on in Colab).
- **Basic kernel:** uncoalesced loads and stores and dependent loads in the search.
  Expect a fraction of the peak bandwidth.
- **Tiled kernel:** coalesced loads, irregular accesses moved to shared memory, but each
  element is loaded about twice; the stores to `C` are still per-thread runs (stride
  `TILE / blockDim.x`), a smaller stride than in the basic kernel. Shared memory bank
  conflicts appear because the threads' sub-pieces start at data-dependent places.
- **Circular kernel:** one load per element; more index arithmetic.
- **Load balance** is perfect by construction: every thread and every block gets the same
  number of outputs, whatever the data. That is the great advantage of splitting the
  output instead of the input.

**About the numbers in emulation.** On a laptop the "kernels" run on CPU threads that are
created for every launch and meet at real barriers; a barrier costs a fraction of a
millisecond. Those timings measure the emulation, not the algorithm (the tiled kernels,
with many barriers, look slowest there). Only compare timings measured on a GPU.

## 10. Pitfalls

- **Tie rule mismatch.** Exactly one of the two co-rank tests must be strict, and the
  strict one decides who wins a tie. With `>` and `>=` (our choice) ties go to `A`, as in
  `merge_sequential`. With `>` in both tests the search may stop at a split that takes an
  equal element of `B` before one of `A`: the keys are still sorted, but the merge is no
  longer stable, and plain `int` tests cannot see it. With `>=` in both tests no split
  satisfies the conditions when the keys are equal, and the search can loop forever
  (`A = [5]`, `B = [5]`, `k = 1` does). Test with many duplicates (the edge cases use values
  in `[0, 3]`) and with records, as `bikeshare_station_logs.cu` does.
- **Rounding the piece size.** `ceil((m + n) / T)` with integers computes the floor first.
  Write `(m + n + T - 1) / T`, and clamp `k` with `imin(..., m + n)` because the last
  threads can get positions past the end.
- **Searching outside the arrays.** The `i > 0`, `j < n`, `j > 0`, `i < m` guards are what
  keep the search inside `A` and `B`; `k` must be in `[0, m + n]`.
- **The second barrier.** Without the `__syncthreads()` at the end of the tile loop a fast
  thread starts loading the next tiles over data a slow thread is still merging from.
- **No early return.** A thread whose piece is empty must still reach every
  `__syncthreads()`. The kernels guard work with `if` and loop bounds, never with `return`.
- **Overflow.** `tid * elementsPerThread` is an `int`; for arrays of more than 2^31 elements use
  64-bit indices.
- **Do not trust one random test.** Empty inputs, one-element inputs, very unbalanced sizes
  (`m = 4097, n = 1`), and pieces that end in the middle of a tile are where merge kernels
  break. `merge.cu` runs all of them for all three kernels.

## 11. Ideas for your project

### The bike-sharing example in this folder

`bikeshare_station_logs.cu` builds the city log of a bike-sharing system from the logs of its
48 docking stations. Each station records every time a bike is undocked (a trip starts there)
or docked (a trip ends there). An `Event` holds `timeMs`, `kind` (`EventKind::Undock` or
`EventKind::Dock`), `stationId` and `bikeId`. A station that appends to its **own** list needs
no lock, and its list is sorted by `t` because its clock only moves forward. The city log
needs all events sorted by `t`: that is a merge.

The synthetic week (SplitMix64, fixed seed) follows each bike from trip to trip: it is undocked
at the station where it was parked, rides for 3 to 25 minutes, is docked at a random station
and stays there for up to 45 minutes. The station clocks count whole seconds
(`STATION_CLOCK_TICK_MS`), so different stations often log events in the same second, and a
bike can be docked and undocked again in the same second: plenty of ties for the stability
checks.

- the inputs are already sorted, so merging them (linear work) beats sorting them again
  (`n log n`);
- the 48 station lists are merged **pairwise in a tree**, `ceil(log2 48) = 6` rounds: round 1
  merges stations (0, 1), (2, 3), ...; round 2 merges those results pairwise again, and so on
  (48, 24, 12, 6, 3, 2, 1 lists). The lists sit one after the other in one array, and
  `listBegin[L]` is where list `L` starts. After a round, merged list `p` starts where list
  `2p` started (`listBeginAfterRound`). When the number of lists is odd, as in the round that
  starts with 3, the last pair has an empty right list (`list_pair` clamps its bounds), so the
  last list is simply copied;
- the merge is **stable** with ties going to the left input, and the left input is always the
  group of lower station ids, so events in the same second come out in station-id order, and
  two events of one station in the same second keep the station's own sequence. The result
  equals `std::stable_sort` of the concatenation, which the program checks;
- in the last rounds there are only a few long pairs. Instead of one block per pair (which
  would leave the GPU almost empty in the last round), every thread takes a fixed slice of the
  whole output, finds its pair with a small binary search over the list starts
  (`last_pair_starting_at_or_before`), and finds its place inside the pair with
  `co_rank_events`. A slice that crosses from one pair into the next is merged one piece per
  pair. Every round keeps all threads busy;
- the program also checks a property that only holds if events from different stations are
  interleaved correctly: for every bike, undock and dock alternate, `t` never decreases, and
  each undock happens at the station where the previous dock left the bike.

Output of the emulation:

```
Merging per-station dock logs on CPU emulation (threads)
48 stations, 240 bikes, 72000 events in all

first lines of the city log:
  day 0 00:00:05  station s10  undock  bike b0173
  day 0 00:00:06  station s13  undock  bike b0157
  ...
a group of events in the same second (lower station ids first):
  day 0 00:44:16  station s08  dock    bike b0229
  day 0 00:44:16  station s14  dock    bike b0114
  day 0 00:44:16  station s39  undock  bike b0075
...
rounds: 6 (ceil(log2 48))

same as stable sort:                       PASS
sorted by t:                               PASS
equal t, lower station first:              PASS
per bike, undock/dock alternate in place:  PASS
ALL PASS
```

### Where this pattern may appear in your project

- If every worker thread keeps its own log, it needs no lock and the log is already sorted by
  time. Building one global timeline from those logs is then a merge, not a sort.
- With many sorted lists, a tree of pairwise merges needs about log2 of the number of lists
  rounds. How would you keep all threads busy in the last rounds, when only one or two long
  pairs are left?
- When two records have the same timestamp, which one should come first, and does your merge
  keep that rule every time (stability)?
- Once there is one timeline, can you check a property per entity along it, for example that
  each order or each courier only moves forward through its states?
- Adding the sorted results of a new run to a sorted archive of earlier runs is also a merge.

## 12. Files and how to run

| File | What it shows |
|------|---------------|
| `01_*.cu` to `05_*.cu` | The step files of the [learning path](#learning-path), from the sequential merge to the circular buffer |
| `merge.cu` | `co_rank`, the sequential merge, the basic, tiled and circular-buffer kernels; edge-case tests; a timing table against `std::merge` |
| `bikeshare_station_logs.cu` | Stable tree merge of per-station dock/undock logs into one city log sorted by `t` |

The same file runs on a GPU and on any laptop (see `../common/cuda_compat.h`).

> **Easiest way on Colab:** this folder has two notebooks. Upload [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) for the step files of the learning path, or [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) for `merge.cu` and the bike-sharing example, to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

```sh
# laptop (no GPU): the CUDA parts are emulated with std::thread
cd ec2/11_merge
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_merge_sequential.cu -o step && ./step
for f in 0*.cu; do clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common $f -o step && ./step || break; done
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common merge.cu -o merge && ./merge
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_station_logs.cu -o station_logs && ./station_logs
# g++ works the same way; add -fsanitize=thread -g to check for data races

# GPU: the two notebooks already contain these cells; by hand, in a Colab cell (T4 GPU):
!nvcc -O2 -I../common 01_merge_sequential.cu -o step && ./step
!nvcc -O2 -I../common merge.cu -o merge && ./merge
!nvcc -O2 -I../common bikeshare_station_logs.cu -o station_logs && ./station_logs
```

Problem sizes are chosen with `BY_BACKEND(gpuSize, cpuSize)`: millions of elements on a GPU,
fewer in the emulation so each program finishes in a few seconds. `bikeshare_station_logs.cu`
takes `-DCPU_BIKES=...` and `-DCPU_TRIPS_PER_BIKE=...` to change the emulation size.

## 13. Exercises

1. **Co-rank by hand.** With `A = [2, 4, 4, 7]` and `B = [1, 4, 5, 7, 9]`, give the co-rank
   `i` of every `k` from 0 to 9, then trace the binary search for `k = 5`.
2. **Breaking stability.** In `co_rank` change `B[j - 1] >= A[i]` into `B[j - 1] > A[i]`
   and keep `merge_sequential` as it is. Show that the output of the basic kernel is still
   sorted, but find inputs (think of records `(key, origin)`) for which it is no longer
   stable. What happens if you make both tests non-strict instead?
3. **Cost of the searches.** The basic kernel runs with `T` threads on `m + n = 2^24`
   outputs. How many co-rank steps does it make in total (roughly) for `T = 2^21` and for
   `T = 2^15`? Compare with the `2^24` steps of the merge itself.
4. **Tile traffic.** In the tiled kernel each step loads up to `2 * TILE` elements and
   consumes `TILE`. What is the total number of elements loaded from global memory for
   `m + n` outputs, ignoring the last step? What is it for the circular-buffer kernel?
5. **The second barrier.** Describe an interleaving of two threads of the tiled kernel that
   produces a wrong result if the `__syncthreads()` at the end of the loop is removed.
6. **Unequal tiles.** Suppose `A_S` had `TILE` elements but `B_S` only `TILE / 2`. Can a
   step still produce `TILE` outputs? What is the largest safe number of outputs per step?
7. **K-way merge.** You must merge 64 sorted lists of different lengths. Compare (a) six
   rounds of pairwise merges and (b) one pass in which each thread merges its slice of all
   64 lists directly. What does each thread need to know in (b), and what does it cost?

### Short solutions

1. Merged: `1(B) 2(A) 4(A) 4(A) 4(B) 5(B) 7(A) 7(B) 9(B)`. Co-ranks for `k = 0..9`:
   `0 0 1 2 3 3 3 4 4 4`. For `k = 5`: start `i = 4, j = 1`; `A[3]=7 > B[1]=4`, so i is too
   large: `delta = 2` (with `i_low = 0`), `i = 2, j = 3`; `B[2]=5 >= A[2]=4`, so i is too
   small: `delta = 1` (with `j_low = 1`), `i = 3, j = 2`; `A[2]=4 > B[2]=5`? no;
   `B[1]=4 >= A[3]=7`? no: `i = 3`.
2. The splits are still shared by neighbouring threads (thread `t`'s `k_next` is thread
   `t+1`'s `k_curr`), so every element is still output exactly once, and every accepted split
   still has no larger key in the prefix than outside it: the keys come out sorted. But the
   search may now accept a split that takes an equal `B` element and leaves an `A` one out.
   With `A = [5, 5, 5, 9]`, `B = [5, 5, 5, 5]` and `k = 4` the original search returns
   `i = 3` and the modified one `i = 2`: a `5` of `B` lands before the third `5` of `A`.
   With both tests non-strict, ties are "wrong" in both directions, no split is accepted,
   and for `A = [5]`, `B = [5]`, `k = 1` the search jumps between `i = 1` and `i = 0` forever.
3. Each thread makes two searches of about `log2(2^24) = 24` steps: `48 T` steps.
   `T = 2^21`: about `10^8` steps, six times the merge itself. `T = 2^15`: about
   `1.6 * 10^6`, a tenth of the merge. Coarsening matters.
4. Tiled: about `2 * (m + n)` loads (two tiles of `TILE` per `TILE` outputs) plus the
   `m + n` stores. Circular: `m + n` loads, each element once.
5. Thread 0 finishes its merge, loops, and writes new values into `A_S[0..]`; thread 1 is
   still in `merge_sequential` reading the old `A_S[3..10]`, and copies some of the new
   values into its part of `C`.
6. No. All of the next `TILE` outputs could come from `B`, so with only
   `TILE / 2` elements of `B` in shared memory a step can safely produce at most
   `TILE / 2` outputs (the smaller of the two tile sizes).
7. (a) costs 6 reads and writes of all the data, each round simple. (b) reads and writes
   once, but each thread must find, for its slice, a split point in each of the 64 lists:
   a 64-way co-rank (for example a binary search on the value plus 64 counts), and its
   sequential part is a 64-way merge (a small heap). Fewer passes, much more complex threads.
