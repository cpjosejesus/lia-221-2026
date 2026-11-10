# Parallel scan (prefix sum) on the GPU: chapter 11

In session 09 you parallelized a prefix sum on the CPU with `std::async` and three phases
(`09_task_parallelism/example_54.cpp`, `exercise_9_2.cpp`). This folder does the same thing
on a GPU. There are thousands of threads instead of eight, so the question changes. It is no
longer "how do I split the array into a few chunks" but "how do 256 threads that share a small
scratchpad scan their piece in a few steps, and how do thousands of such pieces agree on their
offsets".

Scan is worth the effort because it is the standard way to answer, in parallel, the question
"where does my result go?". Filtering (`../10_filter`), radix sort, counting sort, building
adjacency lists and allocating output space all reduce to a scan.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_sequential_scan.cu`](01_sequential_scan.cu) | The sequential inclusive and exclusive scan of 8 values, run by a single GPU thread: n − 1 additions and nothing in parallel yet. | The exclusive row is the inclusive row shifted one place to the right with 0 in front: inclusive[2] = 11 is 3+1+7, and exclusive[3] is the same 11 because element 3 itself is left out. The last inclusive value, 25, is the total of all 8 values, reached after 7 additions in a row. |
| 2 | [`02_kogge_stone.cu`](02_kogge_stone.cu) | Kogge-Stone in one block, one thread per element, two barriers per step (read, wait, write). The array is printed after each stride; the barrier before each print is only there for the printing. | After stride 1 every position holds the sum of up to 2 elements (position 2 = 1+7 = 8), after stride 2 up to 4 (position 7 = 4+1+6+3 = 14), and after stride 4 the whole prefix. Positions below the stride no longer change: 3 4 11 11 is already final after stride 2. Try: change the loop to `stride < 4`; the run stops after stride 2, positions 4 to 7 keep the partial sums 12 12 11 14 and the check fails. |
| 3 | [`03_kogge_stone_double_buffer.cu`](03_kogge_stone_double_buffer.cu) | Two shared buffers that swap roles, so one barrier per step is enough. Each stride prints the buffer it reads, which nobody writes in that step. | Each line shows the buffer that a stride reads, so the rows are one step behind step 2: the stride 2 row `3 4 8 7 4 5 7 9` is what step 2 printed after stride 1. The reads alternate between buf[0] and buf[1]; because each step writes into the other buffer, one barrier per step is enough and the result is the same. |
| 4 | [`04_brent_kung.cu`](04_brent_kung.cu) | Brent-Kung with two elements per thread: the up-sweep builds the reduction tree, the down-sweep fills in the positions it skipped. Every step of both sweeps is printed. | The up-sweep touches only some positions: after stride 1 positions 1, 3, 5 and 7 hold pair sums (4, 7, 5, 9), and after stride 4 position 7 holds the total 25 while positions 2, 4 and 6 still hold their inputs. The down-sweep fills the gaps: stride 2 puts 5+11 = 16 into position 5, and stride 1 completes positions 2, 4 and 6. Try: remove the down-sweep loop; the result is `3 4 7 11 4 5 6 25`, right only at positions 0, 1, 3 and 7. |
| 5 | [`05_coarsened_scan.cu`](05_coarsened_scan.cu) | Thread coarsening, 4 threads with 4 elements each: a coalesced load, phase 1 scans each thread's run, phase 2 runs Kogge-Stone over the run totals, phase 3 adds the totals to the left. | Phase 1 restarts the sum every 4 elements: the second run reads 4 5 11 14, the scan of 4 1 6 3 on its own. Phase 2 scans the 4 run totals 11 14 15 13 into 11 25 40 53, and phase 3 adds the total to the left to each run, so the second run becomes 4+11 = 15 up to 14+11 = 25. |
| 6 | [`06_hierarchical_scan.cu`](06_hierarchical_scan.cu) | Any length: scan 4-element blocks and write each block's sum, scan the block sums with the same routine (recursively), then add them back. Every level is printed. | Level 0 cuts the 19 values into 5 blocks of 4; the last block holds only 4 6 1, so its sum is 11. The 5 block sums do not fit in one block, so level 1 scans them as two blocks (11 25 40 53 and 11) and level 2 scans its two sums 53 11 into 53 64. On the way back each block adds the scanned sum of the blocks to its left: the fifth block adds 53 and ends at 64, the total. Try: keep only the first 16 values of `x`; the 4 block sums fit in one block and level 1 is the last level. |
| 7 | [`07_single_pass_domino.cu`](07_single_pass_domino.cu) | One kernel: each block takes a ticket, scans its section, waits for the total of the sections before it, publishes its own total, and stores its elements once. | Each line is one block: it receives the total of all earlier sections and publishes its own running total, for example block 2 gets 25 = 11+14 and publishes 25+15 = 40. The published totals 11 25 40 53 64 are the numbers that step 6 needed extra levels and kernels for. On a GPU blocks start in any order, so the ticket and `blockIdx.x` on a line can differ; the section a block scans is the one given by its ticket. |

After the steps come the complete programs: `scan.cu` runs every block scan above on 16 M elements with checks and timings, and the bike-sharing examples (`bikeshare_group_by_station.cu`, `bikeshare_rentals_in_progress.cu`) use the scan in an application.

---

## 1. The pattern

Take an array `x[0..n)` and an associative operator `⊕` with an identity `e`. For `+` the
identity is 0, for `max` it is −∞, for `*` it is 1.

| kind | definition | example for `x = 3 1 7 0 4 1 6 3` with `+` |
|------|------------|---------------------------------------------|
| **inclusive** scan | `y[i] = x[0] ⊕ x[1] ⊕ … ⊕ x[i]` | `3 4 11 11 15 16 22 25` |
| **exclusive** scan | `y[i] = x[0] ⊕ … ⊕ x[i-1]`, `y[0] = e` | `0 3 4 11 11 15 16 22` |

Each one is easy to get from the other. For an exclusive scan, shift the inclusive result one
place to the right and put `e` in front. For an inclusive scan, apply `⊕ x[i]` to each
exclusive value. The two answer different questions:

- inclusive: "how much has accumulated **up to and including** me?" Running totals, the
  number of bikes out on a rental after each dock or undock event.
- exclusive: "how much comes **before** me?" That is the starting position of my data in a
  packed output: the offset of bucket `r`, the output slot of a kept element.

The operator must be **associative**, because the parallel algorithms group the operands
differently from the sequential loop. This is the same condition session 09 placed on reduce
(`example_56`). Integer `+` gives bit-identical results. Floating-point `+` gives slightly
different last digits from one algorithm to the next, which is why every program here scans
`int`s and can compare with `==`.

## 2. The sequential baseline and work efficiency

```cpp
int run = 0;
for (size_t i = 0; i < x.size(); ++i) {
  run += x[i];
  y[i] = run;
}
```

This loop does `n − 1` additions and reads and writes each element once. Nothing can do fewer
operations. A parallel algorithm is **work-efficient** if its total number of operations is
still O(n), i.e. a constant multiple of the sequential loop. That matters on a GPU because the
machine is busy all the time: every extra addition takes an execution slot that some other
useful addition could have used. An algorithm with fewer steps but much more work is only
faster while there are idle lanes to absorb the extra work.

Two numbers describe each algorithm:

- **steps** (depth): how many rounds that depend on each other are needed. Each round costs a
  barrier.
- **work**: the total number of additions over all threads.

## 3. Kogge-Stone: few steps, a lot of work

Each element starts with its own value. In step `k`, element `i` adds the value that sits
`stride = 2^k` places to its left, if there is one. After the step with stride `s`, element
`i` holds the sum of the `2s` elements ending at `i` (or of everything from index 0, when that
window is cut off at the left edge).

```
x          3   1   7   0   4   1   6   3
stride 1   3   4   8   7   4   5   7   9      each i >= 1 adds i-1
stride 2   3   4  11  11  12  12  11  14      each i >= 2 adds i-2
stride 4   3   4  11  11  15  16  22  25      each i >= 4 adds i-4   -> inclusive scan
```

- steps: `log2 N`. For a 256-element section that is 8 steps.
- work: in the step with stride `s`, `N − s` threads add, so the total is
  `N·log2 N − (N − 1)`. For N = 256 that is 1793 additions, against 255 for the loop: **seven
  times the work**.

### The write-after-read race and its two fixes

In place, thread `i` writes `XY[i]` while thread `i + stride` may still be about to read the
old `XY[i]`. That is a write-after-read race. If the write lands first, the reader adds a value
that already contains this step's contribution. The kernel in `scan.cu` fixes it with a second
barrier inside the step: everybody reads into a register, everybody waits, then everybody
writes. The first barrier of each step makes the previous step's writes visible; the second
one guarantees that every thread has *read* before any thread *writes*.

```cpp
for (int stride = 1; stride < KS_SECTION; stride *= 2) {
  __syncthreads();
  int temp = XY[t];
  if (t >= stride) {
    temp += XY[t - stride];
  }
  __syncthreads();
  XY[t] = temp;
}
```

The **double-buffered** version uses two shared arrays. A step reads only from `src` and
writes only to `dst`, and then the two swap. No array is read and written in the same step, so
one barrier per step is enough. The price is twice the shared memory, plus every thread copies
its value to `dst` even when it adds nothing:

```cpp
for (int stride = 1; stride < KS_SECTION; stride *= 2) {
  __syncthreads();
  int v = buf[src][t];
  if (t >= stride) {
    v += buf[src][t - stride];
  }
  const int dst = 1 - src;
  buf[dst][t] = v;
  src = dst;
}
```

Kogge-Stone has two more drawbacks. It needs one thread per element, and the number of
threads that do useful work shrinks with every step, but only slowly (`N − s`). So the warps
stay occupied while each one does a little less useful work.

## 4. Brent-Kung: more steps, about 2N work

Brent-Kung does the work in two sweeps over a balanced tree.

**Up-sweep (reduction tree).** With stride 1, 2, 4, … each position `2·s·m − 1` adds the
position `s` places to its left. Afterwards the last element of every aligned group of `2s`
elements holds that group's sum, and the last element of the section holds the total.

**Down-sweep (distribution tree).** With strides N/4, …, 2, 1, each position `index` passes its
partial sum to `index + stride`. That fills in all the positions the up-sweep skipped.

```
x                3   1   7   0   4   1   6   3
up   stride 1    3   4   7   7   4   5   6   9      positions 1,3,5,7
up   stride 2    3   4   7  11   4   5   6  14      positions 3,7
up   stride 4    3   4   7  11   4   5   6  25      position 7  (the total)
down stride 2    3   4   7  11   4  16   6  25      3 -> 5
down stride 1    3   4  11  11  15  16  22  25      1 -> 2, 3 -> 4, 5 -> 6
```

The kernel gives each thread two elements, so 256 threads scan a 512-element section. The index
formula `(t + 1) * 2 * stride - 1` keeps the active threads **contiguous** (threads 0, 1, 2, …),
not spread out. As the tree narrows, whole warps go idle and stop taking execution slots, and
there is less divergence inside a warp. The first loop is the up-sweep, the second the
down-sweep:

```cpp
for (int stride = 1; stride <= BK_THREADS; stride *= 2) {
  __syncthreads();
  const int index = (t + 1) * 2 * stride - 1;
  if (index < BK_SECTION) {
    XY[index] += XY[index - stride];
  }
}

for (int stride = BK_SECTION / 4; stride > 0; stride /= 2) {
  __syncthreads();
  const int index = (t + 1) * 2 * stride - 1;
  if (index + stride < BK_SECTION) {
    XY[index + stride] += XY[index];
  }
}
```

Before these loops each thread loads its two elements in two coalesced passes,
`XY[t]` and then `XY[t + BK_THREADS]`, so neighbouring threads always read neighbouring
addresses.

- steps: `2·log2 N − 1` (17 for N = 512).
- work: `(N − 1)` in the up-sweep plus `(N − 1 − log2 N)` in the down-sweep, so
  `2N − 2 − log2 N`. That is about **twice** the loop, independent of N. Brent-Kung is
  work-efficient.

## 5. The numbers side by side

`scan.cu` prints this table from the formulas above:

| N (section) | loop adds / steps | Kogge-Stone adds / steps | Brent-Kung adds / steps |
|------:|------------:|-------------:|------------:|
| 16    | 15 / 15     | 49 / 4       | 26 / 7      |
| 256   | 255 / 255   | 1793 / 8     | 502 / 15    |
| 512   | 511 / 511   | 4097 / 9     | 1013 / 17   |
| 2048  | 2047 / 2047 | 20481 / 11   | 4083 / 21   |

Read the table this way:

- With **plenty of idle hardware** (a small array, or a few blocks on a large GPU), steps
  dominate and Kogge-Stone wins, because its extra additions run on lanes that would be idle
  anyway.
- With **saturated hardware** (millions of elements and every SM full), the extra work turns
  directly into time. Kogge-Stone's 10× work for N = 2048 is a real loss there, and the
  work-efficient algorithm wins.
- Each step costs a `__syncthreads()`. Brent-Kung has about twice as many steps, so for one
  small section its latency is higher.

## 6. Thread coarsening: the three-phase scan inside one block

Both tree algorithms make a thread do one addition per barrier. Coarsening gives each thread a
**contiguous run of C elements** (C = 8 in `scan.cu`, so a 256-thread block scans 2048
elements). This is the session-09 algorithm again, with threads in place of tasks:

```
load    2048 elements into shared memory, coalesced: thread t reads t, t+256, t+512, ...
phase 1 thread t scans XY[8t .. 8t+7] sequentially, no barriers          (7 adds per thread)
phase 2 Kogge-Stone over the 256 subsection totals                         (8 steps)
phase 3 thread t adds the total of everything left of its subsection      (8 adds per thread)
store   coalesced, the same way as the load
```

The load does not have thread `t` read its own elements `8t..8t+7` straight from global memory,
because the 32 threads of a warp would then touch addresses 8 ints apart in each instruction.
The load goes through shared memory in "corner-turning" order instead. Neighbouring threads read
neighbouring addresses, and the reordering to "each thread owns a contiguous run" happens in
shared memory, where it is cheap.

For one 2048-element section, the work is 1792 (phase 1) + 1793 (phase 2 on 256 values) + 2040
(phase 3) = 5625 additions, about 2.7× the loop. There are about 23 dependent steps and 19
barriers, 16 of them in the two-barrier Kogge-Stone of phase 2 (double buffering would halve
those). Compare a plain Kogge-Stone over 2048 elements: it would need 2048 threads (the
maximum block is 1024) and 20481 additions. Coarsening also cuts the number of blocks by 8, and
therefore the number of block sums the next level has to scan. That is why it is the fastest
block scan here.

## 7. Arbitrary length: the hierarchical scan

A block can only cooperate through its own shared memory, and it scans one section. For n
elements, `scan.cu` uses three steps, exactly the three phases of session 09 with blocks as
chunks:

```
 X:   [ section 0 ][ section 1 ][ section 2 ] ...            n elements
          | kernel 1: scan every section, write its total S[b]
 Y:   [ local scan][ local scan][ local scan] ...    S: [s0 s1 s2 ...]   n / section totals
          | scan S (the same routine, recursively, until one block is enough)
                                                       S': [s0, s0+s1, s0+s1+s2, ...]
          | kernel 3: block b adds S'[b-1] to all of its elements
 Y:   [ final     ][ final     ][ final     ] ...
```

```cpp
void scanHierarchical(BlockScanKernel blockScan,
                      int section,
                      int threads,
                      const int* d_in,
                      int* d_out,
                      int n,
                      int* scratch) {
  const int blocks = blocksFor(n, section);
  int* blockSums = scratch;
  launch(blockScan, dim3(blocks), dim3(threads), d_in, d_out, blockSums, n);

  if (blocks > 1) {
    int* scannedBlockSums = scratch + blocks;
    int* scratchForNextLevel = scratch + 2 * blocks;
    scanHierarchical(
        blockScan, section, threads, blockSums, scannedBlockSums, blocks, scratchForNextLevel);
    launch(
        add_block_offsets, dim3(blocks), dim3(OFFSET_THREADS), d_out, scannedBlockSums, n, section);
  }
}
```

The scratch buffer holds, level after level, the block sums and their scan.
`scratchIntsForHierarchicalScan` computes its size with the same recursion.

In session 09 the middle phase was a sequential loop over about 8 chunk totals. Here there are
thousands of block sums. For 16 M elements and 2048-element sections there are 7813, then 4,
then 1. So the middle phase is itself a hierarchical scan, and the depth grows by one level for
every factor of 2048. The kernel boundaries do the job of a grid-wide barrier: kernel 3 cannot
start before every block of kernel 1 has written its sum.

**Cost.** Kernel 1 reads n elements and writes n elements. Kernel 3 reads and writes n again.
That is about 4n global memory accesses, where 2n is the minimum. Scan is memory-bound (one
addition per element loaded), so the second pass costs roughly as much as the first.

The name "segmented scan" is also used for something else: scanning many independent segments
in one array, where the sum restarts at every segment head flag. That is done by scanning
(flag, value) pairs with the operator in exercise 6. It is a different algorithm.

## 8. Single-pass scan: the domino

The second pass exists only because block b does not know the total of blocks 0..b−1 while it
is running. A **single-pass** scan passes that total along a chain, inside one kernel:

```
 block 0: scan section, publish P1 = s0          -> flag[1] = 1
 block 1: scan section, wait flag[1], read P1, publish P2 = P1 + s1   -> flag[2] = 1
 block 2: scan section, wait flag[2], read P2, publish P3 = P2 + s2   -> flag[3] = 1
 ...      every block then adds its P to its own elements and stores them once
```

The local scans of all blocks still run in parallel. Only the hand-off (one thread, one number
per block) is serial. In `scan.cu` that part reads:

```cpp
if (threadIdx.x == 0) {
  const int before = wait_for_sum_of_previous_sections(flags, prefix, bid);
  publish_sum_of_sections_up_to(bid, before + part[CO_THREADS - 1], flags, prefix);
  before_s = before;
}
```

with the two halves of the hand-off written as small device functions:

```cpp
__device__ int wait_for_sum_of_previous_sections(int* flags, int* prefix, int bid) {
  if (bid == 0) {
    return 0;
  }
  wait_until_ready(&flags[bid]);
  return atomic_read(&prefix[bid]);
}

__device__ void publish_sum_of_sections_up_to(int bid, int sum, int* flags, int* prefix) {
  atomicExch(&prefix[bid + 1], sum);
  __threadfence();
  atomicExch(&flags[bid + 1], FLAG_READY);
}
```

`atomic_read(p)` is `atomicAdd(p, 0)`, and `wait_until_ready` spins on `atomic_read` until the
flag equals `FLAG_READY`. Three details make it correct:

1. **Reads and writes of the chain are atomics.** A plain load could be served from a stale L1
   line, so the spinning thread might never see the flag change; a plain store could become
   visible after the flag.
2. **`__threadfence()` between the value and the flag.** Without it, another SM could see
   `flag = FLAG_READY` before it sees the new `prefix`. The fence gives the flag store release
   semantics: whoever sees the flag also sees the value written before it.
3. **A dynamic block index.** The CUDA model gives no guarantee about the order in which blocks
   start, and does not promise that all of them are resident at once. Suppose block 500 is
   scheduled first and spins waiting for block 499. If all SMs are full of waiting blocks like
   it, block 499 never gets an SM, and the GPU hangs. So the section a block works on is not
   `blockIdx.x` but a **ticket**:

   ```cpp
   if (threadIdx.x == 0) {
     bid_s = next_ticket(ticket);
   }
   __syncthreads();
   const int bid = bid_s;
   ```

   `next_ticket(ticket)` is `atomicAdd(ticket, 1)`: the dynamic block index.

   A block that holds ticket b knows that tickets 0..b−1 went to blocks that have **already
   started**. The block it waits for is running, so it will publish.

*The CPU emulation runs blocks one at a time in index order.* So the version with `blockIdx.x`
would happen to work on a laptop. That is exactly why the ticket matters: code that only works
because of a particular scheduler order is wrong.

**Performance.** One pass: 2n memory accesses instead of 4n. But the chain is serial, and each
link costs a round trip through L2 (on the order of a microsecond). With 7813 sections that is
milliseconds of pure latency, which can cost more than the pass it saved. Production libraries
(CUB's `DeviceScan`) use **decoupled look-back**. Each block first publishes its *local* total
with an "aggregate available" flag. A waiting block walks backwards and adds up the aggregates
of its predecessors until it reaches one that has published a full prefix, so it never waits
for the whole chain. Inside a warp those libraries also replace the shared-memory steps with
register shuffles (`__shfl_up_sync`, 5 steps for 32 values). The code here keeps the book's
shared-memory versions; the emulation now supports the warp shuffles (see `../08_reduction`), so a
warp-shuffle scan is a good exercise.

## 9. Performance analysis

- **Memory bound.** One addition per 4-byte element, so the arithmetic intensity is 0.25
  ops/byte. A GPU scan is limited by DRAM bandwidth, and the fair comparison is "time ÷ time to
  copy the array once".
- **Kogge-Stone** on 256-element sections does 7× the additions and 16 barriers per block, and
  it creates 8× as many block sums as the coarsened version. It is the slowest here once n is
  large.
- **Double buffering** halves the barriers of Kogge-Stone (8 instead of 16 per block) and costs
  shared memory. Fewer barriers usually win.
- **Brent-Kung** halves the number of blocks (two elements per thread) and does about 2N work.
  Its 18 barriers per block keep it from being much faster than double-buffered Kogge-Stone.
- **Coarsened three-phase**: most of the work happens in phase 1, which has no barriers, and
  there are 8× fewer blocks. Expect the best block-level time.
- **Single pass**: saves a full pass over memory, and loses it again to the serial chain when
  sections are small. Larger sections (more coarsening) mean fewer links.
- **Emulation timings** measure the cost of `std::thread` barriers, not GPU behaviour. On the
  laptop they only show that every variant runs. On Colab (`colab_2_full_programs.ipynb`),
  write down your own table (exercise 5).

## 10. Pitfalls

- **One barrier in in-place Kogge-Stone.** A write-after-read race, and it shows up only now
  and then, depending on warp timing. Use two barriers or double buffering.
- **Forgetting padding.** The last section is partial. Load the identity (0) beyond `n`, and
  never store beyond `n`.
- **An early `return` before `__syncthreads()`.** Threads past the end of the array would skip
  the barrier, and the rest of the block would wait forever (undefined behaviour). Guard with
  `if`, as every kernel here does.
- **The block sum read by the wrong thread.** In Brent-Kung the total ends up in
  `XY[BK_SECTION - 1]`, written by another thread. It may only be read after the final barrier.
  In Kogge-Stone the last thread wrote `XY[KS_SECTION - 1]` itself, so it may store the block
  sum without another barrier.
- **Spin-waiting on `blockIdx.x` order** in a single-pass scan: a possible deadlock (section 8).
- **Overflow.** A prefix sum grows with n. For 16 M values of up to 9 the total is
  1.4 × 10⁸, safe in `int`. Byte counts or times in ms can overflow and need `long long`.
- **Floating-point scans differ** between algorithms in the last bits. Compare with a
  tolerance, or scan integers.
- **Bank conflicts** in phase 1 of the coarsened scan: thread t reads `XY[8t + k]`, so the 32
  threads of a warp hit only 4 different banks (8-way conflict). Padding the shared array, or
  using a transposed layout, removes the conflicts (exercise 8).

## 11. Ideas for your project

### The bike-sharing examples in this folder

The two programs below apply the scan to a city bike-sharing system: 48 docking stations on a
6 km × 6 km map, bikes that are docked, out on a ride or in repair, and a log of trips. The data
is synthetic and generated deterministically with a small SplitMix64 generator inside each
file, so every run prints the same numbers.

**Group trips by start station (`bikeshare_group_by_station.cu`).** The trip log lists trips by
start time. To study one station at a time (how its departures spread over the day, which bikes
leave from it), you want each station's trips as one contiguous slice. The program counts the
trips of every start station (a histogram), **exclusive-scans** the counts into start offsets,
and scatters every trip index into its slice. The result is a compressed-sparse-row layout:
station s owns `grouped[start[s] .. start[s+1])`. Version A takes slots with one atomic cursor
per station. It is fast, but the sequence inside a slice depends on which thread won the
atomic. Version B is stable. It counts per (station, block), scans that matrix in
station-major order with the hierarchical Brent-Kung scan, and ranks each trip among the
earlier trips of the same station in its block, so every slice stays sorted by start time.
That is one pass of a radix sort (chapter 13). The synthetic day has 4 M trips on the GPU and
60 k in the emulation, a few busy stations and a long tail of quiet ones, and start times that
peak around 08:00 and 18:00. In version A the 49 start offsets are scanned by a single
Brent-Kung block, which the program checks with a `static_assert` against `MAX_STATIONS`. The
output is a timing table with a check per version (A must hold the same trips in every slice
and reports how many slices happen to be sorted; B must be identical to the CPU counting sort),
followed by the five busiest start stations with their trip counts, slice offsets and first
trips.

**Bikes out on a rental over the day (`bikeshare_rentals_in_progress.cu`).** Every trip
produces two station events: `Undock` (+1) and `Dock` (−1). Sorted by time, the **inclusive**
scan of that sequence is the number of bikes out on a rental after each event. The stream is
long (8 M events on the GPU), so the program uses the coarsened three-phase scan, which creates
the fewest block sums. A privatized `atomicMax` reduction then finds the peak, and a second
pass with `atomicMin` finds the first event that reached it, which gives the time of day when
the most bikes were away from their docks. The program also checks that the curve ends at 0:
every bike that left a dock came back to one. The synthetic day has the two rush hours, about
3 % faulty undocks that are docked again within 90 s, and some long leisure rides. When two
events share a timestamp, `Undock` comes first, so a bike docked again in the same millisecond
never counts −1 before its +1. The output is the timing table with the scan, peak and
"ends at 0" checks, a line with the peak and its clock time, and an hourly bar chart of bikes
out.

### Where this pattern may appear in your project

- If you need all the items of one group next to each other (all the events of one worker, all
  the requests handled at one place), counting per group and scanning the counts tells every
  item where its group's slice begins. Does the consumer care about the original sequence
  inside a slice? Then you need the stable variant.
- A quantity that goes up and down over time (things open, waiting or in progress) is a running
  sum of +1 and −1 events, and its maximum is your peak load. What rule for equal timestamps
  keeps that sum from going negative?
- When many workers each produce output of a different size (formatted log lines, records), an
  exclusive scan of the sizes gives each one its offset in a shared buffer, with no lock.
- A graph such as a street network can be stored without pointers: how many edges leave each
  node, and where do the edges of node v start?
- A running total per entity inside one long array (per worker, per place) is a segmented scan;
  exercise 6 shows the operator.

## 12. Files and how to run

| file | what it shows |
|------|---------------|
| `01_*.cu` … `07_*.cu` | the learning path above, one idea per step on 8 to 19 values, each printing its input, its intermediate arrays and PASS/FAIL |
| `scan.cu` | every block scan (Kogge-Stone, double-buffered, Brent-Kung, coarsened), the hierarchical scan of any length, the single-pass domino scan with a ticket, exclusive from inclusive, work/step table, verification and timing |
| `bikeshare_group_by_station.cu` | counting-sort bucketing of trips by start station: histogram → exclusive scan → scatter (atomic and stable) |
| `bikeshare_rentals_in_progress.cu` | +1 undock / −1 dock events → inclusive scan → peak of bikes out and the hourly curve |

All programs use `../common/cuda_compat.h` and exit non-zero if a check fails. The step files
have a fixed tiny input; the complete programs take an optional size argument and print a
PASS/FAIL table against a CPU reference.

Laptop (no GPU, the header emulates CUDA with threads), the steps first:

```sh
cd ec2/09_scan
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_sequential_scan.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_kogge_stone.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_kogge_stone_double_buffer.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_brent_kung.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_coarsened_scan.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_hierarchical_scan.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_single_pass_domino.cu -o step && ./step
```

Then the complete programs and the bike-sharing examples:

```sh
cd ec2/09_scan
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common scan.cu -o scan && ./scan
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_group_by_station.cu -o group && ./group
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_rentals_in_progress.cu -o inprogress && ./inprogress
# race check (smaller size: TSan is slow)
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common scan.cu -o scan_tsan && ./scan_tsan 5000
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common -DCPU_TRIPS=8000 bikeshare_group_by_station.cu -o group_tsan && ./group_tsan
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common -DCPU_TRIPS=8000 bikeshare_rentals_in_progress.cu -o inprogress_tsan && ./inprogress_tsan
```

The bike-sharing programs read their emulation size from `CPU_TRIPS` (default 60 000 and
40 000 trips). They also run under ThreadSanitizer at the default size, in a few seconds;
`-DCPU_TRIPS=8000` only makes the race check faster.

> **Easiest way on Colab:** this folder has two notebooks. [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) runs the steps of the learning path in order, and [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) runs the complete programs and the bike-sharing examples. Upload one to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

Google Colab (GPU runtime), from a cell:

```sh
!cd ec2/09_scan && nvcc -O2 -I../common 02_kogge_stone.cu -o step && ./step
!cd ec2/09_scan && nvcc -O2 -I../common scan.cu -o scan && ./scan
!cd ec2/09_scan && nvcc -O2 -I../common bikeshare_group_by_station.cu -o group && ./group
!cd ec2/09_scan && nvcc -O2 -I../common bikeshare_rentals_in_progress.cu -o inprogress && ./inprogress
```

On the GPU the default sizes are 16 M elements and 4 M trips. In the emulation they are about
100 k elements and 40 k to 60 k trips, so each program finishes in a few seconds (scan.cu ≈ 10 s).

## 13. Exercises

1. **Trace Kogge-Stone.** Scan `2 5 1 4 3 0 6 2` by hand with Kogge-Stone. Write the array after
   each step and count the additions. Check the count against `N·log2 N − (N − 1)`.
2. **Trace Brent-Kung.** Do the same with Brent-Kung, marking which positions change in each
   up-sweep and down-sweep step. Which step has the fewest active threads?
3. **One barrier is not enough.** Remove the second `__syncthreads()` from `kogge_stone_scan`.
   Describe an interleaving of threads 3 and 4 in the stride-1 step that gives a wrong `XY[4]`.
   Why might the emulation still print PASS?
4. **Exclusive in one kernel.** Change `kogge_stone_scan` so it produces the exclusive scan
   directly, without `inclusive_to_exclusive`. Which single line changes, and what does the
   block-sum line become?
5. **Your own GPU table.** On Colab (`colab_2_full_programs.ipynb`), run `scan.cu` and add a column "GB/s = 8·n bytes ÷ time"
   (one read and one write per element). Which variant comes closest to `cudaMemcpy`
   device-to-device bandwidth? Explain the ranking with sections 5 and 9.
6. **Segmented scan.** The array holds the trips of each bike one after another, and
   `head[i] = 1` marks the first trip of a bike. Define an operator on pairs `(h, v)` such
   that an ordinary scan restarts at every head, and show that it is associative.
7. **Domino deadlock.** In `domino_scan`, replace the ticket with `blockIdx.x`. Describe a
   schedule on a GPU with 2 SMs, one resident block each, that never finishes. Why does the
   ticket rule it out?
8. **Bank conflicts.** In phase 1 of `coarsened_block_scan` (`phase1_scan_own_subsection`),
   which banks do threads 0..31 touch when they read `XY[first]`? Propose a padded index `j + j / 32` and say what must change in the load and the
   store.

### Short solutions

1. `2 7 6 5 7 3 6 8` → `2 7 8 12 13 8 13 11` → `2 7 8 12 15 15 21 23`.
   7 + 6 + 4 = 17 additions = 8·3 − 7.
2. Up-sweep: positions 1, 3, 5, 7 (4 threads), then 3, 7 (2), then 7 (1). Down-sweep: 3→5
   (1 thread), then 1→2, 3→4, 5→6 (3 threads). The last up-sweep step and the first down-sweep
   step each have a single active thread.
3. Thread 3 writes `XY[3] = x2 + x3` before thread 4 reads `XY[3]`. Thread 4 then computes
   `x2 + x3 + x4` instead of `x3 + x4`. Whether that happens depends on timing, so a run can
   print PASS by luck and the next one FAIL. ThreadSanitizer reports the race even in a run
   whose output is right. A test that passes is not a proof.
4. The load becomes `XY[t] = (i > 0 && i - 1 < n) ? X[i - 1] : 0;`. Then `XY[t]` ends as the sum
   of the block's elements before `i`. The block total no longer sits in `XY[KS_SECTION - 1]`,
   which misses the last input. Thread `KS_SECTION - 1` must write
   `sums[b] = XY[t] + (i < n ? X[i] : 0)`.
5. Expect: coarsened > Brent-Kung ≈ Kogge-Stone 2-buffer > Kogge-Stone. The single-pass version
   moves up or down depending on n, because the chain latency is about (number of sections) ×
   (one L2 round trip). Even the best hierarchical variant stays well below copy bandwidth,
   because it makes two passes over memory. Around half of it is a good result.
6. `(h1, v1) ⊕ (h2, v2) = (h1 | h2, h2 ? v2 : v1 + v2)`. To check associativity, expand both
   groupings of three pairs. Both give `h1|h2|h3` and "the sum since the last head among them".
7. Block 1 starts on SM 0 and block 2 on SM 1. Both spin waiting for blocks 0 and 1. Block 0 is
   never scheduled, because no SM ever frees up. With tickets, whoever starts first gets ticket
   0, and it waits for nobody.
8. With `j = 8t + k`, the bank is `(8t) mod 32`, so only banks 0, 8, 16 and 24 are used: 8
   threads per bank. With `j + j/32` the address shifts by one bank every 32 elements, and the
   eight threads that shared a bank now spread over eight banks. The shared array needs
   `CO_SECTION + CO_SECTION/32` entries, and the load, the store and phases 1 and 3 must all use
   the padded index.
