# Parallel sorting (chapter 14)

Based on chapter 14 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj).
Sorting is where several earlier patterns meet. The parallel **radix sort** of this chapter
is built from a map (extract a digit), an exclusive **scan** (chapter 11) to compute where
every element goes, and a scatter. Its optimized version adds tiling in shared memory and a
small table of bucket counts. The parallel **merge sort** is built from the co-rank merge of
chapter 13.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_radix_pass_sequential.cu`](01_radix_pass_sequential.cu) | One radix pass done sequentially by a single GPU thread: count the zeros, then move the keys with bit 0 to the front and the keys with bit 1 after them, in input order. Each key is printed with its original position in parentheses, so stability is visible. | Follow the numbers in parentheses: after the pass the even keys 2(1) 2(3) 4(4) 6(6) are at the front and the odd keys 5(0) 7(2) 1(5) 3(7) after them, each group in its original order. The two 2s keep their order, 2(1) before 2(3); that is stability, and the later passes rely on it. |
| 2 | [`02_radix_sort_1bit.cu`](02_radix_sort_1bit.cu) | The parallel 1-bit radix sort: a map extracts the bits, an exclusive scan (one block, Kogge-Stone) gives `onesBefore`, and the scatter kernel computes every destination. The bits, the scan and the array are printed after every bit. | Check one destination per kind in bit 0: key 5(0) has bit 1 and `onesBefore` 0, so it goes to 4 + 0 = 4, right after the 4 zeros; key 2(3) has bit 0 and `onesBefore` 2, so it goes to 3 − 2 = 1. After bit 1 the keys are sorted by their two lowest bits, and after bit 2 by all three, with 2(1) still before 2(3). Try: set `KEY_BITS = 2`; only two passes run, the result 4 5 1 2 2 6 7 3 is sorted by the lowest two bits only and the check fails. |
| 3 | [`03_radix_local_sort.cu`](03_radix_local_sort.cu) | Local sort in shared memory: each block splits its tile of 4 keys by the bit with a block scan and writes its bucket counts into a bucket-major table; the table is scanned and the coalesced scatter writes each bucket of each tile as one run (the worked example of the README). | In bit 0 each tile of 4 keys is split on its own (5 2 7 2 becomes 2 2 5 7), and the bucket-major table lists the zeros of tile 0, the zeros of tile 1, the ones of tile 0 and the ones of tile 1: 2 2 2 2. Its exclusive scan 0 2 4 6 is where each run starts, so the ones of tile 0 (5 7) land at positions 4 and 5. In bit 1 the counts become 1 2 3 2 and the offsets 0 1 3 6; the final array is the same as in step 2. |
| 4 | [`04_radix_multibit.cu`](04_radix_multibit.cu) | Multi-bit radix with 4-bit digits: the local sort does 4 one-bit splits per pass and finds where each of the 16 buckets starts and ends in the tile, so 8-bit keys need 2 passes instead of 8. Keys are printed in hex, one hex digit per pass. | Read only the low hex digit of the keys after the first pass: 0 0 1 2 3 5 5 6 7 8 a a b b c d, so one pass sorted by 4 bits at once. Each counts row has 16 buckets for one block (in the second pass block 0 has 2 keys in bucket 9, the two 95s). The second pass sorts by the high digit and keeps the low-digit order inside each bucket, so 3a 3a come before 4b 4b and the 8-bit keys are sorted after 2 passes. |
| 5 | [`05_merge_sort.cu`](05_merge_sort.cu) | Parallel merge sort: each thread sorts a run of 4 keys with insertion sort, then every pass merges pairs of runs with `co_rank`, each thread taking 3 outputs even when its slice crosses into the next pair. | Each vertical bar marks a run boundary: 4 sorted runs of 4 become 2 runs of 8 and then 1 run of 16, so 16 keys need 2 merge passes after the run sort. In the pass that builds runs of 8, 6 threads take 3 outputs each, and thread 2 owns outputs 6 to 8, so its slice crosses from the first pair of runs into the second and it calls `co_rank` for each piece. Try: set `OUTPUTS_PER_THREAD = 5`; 4 threads cover the 16 outputs and the result is the same. |

The steps use tiny inputs so the output can be checked by eye; steps 3 to 5 sort keys only, without
values. After them, `sorting.cu` adds the device-wide hierarchical scan, thread coarsening
(`ITEMS_PER_THREAD = 8`), key-value pairs, r = 1, 4 and 8 on large inputs with timings, and the two
bike-sharing programs apply the 4-bit radix sort.

## Contents

1. [Comparison and non-comparison sorts](#1-comparison-and-non-comparison-sorts)
2. [Radix sort, one bit at a time](#2-radix-sort-one-bit-at-a-time)
3. [The parallel radix sort kernel](#3-the-parallel-radix-sort-kernel)
4. [A device-wide exclusive scan](#4-a-device-wide-exclusive-scan)
5. [Why the scatter is slow: uncoalesced writes](#5-why-the-scatter-is-slow-uncoalesced-writes)
6. [Local sort in shared memory and coalesced writes](#6-local-sort-in-shared-memory-and-coalesced-writes)
7. [Choosing the radix size](#7-choosing-the-radix-size)
8. [Thread coarsening](#8-thread-coarsening)
9. [Parallel merge sort](#9-parallel-merge-sort)
10. [Other parallel sorts](#10-other-parallel-sorts)
11. [Performance analysis](#11-performance-analysis)
12. [Pitfalls](#12-pitfalls)
13. [Ideas for your project](#13-ideas-for-your-project)
14. [Files and how to run](#14-files-and-how-to-run)
15. [Exercises](#15-exercises)

---

## 1. Comparison and non-comparison sorts

A **comparison sort** learns about the data only by asking "is `x < y`?". Quicksort, merge
sort, heap sort and `std::sort` are comparison sorts. They work for any type with an order
(strings, floats, tuples, records with a custom comparator), but there is a price: any
comparison sort needs on the order of `n log n` comparisons in the worst case, because
each comparison gives at most one bit of information about which of the `n!` orders is the
right one.

A **non-comparison sort** looks inside the keys. Radix sort treats a key as a string of
digits (bits, groups of bits, bytes) and distributes the elements into buckets by one digit
at a time. For keys of `b` bits handled `r` bits at a time it makes `b / r` passes over the
data, each of linear work: `O(n * b / r)` in total, with no `log n`. The price is
generality: the key must be (or be mapped to) an integer.

Both have good parallel versions. Radix sort is usually the fastest GPU sort for integer
keys; merge sort is the usual choice when you need a comparator.

**Stability** matters for both. A sort is stable when elements with equal keys keep their
input order. Radix sort *needs* a stable step to work at all (section 2), and stability is
what lets you sort records by a second key without destroying the order of a first one.
All variants in this folder are stable and are checked against `std::stable_sort`: every key
carries a value (its original position), and the values must come out in exactly the same
order as in the reference.

## 2. Radix sort, one bit at a time

The **least significant digit first** (LSD) radix sort with one-bit digits:

```
for bit = 0, 1, ..., b-1:
    stably move all keys whose bit is 0 to the front, and all keys whose bit is 1 after them
```

After the pass for bit `p`, the keys are sorted by their lowest `p + 1` bits. Why: two keys
that differ in bit `p` are put in the right order by this pass; two keys that agree in bit
`p` keep the order of the previous pass (stability), which was already right for the lower
bits. If the split were not stable the earlier passes would be undone.

Example with 3-bit keys (the second `2` is written `2'` to follow it):

```
input           5    2    7    2'   4    1    6    3
bit 0           1    0    1    0    0    1    0    1
after bit 0     2    2'   4    6    5    7    1    3      (zeros 2 2' 4 6, ones 5 7 1 3)
after bit 1     4    5    1    2    2'   6    7    3
after bit 2     1    2    2'   3    4    5    6    7      2 is still before 2': stable
```

## 3. The parallel radix sort kernel

How does a thread know where its element goes in a split? It needs only two numbers:

- `onesBefore[i]`: how many elements before position `i` have bit 1. This is the
  **exclusive scan** of the bit array;
- `zeros`: the total number of zeros, `n - (onesBefore[n-1] + bit[n-1])`.

Then

```
if bit[i] == 0:  destination = i - onesBefore[i]           (the zeros before i)
if bit[i] == 1:  destination = zeros + onesBefore[i]       (after all zeros, then the ones before i)
```

Both formulas keep the input order inside each group, so the split is stable. For bit 0 of
the example: bits `1 0 1 0 0 1 0 1`, `onesBefore = 0 1 1 2 2 2 3 3`, `zeros = 4`; the `2` at
position 1 goes to `1 - 1 = 0`, the `5` at position 0 goes to `4 + 0 = 4`, and so on.

One pass of `sorting.cu` is three steps: a map that extracts the bits, the device-wide
exclusive scan, and the scatter:

```cpp
__global__ void radix_scatter_kernel(const unsigned* keysIn,
                                     const unsigned* valsIn,
                                     const unsigned* bits,
                                     const unsigned* onesBefore,
                                     int n,
                                     unsigned* keysOut,
                                     unsigned* valsOut) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const unsigned totalOnes = onesBefore[n - 1] + bits[n - 1];
    const unsigned zeros = n - totalOnes;
    const unsigned dst = bits[i] ? zeros + onesBefore[i] : i - onesBefore[i];
    keysOut[dst] = keysIn[i];
    valsOut[dst] = valsIn[i];
  }
}
```

The two stores at the end are the slow part of the pass: neighbouring threads write to two
far-apart regions of the output (section 5). The scatter cannot work in place (an element could
overwrite one that has not moved yet), so the driver ping-pongs between two pairs of buffers.

## 4. A device-wide exclusive scan

Every pass needs an exclusive scan of `n` values, across all blocks. `sorting.cu` contains a
compact hierarchical scan, the scheme of chapter 11:

1. `scan_block_kernel`: each block loads a tile of `TILE = 1024` values (coalesced) into
   shared memory; each of its 128 threads adds up its 8 consecutive values; a Kogge-Stone
   scan over the 128 thread sums (double-buffered: one barrier per step) gives each thread
   its starting offset; the thread writes the exclusive prefixes of its 8 values. The tile
   total goes to `blockSums[blockIdx.x]`;
2. the array of block sums is scanned by the same function, recursively (it is 1024 times
   shorter, so one more level is usually enough);
3. `add_block_offsets_kernel` adds to each element the scanned sum of all tiles before its own.

```cpp
static void exclusiveScan(const unsigned* in, unsigned* out, int n, unsigned* workspace) {
  if (n <= 0) {
    return;
  }

  const int blocks = cdiv(n, TILE);
  unsigned* sums = workspace;
  unsigned* sumsScanned = sums + blocks;
  unsigned* deeperWorkspace = sumsScanned + blocks;
  launch(scan_block_kernel, dim3(blocks), dim3(BLOCK_THREADS), in, out, n, sums);
  if (blocks > 1) {
    exclusiveScan(sums, sumsScanned, blocks, deeperWorkspace);
    launch(add_block_offsets_kernel, dim3(blocks), dim3(BLOCK_THREADS), out, n, sumsScanned);
  }
}
```

Each level of the recursion takes two arrays of block sums from the workspace (the sums and
their scan) and hands the rest to the next level; `scanWorkspace(n)` adds up those sizes
for every level, plus a little slack, so the drivers can allocate it once.

Two index patterns appear in the kernels, each with its own helper. `coalesced_position(item)`
is `item * BLOCK_THREADS + threadIdx.x`: consecutive threads touch consecutive addresses, which
is how every tile is loaded from and stored to global memory. `thread_owned_position(item)` is
`threadIdx.x * ITEMS_PER_THREAD + item`: each thread owns `ITEMS_PER_THREAD` consecutive
elements of the tile in shared memory, which it scans sequentially.

`block_inclusive_scan` ends with a `__syncthreads()` after every thread has read its result.
The caller reuses the same shared buffer right away (the local sort calls it once per bit), and
without that barrier a fast thread could start the next scan and overwrite values that a slow
thread has not read yet. Its `readHalf` and `writeHalf` name the two halves of the
double-buffered Kogge-Stone scan.

The same scan is used by the optimized radix sort, on a different array (section 6).

## 5. Why the scatter is slow: uncoalesced writes

In the scatter, the 32 threads of a warp read 32 consecutive elements, but they write them
to two separate places (the zeros' region and the ones' region), and inside each region
only the threads with that bit value write, at consecutive addresses. With random bits a
warp's store touches two partially used memory segments, and the hardware moves whole
segments. With `r`-bit digits instead of single bits the same direct scatter would send a
warp's elements to up to `2^r` regions: worse still.

There is a second cost: every pass reads and writes the whole array several times (bits,
scan, scatter), and a 32-bit key needs 32 passes.

## 6. Local sort in shared memory and coalesced writes

The optimization splits each pass into a block-local part, done in shared memory, and a
global part that writes long contiguous runs:

1. **Local sort** (`radix_local_sort_kernel`). Each block loads its tile of `TILE` pairs into
   shared memory and sorts it by the current digit with stable 1-bit splits (the same
   scan-and-move as in section 3, but with a block-level scan in shared memory). In the
   sorted tile all elements of bucket 0 are together, then bucket 1, and so on. The block
   records, for each bucket `d`, its size `counts[d][block]` and where it starts in the tile
   `localStart[block][d]`, and stores the sorted tile (coalesced).
2. **Scan of the bucket table.** The counts are stored **bucket-major**:
   `counts[d * numBlocks + block]`. Its exclusive scan gives, for every (bucket, block), the
   global position of that block's run of that bucket: first bucket 0 of block 0, then
   bucket 0 of block 1, ..., then bucket 1 of block 0, ... This is exactly the order of the
   sorted output, and it keeps the sort stable (lower blocks first inside each bucket).
3. **Coalesced scatter** (`radix_scatter_coalesced_kernel`). The element at local position
   `pos` in bucket `d` goes to `offset[d][block] + (pos - localStart[block][d])`.
   Consecutive threads hold consecutive elements of the same bucket, so they write
   consecutive addresses; only the few bucket boundaries break a warp's store.

The same example (`r = 1`, bit 0, tiles of 4):

```
block 0: 5 2 7 2'  -> local sort -> 2 2' | 5 7       counts: bucket0 = 2, bucket1 = 2
block 1: 4 1 6 3   -> local sort -> 4 6  | 1 3       counts: bucket0 = 2, bucket1 = 2

table (bucket-major):   b0.blk0  b0.blk1  b1.blk0  b1.blk1
counts                      2        2        2        2
exclusive scan              0        2        4        6     <- where each run starts

output:  [2 2'] [4 6] [5 7] [1 3]      same as the naive pass, written as four runs
```

The heart of the local sort, one 1-bit split in shared memory (each thread owns
`ITEMS_PER_THREAD` consecutive elements of the tile). `keys_S` and `vals_S` hold two copies
of the tile; `current` is the copy that holds the order so far and `next` the one the split
writes:

```cpp
    unsigned totalOnes;
    unsigned onesBefore = block_inclusive_scan(myOnes, buf, &totalOnes) - myOnes;
    const unsigned zeros = TILE - totalOnes;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const unsigned pos = thread_owned_position(item);
      const unsigned dst = myBits[item] ? zeros + onesBefore : pos - onesBefore;
      onesBefore += myBits[item];
      keys_S[next][dst] = keys_S[current][pos];
      vals_S[next][dst] = vals_S[current][pos];
    }
    __syncthreads();
    current = next;
```

The two layouts of the tables have their own index helpers, so the code says which one it
uses: `counts` and `offsets` are read and written through `bucket_major_index(bucket, block,
numBlocks)`, `localStart` through `block_major_index(block, bucket, buckets)`. In the
coalesced scatter each block turns the two tables into one number per bucket,
`localToGlobal[d] = offset - localStart`, and adds it to the local position.

A partial last tile is padded with `PADDING_KEY` (`0xFFFFFFFF`). Their digit is always the
largest and they come after all real elements, so the stable splits leave them at the end of
the tile, where they are never counted or written.

## 7. Choosing the radix size

With `r`-bit digits there are `2^r` buckets and `ceil(b / r)` passes. For `b = 32`:

| r | buckets | passes | table size (n = 2^24, TILE = 1024) | average run per (block, bucket) |
|---|---------|--------|------------------------------------|---------------------------------|
| 1 | 2       | 32     | 32 K entries                       | 512 elements                    |
| 4 | 16      | 8      | 256 K entries                      | 64 elements                     |
| 8 | 256     | 4      | 4 M entries                        | 4 elements                      |

- **Larger `r`, fewer passes.** Each pass reads and writes all keys and values a few times,
  so cutting the passes from 32 to 8 cuts most of the global traffic.
- **Larger `r`, more buckets.** The bucket table grows as `2^r * numBlocks` and must be
  scanned every pass. And the runs get short: with 256 buckets a tile of 1024 elements has
  runs of about 4 elements, so a warp's store touches many segments again and the coalescing
  that motivated the local sort is lost.
- The local sort in our code does `r` one-bit splits per pass, so its total cost is the same
  for every `r` (`b` splits in all); production sorts instead count digits directly with a
  per-block histogram (chapter 9) to make larger `r` cheaper.

Values around 4 to 8 bits are the usual compromise. Run `sorting.cu` on a GPU to see
`r = 1, 4, 8` side by side.

## 8. Thread coarsening

In the scan and the local sort each thread handles `ITEMS_PER_THREAD = 8` elements, so a block
of 128 threads covers a tile of 1024 elements. Coarsening pays off three times here:

- the expensive part of a block scan (the Kogge-Stone steps with barriers) runs over 128
  values instead of 1024, while each thread does a cheap sequential scan of its 8;
- bigger tiles mean fewer blocks, so the bucket table (`2^r` entries **per block**) is
  smaller and cheaper to scan;
- bigger tiles mean longer runs per bucket, so the final writes are better coalesced.

The limits are the usual ones: shared memory per block (our local sort keeps two copies of
keys and values: 16 KB) and enough blocks to fill the GPU.

## 9. Parallel merge sort

Merge sort works bottom-up:

1. `sort_runs_kernel`: each thread sorts a short run of 16 elements with insertion sort
   (stable);
2. `log2(n / 16)` passes of `merge_pass_kernel`: pass `w` merges pairs of sorted runs of
   width `w` into runs of width `2w`.

Each merge pass is the basic kernel of chapter 13 applied to many pairs at once. Every
thread takes the same number of outputs of the whole array; if its slice crosses from one
pair into the next, it handles each part separately. Inside a pair it finds its sub-range
with `co_rank` and merges sequentially:

```cpp
  int k = sliceBegin;
  while (k < sliceEnd) {
    const int pairStart = (k / (2 * width)) * (2 * width);
    const int mid = imin(pairStart + width, total);
    const int pairEnd = imin(pairStart + 2 * width, total);
    const int pieceEnd = imin(sliceEnd, pairEnd);

    const int m = mid - pairStart;
    const int n = pairEnd - mid;
    const int k_curr = k - pairStart;
    const int k_next = pieceEnd - pairStart;
    const int i_curr = co_rank(k_curr, keysIn + pairStart, m, keysIn + mid, n);
    const int i_next = co_rank(k_next, keysIn + pairStart, m, keysIn + mid, n);
    ...
    k = pieceEnd;
  }
```

Here `total` is the length of the whole array, and `m`, `n`, `k_curr`, `k_next`, `i_curr` and
`i_next` mean what they mean in the basic merge kernel of chapter 13, measured from the start
of the current pair: the two runs of the pair play the roles of `A` and `B`. The runs of the
first step are sorted by `stable_insertion_sort`, one run per thread.

Because the work is split by output, the last passes (a few very long pairs) keep all
threads busy, just like the first ones. Ties go to the left run, so the sort is stable. The
cost is `O(n log n)` work and `log2(n / 16)` passes over memory; with the tiled merge of
chapter 13 each pass would read memory more efficiently. Merge sort needs only a comparison,
so the same code sorts floats, strings or records with a custom order.

## 10. Other parallel sorts

- **Bitonic sort** and **odd-even merge sort** are *sorting networks*: a fixed sequence of
  compare-and-swap steps that does not depend on the data. `O(n log^2 n)` work, but perfectly
  regular, which makes them good for small arrays inside one block (in shared memory or
  registers).
- **Odd-even transposition sort**: `n` rounds of neighbour swaps. Simple, too slow for large `n`.
- **Sample sort**: pick splitters from a sample, distribute the elements into buckets (one
  per splitter interval), sort each bucket independently. A parallel generalization of
  quicksort; good for distributed memory.
- **Parallel quicksort** with dynamic parallelism: the kernel launches kernels for the two
  partitions. Load balance depends on the pivots.
- **Library sorts.** In practice use `thrust::sort` / `cub::DeviceRadixSort`. CUB's radix
  sort uses the ideas of this chapter plus a single-pass scan with decoupled look-back, so a
  whole digit pass needs only one sweep over memory.

## 11. Performance analysis

- **Naive 1-bit radix.** `b` passes; each pass reads the keys twice, writes and reads the
  bits, runs a scan (another read and write) and writes keys and values with poorly
  coalesced stores. Expect it to be clearly the slowest radix variant on a GPU.
- **Local sort, r = 1.** Same 32 passes, but the stores are coalesced; the local sort adds
  shared-memory work and barriers.
- **Local sort, r = 4.** 8 passes: much less global traffic. Usually the sweet spot of these
  three on a GPU.
- **Local sort, r = 8.** 4 passes, but a table 16 times larger and runs of ~4 elements.
- **Merge sort.** `log2(n/16) = 20` passes for `n = 2^24`, each with the uncoalesced accesses
  of the basic merge kernel; competitive only if the keys are not integers.
- **CPU reference.** `std::sort` and `std::stable_sort` on one core. On a GPU the radix
  sorts should win by a large factor for millions of keys; for a few thousand keys the
  launch overheads dominate and the CPU wins.

**About the numbers in emulation.** On a laptop each launch creates CPU threads, and each
`__syncthreads()` is a real barrier between them. The local-sort kernels have dozens of
barriers per block, so in the emulation they look slow and merge sort (no barriers) looks
fast. That says nothing about a GPU. Use the emulation to check results; compare timings
only on a GPU (Colab, with `colab_2_full_programs.ipynb`).

## 12. Pitfalls

- **Unstable split.** Any change that does not preserve the input order inside a bucket
  (for example assigning destinations with an `atomicAdd` per bucket) breaks LSD radix sort:
  later passes undo earlier ones. The value column in `sorting.cu` exists to catch that.
- **Reading the total in the scatter.** `zeros` needs the *inclusive* total
  (`onesBefore[n-1] + bits[n-1]`); forgetting the last bit is an off-by-one that only shows
  when the last key has bit 1.
- **Sorting in place.** The scatter must write to a different buffer; and the driver must
  remember which buffer holds the result after an odd number of passes.
- **Padding a partial tile** with keys that would sort *before* real keys (for example 0)
  puts garbage in the output. Pad with the largest key and never count the padding.
- **Table layout.** The bucket table must be bucket-major to get the global order of the
  runs from one scan; block-major gives a different (wrong) order.
- **Shifting by 32.** `key >> 32` is undefined behaviour in C++ and CUDA. When `b` is not a
  multiple of `r`, the last digit can reach past bit 31; `digit_of` guards
  `shift < UNSIGNED_BITS` (32).
- **Passes for the key range.** Radix sort only needs as many bits as the largest key has
  (`bikeshare_trip_percentiles.cu` computes it): 23 bits instead of 32 saves two 4-bit passes.
- **Signed and floating-point keys.** The bit order of negative `int`s and of floats is not
  their numeric order; map them first (flip the sign bit for `int`; for IEEE floats flip
  the sign bit of positives and all bits of negatives).
- **Early return before a barrier.** Threads of the last, partial tile still take part in
  every `__syncthreads()`; out-of-range positions are guarded with `if`, never `return`.

## 13. Ideas for your project

### The bike-sharing examples in this folder

Both programs model a city bike-sharing system (a 6 km × 6 km map in metres, 48 docking
stations) with synthetic data from the file's own SplitMix64 generator and a fixed seed.

**Trip-duration percentiles (`bikeshare_trip_percentiles.cu`).** The average trip duration
hides what riders feel and what they pay: the fare is Bs 3 plus Bs 1 per started 10 minutes
(integer cents), and a trip is late when it runs over the 45 included minutes. The median,
p90, p95 and p99 say much more. The program generates trips with a bike, a start and an end
station and start and end times: commute trips of 4–25 minutes around the 08:00 and 18:00
rush hours, a mix of commute and leisure trips (10–70 minutes) at other times, and one trip in
25 that ends at a full station and adds 5–40 minutes of looking for a free dock. Durations are
whole seconds, so many trips tie. The key (duration in ms) is a small integer: radix sort
territory. The program computes the number of key bits from the longest trip (23 bits for
108 minutes, so 6 passes of 4 bits), sorts (duration, trip id) pairs, and reads every
percentile with one access (nearest rank: `sorted[ceil(p/100 * n) - 1]`), printed with its
fare. Because the array is sorted, the number of late trips is a single binary search for the
first duration over 45 minutes; it is checked against a direct count. The five longest trips
are listed with their bike, stations and fare, and stability makes that list deterministic:
trips with equal durations keep ascending trip ids. The checks compare the keys with
`std::sort`, verify that every id kept its key and that ties keep ascending ids, and compare
the percentiles and the late count with the reference, ending with `ALL PASS`.

**Nearest bikes (`bikeshare_nearest_bikes.cu`).** In a rush many riders open the app at the
same time, and each one must be offered the `k = 3` nearest bikes that are docked and have at
least 30 % battery. `R` requests and `B` bikes give `R x B` pairs: a map computes the walking
distance (Manhattan metres) of every pair, one thread per pair, and one radix sort on the
**composite key** `(request << 14) | distance` sorts every request's bikes by distance in a
single sort (a segmented sort without segment logic). The `k` nearest bikes of request `r`
are then the first `k` pairs of its group, positions `[r*B, r*B + k)` of the sorted array: no
search is needed. Bikes docked at the same station are at the same distance; they keep
ascending bike ids (the pairs are generated with ascending bike ids and the sort is stable), so the
offer is the same in every run. A bike that is riding, in repair, or docked below 30 % battery
gets `DISTANCE_UNAVAILABLE`, the largest 14-bit value, so it sorts after every bike that can
be offered, and the printed offers stop there. The key has `DISTANCE_BITS + bits(R - 1)` bits:
20 bits and 5 passes for 64 requests x 256 bikes in the emulation, 24 bits and 6 passes for
1024 x 4096 on a GPU (`-DCPU_REQUESTS=...` and `-DCPU_BIKES=...` change the emulation sizes).
Both the offered bikes and the whole ranking are checked against a `std::stable_sort` per
request on the CPU.

### Where this pattern may appear in your project

- Percentiles say more than the average when a design change can help most cases and hurt the
  slowest ones. If your simulation measures a duration per item, how would you report its tail?
- When the key is a small integer (milliseconds below a few hours, an id below a few thousand),
  count its bits first: fewer bits means fewer radix passes.
- Many requests that each need their best few candidates (couriers for an order, for example)
  are a segmented sort in disguise: a composite key (group, score) sorts every group at once,
  and the first `k` of each group are the answer.
- Stability is what makes ties reproducible. If two candidates are equally good, which one
  should win, and does your program pick the same one in every run?
- A log already sorted by time and then stably sorted by an id is grouped by that id, with the
  time sequence kept inside each group. Does any report or check in your project need events
  grouped per item?

## 14. Files and how to run

| File | What it shows |
|------|---------------|
| `01_*.cu` to `05_*.cu` | The step files of the [learning path](#learning-path), from one sequential radix pass to merge sort |
| `sorting.cu` | Hierarchical exclusive scan; naive 1-bit radix sort; local-sort radix sort with r = 1, 4, 8; merge sort with co-rank; tests and a timing table against `std::sort` / `std::stable_sort` |
| `bikeshare_trip_percentiles.cu` | Radix sort of bike-share trip durations to get median, p90, p95, p99, their fares, the trips over the included time and the longest trips |
| `bikeshare_nearest_bikes.cu` | Distance map plus one radix sort on a composite key: the k nearest charged bikes for every rider request of a rush |

The same file runs on a GPU and on any laptop (see `../common/cuda_compat.h`).
`sorting.cu` and `bikeshare_trip_percentiles.cu` take an optional problem size as argument.
Both bike-sharing programs carry their own copy of the 4-bit local-sort radix sort of
`sorting.cu` (with `RADIX_BITS = 4` fixed instead of a template parameter), so that each file
compiles on its own.

> **Easiest way on Colab:** this folder has two notebooks. Upload [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) for the step files of the learning path, or [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) for `sorting.cu` and the bike-sharing examples, to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

```sh
# laptop (no GPU): the CUDA parts are emulated with std::thread
cd ec2/12_sorting
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_radix_pass_sequential.cu -o step && ./step
for f in 0*.cu; do clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common $f -o step && ./step || break; done
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common sorting.cu -o sorting && ./sorting
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_trip_percentiles.cu -o bikeshare_trip_percentiles && ./bikeshare_trip_percentiles
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_nearest_bikes.cu -o bikeshare_nearest_bikes && ./bikeshare_nearest_bikes
# g++ works the same way; add -fsanitize=thread -g (and a small size: ./sorting 3000) to check for races
# for example: ./bikeshare_trip_percentiles 3000, or build bikeshare_nearest_bikes.cu with -DCPU_REQUESTS=16 -DCPU_BIKES=128

# GPU: the two notebooks already contain these cells; by hand, in a Colab cell (T4 GPU):
!nvcc -O2 -I../common 01_radix_pass_sequential.cu -o step && ./step
!nvcc -O2 -I../common sorting.cu -o sorting && ./sorting
!nvcc -O2 -I../common bikeshare_trip_percentiles.cu -o bikeshare_trip_percentiles && ./bikeshare_trip_percentiles
!nvcc -O2 -I../common bikeshare_nearest_bikes.cu -o bikeshare_nearest_bikes && ./bikeshare_nearest_bikes
```

Sizes come from `BY_BACKEND(gpuSize, cpuSize)`: 16 M pairs with 32-bit keys on a GPU, 16 K
pairs with 16-bit keys in the emulation, so each program finishes in seconds.

## 15. Exercises

1. **Trace.** Sort `[6, 3, 6', 0, 5, 2, 7, 1]` (3-bit keys) with the 1-bit radix sort. For
   the first pass, write the bit array, `onesBefore`, `zeros` and every destination.
2. **Why stable?** Give a two-pass example (2-bit keys) where an unstable split in the
   second pass produces an unsorted result.
3. **Passes and tables.** For 32-bit keys, `n = 2^22` and tiles of 512, compute the number of
   passes, the size of the bucket table and the average run length for `r = 2` and `r = 6`.
4. **Fewer bits.** Trip durations are below 30 minutes and stored in ms. How many 4-bit
   passes does the radix sort need? And if the times are first rounded to the 100 ms tick
   and divided by 100?
5. **Composite keys.** A report wants trips grouped by start station (48 stations), and
   inside each station by duration (below 2^23 ms), longest first. Design one key for one
   ascending radix sort.
6. **Floats.** Explain how to map IEEE-754 `float` keys to `unsigned` so that the unsigned
   order equals the float order, including negative numbers.
7. **Merge sort or radix sort?** For `n = 2^24` 32-bit keys, count the passes over memory of
   the 4-bit radix sort and of the merge sort of `sorting.cu`. When would you still pick merge
   sort?

### Short solutions

1. Bits of bit 0: `0 1 0 0 1 0 1 1`; `onesBefore = 0 0 1 1 1 2 2 3`; `zeros = 4`.
   Destinations: `6->0, 3->4, 6'->1, 0->2, 5->5, 2->3, 7->6, 1->7`, giving
   `6 6' 0 2 3 5 7 1`. After bit 1: `0 5 1 6 6' 2 3 7`. After bit 2:
   `0 1 2 3 5 6 6' 7`.
2. Input `[3, 2]` (`3 = 11`, `2 = 10`). The pass for bit 0 puts `2` (bit 0 = 0) before `3`:
   `[2, 3]`. In the pass for bit 1 both keys have bit 1 = 1 and fall into the same bucket. A
   stable split keeps `[2, 3]`; an unstable one is free to output `[3, 2]`, which is not
   sorted: the order established by bit 0 was lost.
3. 512-element tiles give `2^13` blocks. `r = 2`: 16 passes, table `4 * 2^13 = 32 K`, runs of
   `512 / 4 = 128`. `r = 6`: 6 passes (the last one with 2 bits), table `64 * 2^13 = 512 K`,
   runs of `512 / 64 = 8`.
4. 30 min = 1,800,000 ms < 2^21: 21 bits, 6 passes. Divided by 100: 18,000 < 2^15:
   15 bits, 4 passes.
5. `key = (startStation << 23) | (2^23 - 1 - durationMs)`: the high 6 bits (48 < 64) group
   by station, and subtracting from the maximum turns "longest first" into ascending order.
   29 bits: 8 passes of 4 bits.
6. Take the bits `u` of the float. If the sign bit is 0, set it (`u ^ 0x80000000`), so
   positives go above negatives. If the sign bit is 1, flip all bits (`~u`), which reverses
   the order of negatives (a more negative float has a larger magnitude, so it must become
   smaller).
7. Radix r = 4: 8 passes, each reading keys and values twice and writing them twice, plus
   the table scan. Merge sort: 1 pass for the runs plus 20 merge passes. Pick merge sort when
   the keys are not integers or the order is given by a comparator (strings, tuples, a custom
   order of records), or when only a merge of already sorted parts is needed.
