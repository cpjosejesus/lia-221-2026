# Chapter 9 — Parallel histogram

Based on chapter 9 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj).
Every program in this folder is a single `.cu` file that runs on an NVIDIA GPU (Google Colab)
**and** on any laptop without a GPU, through the CPU emulation in
[`../common/cuda_compat.h`](../common/cuda_compat.h).

The slides are session 17 (`17_histogram`). The numbered step files below follow them: the
same 8 x 8 grayscale tree, 4 bins of 64 intensities each (`bin = intensity / 64`, so the
correct counts are 6, 12, 14 and 32), and the same kernels with the book's names, one idea at
a time. Each step prints the image and the bins and checks them against
`histogram_sequential`.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_histogram_sequential.cu`](01_histogram_sequential.cu) | The sequential loop `histogram_sequential` on the tree image; it is called row by row so the bins are printed after every row. (slide: *What you already know*) | The bins grow row by row: rows 0 to 3 add only to the leaves and sky bins, and the last line equals the expected 6, 12, 14, 32. |
| 2 | [`02_race_and_atomic.cu`](02_race_and_atomic.cu) | One thread per pixel. `histogram_racy` uses a plain `++bins[b]` and loses updates on a GPU (the CPU emulation often loses none); `histogram_racy_all_read_first` puts a barrier between the read and the write, so every thread reads before anyone writes and each bin ends at 1 (60 updates lost). `histogram_basic` uses `cuda::atomic_ref` and `fetch_add` and is exact. The two racy kernels are wrong on purpose: ThreadSanitizer reports them, and the PASS refers to the atomic kernel. (slides: *Atomic operations in CUDA C++*, *Code: the basic histogram kernel*, *Why `memory_order_relaxed` is enough*) | The `lost` column. The `++bins[b]` row may lose nothing on the CPU emulation but loses updates on a GPU; the "all read, then write" row has 1 in every bin (60 lost) on both; the atomic row equals the sequential one. |
| 3 | [`03_private_global.cu`](03_private_global.cu) | `histogram_private_global`: 4 blocks of 16 threads, each block counts into its own slice of `bins_pool` (block scope), then commits it to `bins` (device scope). Every block's private copy is printed next to the total. (slide: *Code: private copies in global memory*) | Each private copy counts only its block's two rows (block 0 sees only sky and leaves), and the four copies add up to the final bins. |
| 4 | [`04_private_shared.cu`](04_private_shared.cu) | `histogram_private_shared`: the private copy moves to `__shared__` memory; the block zeroes it itself, and two barriers separate zeroing, counting and committing. (slide: *Code: private copies in shared memory*) | The same bins as in step 3, now without a pool of private copies in global memory. |
| 5 | [`05_coarsening.cu`](05_coarsening.cu) | `histogram_coarse_contig` and `histogram_coarse_interl`: 8 threads per block with `COARSE_FACTOR` = 4 pixels each, one line apart. The program prints the pixels block 0 reads in each iteration: 4 apart in the contiguous kernel, consecutive (coalesced) in the interleaved one. (slide: *Code: coarsening, one line apart*) | In iteration c = 0 the contiguous threads read pixels 0, 4, 8, ... and the interleaved threads read 0, 1, 2, ...; both kernels give the same bins. |
| 6 | [`06_most_recent_bin.cu`](06_most_recent_bin.cu) | `histogram_thread_run`: each thread keeps the most recent bin in a register (`bin_r`) and issues one shared-memory atomic per run of equal bins. It prints the bins each thread sees (a column of the image) and counts 30 atomics instead of 64. (slide: *Code: privatizing the most recent bin*) | Threads whose column is all sky need a single atomic; the total line shows 30 shared-memory atomics instead of 64. |
| 7 | [`07_thread_private_bins.cu`](07_thread_private_bins.cu) | `histogram_thread_private`: each thread counts its 4 interleaved pixels in a private array `bins_t[NUM_BINS]` with plain `+=` (the unrolled compare-and-add keeps it in registers), then adds each nonzero entry to `bins_s` with one block-scope atomic. It prints every thread's `bins_t` and counts the shared-memory atomics. (slide: *Code: thread-level privatization*) | Each thread's `bins_t` adds up to its 4 pixels and needs at most 4 atomics (one per nonzero bin); the total line shows 30 shared-memory atomics instead of 64, and the counting loop has none. |

After the steps come the complete programs: `histogram_image.cu` runs every kernel of the
chapter on large images, `histogram.cu` does the same on text with timing and atomic counts,
`race_demo.cu` shows the race on 64 K pixels, and the bike-sharing examples
(`bikeshare_trip_durations.cu`, `bikeshare_station_hour.cu`) use the pattern in an application.

## 1. The pattern in one paragraph

A histogram counts how many input elements fall in each of a fixed set of bins. It looks like
a map (one small computation per element), but it is not: in a map each thread knows *where* it
writes before it starts, so no two threads ever write the same place. In a histogram the output
location is decided by the data. Two threads that read two `e`s both want to increment the same
counter, and neither knows the other exists. This is **output interference**, and it is the
whole subject of the chapter: first make the updates correct (atomic operations), then make
them fast (privatization, coarsening, aggregation). The same structure appears whenever you
"count by category": word frequencies, image intensity histograms, events per type, rentals per
station, trip durations by minute.

## 2. The running example

The input is a text. Each lowercase letter is counted in a bin of four consecutive letters:

```
bin    0     1     2     3     4     5     6
     a-d   e-h   i-l   m-p   q-t   u-x   y-z        (26 letters -> 7 bins)
```

Anything that is not a lowercase letter (spaces, punctuation) is skipped. With only 7 bins and
English text, two bins (`e-h` and `q-t`) receive almost half of all updates: little output, a
lot of contention. That is what makes this tiny example a good stress test.

## 3. The sequential baseline

```cpp
static void histogram_sequential(const char* data, unsigned length, unsigned* histo) {
  for (unsigned b = 0; b < NUM_BINS; ++b) {
    histo[b] = 0;
  }

  for (unsigned i = 0; i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) {
      histo[alphabet_position / LETTERS_PER_BIN]++;
    }
  }
}
```

`ALPHABET_SIZE` is 26 and `LETTERS_PER_BIN` is 4; `NUM_BINS` is derived from them (26 / 4
rounded up = 7). The kernels classify a character with `binOf`, which does the same test and
returns `NOT_A_LETTER` (-1) for anything outside `a..z`.

One pass, one load and one counter update per character: O(n) work, perfectly cache friendly
(the 7 counters stay in L1). On a laptop it processes hundreds of millions of characters per
second. A parallel version has to beat that, and the counters are the problem.

## 4. Kernel 1 — one thread per character, and why `histo[b]++` is wrong

The obvious parallelisation assigns one thread to each character:

```cpp
__global__ void histo_basic(const char* data, unsigned length, unsigned* histo) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int b = binOf(data[i]);
    if (b >= 0) {
      atomicAdd(&histo[b], 1u);
    }
  }
}
```

Why the `atomicAdd`? Because `histo[b]++` is not one operation but three: **read** the counter
into a register, **add** one, **write** it back. Two threads doing that on the same counter can
interleave like this:

```
histo[1] = 5
thread A: read  -> 5
thread B: read  -> 5
thread A: add, write 6
thread B: add, write 6          final value 6, but two letters were counted: one is lost
```

This is a **read-modify-write race**: the result depends on the timing, and the error is
silent. On a GPU with thousands of threads hitting 7 counters, the plain version loses most of
the increments.

An **atomic operation** performs the read, the modification and the write as one indivisible
step on that memory location: while one thread's `atomicAdd` on `histo[1]` is in progress, no
other thread can read or write `histo[1]`. CUDA offers `atomicAdd`, `atomicSub`, `atomicMin`,
`atomicMax`, `atomicExch`, `atomicCAS` and others, for 32- and 64-bit integers and (for add)
floats. Each returns the **old** value, which is useful for things like claiming slots in a
queue. Two things an atomic does **not** give you: an order between threads (which thread's
add happens first is still arbitrary, which is fine for a sum because addition is
commutative), and protection of *other* memory locations (it is not a lock).

## 5. Why atomics are slow: latency, throughput and contention

An atomic on global memory is executed close to the memory (in the L2 cache on current GPUs),
not in the SM. A single atomic has a long **latency**: the request travels to L2, the old value
is read, modified, written, and the reply travels back. While one atomic on an address is in
progress, the next atomic on **the same address** has to wait. So the throughput on one
location is bounded by

```
atomics per second on one address  <=  1 / (latency of one atomic)
```

A model with round numbers (illustrative, not a measurement): if an atomic on a DRAM location
took 1000 cycles at 1 GHz, one address could absorb only 1 million atomics per second. With 7
bins, at most 7 million letters per second, slower than the CPU loop above. Caching the
atomics in L2 cuts the latency by an order of magnitude or more, but the shape of the problem
is the same: **the fewer distinct bins the updates go to, the more they serialise**. Input
with a skewed distribution (an image of mostly sky, a text where most characters are `a`) is the
worst case: almost every thread queues on one address. Atomics on *different* addresses
proceed in parallel, so large histograms with evenly spread data suffer much less.

The program's "sky" input (95% of the characters in bin 0) exists to show this.

## 6. Privatization

The cure for contention is to give groups of threads their **own copy** of the histogram.
Threads of one group update their private copy; at the end the copies are added into the
final histogram. Contention per copy drops by the number of copies, and the merge costs only
`NUM_BINS` atomics per copy.

### Kernel 2 — one private copy per block, in global memory

`histo` now has `gridDim.x * NUM_BINS` counters. Block `k` updates `histo[k*NUM_BINS + b]`.
After a barrier, which guarantees that the block's whole private copy is complete before any
thread reads it, every block except block 0 adds its copy into copy 0:

```cpp
  __syncthreads();

  if (blockIdx.x > 0) {
    for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
      unsigned v = histo[blockIdx.x * NUM_BINS + bin];
      if (v > 0) {
        atomicAdd(&histo[bin], v);
      }
    }
  }
```

Block 0 does not commit: its copy *is* the final histogram. That is also why the program's
atomic count for kernel 2 leaves out the bins of block 0 (`FirstCopyIsFinal::Yes` in
`countCommitAtomics`).

Why per block? Threads of a block can synchronise with `__syncthreads()`, so the block knows
when its copy is complete. The commit still uses atomics because many blocks add into copy 0
concurrently. The price: `gridDim.x * NUM_BINS` counters in global memory that must be zeroed
before every launch: with 1024 blocks and 7 bins about 28 KB, with the 256 K blocks of the GPU
run about 7 MB, and with 4096 bins instead of 7 it would be 4 GB.

### Kernel 3 — the private copy in shared memory

The private copy is only used by the threads of one block, and that is exactly the scope of
shared memory:

```cpp
  __shared__ unsigned histo_s[NUM_BINS];
  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  ...  atomicAdd(&histo_s[b], 1u);  ...
  __syncthreads();

  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned v = histo_s[bin];
    if (v > 0) {
      atomicAdd(&histo[bin], v);
    }
  }
```

The first barrier makes sure all bins are zero before the first increment; the second, that
all increments of the block are done before the commit. Shared-memory atomics are executed inside the SM with a latency of tens of cycles instead of
hundreds, every SM works on its own copies in parallel, and no global space is needed. The two
barriers are both essential: without the first, a thread may increment a bin that another
thread has not zeroed yet (its `0` would overwrite the count); without the second, a thread may
commit a bin while others are still adding to it.

**When the bins do not fit.** Shared memory per block is tens of KB, so a histogram of a few
thousand 32-bit bins fits; one with millions of bins does not. Options: privatize only the
"hot" bins, use 16-bit private counters with an overflow flush, privatize per group of
blocks in global memory (kernel 2), or skip privatization, because with many bins and spread
data the contention it removes was small anyway.

## 7. Thread coarsening

Kernel 3 still launches one thread per character, so with 64 M characters it creates 256 K
blocks, and each block zeroes and commits its own copy. If the GPU can only run, say, 2 560
blocks at a time, most of that overhead buys nothing. **Coarsening** launches fewer threads and
gives each one many characters: fewer blocks, fewer private copies, fewer commits. There are two
ways to split the work.

**Contiguous** (kernel 4): thread `t` takes `CFACTOR` consecutive characters.

**Interleaved** (kernel 5): thread `t` takes characters `t, t+S, t+2S, ...` with `S` the total
number of threads.

```
contiguous, CFACTOR = 4         interleaved, S = 4 threads
chars: 0 1 2 3 4 5 6 7 ...      chars: 0 1 2 3 4 5 6 7 ...
       T0T0T0T0T1T1T1T1                T0T1T2T3T0T1T2T3
step 0: T0 reads 0, T1 reads 4  step 0: T0 reads 0, T1 reads 1   <- neighbours read neighbours
```

On a CPU, contiguous is the natural choice (each thread streams through its own chunk). On a
GPU it is the wrong one: the 32 threads of a warp execute the same load at the same time, and
with contiguous partitioning their addresses are `CFACTOR` bytes apart, so one load instruction
touches 32 different memory segments. With interleaved partitioning the warp reads 32
consecutive bytes, one segment: the access is **coalesced**. Same work, same atomics, much less
memory traffic. Note also that the commit count of the coarsened kernels is at most
`NUM_BINS * blocks`: 224 atomics instead of 7 156 in the run below.

## 8. Aggregation

Some data has long runs of the same value: the sky in a photo, the background of a scanned
document, events that are sorted by time and binned by hour. A thread that sees the same bin
again and again can count in a **register** and issue a single atomic when the run ends:

```cpp
    if (b >= 0) {
      if (b == prevBin) {
        ++accumulator;
      } else {
        if (accumulator > 0) {
          atomicAdd(&histo_s[prevBin], accumulator);
        }
        accumulator = 1;
        prevBin = b;
      }
    }
  ...
  if (accumulator > 0) {
    atomicAdd(&histo_s[prevBin], accumulator);
  }
```

While the bin stays the same (`b == prevBin`) the thread only increments `accumulator`, with no
atomic. When the bin changes, it flushes the finished run with one atomic and starts a new run
of length 1. After the loop, the last run is still in the register, so one more atomic flushes
it.

It helps exactly where contention is worst (one dominant bin) and costs a compare and some
divergence when the data has no runs. Remember that with interleaved partitioning a thread's
consecutive elements are `S` positions apart in the input, so what matters is whether the data
is uniform at that distance.

## 9. Performance analysis

The program prints, for every kernel, the time and two numbers that do not depend on the
machine: how many atomics each kernel issues on global memory and on shared memory (counted
on the host by replaying the partitioning). Output of the CPU emulation (256 K characters,
a 10-core laptop; the milliseconds change from run to run, the atomic counts do not):

```
=== input: English-like text, 262144 characters, 219085 letters ===
kernel                         blocks        ms  global atomics  shared atomics  check
CPU sequential                      -     0.375               -               -  reference
1 basic (global atomics)         1024   294.989          219085               0  PASS
2 private copies, global         1024   631.463          226234               0  PASS
3 private copy, shared           1024   910.484            7156          219085  PASS
4 + coarsening contiguous          32    30.415             224          219085  PASS
5 + coarsening interleaved         32    58.989             224          219085  PASS
6 + aggregation                    32    35.632             224          181168  PASS

=== input: sky (95% in one bin), 262144 characters, 262144 letters ===
kernel                         blocks        ms  global atomics  shared atomics  check
CPU sequential                      -     0.556               -               -  reference
1 basic (global atomics)         1024   301.433          262144               0  PASS
2 private copies, global         1024   570.186          267665               0  PASS
3 private copy, shared           1024   875.304            5526          262144  PASS
4 + coarsening contiguous          32    37.206             224          262144  PASS
5 + coarsening interleaved         32    40.270             224          262144  PASS
6 + aggregation                    32    29.991             224           28830  PASS
```

How to read it:

- **Atomic counts are the real lesson.** The basic kernel sends every letter to global
  memory (219 K contended atomics on 7 addresses). Shared privatization moves them all to shared
  memory and leaves 7 156 global ones; coarsening cuts the global ones to 224. On the sky input
  aggregation divides the shared atomics by 9 (262 K to 29 K); on the English text it saves
  only 17%, because at distance `S` neighbouring letters are unrelated.
- **The emulation's milliseconds measure something else.** On the CPU, a block costs a
  round of thread wake-ups and barriers, so the time is roughly `blocks x barriers`. That is
  why kernels 2 and 3 (two barriers per block) are slower than kernel 1 there, and why
  coarsening (32 blocks instead of 1024) wins by 10-30x. Contiguous versus interleaved makes no
  systematic difference on a CPU, which has no warps and no coalescing.
- **On a GPU** (run it on Colab) expect: kernel 1 slowest, much slower on the sky input than
  on the text (contention); kernel 2 better but paying for zeroing `blocks x 7` counters;
  kernel 3 clearly better; kernel 5 faster than kernel 4 (coalescing); kernel 6 the fastest on
  the sky input and about equal to kernel 5 on the text. The input is 64 MB; at a few hundred
  GB/s just reading it takes a fraction of a millisecond, which is the "speed of light" against
  which to judge the best kernel.

## 10. The book's version: an image, `cuda::atomic_ref` and thread-level privatization

The current edition of the book writes the chapter on a **grayscale image** (one byte per
pixel, 256 bins, one per intensity) and expresses every atomic with the libcu++ class
`cuda::atomic_ref` instead of `atomicAdd`. [`histogram_image.cu`](histogram_image.cu) has every
kernel in that form, checked against the sequential C function, on two images: uniform noise
(little contention) and a tree under a bright sky (60 % of the pixels in one bin).

`cuda::atomic_ref` comes from `<cuda/atomic>` (libcu++, CUDA 11.6 or later). `cuda_compat.h`
includes that header under `nvcc` and provides the same class on the CPU, so the programs do
not include it themselves.

```cpp
unsigned char b = image[i];
cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
bins_ref.fetch_add(1, cuda::memory_order_relaxed);
```

- **The reference.** `atomic_ref<T, scope>` wraps an ordinary variable (`bins[b]`) so that its
  member functions (`fetch_add`, `fetch_min`, `fetch_max`, `exchange`, `compare_exchange_*`,
  ...) are atomic. Building it costs nothing at run time; it only tells the compiler which
  instruction to emit. `fetch_add` returns the old value, which a histogram ignores (the scan and
  filter chapters use it).
- **The scope** says which threads must see the operation as indivisible:
  `thread_scope_device` for the public bins (every block may update them),
  `thread_scope_block` for a private copy that only one block touches (in global memory, as in
  `histogram_private_global`, or in shared memory). A narrower scope can be cheaper on hardware
  that implements it closer to the SM. `atomicAdd` is always device scope.
- **The memory order** says how the atomic may be reordered with the thread's *other* memory
  accesses. The only other access is the load of `image[i]`, and the atomic already depends on
  it through `b`, so `memory_order_relaxed` (no extra ordering) is enough. Code that publishes
  data through an atomic (a flag that says "my result is ready") needs `release`/`acquire`, as in
  the single-pass scan of chapter 11.

Kernels 1–5 of `histogram_image.cu` are the chapter's listings (basic, private copies in global
memory with `bins_pool`, private copy in shared memory, contiguous and interleaved coarsening).
The two coarsened kernels differ in one line. In `histogram_coarse_contig`:

```cpp
unsigned int i = segment + threadIdx.x * COARSE_FACTOR + c;
```

so in iteration `c` the 32 threads of a warp read 32 addresses `COARSE_FACTOR` bytes apart. In
`histogram_coarse_interl`:

```cpp
unsigned int i = segment + c * blockDim.x + threadIdx.x;
```

so the warp reads 32 consecutive bytes, one coalesced transaction. In `histogram_private_global`
each block's private copy is `bins_priv`, its slice of `bins_pool`; the host zeroes the whole
pool before every launch.

**Thread-level privatization** (kernels 6 and 7) is the next level after the block. Once a thread
processes many pixels it can keep a histogram of its own, updated with plain increments (no other
thread sees it), and commit it to the block's copy at the end: per pixel, a register increment
instead of a shared-memory atomic.

- *Few bins:* kernel 6 (`histogram_thread_private`) counts `NUM_BINS_SMALL` = 16 intervals of
  `INTENSITIES_PER_SMALL_BIN` = 16 intensities, in a per-thread array `bins_thread`. A 16-entry
  array per thread fits in registers (or, if the compiler cannot keep it there, in the thread's
  private local memory, which is still private and cached). The thread counts its pixels with
  plain increments, since no other thread sees this copy, and then commits it: thread → block (at
  most 16 block-scope atomics per thread) → public copy.
- *Many bins:* 256 counters per thread would exhaust the registers, so kernel 7 is the book's
  answer: each thread privatizes **one** bin, the most recently accessed one, as a counter
  `bin_r` in a register. The first pixel is loaded before the loop (after a boundary check) and
  `bin_r` starts at 1; for every next pixel `bNext`, equal to the private bin means `++bin_r`,
  different means one block-scope atomic that adds `bin_r` to the block's bin, then switching to
  `bNext` with `bin_r = 1`. The update is always one element behind, so one more atomic after the
  loop flushes the last run. It needs more statements and variables, so on data with little
  similarity it can be slower than kernel 5; on data with large uniform regions (the sky) it
  removes most of the shared-memory atomics. Its `if` may diverge, but with no similarity or heavy
  similarity all threads take the same path. This is the aggregation idea of section 8.

[`race_demo.cu`](race_demo.cu) shows why any of this is needed. Every thread reads one pixel of
an image with only 4 distinct values (trunk, leaves, grass, sky), so thousands of threads update
the same 4 bins. `racy_kernel` writes `++bins[b]` out as its three steps: `readFromMemory`
(Old ← `bins[b]`), `newValue = oldValue + 1`, `writeToMemory` (`bins[b]` ← New). Both helpers
take a `volatile` pointer, which forbids the compiler from merging the read and the write into
one instruction or keeping the value in a register, so the three steps really are separate
memory operations. Two threads that read the same Old both write Old + 1: one increment is lost.
`atomic_kernel` is the `atomic_ref` version, which is exact.

The racy kernel is **wrong on purpose**: it is a data race, undefined behaviour in C++ and in
CUDA, and ThreadSanitizer reports it, as it should. It exists only to make the lost updates
visible. On a GPU the racy kernel loses most of the increments; on the CPU emulation it loses a
handful, the number changes from run to run, and it can even be 0 on a lucky run.

## 11. Pitfalls

- **Forgetting the atomic** (`histo[b]++`): compiles, runs, produces plausible but wrong counts.
  Always compare with a sequential reference, as every program here does.
- **Missing one of the two barriers** around a shared-memory private copy (zeroing races with
  counting, or committing races with counting).
- **Not zeroing** the histogram (or all private copies) before each launch. Atomics *add* to
  whatever is there.
- **Shared memory too small** for the bins: the kernel fails to launch, or, with a dynamic size
  computed wrong, writes out of bounds.
- **Counter overflow**: 32-bit counters overflow after 4 G increments; use 64-bit bins for
  very large inputs (CUDA has `atomicAdd` on `unsigned long long`).
- **Contiguous coarsening on a GPU**: the textbook CPU partitioning, uncoalesced on a GPU.
- **Bin index out of range**: a character outside `a..z` must be filtered out *before* indexing,
  or the kernel corrupts neighbouring memory.
- **Reading a bin with a plain load while others still update it** is a data race; only read
  after a barrier (inside a block) or after the kernel ends (across blocks).

## 12. Ideas for your project

### The bike-sharing examples in this folder

Two programs apply the chapter to a city bike-sharing system: 48 docking stations on a
6 km x 6 km map, bikes that are undocked at one station and docked at another, and a fare that
includes the first 45 minutes of each trip. Both generate their data with a fixed seed and
SplitMix64 rather than the `<random>` distributions, whose output differs between libstdc++,
libc++ and `nvcc`'s host compiler: every machine sees the same trips.

**`bikeshare_trip_durations.cu`** counts trip durations in 1-minute buckets. `NUM_BUCKETS` is
61: one bucket per minute below `MINUTES_WITH_OWN_BUCKET` (60), plus `OVERFLOW_BUCKET` for an
hour or more. A synthetic trip is the time to unlock the bike, the ride itself (600 m to
4.5 km, faster on an e-bike), traffic lights and hills, sometimes a search for a free dock when
the end station is full, and for 3 % of the trips a leisure detour of 10 to 40 minutes. The
constants (`RIDE_DISTANCE_M`, `E_BIKE_PERCENT`, `LEISURE_RIDE_PERCENT`, ...) are at the top of
the host code. Most trips last 7 to 17 minutes, so about ten buckets receive most of the
updates: the contended case, where privatization pays. The program compares the basic kernel,
one global atomic per trip, with `tripHistoPrivate`, which keeps a copy of the 61 buckets in
shared memory per block and coarsens with interleaved partitioning (32 trips per thread, with
coalesced reads), so that each block commits at most 61 global atomics. From the histogram the
host derives the median, p90 and p99 and the share of trips within the included 45 minutes,
with a running sum over the buckets (a prefix sum, chapter 11): a percentile is the first
bucket where the running sum reaches that fraction of all trips. A trip in bucket 45 or above
counts as late. Output of the emulation:

```
Bike-share trip durations (chapter 9 use case), backend: CPU emulation (threads)
262144 synthetic trips (seed 42)

kernel                         blocks         ms  check
CPU sequential                      -      0.203  reference
basic, global atomics            1024    611.358  PASS
shared private + coarsened         32     62.124  PASS
...
median 12 min, p90 19 min, p99 42 min
within the included 45 min: 99.2%, late (45 min or more): 2011 trips
ALL PASS
```

**`bikeshare_station_hour.cu`** answers two questions about a synthetic week of rentals: how
many rentals start in each hour of the day (histogram A, 24 bins), and how many start at each
station in each hour (histogram B, 48 x 24 = 1 152 bins, with the combined index
`station * HOURS + hour`). The rentals arrive as a Poisson process whose rate follows the
commuter day (`RELATIVE_RENTAL_RATE_BY_HOUR`: almost nothing at night, peaks around 08:00 and
18:00), and stations near the city centre are more popular than those at the edge. The rentals
are in time order, so consecutive rentals fall in the same hour, which is exactly when
aggregation helps: a thread keeps a run counter and issues one shared atomic per run of equal
hours instead of one per rental. With interleaved partitioning the rentals a thread sees are a
whole grid apart and the runs are short; with contiguous partitioning a thread sees 64
consecutive rentals and the runs are long (8 350 atomics instead of 506 787 for 512 K
rentals). `countAggregatedAtomics` replays each partition on the host to count those atomics.
Histogram B is privatized whole in shared memory (4.5 KB per block) without aggregation, since
consecutive rentals rarely start at the same station. The report prints rentals per hour,
rentals per station laid out as on the map, and the busiest station-hour, and checks that the
column sums of B equal A:

```
kernel                              blocks         ms  shared atomics  check
CPU sequential (A and B)                 -      1.136               -  reference
A per hour, interleaved + aggr.         32     52.437          506787  PASS
A per hour, contiguous + aggr.          32     57.700            8350  PASS
B station x hour, shared priv.          32     61.364          524288  PASS
...
busiest station-hour: s27 at 08:00 with 2242 rentals
columns of B equal A: yes
ALL PASS
```

### Where this pattern may appear in your project

- When many threads count events into the same few categories, one shared counter (locked or
  atomic) becomes the bottleneck. Could each thread count into its own small array and add
  the arrays once at the end? That is privatization, and it works with `std::thread` as well
  as with blocks.
- A histogram of how long something takes, bucketed by minute, gives you the median and the
  percentiles cheaply, with a running sum, without sorting or keeping every measurement.
- If the data arrives in time order, consecutive items often fall in the same bucket. Is it
  worth counting a run locally before touching the shared counter?
- Counting by two keys at once, for example per restaurant and per hour, is still one
  histogram with a combined bin index, and its row and column sums give you a cross-check
  for free.
- Which questions about a run of your simulation are really "how many per category"?
  Rejections per reason, events per type, orders per minute, couriers busy at each moment?

## 13. Files and how to run

| File | Contents |
|------|----------|
| `01_*.cu` … `07_*.cu` | the learning path above, one idea per step on the 8 x 8 tree, each printing the image, the bins (and its intermediate copies) and PASS/FAIL |
| `histogram.cu` | the six kernels of the chapter (`atomicAdd`, text input), two inputs, CPU reference, atomic counts, timing table |
| `histogram_image.cu` | the book's listings on a 256-bin grayscale image with `cuda::atomic_ref`, plus thread-level privatization (16 bins per thread; the most recent bin in a register) |
| `race_demo.cu` | the lost-update race of a plain read-modify-write, and the atomic that fixes it |
| `bikeshare_trip_durations.cu` | bike-share trip durations in 1-minute buckets, with percentiles and the share within the included 45 minutes |
| `bikeshare_station_hour.cu` | bike rentals per hour (aggregation, two partitionings) and per station x hour |

On a laptop (no GPU; `g++` works the same), the steps first:

```sh
cd ec2/07_histogram
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_histogram_sequential.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_race_and_atomic.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_private_global.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_private_shared.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_coarsening.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_most_recent_bin.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_thread_private_bins.cu -o step && ./step
```

Then the complete programs and the bike-sharing examples:

```sh
cd ec2/07_histogram
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common histogram.cu -o histogram && ./histogram
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common histogram_image.cu -o histogram_image && ./histogram_image
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common race_demo.cu -o race_demo && ./race_demo
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_trip_durations.cu -o trip_durations && ./trip_durations
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_station_hour.cu -o station_hour && ./station_hour
```

> **Easiest way on Colab:** this folder has two notebooks. [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) runs the steps of the learning path in order, and [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) runs the complete programs and the bike-sharing examples. Upload one to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

On Google Colab (Runtime > Change runtime type > GPU), in a cell:

```sh
!cd ec2/07_histogram && nvcc -O2 -arch=native -I../common 02_race_and_atomic.cu -o step && ./step
!cd ec2/07_histogram && nvcc -O2 -I../common histogram.cu -o histogram && ./histogram
!cd ec2/07_histogram && nvcc -O2 -arch=native -I../common histogram_image.cu -o histogram_image && ./histogram_image
!cd ec2/07_histogram && nvcc -O2 -arch=native -I../common race_demo.cu -o race_demo && ./race_demo
!cd ec2/07_histogram && nvcc -O2 -I../common bikeshare_trip_durations.cu -o trip_durations && ./trip_durations
!cd ec2/07_histogram && nvcc -O2 -I../common bikeshare_station_hour.cu -o station_hour && ./station_hour
```

The problem size is chosen with `BY_BACKEND(gpu, cpu)`: 64 M characters on the GPU, 256 K in
the emulation. To run under ThreadSanitizer, shrink it:
`clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common -DCPU_CHARS=20000 histogram.cu`
(the bike-sharing programs take `-DCPU_TRIPS=...` and `-DCPU_RENTALS=...`). `histogram_image.cu`,
`race_demo.cu` and the two bike-sharing programs are small enough in the emulation to run under
ThreadSanitizer as they are; it reports no race except the intentional one in `racy_kernel`.
The step files are tiny and run under ThreadSanitizer as they are; it reports races only in
the two intentionally racy kernels of `02_race_and_atomic.cu`.

## 14. Exercises

1. **Lost updates.** Four threads execute `histo[2]++` (not atomic) once each, starting from
   `histo[2] == 10`. Which final values are possible? Draw an interleaving for the smallest.
2. **Change the bins.** Modify `histogram.cu` to use bins of 3 letters. How many bins are
   there, which constants and loops must change, and does any kernel need a new barrier?
3. **Atomic throughput.** Suppose a global atomic has a latency of 500 cycles on a 1.5 GHz GPU
   and atomics on different addresses never interfere. What is the maximum rate (letters per
   second) of kernel 1 on (a) English text spread over 7 bins, (b) the sky input? What does
   this predict for 64 M characters?
4. **Memory of privatization.** For 1 M characters, 256 threads per block and 256 bins, how
   many bytes do the private copies of kernel 2 need? How many commit atomics does kernel 3
   issue at most? And kernel 5 with `CFACTOR = 64`?
5. **Coalescing.** With `CFACTOR = 32` and a warp of 32 threads, list the addresses read by
   threads 0, 1, 2 and 31 in the first iteration of kernels 4 and 5. How many 32-byte memory
   segments does each version touch per load instruction?
6. **Aggregation by hand.** One thread sees the bins `0 0 0 2 2 0 0 0 1 1`. How many shared
   atomics does kernel 5 issue, and how many does kernel 6 issue? Give an input of the same
   length where kernel 6 issues as many atomics as kernel 5.
7. **Big histograms.** You need a histogram of the 65 536 possible values of a 16-bit sensor.
   It does not fit in shared memory as 32-bit counters. Propose two designs and say on which
   input distribution each one is better.
8. **Bike sharing.** Change `bikeshare_trip_durations.cu` to 30-second buckets up to one
   hour. How many buckets, how much shared memory per block, and what must change in the
   report? Then add a second histogram, of trips per bike type (pedal or e-bike), filled in
   the same kernel launch.

**From the book (section 9.8).**

9. Each atomic operation in a DRAM system has a total latency of 100 ns. What is the maximal
   throughput of atomic operations on the same global memory variable?
10. A processor supports atomics in L2: 4 ns in L2, 100 ns in DRAM, and 90 % of the atomics hit
    in L2. What is the approximate throughput of atomics on the same global memory variable?
11. In exercise 9, a kernel performs 5 floating-point operations per atomic operation. What is
    its maximal floating-point throughput, as limited by the atomics?
12. In exercise 9, the variable is privatized into shared memory with a 1 ns access latency;
    every global atomic becomes a shared one, and the extra global atomics that merge the
    private copies add 10 % to the execution time. With 5 floating-point operations per atomic,
    what is the maximal floating-point throughput?
13. A histogram kernel processes 524 288 elements into 128 bins with 1024 threads per block.
    How many atomics on global memory does (a) the basic kernel perform, (b) the shared-memory
    privatized kernel without coarsening perform at most, (c) the same with a coarsening factor
    of 4 perform at most? Check (b) and (c) by changing the sizes in `histogram_image.cu`.

## 15. Solutions (short)

1. Any value from 11 to 14. 11 happens when all four read 10 before anyone writes; each then
   writes 11.
2. `ceil(26/3) = 9` bins (the last holds only `y-z`). Change `LETTERS_PER_BIN` to 3: `NUM_BINS`
   is derived from it, and `binOf` and `histogram_sequential` divide by it. Only the bar labels
   (`BIN_LABELS`) must be rewritten by hand. The loops that zero and commit already iterate up to
   `NUM_BINS`; no new barrier is needed.
3. One address absorbs `1.5e9 / 500 = 3 M` atomics per second. (a) Upper bound about
   `7 x 3 M = 21 M` letters/s if the load were even, in practice less because two bins dominate;
   (b) about 3 M/s, since almost everything goes to one bin. 64 M characters: at least 3 s on
   the text and about 20 s on the sky, slower than a CPU loop. Hence privatization.
4. `ceil(1M/256) = 4096` blocks x 256 bins x 4 bytes = 4 MB. Kernel 3 commits at most
   `4096 x 256 = 1 M` atomics (only non-zero bins). Kernel 5 has `ceil(1M/(256x64)) = 64` blocks,
   so at most `64 x 256 = 16 384`.
5. Kernel 4 (contiguous): thread `t` reads `32t`, i.e. 0, 32, 64, ..., 992: 32 different
   segments. Kernel 5 (interleaved): thread `t` reads `t`, i.e. 0..31: one segment.
6. Kernel 5: 10 atomics (one per element). Kernel 6: 4 runs, so 4 atomics. Alternating bins,
   e.g. `0 1 0 1 0 1 0 1 0 1`, gives 10 with both.
7. (a) Global atomics without privatization: 65 536 addresses spread the contention, fine for
   roughly uniform data. (b) Privatize only the hottest few hundred values in shared memory and
   send the rest to global memory, or use 16-bit shared counters flushed when they approach
   overflow; better for very skewed data (most readings near one value). Also possible: sort
   the data and count run lengths.
8. 120 buckets of 30 s plus the overflow bucket, `121 x 4 = 484` bytes of shared memory. The
   report counts in buckets, not minutes: the percentiles must be divided by 2, and the
   included time is now bucket 90. For the bike types, add a second shared array of 2
   counters, zero it in the same loop, count in it, and commit it after the same second
   barrier; the generator must also return the type of each trip's bike (today it only uses
   it to pick the speed) as a second input array.

9. Atomics on one variable are serialized: 1 / 100 ns = **10 M atomic operations per second**.
10. Average latency 0.9 × 4 + 0.1 × 100 = 13.6 ns, so about **73.5 M atomics per second**.
11. 5 × 10 M = **50 MFLOP/s**.
12. 1 / 1 ns = 1 G shared atomics per second; the merge adds 10 % to the time, so
    1 G / 1.1 ≈ 0.91 G atomics per second × 5 ≈ **4.5 GFLOP/s**.
13. (a) One per element: **524 288**. (b) 524 288 / 1024 = 512 blocks, each commits at most 128
    bins: **≤ 65 536**. (c) 128 blocks × 128 bins: **≤ 16 384**.
