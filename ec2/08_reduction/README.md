# Chapter 10 — Reduction

Based on chapter 10 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj).
Every program in this folder is a single `.cu` file that runs on an NVIDIA GPU (Google Colab)
**and** on any laptop without a GPU, through the CPU emulation in
[`../common/cuda_compat.h`](../common/cuda_compat.h).

The slides are session 18 (`18_reduction`). The numbered step files below follow them with the
book's kernel names, one idea at a time, on 16 to 256 small whole numbers stored as `float`
(so every order of the additions gives the exact same sum). The first kernels print the array
after every iteration. A kernel never prints in the book, so each step keeps the book's kernel
unchanged and adds a copy with `Printing` in its name that does the printing; both results are
checked against `sum_sequential`.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_sequential_reduction.cu`](01_sequential_reduction.cu) | `sum_sequential` and the general form `reduce_sequential` with an operator and its identity (0 for the sum, minus infinity for the maximum) on 16 values; prints the accumulator after every element. (slide: *What you already know: operator and identity*) | The sum accumulator changes at every one of the 16 steps and ends at 62; the max accumulator changes only when a larger value appears and ends at 9. |
| 2 | [`02_simple_kernel.cu`](02_simple_kernel.cu) | `SimpleSumReductionKernel`: one block of 8 threads for 16 elements, thread t owns `input[2t]`, the stride doubles 1, 2, 4, 8. `SimpleSumReductionKernelPrinting` prints the array after every stride; its second barrier is there only for the printing. (slide: *Code: the simple kernel*) | After stride 1 the pair sums sit at the even positions, after stride 2 at the multiples of 4, and so on; the total 62 ends in `input[0]` and the odd positions never change. |
| 3 | [`03_convergent_kernel.cu`](03_convergent_kernel.cu) | `ConvergentSumReductionKernel`: thread t owns `input[t]` and the stride halves 8, 4, 2, 1, so the active threads stay consecutive. Printed after every stride, as in step 2. (slide: *Code: the convergent kernel*) | The partial sums stay packed on the left: positions 0 to 7 after stride 8, 0 to 3 after stride 4; `input[0]` ends at 62, as in step 2. |
| 4 | [`04_shared_memory_kernel.cu`](04_shared_memory_kernel.cu) | `SharedMemorySumReductionKernel`: the first addition happens while loading into `input_s`, the partial sums stay in shared memory, and the input is not modified. The printing copy shows `input_s` after the load and after every stride. (slide: *Code: partial sums in shared memory*) | `input_s` holds only 8 values because the load already added the two halves; the stride rows match step 3, but the input array is not modified. |
| 5 | [`05_warp_reduce.cu`](05_warp_reduce.cu) | `warp_reduce` with `__shfl_down_sync` on one warp of 32 values: five shuffles, no shared memory, no barrier. The printing copy shows every lane after each stride; only lane 0 ends with the total, because a lane whose partner lane i + stride is outside the warp adds its own value. (slide: *Warp-level primitives: `__shfl_down_sync`*) | After each stride only the lower lanes hold useful partial sums; lane 0 ends at 125 and the other lanes hold values that nobody uses. |
| 6 | [`06_warp_level_kernel.cu`](06_warp_level_kernel.cu) | `WarpLevelSumReductionKernel`: the shared-memory tree stops at stride 32 and warp 0 finishes with `warp_reduce`. 64 threads and 128 elements; the printing copy shows the 32 partial sums that warp 0 receives. (slide: *Code: the warp-level kernel*) | The 32 values of `input_s` add up to the final 570. |
| 7 | [`07_two_stage_warp_kernel.cu`](07_two_stage_warp_kernel.cu) | `TwoStageWarpLevelSumReductionKernel`: every warp reduces its 64 elements in registers, lane 0 stores the warp sum in `partialSums_s`, one barrier, then warp 0 reduces the warp sums. 128 threads (4 warps) and 256 elements; with fewer than 1024 threads the second stage needs the guard from the slides (`t < NUM_WARPS`, else 0). (slide: *Code: two-stage warp-wide reduction*) | The four warp sums of stage 1 add up to the stage 2 result, 1148. |
| 8 | [`08_multi_block_atomic.cu`](08_multi_block_atomic.cu) | `reduce_kernel`: 4 blocks, each reduces its segment of 2 x BLOCK_DIM elements and its thread 0 adds the block sum to `*output` with a device-scope `cuda::atomic_ref` `fetch_add`. The printing copy shows each block sum and the old value `fetch_add` returned, which depends on the order in which the blocks arrive. (slide: *Code: multi-block reduction with `atomic_ref`*) | The four block sums add up to 1148, and each `fetch_add` returns the running total before that block (on a GPU the blocks may arrive in any order). |
| 9 | [`09_coarsened_kernel.cu`](09_coarsened_kernel.cu) | `CoarsenedSumReductionKernel` with `COARSE_FACTOR` = 2: each thread first adds 4 elements BLOCK_DIM apart without any barrier, so the same 256 elements need 2 blocks instead of 4. Printed as in step 8. (slide: *Code: the coarsened kernel*) | Two blocks of 128 elements each instead of four of 64, and the same total, 1148. |
| 10 | [`10_reversed_kernel.cu`](10_reversed_kernel.cu) | Exercise 18.3, `ReversedSumReductionKernel`: thread t owns `input[8 + t]`, the stride halves 8, 4, 2, 1 and points left, and the active threads are the last `stride` ones, so they stay consecutive. Same input as step 3, printed after every stride. (slide: *Solutions 18.1, 18.2, 18.3, 18.6*) | The mirror image of step 3: the partial sums stay packed on the right (positions 8 to 15 after stride 8, 12 to 15 after stride 4), the left half never changes, and `input[15]` ends at 62. |
| 11 | [`11_coarsened_max_kernel.cu`](11_coarsened_max_kernel.cu) | Exercise 18.4, `CoarsenedMaxReductionKernel`: the coarsened kernel with `fmaxf` instead of +, `warp_max`, the identity minus infinity (the host sets `*output = -INFINITY`, the input is all negative) and an atomic maximum for `float` written as a `compare_exchange_strong` loop. The printing copy shows each block's maximum and the value the loop found in `*output`. (slide: *Solutions 18.4 and 18.5*) | Block 0's maximum, -45, replaces -inf, and block 1's, -44, replaces -45; on a GPU the blocks may arrive in the other order, and then the second one finds a larger value and keeps it. The result is -44 either way. |
| 12 | [`12_any_length_kernel.cu`](12_any_length_kernel.cu) | Exercise 18.5, `CoarsenedSumReductionKernel` with a parameter `N`: every load is guarded by `i + c*BLOCK_DIM < N` and starts from the identity 0, the grid is `cdiv(N, COARSE_FACTOR*2*BLOCK_DIM)`, and no thread returns early, so the shuffles keep their full mask. It runs N = 256, 200, 129, 100 and 1 on buffers of exactly N elements. (slide: *Solutions 18.4 and 18.5*) | The last block reads only the elements that exist (72 of 128 for N = 200, 1 of 128 for N = 129), and every N matches the sequential sum. |

After the steps come the complete programs: `reduction_book.cu` runs the book's kernels on
64 K to 32 M elements with the exercises and the resource counts, `reduction.cu` compares every
variant with timings, and the bike-sharing examples (`bikeshare_daily_summary.cu`,
`bikeshare_nearest_truck.cu`, `bikeshare_kpis.cu`, `bikeshare_bike_totals.cu`) use reductions
in an application.

## 1. The pattern in one paragraph

A reduction turns an array into a single value by combining all its elements with one binary
operator: sum, product, minimum, maximum, logical and/or, a count (a sum of 0/1 values), an
argmin (a minimum that remembers where it was). The operator must be **associative**, so the
combinations can be regrouped, and it needs an **identity** value (0 for +, 1 for x, -inf for
max, +inf for min) to start from and to pad with. If it is also **commutative** the elements
may be combined in any order, which the fastest kernels rely on. Reductions are everywhere:
the total of a bill, the error of a numerical method, the best move of a search, the summary
line printed at the end of a simulation. The chapter is about computing one with thousands of
threads while keeping them all busy, and it is the best place to see control divergence,
memory coalescing, shared memory, atomics and thread coarsening working together.

## 2. The sequential baseline

```cpp
  for (float value : data.large) {
    references.largeSum += value;
  }
```

The accumulator starts at the identity of `+` (0) and every addition waits for the previous
one. O(n) work, and it is already **work-efficient**: no algorithm can do fewer than `n - 1`
combinations. The problem is the chain of dependences: each addition needs the result of the
previous one, so the loop is inherently sequential. A parallel reduction must break that
chain, and it can, because associativity lets us regroup:

```
((((((x0 + x1) + x2) + x3) + x4) + x5) + x6) + x7      7 steps, one after the other
 ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))     3 steps, 4 + 2 + 1 additions
```

## 3. The reduction tree

The second line is a **reduction tree**. With `n` inputs:

```
step 1:   x0  x1  x2  x3  x4  x5  x6  x7        n/2 additions, in parallel
            \/      \/      \/      \/
step 2:    s01     s23     s45     s67          n/4 additions
              \    /          \    /
step 3:       s0123           s4567             ...
                   \         /
                      total                     log2(n) steps, n - 1 additions in all
```

- **Work**: `n/2 + n/4 + ... + 1 = n - 1`, the same as the sequential loop. The tree is
  work-efficient: parallelism costs no extra operations.
- **Steps**: `log2(n)`. A million elements need 20 steps instead of a million.
- **Parallelism** halves at every step: `n/2` threads are useful in the first step, one in the
  last. The *average* parallelism is `(n - 1) / log2(n)`, about 50 000 for a million elements;
  plenty for a GPU, but it means many threads idle in the late steps. Every kernel below is a
  different answer to "how do we keep those idle threads from costing us".

## 4. Kernel 1 — the simple kernel

One block of 1024 threads reduces 2048 elements in place. Thread `t` owns position `2t`. In the
step with stride `s`, the threads whose index is a multiple of `s` add the element `s` positions
to their right:

```cpp
__global__ void SimpleSumReductionKernel(float* input, float* output) {
  unsigned i = 2 * threadIdx.x;
  for (unsigned stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}
```

```
positions:  0   1   2   3   4   5   6   7
stride 1:   T0 <-+  T1 <-+  T2 <-+  T3 <-+      threads 0,1,2,3 active
stride 2:   T0 <-----+      T2 <-----+          threads 0,2   (1 and 3 idle)
stride 4:   T0 <-------------+                  thread 0      (1, 2 and 3 idle)
```

The barrier is required: the partial sums of one step must all be written before the next step
reads them, and in the next step a thread reads a partial sum written by another thread. The kernel is correct but wasteful in two ways.

**Control divergence.** The 32 threads of a warp execute together. A warp in which only some
threads pass the `if` still spends a full step. In this kernel the active threads are spread
out: with stride 16, only threads 0 and 16 of each warp work, yet all 32 warps execute the step.
Until the stride passes 32, *every* warp stays busy doing mostly nothing. The program counts
this on the host (output below): 223 warp-steps for 2047 additions, so only 28.7% of the lanes
the hardware runs do useful work.

**Memory.** Every step reads and writes global memory, and the active threads touch addresses
`2s` apart, so the warps' accesses are spread over many memory segments (not coalesced).

## 5. Kernel 2 — the convergent kernel

Change only the assignment of threads to data: thread `t` owns position `t`, and the stride
starts at `blockDim.x` and halves. The active threads are always the *first* `stride` threads:

```cpp
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
  unsigned i = threadIdx.x;
  for (unsigned stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}
```

```
positions:  0   1   2   3   4   5   6   7
stride 4:   T0  T1  T2  T3 <- add 4..7          threads 0-3 active
stride 2:   T0  T1 <- add 2..3                  threads 0-1 active
stride 1:   T0 <- add 1                         thread 0 active
```

Now a warp is either fully active or fully idle until fewer than 32 threads remain. The same
2047 additions need only 68 warp-steps (94.1% of the lanes useful). The idle warps finish their
step immediately, so the SM can schedule other work. As a bonus, the active threads read
consecutive addresses, so loads and stores are **coalesced**. This kernel needs commutativity
(it adds `x0 + x4`, not `x0 + x1`), which addition and max have.

The divergence table printed by `reduction.cu`:

```
  step | simple: threads / warps    | convergent: threads / warps
     0 |       1024 / 32            |       1024 / 32
     1 |        512 / 32            |        512 / 16
     2 |        256 / 32            |        256 / 8
     3 |        128 / 32            |        128 / 4
     4 |         64 / 32            |         64 / 2
     5 |         32 / 32            |         32 / 1
     6 |         16 / 16            |         16 / 1
   ...
    10 |          1 / 1             |          1 / 1
total additions 2047; warp-steps: simple 223, convergent 68
lane utilisation (additions / (warp-steps*32)): simple 28.7%, convergent 94.1%
```

## 6. Kernel 3 — shared memory

Kernels 1 and 2 write every partial sum to global memory and read it back in the next step.
Only the first step needs global memory: it reads the two halves of the input, and its results
can go to shared memory, where the rest of the tree runs:

```cpp
__global__ void SharedMemorySumReductionKernel(const float* input, float* output) {
  __shared__ float input_s[SINGLE_BLOCK_THREADS];
  unsigned t = threadIdx.x;
  input_s[t] = input[t] + input[t + SINGLE_BLOCK_THREADS];
  for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    *output = input_s[0];
  }
}
```

Global traffic falls from `2 x 2047` loads plus `2047` stores to `2048` loads plus `1` store, and
the input array is no longer destroyed (it is `const`). The barrier moved to the *top* of the
loop: it separates the first write to `input_s` from the first read of the tree, and each tree
step from the next.

## 7. Kernel 4 — hierarchical (segmented) reduction for any length

A block has at most 1024 threads, and only threads of one block can use `__syncthreads()`, so
kernels 1-3 stop at 2048 elements. For longer inputs, split the input into **segments** of
`2 x BLOCK_DIM` elements, let each block reduce its segment with the shared-memory tree, and
combine the block results with one `atomicAdd` per block. Between the first step and the atomic
runs the same tree as in kernel 3:

```cpp
__device__ inline float valueOrSumIdentity(const float* input, unsigned index, unsigned n) {
  return index < n ? input[index] : SUM_IDENTITY;
}
```

```cpp
  unsigned segment = 2 * blockDim.x * blockIdx.x;
  unsigned i = segment + threadIdx.x;
  unsigned t = threadIdx.x;
  float firstHalf = valueOrSumIdentity(input, i, n);
  float secondHalf = valueOrSumIdentity(input, i + BLOCK_DIM, n);
  input_s[t] = firstHalf + secondHalf;
  ...
  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
```

```
input:  [ segment 0 ][ segment 1 ][ segment 2 ] ... [ last, partial ]
          block 0      block 1      block 2           block k   (missing elements = 0)
             |            |            |                 |
             +------------+---- atomicAdd ---------------+----> *output (zeroed before launch)
```

Elements past the end are replaced by the identity (0), so any `n` works. The atomics are few
(one per 512 elements) and spread over time, so contention on `*output` is negligible. The
alternative without atomics is a second kernel launch that reduces the array of block results.

## 8. Kernel 5 — thread coarsening

Kernel 4 launches `n / 512` blocks: 65 536 for 32 M elements. The GPU cannot run them all at
once; the hardware runs as many as fit and the rest wait. Each block spends 9 steps with
barriers, and in most of them most threads are idle. That is paying for parallelism that the
hardware cannot use. **Coarsening** gives each thread more elements to add *sequentially,
in a register*, before the tree starts. After these lines come the same tree and the same single
`atomicAdd` as in kernel 4:

```cpp
template <unsigned COARSE_FACTOR>
__global__ void CoarsenedSumReductionKernel(const float* input, float* output, unsigned n) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned i = segment + threadIdx.x;
  unsigned t = threadIdx.x;
  float partialSum = SUM_IDENTITY;
  for (unsigned tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
    partialSum += valueOrSumIdentity(input, i + tile * BLOCK_DIM, n);
  }
  input_s[t] = partialSum;
  ...
```

During the sequential phase every thread is busy, there are no barriers, and the reads are
coalesced (in each iteration consecutive threads read consecutive addresses, `BLOCK_DIM` apart
between iterations). A small example of why it pays: 4096 elements, 256 threads per block, a
GPU that runs 2 blocks at a time. Without coarsening, 8 blocks of 512 elements, each taking 1
load step and 8 tree steps, run in 4 rounds: `4 x 9 = 36` steps. With factor 4, 2 blocks run in
one round: 8 sequential additions per thread, then 8 tree steps: about 16 steps, and the
hardware was equally full in both cases. Too large a factor launches too few blocks to fill the
GPU and leaves its bandwidth unused; the right value is the one that just fills the machine.

## 9. Other operators: max, min, argmin

Nothing in the kernels depends on `+` except the operator and the identity. Kernel 6 is the
coarsened kernel with `imax`, identity `MAX_IDENTITY = INT_MIN`, and `atomicMax` for the final
combine:

```cpp
  int partialMax = MAX_IDENTITY;
  for (unsigned tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
    partialMax = imax(partialMax, valueOrMaxIdentity(input, i + tile * BLOCK_DIM, n));
  }
```

An **argmin** also needs the position of the minimum. Instead of reducing pairs (value, index)
with a custom operator, pack both in one 64-bit integer, value in the high bits, index in the
low bits: comparing keys compares the values first and breaks ties by the lowest index. Then a
plain min-reduction and the 64-bit `atomicMin` do the job, deterministically
(`bikeshare_nearest_truck.cu`).

## 10. The book's version: warp-level primitives, two stages, `atomic_ref`, coarsening

`reduction_book.cu` holds every kernel of the chapter with the book's names and line structure,
and the book's exercises (section 16). Its kernels are numbered in the book's order, which this
section follows: book kernels 1-3 are kernels 1-3 of sections 4-6, then 4 is `warp_reduce`, 5 the
warp-level kernel, 6 the two-stage kernel, 7 `reduce_kernel` and 8 the coarsened kernel. The book
writes its later kernels for `BLOCK_DIM = 1024` threads, 32 warps of 32, so that is the block size
used. (The book's shared-memory kernel takes only `input` although its last line writes
`*output`; the program adds the parameter.) The new kernels attack what kernel 3 still pays for: a
`__syncthreads()` and shared-memory traffic in every late step of the tree, where few threads work.

**Warp-level primitives.** The threads of one warp can exchange registers directly.
`__shfl_down_sync(mask, var, delta)` returns the `var` of the lane `delta` positions higher (a
lane with no such partner gets its own value back). A warp-wide sum takes 5 shuffles:

```cpp
__device__ unsigned int warpIdx() {
  return threadIdx.x / WARP_SIZE;
}

__device__ unsigned int laneIdx() {
  return threadIdx.x % WARP_SIZE;
}

__device__ __inline__ float warp_reduce(float val) {
  float partialSum = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
  }
  return partialSum;
}
```

```
lane:        0      1     ...   15      16  ...  31
stride 16:   0+16   1+17  ...   15+31   (lanes 16-31 add values nobody will read)
stride 8:    lanes 0-7 add lanes 8-15
stride 4, 2, 1:  ...                    lane 0 now holds the sum of the 32 values
```

The mask `0xffffffff` says that all 32 lanes take part, so all 32 must execute the call: never
put it in a branch that some lanes of the warp skip. Only lane 0's result is the sum. The shuffle
itself synchronizes the lanes, so no shared memory and no barrier are needed between steps.

**Book kernel 5 — warp-level finish (`WarpLevelSumReductionKernel`).** Kernel 3 with the loop
stopping at `stride >= WARP_SIZE`. After the stride-32 step the 32 remaining partial sums are in
warp 0, which loads them into registers and finishes with `warp_reduce`:

```cpp
  for (unsigned int stride = blockDim.x / 2; stride >= WARP_SIZE; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  __syncthreads();
  if (warpIdx() == 0) {
    float partialSum = input_s[t];
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      *output = partialSum;
    }
  }
```

**Book kernel 6 — two-stage warp reduction (`TwoStageWarpLevelSumReductionKernel`).** Use shuffles
from the start: in stage 1 all 32 warps reduce their own values at once; after the only
barrier, in stage 2 one warp reduces the 32 warp sums.

```cpp
  unsigned int t = threadIdx.x;
  float partialSum = input[t] + input[t + BLOCK_DIM];
  partialSum = warp_reduce(partialSum);

  __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float partialSum = partialSums_s[t];
    partialSum = warp_reduce(partialSum);
    if (threadIdx.x == 0) {
      *output = partialSum;
    }
  }
```

```
global -> registers:  32 warps x 32 lanes, each lane adds its 2 inputs
stage 1:              every warp: 5 shuffles -> lane 0 has the warp's sum
shared memory:        32 warp sums, one __syncthreads()
stage 2:              warp 0: 5 shuffles -> thread 0 has the block's sum
```

| one block of 1024 threads | `__syncthreads()` | tree steps in shared memory | shuffle steps |
|---------------------------|-------------------|-----------------------------|---------------|
| kernel 3, shared memory   | 10                | 10                          | 0             |
| book kernel 5, warp-level | 6                 | 5                           | 5             |
| book kernel 6, two-stage  | 1                 | 0 (32 writes, 32 reads)     | 10            |

The price of book kernel 6: in stage 1 every warp is busy, but in each shuffle step half of the lanes
(then three quarters, ...) add values nobody reads; book kernel 5 instead idles whole warps. Once the
input is loaded, the rest of a reduction is limited by latency (barriers, shared-memory round
trips), not by arithmetic, so trading idle lanes for fewer barriers pays off.

**The `BLOCK_DIM == 1024` assumption.** In stage 2 the 32 lanes of warp 0 read
`partialSums_s[t]`, but the array has `BLOCK_DIM / WARP_SIZE` entries: exactly 32 only for 1024
threads. With 256 threads, lanes 8-31 would read past the end. The program asserts the block
size (`static_assert`); its exercise kernels and the bike-share programs use the guarded read, with
the identity for the missing warps, which is correct for any `BLOCK_DIM` up to 1024:

```cpp
__device__ bool laneHasWarpResult() {
  return laneIdx() < BLOCK_DIM / WARP_SIZE;
}
```

```cpp
    float warpSum = laneHasWarpResult() ? partialSums_s[laneIdx()] : SUM_IDENTITY;
```

**Book kernel 7 — any number of blocks (`reduce_kernel`).** Each block reduces its segment of
`2 * blockDim.x` elements exactly like book kernel 6 (the `...` below stands for its two stages),
and its leader adds the block sum to the global result with an atomic:

```cpp
  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i] + input[i + BLOCK_DIM];
  ...
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
    }
```

`cuda::atomic_ref` (libcu++, `<cuda/atomic>`) turns an ordinary `float` into an atomic object
for the duration of the call. `thread_scope_device`: every block of the grid updates it, so the
atomic must be coherent across the whole GPU (a block-scope atomic would be cheaper but wrong
here). `memory_order_relaxed`: only the read-modify-write must be indivisible; nothing else is
ordered by it. The blocks finish in any order, so the operator must be associative **and**
commutative, and `*output` must hold 0 before the launch. The book's `reduce_kernel` takes `N`
but never reads it: there is no bounds check, so the length must be a multiple of
`2 * BLOCK_DIM` (the program writes `(void)N;` to keep `-Wextra` quiet). Instead of the atomic,
the leaders can write their sums to an array that a second launch (or the host) reduces: one more
step, but the order of the additions becomes fixed (`bikeshare_kpis.cu` shows both).

**Book kernel 8 — coarsening (`CoarsenedSumReductionKernel`).** The book shows the first lines and
leaves the rest as `...`. In the loop every thread is busy, there is no barrier, and in every
iteration a warp reads 32 consecutive floats (coalesced). The book's lines end with the loop; the
rest, from the first `warp_reduce` on, is completed here with the two stages of book kernel 7:
stage 1 in every warp, then stage 2 in warp 0 and one atomic per block.

```cpp
__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  float partialSum = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialSum += input[i + c * BLOCK_DIM];
  }

  partialSum = warp_reduce(partialSum);
  __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
  if (laneIdx() == 0) {
    partialSums_s[warpIdx()] = partialSum;
  }
  __syncthreads();

  if (warpIdx() == 0) {
    float blockSum = warp_reduce(partialSums_s[laneIdx()]);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
      output_ref.fetch_add(blockSum, cuda::memory_order_relaxed);
    }
  }
}
```

With `COARSE_FACTOR = 4` a block takes 8192 elements: 8 sequential additions per thread, then 10
shuffle steps and one barrier, instead of 4 blocks each paying the full tree. Like book kernel 7 it
needs a length that is a multiple of the segment; book exercise B5 removes that.

**The chapter's analysis, recomputed.** The book derives by hand, for 256 elements (128 threads,
4 warps), how much of the hardware the two in-place kernels use and how many global memory
requests they issue. `reduction_book.cu` recounts it on the host from the kernels' index
arithmetic and checks the book's numbers:

```
Execution resources, 256 elements = 128 threads = 4 warps
 iter | simple: stride threads warps   | convergent: stride threads warps
    1 |        1      128      4        |      128      128      4
    2 |        2       64      4        |       64       64      2
    3 |        4       32      4        |       32       32      1
    4 |        8       16      4        |       16       16      1
    5 |       16        8      4        |        8        8      1
    6 |       32        4      4        |        4        4      1
    7 |       64        2      2        |        2        2      1
    8 |      128        1      1        |        1        1      1
resources consumed (active warps x 32): simple 864, convergent 384, reversed 384
results committed (active threads):     simple 255, convergent 255, reversed 255
efficiency: simple 255/864 = 0.30, convergent 255/384 = 0.66 (66%)
book: 864 and 384 resources, 255 results: PASS

Global memory requests, 256 elements (128-byte segments per warp access)
  simple 141, convergent 36, reversed (ex. 3) 36, shared memory 9 (= 256/32 + 1)
book: 141, 36 and 9: PASS
```

- **Resources**: an active warp occupies all 32 lanes, however few of them work. The simple
  kernel keeps all 4 warps busy for 6 iterations: `(4 x 6 + 2 + 1) x 32 = 864` lane-steps for 255
  useful additions (30%). The convergent kernel: `(4 + 2 + 1 + 5 x 1) x 32 = 384` (66%). (Right
  after computing 864, the book's text compares 384 with "almost half of 736"; 736 matches no
  count of these kernels: 384 is 44% of 864.)
- **Requests**: one request per 128-byte segment that a warp touches in one access, three
  accesses per iteration (read own, read partner, write own). In the simple kernel a warp's
  active lanes span 64 floats = 2 segments while two or more lanes are active, 1 segment after
  that: `(4 x 5 x 2 + 4 + 2 + 1) x 3 = 141`. The convergent kernel's warps read 32 consecutive
  floats: `(4 + 2 + 1 + 5) x 3 = 36`. The shared-memory kernel reads the input once, coalesced,
  and writes one value: `256/32 + 1 = 9`, a 4x improvement over 36.

**Warps in the CPU emulation.** `cuda_compat.h` emulates warps: the threads `32w .. 32w+31` of a
block form warp `w`, and a shuffle is an exchange through 32 slots between two warp barriers.
As in CUDA with mask `0xffffffff`, every lane must call the primitive; the emulation supports only
full-warp masks (a shuffle that some lanes skip typically hangs, a useful symptom). It does not
run the lanes in lockstep, so divergence costs nothing measurable there: the tables count it.
Each emulated shuffle costs two barriers among OS threads, so warp-heavy kernels are slow on a
laptop (one block of `reduce_kernel` takes tens of ms) and the sizes are small (64 K elements).

Libraries package all of this: CUB's `WarpReduce`, `BlockReduce` and `DeviceReduce`, Thrust's
`reduce`, cooperative groups' `reduce`; newer GPUs even provide a warp-wide reduce instruction for
integers (`__reduce_add_sync`).

## 11. Floating point: the answer depends on the order

Floating-point addition is **not** associative: `(a + b) + c` and `a + (b + c)` can differ in
the last bits. A parallel reduction changes the order of the additions, so its result differs
slightly from the sequential one, and with `atomicAdd` the order of the block contributions
changes from run to run, so the last digits can change between runs. The tree order is usually
*more* accurate than the sequential loop: the loop adds tiny values to an ever larger sum. With
the 32 M uniform values of the GPU run, a sequential `float` loop gets stuck at exactly
16 777 216 (2^24: past that point, adding a number below 1 rounds back to the same value), a
relative error of 6 x 10^-5; adding per-block tree results gives 5 x 10^-6. The checks in
`reduction.cu` therefore compare against a `double` reference with a relative tolerance. When
results must be exact and reproducible, as in a day report, reduce integers
(`bikeshare_daily_summary.cu`).

`reduction_book.cu` shows the effect in miniature: its six single-block kernels add the same 2048
numbers and print `1034.5574` or `1034.5575` depending on the shape of their tree; all are within
`1e-7` of the `double` reference, so each passes a relative tolerance of `1e-5`
(`SINGLE_BLOCK_TOLERANCE`) and none would pass an equality test. The multi-block kernels get
`1e-4` (`MULTI_BLOCK_TOLERANCE`): every block adds its result into one `float` with an atomic, one
more rounding per block, and the inputs are much longer. `bikeshare_kpis.cu` shows it with money:
the day's revenue as a `float` sum of bolivianos comes out different from two kernels with
different coarsening, from the CPU loop, and from the same 8 block results added in different
orders, and never equals the exact total; the same revenue in integer centavos is identical
everywhere. **Keep money in integer cents.**

## 12. Performance analysis

Output of the CPU emulation (a 10-core laptop; the milliseconds vary between runs):

```
kernel                                  n  blocks         ms             result    rel.err  check
CPU double (small input)             2048       -          -          1034.5574          -  reference
1 simple (in place)                  2048       1     43.777          1034.5574   7.04e-08  PASS
2 convergent (in place)              2048       1     72.161          1034.5575   4.76e-08  PASS
3 shared memory                      2048       1     25.144          1034.5575   4.76e-08  PASS
CPU double (large input)           262144       -      0.260        131110.0563          -  reference
CPU float, sequential              262144       -      0.260        131110.0781   1.67e-07  (for comparison)
4 segmented + atomicAdd            262144     512   1393.509        131110.1094   4.05e-07  PASS
5 coarsened, factor 4              262144     128    379.413        131110.0625   4.75e-08  PASS
5 coarsened, factor 32             262144      16     44.268        131110.0625   4.75e-08  PASS
CPU max, sequential                262144       -      0.015         1999999999          -  reference
6 max, coarsened factor 8          262144      64    197.236         1999999999      exact  PASS
```

How to read it:

- **Hardware-independent numbers first.** The divergence table (section 5) and the global
  traffic line (`4094 loads + 2047 stores` versus `2048 loads + 1 store`) are what the chapter
  is about, and they are the same on every machine.
- **Emulation timings** count blocks and barriers. A block costs a round of thread wake-ups,
  every `__syncthreads()` a CPU barrier among 256 threads. That is why going from 512 blocks to
  128 to 16 (kernel 4, factor 4, factor 32) divides the time almost proportionally: coarsening
  removes blocks and barriers, which is precisely its point on a GPU too, though there the
  saving is smaller. Kernels 1-3 run a single block, so their times (tens of ms) are dominated
  by creating 1024 threads and are noise. The emulation runs the lanes of a warp as independent
  threads, not in lockstep, so it cannot show the divergence penalty in time; the table shows it
  in counts.
- **On a GPU** (Colab), the large input is 32 M floats = 128 MB. A reduction is memory-bound:
  one addition per 4 bytes read. At 300 GB/s (a T4-class GPU) the "speed of light" is about
  0.45 ms. Expect kernel 4 to be well above that (65 536 blocks, barrier-heavy), the coarsened
  kernels to approach it, and the CPU `double` loop to take tens of ms. Kernels 1-3 on 2048
  elements take microseconds and mostly measure the launch overhead.

`reduction_book.cu` in the emulation (64 K elements for the multi-block kernels):

```
kernel                                           n blocks        ms           result   rel.err  check
CPU, double accumulation                      2048      -         -        1034.5574         -  reference
1 SimpleSumReductionKernel                    2048      1    50.102        1034.5574  7.04e-08  PASS
2 ConvergentSumReductionKernel                2048      1    48.430        1034.5575  4.76e-08  PASS
ex3 ReversedConvergentSumReductionKernel      2048      1    55.119        1034.5575  4.76e-08  PASS
3 SharedMemorySumReductionKernel              2048      1    52.663        1034.5575  4.76e-08  PASS
5 WarpLevelSumReductionKernel                 2048      1    50.728        1034.5575  4.76e-08  PASS
6 TwoStageWarpLevelSumReductionKernel         2048      1    63.949        1034.5574  7.04e-08  PASS
CPU, double accumulation                     65536      -         -       32764.0218         -  reference
7 reduce_kernel (atomic_ref)                 65536     32  2120.071       32764.0234  5.04e-08  PASS
8 CoarsenedSumReductionKernel                65536      8   506.412       32764.0195  6.88e-08  PASS
ex4 CoarsenedMaxReductionKernel              65536      8   499.257       1999999999     exact  PASS
ex5 CoarsenedSumReductionKernelN                 1      1    72.659           0.2384  0.00e+00  PASS
ex5 CoarsenedSumReductionKernelN              1000      1    69.487         496.2630  4.44e-09  PASS
ex5 CoarsenedSumReductionKernelN              8191      1    75.838        4081.4753  4.51e-08  PASS
ex5 CoarsenedSumReductionKernelN              8193      2   133.946        4083.2266  4.94e-08  PASS
ex5 CoarsenedSumReductionKernelN             32773      5   315.642       16381.8105  1.22e-07  PASS
ex5 CoarsenedSumReductionKernelN             64759      8   518.988       32379.8809  8.68e-08  PASS
```

- In the emulation a block of 1024 threads costs tens of ms whatever it computes, so the time
  follows the block count: 32 blocks of `reduce_kernel` against 8 of the coarsened kernel. The
  emulated shuffles are expensive (two barriers among OS threads each), which hides the gain of
  book kernels 5 and 6 over kernel 3: on the laptop, count barriers instead (table in section 10).
- On a GPU expect book kernels 5 and 6 to beat kernel 3 slightly on one block (fewer barriers), and
  `CoarsenedSumReductionKernel` (4096 blocks for 32 M elements) to beat `reduce_kernel` (16 384
  blocks) and approach the bandwidth limit. Measuring it is book exercise B7 below.

## 13. Pitfalls

- **A barrier inside a branch that not all threads take** (`if (t < stride) { ...;
  __syncthreads(); }`) deadlocks or is undefined. Keep barriers where every thread of the block
  reaches them, as in all kernels here, and never `return` early before a barrier.
- **Missing the barrier** between steps: a thread reads a partial sum that is not written yet.
- **Forgetting to initialise `*output`** with the identity (0 for sums, `INT_MIN` for max) before
  launching a kernel that combines with atomics.
- **Wrong identity for padding**: padding a max-reduction with 0 gives the wrong answer for
  all-negative data.
- **Lengths that are not a multiple of the segment**: without the `i < n` guards the last block
  reads past the end.
- **Destroying the input** with the in-place kernels 1-2 and then reusing it.
- **Expecting bit-identical float results** between runs or between CPU and GPU.
- **Integer overflow** of a sum: 16 M ride times of ~1 M ms each overflow 32 bits; the
  summary kernel accumulates them in `unsigned long long`.
- **Too much or too little coarsening**: too few blocks leave SMs idle; measure.
- **A full-mask shuffle that not every lane reaches** (`if (t < 16) x += __shfl_down_sync(
  0xffffffff, x, 8);`) is undefined behaviour on a GPU and hangs the emulation. Let the whole warp
  call it, each lane with the identity if it has nothing to contribute.
- **Reading the warp sums past the end**: the book's second stage reads `partialSums_s[t]` for 32
  lanes, valid only with 1024 threads. With smaller blocks, read the identity for
  `t >= BLOCK_DIM / WARP_SIZE`.
- **Using another lane's result** after `warp_reduce`: only lane 0 holds the sum.
- **Shuffling a struct**: a shuffle moves one value of at most 64 bits; shuffle each field
  (`bikeshare_kpis.cu`).
- **Ties in an argmax**: `a.value >= b.value ? a : b` keeps whichever operand the tree puts first,
  so the winner depends on the tree. Reduce (value, index) pairs with an explicit tie-break
  (`bikeshare_bike_totals.cu`).

## 14. Ideas for your project

### The bike-sharing examples in this folder

The four `bikeshare_*.cu` programs apply the chapter to a city bike-sharing system: 48 docking
stations on a 6 km x 6 km map, bikes that are docked, ridden or in repair, trips whose fare is
kept in integer centavos (Bs 3 plus Bs 1 per started 10 minutes; a trip is late once it runs over
the 45 included minutes) and rebalancing trucks that carry bikes from full stations to empty
ones. The data is synthetic, generated by each file's own `SplitMix64` with a fixed seed, so every
run prints the same numbers. The structure carries over to `std::thread`: each thread reduces
its share into local variables (coarsening), then the partial results are combined once (the
tree, or a short loop).

**`bikeshare_daily_summary.cu`, the day report.** It prints what an operator reads at closing
time: trips started, bikes returned to a dock, bikes sent to repair (the rider reported a fault
when docking), bikes still out, late returns, the fare collected, the average and the longest
ride, the busiest bike and the bike with the most repairs. Every line is a sum, a count or a max
over the trips. `SummaryKernel` computes seven reductions in **one pass**: it is the coarsened
shared-memory tree of kernel 5, fused (the data is read once and the tree runs seven operators
with the same barriers), with one atomic per field per block. `PerBikeKernel` is a **segmented**
reduction: the trips are grouped by bike (a counting sort on the host), one block reduces each
bike's slice, and no atomics are needed. Independent reductions must agree: the rides of all bikes
add up to the finished trips, their repairs to the repairs, and every trip started is returned, in
repair or still riding. Integers make the result exact and identical in every run.

**`bikeshare_kpis.cu`, the day's KPIs in one pass.** Completed trips, late returns, revenue in
centavos, metres ridden, the shortest and the longest ride. The KPIs are one struct and `combine`
works field by field (sum, count, min, max), so it is associative and commutative with identity
`{0, 0, 0, 0, INT_MAX, INT_MIN, 0}`; trips still riding and false starts (a bike undocked and
docked again at once, not charged) map to the identity. Each thread folds its trips in registers,
then the book's two-stage warp reduction runs on the struct, **each field shuffled on its own**.
Across blocks there are two ways: one atomic per field (one kernel, exact for integers in any
order of the blocks), or per-block partials and a second pass in a fixed sequence (reproducible
for floats). The same revenue is also kept as a `float` sum of bolivianos. Members pay 15% less,
so most fares (Bs 2.55, Bs 3.40, ...) have no exact `float`: the float total changes with the
kernel and with the sequence of the blocks, while the integer centavos never do.

**`bikeshare_bike_totals.cu`, the most used bike and the top earner.** Per bike: metres ridden,
revenue in centavos and completed trips. The trips are grouped by bike and **one coarsened block
per bike** walks its slice 256 trips at a time (any slice length, coalesced reads), with the
two-stage warp reduction of three fields and no atomics. A **second reduction** runs over the
bikes: an argmax of (value, index) pairs where the lowest index wins a tie, both argmaxes (most
kilometres, most revenue) in one pass of one block. A tie test puts three equal maxima where a
higher index sits in a lower lane, so a careless `>=` would fail it.

**`bikeshare_nearest_truck.cu`, a truck for every empty station.** For each empty station, the
rebalancing truck with bikes loaded that can stop there soonest. A truck's ETA is the time it
still needs for its current job, plus the drive in Manhattan metres, but during the rush hour some
curbs are closed to trucks for a few minutes, so the ETA is never earlier than the moment the curb
opens. It is a map (compute each ETA) fused with a min-reduction: the ETA (high 32 bits) and the
truck index (low 32 bits) are packed in one 64-bit key and combined with `atomicMin`. A 2D grid
gives one row of blocks per empty station. Ties are frequent (every truck that arrives before the
curb opens has the same ETA) and are broken by the lowest truck index, deterministically. The
program also counts how many stations chose the same truck: independent argmins conflict.

Emulation output:

```
kernel                              blocks         ms  check
CPU sequential (both)                    -      0.607  reference
SummaryKernel (7 fused reductions)      32    187.779  PASS
PerBikeKernel (segmented)              240   1097.080  PASS
started == returned + repair + riding: yes
sum of rides per bike == finished:     yes
sum of repairs per bike == repair:     yes

Bike-share day report: 48 stations, 240 bikes
  trips started:             262144
  returned to a dock:        250057
  bikes sent to repair:      3951
  still riding at closing:   8136
  late returns (> 45 min):   12828 (5.05% of finished trips)
  fare collected:            Bs 1369344.00
  average ride:              18 min 57 s
  longest ride:              137 min 29 s
  busiest bike:              b173, 1146 rides
  most repairs:              b135, 32 trips ended in the workshop
```

```
version                                       grid         ms  check
CPU sequential argmin                            -      0.490  reference
NearestTruckKernel (packed atomicMin)        8 x 13   1120.457  PASS

empty station  docks  curb opens  nearest truck  bikes  ETA (min:s)  same ETA
s0                25        0:54  t1739              7         0:54        18
s7                19        1:30  t156               5         1:30        78
...
s21               18        0:00  t5811             11         0:04         1
...
s41               25        4:02  t14                5         4:02       487
```

Where the curb is open, one truck is a few seconds away and wins alone; where it is closed, every
truck that gets there first waits for the same moment, and the lowest index among them (487
trucks for `s41`) is chosen.

```
version                                  blocks        ms revenue float (Bs)     - exact  integer KPIs
CPU sequential loop, float                    -     0.231        655022.8750    -83.1750  reference
CPU sequential loop, double                   -         -        655106.0457     -0.0043  -
KpiKernel CF=128, atomics per field           4   286.171        655106.1250      0.0750  PASS  (1 run: float identical, integers identical)
KpiKernel CF=256, atomics per field           2   134.309        655105.8125     -0.2375  PASS  (1 run: float identical, integers identical)
KpiKernel CF=64, partials + 2nd pass          9   682.586        655106.1250      0.0750  PASS  (fixed sequence)
exact: integer centavos / 100                 -         -        655106.0500      0.0000  the right answer

8 block results added in 1000 random sequences (what atomicAdd sees on a GPU):
  float bolivianos: 2 distinct totals, from 655106.0625 to 655106.1250
  integer centavos: 1 distinct total, 655106.05

day KPIs: 120520 completed trips, revenue Bs 655106.05, 4322 late returns (3.59%)
          shortest ride 2 min 16 s, longest 102 min 13 s, 500371.9 km ridden, 4152 m per trip
```

At 655 000 bolivianos one `float` step is 0.0625 Bs, so the float totals can only land on
multiples of 6.25 centavos, and the sequential loop, adding small fares to a large sum, drifts by
83 Bs. Even the `double` loop is off by a hair (the `-0.0043`), because each fare was first
rounded to a `float`. On the GPU (16 M trips, about 84 million Bs) one float step is 8 Bs and the
atomic version can print a different total in every run; the "runs" note on each line says
whether it did.

```
kernel                                blocks         ms  check
CPU sequential (totals + argmax)           -      0.051  reference
BikeTotalsKernel (segmented)              16    711.062  PASS
ArgMaxKernel (value, index) pairs          1     34.385  PASS

bike        trips   revenue (Bs)  ridden (km)
b6           2499      14485.00      10343.1
b4           2480      14386.00      10300.4
b7           2460      14315.00      10199.1
b0           2437      14244.00      10352.8
b5           2444      14235.00      10273.9
top earner: b6, Bs 14485.00
most used:  b0, 10352.8 km

tie test: revenue tied at b2999, b513, b259 -> b259; distance tied at b2050, b1795, b1000 -> b1000  PASS
tie test: all bikes at 0 -> b0 and b0  PASS
```

The most used bike is not the top earner: b0 rode the most kilometres, but b6 made more trips,
and a fare depends on the time ridden and starts with Bs 3 per trip, not on the distance. The
per-bike sums carry information a count alone does not.

Details of each program that the code relies on:

- **`bikeshare_daily_summary.cu`.** Without grouping, the rides per bike would be a histogram with
  one bin per bike (chapter 9, [`../07_histogram`](../07_histogram/)). Grouping costs a counting sort (a
  histogram plus a prefix sum), but then every per-bike statistic becomes an ordinary reduction.
  Why integers: a float sum would change in the last digits from run to run, because the order of
  the `atomicAdd`s changes, which is not acceptable for a report that must be exact. The
  max-reduction of ride times starts at 0 (`MAX_IDENTITY_FOR_DURATIONS`), a valid identity because
  times are never negative. A trip still riding at closing has no duration and no fare yet:
  `addTrip` only counts it. Every fourth bike is an e-bike, whose rides last 25% longer, so the
  per-bike averages differ. The fares are whole bolivianos here; `bikeshare_kpis.cu` adds the
  member discount that breaks `float`. The GPU run uses 480 bikes (10 per station), the emulation
  240, because each per-bike block pays the barriers among OS threads.
- **`bikeshare_nearest_truck.cu`.** Comparing two packed keys compares the ETAs first and, only
  when they are equal, the truck indices, so `min(key)` is the best ETA and, among trucks with the
  same ETA, the lowest index: the same winner in every run and on every backend. The ETA is
  computed with integers only (metres and milliseconds, 25 km/h with a 25% street detour), so the
  host reference and the kernel produce exactly the same numbers; with floats, `nvcc` may fuse
  `a * b + c` into one FMA instruction that rounds differently from the CPU, which could change
  the winner of a tie. For float ETAs the same packing works with the float's bit pattern, which
  orders like an unsigned integer when the value is non-negative. All the threads of a block read
  the same station (one broadcast read), and consecutive threads read consecutive trucks
  (coalesced). Off-duty trucks and empty trucks get `NO_ETA` and never win; a station that no
  truck can serve keeps `NO_KEY`. The stations are generated before the trucks, so
  `./btruck 40` and `./btruck 16384` restock the same 13 empty stations (with 40 trucks some
  stations share their nearest truck). A real city runs a handful of trucks and the argmin is a
  short loop: the GPU version only pays off with tens of thousands of candidates or many queries
  at once. The packing trick works the same with `std::atomic<uint64_t>` and a compare-exchange
  loop.
- **`bikeshare_kpis.cu`.** With atomics, thread 0 of each block applies one atomic per field
  (`combineIntoTotalWithAtomics`). Each field is updated atomically on its own, and that is enough:
  no field depends on another and nobody reads the struct before the kernel ends. A few hundred
  blocks mean one atomic per field per block and no contention worth mentioning. With partials,
  `KpiCombineKernel` (one block) has thread `t` fold partials `t, t + 256, ...` and then runs the
  block tree, so the sequence of every addition is fixed by the code, not by the scheduler; this
  is the second option the book mentions. The partials also let the host replay what the atomics
  do on a GPU: add the block results one after another in the sequence the blocks happen to
  finish, here in 1000 random sequences.
- **`bikeshare_bike_totals.cu`.** `better(a, b)` is a max under a total order (value first, then
  the lower index), so it is associative and commutative: whatever the tree and the warp in which
  two tied bikes meet, the lowest index wins. With a plain `a.value >= b.value ? a : b` the winner
  of a tie would depend on which operand the tree puts first. The identity `(0, NO_INDEX)` loses
  against every real bike. In the tie test, thread `t` of `ArgMaxKernel` folds totals
  `t, t + 256, ...`, so a higher index sits in a lower lane: 513 is in lane 1 and 259 in lane 3;
  2050 is in lane 2, 1795 in lane 3 and 1000 in warp 7. A rule that keeps the left operand on ties
  would answer b513 and b2050. Three trips in ten go to the lower half of the bike numbers (the
  bikes docked downtown), so the slices have different lengths (1394 to 2706 trips). `bikeshare_nearest_truck.cu` packs the pair into
  one 64-bit key to use `atomicMin` instead.

### Where this pattern may appear in your project

- A closing summary is often a handful of counts, sums and maxima over the same records: could one
  pass compute all of them, and which identity does each one start from?
- If many workers must pick the best candidate for each request, think of a reduction over
  (key, index) pairs, and decide how ties are broken before writing it, so that every run picks
  the same one.
- Totals per entity (per worker, per place, per hour) are a segmented reduction: what would it
  cost to group the records first, and what would you save compared with one atomic per record?
- Times and money summed in floating point change with the order of the additions; would integer
  milliseconds and cents give you a total that is the same in every run and with any number of
  threads?
- Two independent reductions that must agree (the parts add up to the whole) are a cheap check
  that a report is right.

## 15. Files and how to run

| File | Contents |
|------|----------|
| `01_*.cu` … `12_*.cu` | the learning path above, one idea per step on 16 to 256 values, each printing its input, its intermediate arrays and PASS/FAIL |
| `reduction.cu` | the kernels of the chapter (simple, convergent, shared, segmented, coarsened x2, max), divergence analysis, CPU references, timing table |
| `bikeshare_daily_summary.cu` | a bike-share day report (trips, late returns, repairs, fare) as fused reductions, plus per-bike totals as a segmented reduction |
| `bikeshare_nearest_truck.cu` | for every empty station, the argmin over the rebalancing trucks' ETAs with packed 64-bit keys |
| `reduction_book.cu` | the chapter's kernels as the book writes them (simple, convergent, shared memory, `warp_reduce`, warp-level, two-stage, `reduce_kernel` with `cuda::atomic_ref`, coarsened), the book's exercises 3-6 as kernels, and the chapter's resource and memory-request counts for 256 elements |
| `bikeshare_kpis.cu` | the day's bike-share KPIs in one pass as a struct reduction; atomics per field versus a second pass; float versus integer revenue |
| `bikeshare_bike_totals.cu` | per-bike distance and revenue (one block per bike), then the most used bike and the top earner with a deterministic tie-break |

On a laptop (no GPU; `g++` works the same), the steps first:

```sh
cd ec2/08_reduction
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_sequential_reduction.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_simple_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_convergent_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_shared_memory_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_warp_reduce.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_warp_level_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_two_stage_warp_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 08_multi_block_atomic.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 09_coarsened_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 10_reversed_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 11_coarsened_max_kernel.cu -o step && ./step
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 12_any_length_kernel.cu -o step && ./step
```

Then the complete programs and the bike-sharing examples:

```sh
cd ec2/08_reduction
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common reduction.cu -o reduction && ./reduction
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_daily_summary.cu -o bsum && ./bsum
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_nearest_truck.cu -o btruck && ./btruck [trucks]
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common reduction_book.cu -o reduction_book && ./reduction_book
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_kpis.cu -o bkpis && ./bkpis
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_bike_totals.cu -o bbikes && ./bbikes
```

> **Easiest way on Colab:** this folder has two notebooks. [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) runs the steps of the learning path in order, and [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) runs the complete programs and the bike-sharing examples. Upload one to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

On Google Colab (Runtime > Change runtime type > GPU), in a cell:

```sh
!cd ec2/08_reduction && nvcc -O2 -arch=native -I../common 05_warp_reduce.cu -o step && ./step
!cd ec2/08_reduction && nvcc -O2 -I../common reduction.cu -o reduction && ./reduction
!cd ec2/08_reduction && nvcc -O2 -I../common bikeshare_daily_summary.cu -o bsum && ./bsum
!cd ec2/08_reduction && nvcc -O2 -I../common bikeshare_nearest_truck.cu -o btruck && ./btruck
!cd ec2/08_reduction && nvcc -O2 -arch=native -I../common reduction_book.cu -o reduction_book && ./reduction_book
!cd ec2/08_reduction && nvcc -O2 -arch=native -I../common bikeshare_kpis.cu -o bkpis && ./bkpis
!cd ec2/08_reduction && nvcc -O2 -arch=native -I../common bikeshare_bike_totals.cu -o bbikes && ./bbikes
```

Sizes come from `BY_BACKEND(gpu, cpu)`: 32 M elements on the GPU, 256 K in the emulation
(64 K for `reduction_book.cu`, whose warp shuffles are slow to emulate). For ThreadSanitizer,
shrink them: `-DCPU_N=30000` for `reduction.cu`, `-DCPU_TRIPS=20000 -DCPU_BIKES=48u` for
`bikeshare_daily_summary.cu`, a small argument (`./btruck 3000`) for the argmin, `-DCPU_N=16384`
for `reduction_book.cu` (a multiple of 8192), `-DCPU_TRIPS=32768` for `bikeshare_kpis.cu`, and
`-DCPU_TRIPS=8192 -DCPU_BIKES=6u` for `bikeshare_bike_totals.cu`. The step files are small enough
to run under ThreadSanitizer as they are, and it reports no race in any of them.

## 16. Exercises

1. **Tree arithmetic.** For `n = 2^20`, how many steps and how many additions does the tree
   need? What is the average parallelism? How many threads are useful in the last 5 steps?
2. **Divergence by hand.** For one block of 256 threads (8 warps) running kernel 1, list how
   many warps execute each step. Do the same for kernel 2. Check against
   `printDivergenceAnalysis()` after changing `SINGLE_BLOCK_THREADS`.
3. **Why commutativity?** Kernel 1 only ever adds neighbouring partial sums in their original
   order; kernel 2 does not. Give an associative but non-commutative operator for which kernel 1
   is correct and kernel 2 is wrong.
4. **Barrier placement.** In kernel 3, move `__syncthreads()` from the top of the loop to the
   bottom. Is the kernel still correct? What exactly could go wrong?
5. **Coarsening factor.** A GPU has 40 SMs, each able to run 4 blocks of 256 threads at once.
   For `n = 2^25`, how many blocks does kernel 4 launch, and how many rounds does the hardware
   need? Which coarsening factor makes the grid fit in exactly one round?
6. **Min with index.** Write the max-reduction of kernel 6 as an argmax for non-negative
   integers using one 64-bit key. Which index wins a tie, and how would you make the *highest*
   index win instead?
7. **Floating point.** Explain why the sequential `float` sum of 32 M values in [0, 1) never
   exceeds 16 777 216. Propose two ways to get an accurate sum on the CPU.
8. **Bike-share report.** Add the shortest ride (`minRideMs`) and the number of rides longer than
   two hours to `SummaryKernel` of `bikeshare_daily_summary.cu`. Which arrays and which identity
   values do you need? Do you need another barrier?

### Exercises from the book

- **B1.** For the simple kernel, with 1024 elements and warps of 32 threads, how many warps of the
  block have divergence during the 5th iteration?
- **B2.** The same for the convergent kernel.
- **B3.** Modify the convergent kernel so that the threads own the **last** half of the array and add
  the element `stride` positions to their left: in the first iteration thread `t` adds
  `input[t]` into `input[blockDim.x + t]`, and the total ends in the last element.
- **B4.** Modify the coarsened kernel to perform a max reduction instead of a sum.
- **B5.** Modify the coarsened kernel to work for an input of any length, not necessarily a multiple of
  `COARSE_FACTOR * 2 * blockDim.x`, with an extra parameter `N`.
- **B6.** Show how the array `{6, 2, 7, 4, 5, 8, 3, 1}` changes after each iteration of (a) the simple
  kernel and (b) the convergent kernel.
- **B7.** (Colab) Run `reduction_book.cu` on a GPU with 32 M elements, compare `reduce_kernel` with
  `CoarsenedSumReductionKernel`, then try `COARSE_FACTOR` 1, 2, 8, 32 and explain the curve.

## 17. Solutions (short)

1. `log2(2^20) = 20` steps, `2^20 - 1` additions, average parallelism `(2^20 - 1)/20`, about
   52 000. The last 5 steps use 16, 8, 4, 2 and 1 threads.
2. Kernel 1, strides 1..256: 8, 8, 8, 8, 8, 8 warps (strides 1-32), then 4, 2, 1 (strides 64,
   128, 256): 55 warp-steps. Kernel 2, strides 256..1: 8, 4, 2, 1, 1, 1, 1, 1, 1: 20 warp-steps.
3. Matrix multiplication (or string concatenation): associative, not commutative. Kernel 1
   combines `x0 x1`, then `(x0x1)(x2x3)`: the order is preserved. Kernel 2 combines `x0` with
   `x[blockDim.x]` in its first step: wrong order.
4. No. In the first iteration a thread would read `input_s[t + stride]` possibly before the
   thread that owns that slot has written it: nothing separates the initial write from the first
   read. The later steps are still separated by the bottom barrier, so adding one barrier before
   the loop would fix it.
5. `2^25 / 512 = 65 536` blocks; `40 x 4 = 160` blocks at a time, so 410 rounds. One round needs
   at most 160 blocks: `2^25 / (2 x 256 x F) <= 160` gives `F >= 410`; in practice a factor
   of 256 or 512 (a few rounds) is fine, or a grid-stride loop over a grid of exactly 160 blocks.
6. `key = ((unsigned long long)value << 32) | index`, reduce with max and `atomicMax` on 64-bit
   (CUDA has it). The highest index wins a tie. For the lowest index, store `~index` (or
   `0xffffffff - index`) in the low bits.
7. From 2^24 on, consecutive floats are 2 apart; adding `x < 1` gives a result whose nearest
   float is the old sum. Remedies: accumulate in `double`; pairwise (tree) summation; Kahan
   compensated summation.
8. One more `int` array for the min (identity `INT_MAX`, combine with `imin` and `atomicMin`),
   one more `unsigned` array for the count (identity 0, `+`, `atomicAdd`). They join the same
   tree loop, so no new barrier is needed. A trip still riding has no duration yet: `addTrip`
   returns before the new fields, so it leaves the min at its identity.

### Solutions to the book's exercises

B1-B6 are computed or run by `reduction_book.cu`; the tables below are its output.

- **B1.** 1024 elements = 512 threads = 16 warps. Iterations have strides 1, 2, 4, 8, **16**: in the
  5th, the threads with `t % 16 == 0` work, 2 per warp, so **all 16 warps diverge** (counting
  the first iteration as iteration 0, the 5th has stride 32 and 1 active thread per warp: still
  16). Only the first iteration is free of divergence.
- **B2.** Strides 512, 256, 128, 64, **32**: in the 5th iteration threads 0-31 work, which is exactly
  warp 0, and the other 15 warps are idle: **0 divergent warps**. Divergence appears only in the
  last 5 iterations (stride 16 and below), always in warp 0 alone. (Counting from iteration 0,
  the 5th has stride 16: 1 divergent warp.)

```
 iter | simple: stride threads warps diverg | convergent: stride threads warps diverg
    1 |        1      512     16      0      |      512      512     16      0
    2 |        2      256     16     16      |      256      256      8      0
    3 |        4      128     16     16      |      128      128      4      0
    4 |        8       64     16     16      |       64       64      2      0
    5 |       16       32     16     16      |       32       32      1      0   <- 5th iteration
    6 |       32       16     16     16      |       16       16      1      1
  ...
   10 |      512        1      1      1      |        1        1      1      1
```

- **B3.** The active threads are the last `stride` ones, still contiguous, so divergence and
  coalescing are those of the convergent kernel (384 resources and 36 requests for 256
  elements, as the program counts):

```cpp
__global__ void ReversedConvergentSumReductionKernel(float* input, float* output) {
  unsigned int i = blockDim.x + threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x >= blockDim.x - stride) {
      input[i] += input[i - stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == blockDim.x - 1) {
    *output = input[2 * blockDim.x - 1];
  }
}
```

- **B4.** Replace `+` by max everywhere, the identity 0 by `INT_MIN`, and the final `fetch_add` by an
  atomic max (`atomicMax`, or `fetch_max` of `cuda::atomic_ref` for integers); `*output` starts
  at `INT_MIN`:

`warp_reduce_max` is `warp_reduce` with `imax` instead of `+`, and `...` stands for the same two
stages as in book kernel 8:

```cpp
  int partialMax = input[i];
  for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
    partialMax = imax(partialMax, input[i + c * BLOCK_DIM]);
  }

  partialMax = warp_reduce_max(partialMax);
  ...
    if (threadIdx.x == 0) {
      atomicMax(output, blockMax);
    }
```

- **B5.** Launch `cdiv(N, COARSE_FACTOR * 2 * BLOCK_DIM)` blocks and replace every element past the end
  by the identity; every thread still runs both stages (no early `return`: the barrier and the
  shuffles need all of them). The partial sum starts at `SUM_IDENTITY` (0), which is correct even
  for a thread none of whose elements exists. Tested with N = 1, 1000, 8191, 8193, 32773 and 64759:

```cpp
  float partialSum = SUM_IDENTITY;
  for (unsigned int c = 0; c < COARSE_FACTOR * 2; ++c) {
    if (i + c * BLOCK_DIM < N) {
      partialSum += input[i + c * BLOCK_DIM];
    }
  }
```

- **B6.** Printed by traced copies of the two kernels (one extra line after the barrier, where each
  thread copies the position it owns and one that nobody writes, so no extra barrier is needed),
  run with one block of 4 threads:

```
  a. SimpleSumReductionKernel (traced copy)
    input:             6   2   7   4   5   8   3   1
    after stride 1:    8   2  11   4  13   8   4   1
    after stride 2:   19   2  11   4  17   8   4   1
    after stride 4:   36   2  11   4  17   8   4   1
  b. ConvergentSumReductionKernel (traced copy)
    input:             6   2   7   4   5   8   3   1
    after stride 4:   11  10  10   5   5   8   3   1
    after stride 2:   21  15  10   5   5   8   3   1
    after stride 1:   36  15  10   5   5   8   3   1
  ex. 3 ReversedConvergentSumReductionKernel (traced copy)
    input:             6   2   7   4   5   8   3   1
    after stride 4:    6   2   7   4  11  10  10   5
    after stride 2:    6   2   7   4  11  10  21  15
    after stride 1:    6   2   7   4  11  10  21  36
```

- **B7.** Expect a large gain from factor 1 to 4-8 (fewer blocks, fewer barriers, the same bytes
  read) and then a flat curve or a slight loss once the grid no longer fills every SM: the best
  factor is the smallest one that keeps the GPU full.
