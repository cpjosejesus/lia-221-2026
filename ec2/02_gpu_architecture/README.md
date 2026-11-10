# Inside the GPU: architecture and scheduling (chapter 4)

Session 12 (`12_gpu_architecture_scheduling`): session 11 launched grids of blocks of threads
without asking where they run. This folder opens the hardware. It asks the GPU what it has, sizes
grids against its limits, makes the threads of a block wait for each other, finds out which warp
and lane each thread is, counts how branches split warps, and computes occupancy from the device
properties.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_device_query.cu`](01_device_query.cu) | The device query of the slides: `cudaGetDeviceCount` and `cudaGetDeviceProperties`, then the SM count, the block and grid limits and the per-SM limits (threads, blocks, registers, shared memory). A one-thread kernel reads the built-in `warpSize`, which must match the property. (slide: *Asking the GPU what it has*) | On Colab's T4 you should see compute capability 7.5, 40 SMs, 1024 threads per block, and per SM 1024 threads, 16 blocks, 65536 registers and 64 KiB of shared memory (the CPU emulation reports fixed values of its own). The last line checks that the `warpSize` a kernel reads is the same 32 that the host property reports. |
| 2 | [`02_blocks_needed.cu`](02_blocks_needed.cu) | How many blocks a grid needs: `(n + threadsPerBlock - 1) / threadsPerBlock` for four launches, the idle threads of the last block, and which block owns each element. A block of 2048 threads is refused because the device allows `maxThreadsPerBlock` = 1024. | For n = 10 with 4 threads per block, the owner list 0 0 0 0 1 1 1 1 2 2 shows that block 2 has only 2 of its 4 threads working; n = 16 is a multiple of 4 and leaves none idle. The 2048-thread launch is printed but refused because it exceeds `maxThreadsPerBlock` = 1024. Try: add `{20, 8}` to `LAUNCHES`: 3 blocks, 24 threads, 4 idle. |
| 3 | [`03_sync_reverse.cu`](03_sync_reverse.cu) | `reverseChunks`: each block copies its chunk of 8 elements to a `__shared__` array, waits at `__syncthreads()`, and reads another thread's slot. The barrier is outside any branch, and blocks never wait for each other. (slide: *Code: a barrier in a real kernel*) | Each half is reversed on its own: 0 … 70 becomes 70 … 0 and 80 … 150 becomes 150 … 80, and no value moves from one chunk to the other, because each block only sees its own `__shared__` array. Thread 0 writes 70, a value loaded by thread 7, which is why the barrier must come before the read. |
| 4 | [`04_warps_and_lanes.cu`](04_warps_and_lanes.cu) | `whoAmI` on 16×16 blocks: the row-major thread index `t = threadIdx.y * blockDim.x + threadIdx.x`, the warp `t / warpSize` and the lane `t % warpSize`. Every two rows of a 16-wide block form one warp; the sample threads are those of the slide's table. (slide: *Code: which warp and which lane am I?*) | In the map every two rows share a warp number (rows 0 and 1 are warp 0, rows 14 and 15 are warp 7), because a 16-wide block packs 32 consecutive values of t into one warp. In the sample table, thread (15, 1) has t = 31, the last lane of warp 0, and (0, 2) has t = 32, lane 0 of warp 1. |
| 5 | [`05_divergence.cu`](05_divergence.cu) | Control divergence counted: the `branches` kernel records which way each thread goes at the boundary check, at `t % 2 == 0` and at `(t / warpSize) % 2 == 0`, and the host counts the lanes of every warp on each side. A branch costs 2 passes in a warp only when both sides have lanes. (slides: *Code: divergent and warp-uniform branches*, *The most common divergence: the boundary check*) | Each cell is "lanes taking the if / lanes taking the else". `t % 2 == 0` splits every running warp 16/16, so it always costs 2 passes; the warp-aligned test `(t / warpSize) % 2 == 0` never splits a warp. The boundary check splits only warp 2 (threads 64 to 79 work, 80 to 95 return), and warp 3 shows 0/0 in the later branches because all its lanes already returned. Try: set `N = 96`; warp 2 no longer diverges at `i >= n`. |
| 6 | [`06_occupancy.cu`](06_occupancy.cu) | An occupancy calculator: blocks per SM limited by threads, block slots and registers, and occupancy as resident warps over the maximum. It reproduces the slides' H100 table (the check), then runs the same table with the limits that `cudaGetDeviceProperties` reports for this device. (slides: *Occupancy in numbers: four ways to lose it*, *Code: let the runtime compute occupancy*; the runtime's `cudaOccupancy*` calls themselves need a real GPU) | In the H100 table, 32-thread blocks reach only 50% because the 32-block limit is hit before the thread limit, and 512 threads at 33 registers drop to 3 blocks (75%) while 31 registers still allow 4. On Colab's T4 the second table changes: the SM holds only 1024 threads and 16 blocks, so the thread limit binds first and the two register rows (256 at 64, 512 at 33) reach 100%. |
| 7 | [`07_barrier_placement.cu`](07_barrier_placement.cu) | The right version of the slide's `step` kernel: even threads double `x[t]`, odd threads add 1, then all 8 threads meet at one `__syncthreads()` after the `if`/`else` and compute `y[t] = x[t] - x[t ^ 1]` from their neighbour's new value. The wrong version, with one barrier inside each branch, is not run: its behaviour is undefined. (slide: *A barrier every thread must reach: the wrong and the right way*) | Each pair (t, t ^ 1) gives opposite values in y: thread 0 gets 20 − 21 = −1 and thread 1 gets 21 − 20 = 1, because thread 0 reads `x[1]` = 21 only after thread 1 has written it. The result goes to a second array `y`: writing it back into `x` would let a thread overwrite `x[t]` while its neighbour is still reading it, a race that ThreadSanitizer reports. |
| 8 | [`08_divergence_2d.cu`](08_divergence_2d.cu) | The 2D boundary check `col < width && row < height` on 16 × 16 blocks, where each warp is two rows of a block. The kernel records which threads are inside the picture; the host counts, for every warp, whether its lanes are all inside, all outside or split, for the 76 × 62 picture of the slide and the 100 × 70 image of exercise 12.8. (slide: *Boundary checks in 2D: which warps diverge?*) | Only the right-hand column of blocks diverges, 8 warps per block, because the edge cuts every pair of rows. The bottom row of blocks never does: each of its warps is fully inside or fully outside. 76 × 62 has 31 divergent warps of 160; 100 × 70 has 35 of 280 (12.5 %), the answer to exercise 12.8. Try: change the first picture to 76 × 63: row 62 now exists but row 63 does not, so the last warp of every bottom block splits and the count rises to 36. |
| 9 | [`09_warp_efficiency.cu`](09_warp_efficiency.cu) | The kernel `k` of exercise 12.1 on N = 2048 with blocks of 128 threads. Every branch also marks the threads that take it, so the host can count, per warp, the active lanes at line 4, at line 6 and in each iteration of the loop on line 7. (slide: *Exercise 12.1: count warps, divergence and SIMD efficiency*) | The answers of exercise 12.1: (a) 4 warps per block and 64 in the grid; (b) 48 active and 32 divergent warps at line 4, with warp 2 of each block inactive but not divergent; (c) every warp runs line 6 with only 8 lanes, 25 %; (d) the loop runs 32, 24, 16 and 8 lanes of each warp, so 1 iteration without divergence and 3 with. |

The next folder, [`../03_memory_tiling`](../03_memory_tiling/), goes inside the SM's memory:
registers, shared memory and the tiled matrix product.

## 1. SMs, blocks and the device query

A GPU is an array of **streaming multiprocessors** (SMs), grouped into GPCs on recent GPUs. Each
SM has many arithmetic lanes (the book's *streaming processors*, marketing's *CUDA cores*),
registers, shared memory and limits on how many threads and blocks it can hold. An H100 has 132
SMs; each holds up to 2048 threads (64 warps), 32 blocks and 65 536 registers.

`cudaGetDeviceProperties` returns those numbers in a `cudaDeviceProp` (step 1):
`multiProcessorCount` (SMs), `maxThreadsPerBlock`, `maxThreadsDim`, `maxGridSize`,
`maxThreadsPerMultiProcessor`, `maxBlocksPerMultiProcessor`, `regsPerMultiprocessor`,
`regsPerBlock`, `sharedMemPerBlock`, `warpSize` and the compute capability `major.minor` (A100:
8.0, H100: 9.0). Despite its name, `regsPerBlock` is usually the register count of a whole SM. On
a laptop the emulation reports a made-up device: one SM per CPU hardware thread, the H100's
per-SM limits on threads, blocks and registers, and compute capability 0.0.

Step 2 sizes grids against those limits. The number of blocks follows the data, rounded up, and
the block size must not exceed `maxThreadsPerBlock`; a launch that does fails and the kernel never
runs.

## 2. Block scheduling and barriers

The run-time assigns blocks to SMs **whole**: all threads of a block go to the same SM and are
there at the same time, and a block only starts when the SM has room for all of it. That is what
lets the threads of one block cooperate: they share fast `__shared__` memory and can wait for each
other at a barrier, `__syncthreads()`. Threads in different blocks get no such guarantee: their
blocks may run at different times, on different SMs, in any order. Because blocks never wait for
each other, the same grid runs on a small GPU (few blocks at a time) and on a large one (many at a
time): **transparent scalability**.

In step 3 thread `t` reads the slot that thread `BLOCK - 1 - t` wrote. Without the barrier it
could read before that thread has written: a race, and a result that changes from run to run.
Each `__syncthreads()` call site is a different barrier, and it must be reached by all threads of
the block or by none. A barrier inside an `if` that only some threads take, or after an early
`return` that only some threads take, is a bug. On the slides, the version with one barrier in each
branch is the wrong one; the version with a single barrier after the `if`/`else` is the right one.

## 3. Warps and lanes

Inside an SM, a block is cut into **warps** of 32 consecutive threads. A warp is what the SM
actually schedules: one instruction is fetched and executed for all 32 lanes at once (SIMD
hardware, programmed as scalar threads: SIMT). For 2D and 3D blocks the threads are first put in a
line in row-major order, `t = threadIdx.y * blockDim.x + threadIdx.x`, and then cut every 32. So
in a 16-wide block every two rows form a warp (step 4). Use `warpSize` in kernels instead of a
literal 32; on the host the same number is `cudaDeviceProp::warpSize`.

## 4. Control divergence

When the lanes of one warp take different sides of a branch, the warp runs both sides, one after
the other, with the lanes of the other side switched off: **control divergence**. Step 5 records
the side each thread takes and counts lanes per warp:

- `t % 2 == 0` splits odd and even lanes in **every** warp: 2 passes everywhere.
- `(t / warpSize) % 2 == 0` is the same for all 32 lanes of a warp: 1 pass. Both branches do the
  same work per thread, but this one costs half the passes. If the mapping of work to threads is
  yours to choose, give each warp a single kind of work.
- The boundary check `if (i >= n) return;` diverges in at most one warp, the one that holds
  `i == n`. Warps entirely past n do nothing at all, which is not divergence. Its cost shrinks as n
  grows.

A rule of thumb: a condition that depends on `threadIdx` can diverge; one that is the same for the
whole warp (`blockIdx`, a kernel argument) cannot. Loops whose trip count depends on the thread
keep the warp busy until its longest lane finishes. From Volta on, the two sides of a branch may be
interleaved, so do not assume the lanes of a warp are back together after a branch: say so with
`__syncwarp()` when it matters. For very short branches, computing both results and selecting one
(`x = cond ? a : b`) avoids the branch, and the compiler often does it by itself (predication).

## 5. Latency hiding and occupancy

An SM holds many more warps than it can execute in one cycle (an H100 SM: 128 lanes, up to 64
warps). When a warp waits for global memory, the scheduler picks another ready warp, at no cost:
**zero-overhead scheduling**. The more warps an SM has, the better it hides memory latency.

**Occupancy** is resident warps divided by the maximum the SM supports. The SM's thread slots,
block slots and registers are shared by the resident blocks, and any of them can be the limit
(step 6):

```
blocks per SM = min( maxThreadsPerSM / threadsPerBlock,
                     maxBlocksPerSM,
                     registersPerSM / (registersPerThread * threadsPerBlock) )
occupancy     = blocks per SM * warps per block / (maxThreadsPerSM / warpSize)
```

On an H100, blocks of 32 threads reach only 50 % (the 32 block slots run out first), blocks of 768
reach 75 % (2048 is not a multiple of 768), and 64 registers per thread allow only 1024 threads.
512-thread blocks with 31 registers fit 4 blocks (100 %), but with 33 registers only 3 (75 %): two
extra local variables cost a quarter of the SM, a **performance cliff**. With 31 registers the
thread slots and the registers both allow exactly 4 blocks, so the program names both limits.

The calculator is the simple model of the book. The real one also rounds registers and shared
memory up to allocation units and counts shared memory (session 13). For a compiled kernel, ask
the runtime with `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`, or let it choose a block size
with `cudaOccupancyMaxPotentialBlockSize()`; `nvcc -Xptxas -v` prints the registers each kernel
uses. High occupancy is a means, not the goal: always measure the running time too.

## 6. How the code differs from the slides

- Kernels are launched with `launch(kernel, grid, block, args...)` instead of `<<<grid, block>>>`;
  every CUDA call is wrapped in `CHECK(...)`.
- The device query does not print `clockRate`: that field of `cudaDeviceProp` is deprecated in
  recent CUDA versions.
- `05_divergence.cu` records the side of each branch instead of updating `x`, so the warp counts
  can be printed; the conditions are those of the slide's `branches` kernel.
- The occupancy calculator is plain host code from the slides' formulas. The runtime's
  `cudaOccupancy*` functions are only mentioned, because the laptop emulation cannot know the
  registers of a compiled kernel.

## 7. How to run

Laptop (no GPU; any C++17 compiler; clang++ works the same way):

```sh
cd ec2/02_gpu_architecture
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_device_query.cu -o 01_device_query && ./01_device_query
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_blocks_needed.cu -o 02_blocks_needed && ./02_blocks_needed
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_sync_reverse.cu -o 03_sync_reverse && ./03_sync_reverse
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_warps_and_lanes.cu -o 04_warps_and_lanes && ./04_warps_and_lanes
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_divergence.cu -o 05_divergence && ./05_divergence
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_occupancy.cu -o 06_occupancy && ./06_occupancy
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_barrier_placement.cu -o 07_barrier_placement && ./07_barrier_placement
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 08_divergence_2d.cu -o 08_divergence_2d && ./08_divergence_2d
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 09_warp_efficiency.cu -o 09_warp_efficiency && ./09_warp_efficiency
```

> **Easiest way on Colab:** upload [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) from
> this folder to [Google Colab](https://colab.research.google.com), choose *Runtime → Change
> runtime type → GPU*, and *Run all*. The notebook runs the nine steps in order and writes
> `cuda_compat.h` and every `.cu` file itself, so nothing needs to be cloned or uploaded. On a
> Colab T4 the device query and the occupancy table show a real GPU's limits (1024 threads and 16
> blocks per SM). The commands below are for a copy of the repository.

Google Colab (Runtime → Change runtime type → GPU), in a cell:

```sh
!nvcc -O2 -arch=native -I../common 01_device_query.cu -o 01_device_query && ./01_device_query
!nvcc -O2 -arch=native -I../common 02_blocks_needed.cu -o 02_blocks_needed && ./02_blocks_needed
!nvcc -O2 -arch=native -I../common 03_sync_reverse.cu -o 03_sync_reverse && ./03_sync_reverse
!nvcc -O2 -arch=native -I../common 04_warps_and_lanes.cu -o 04_warps_and_lanes && ./04_warps_and_lanes
!nvcc -O2 -arch=native -I../common 05_divergence.cu -o 05_divergence && ./05_divergence
!nvcc -O2 -arch=native -I../common 06_occupancy.cu -o 06_occupancy && ./06_occupancy
!nvcc -O2 -arch=native -I../common 07_barrier_placement.cu -o 07_barrier_placement && ./07_barrier_placement
!nvcc -O2 -arch=native -I../common 08_divergence_2d.cu -o 08_divergence_2d && ./08_divergence_2d
!nvcc -O2 -arch=native -I../common 09_warp_efficiency.cu -o 09_warp_efficiency && ./09_warp_efficiency
```

Every step prints its input and its result, then `PASS` or `FAIL`, and returns a non-zero exit
code on `FAIL`.
