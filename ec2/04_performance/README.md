# Performance considerations: coalescing, banks, coarsening (chapter 6)

Session 14 (`14_performance`): sessions 12 and 13 showed how the GPU runs threads and which
memories it has. This folder is about matching a kernel's demands to the hardware's limits. It
counts the DRAM segments one warp touches, makes a matrix product with a column-major input
coalesced again by corner turning, uses `float4` vector loads, removes shared-memory bank
conflicts with padding, and then tries the optimizations that cut overhead inside a thread:
coarsening, unrolling and double buffering. It ends with the data-layout choice between an array
of structures and a structure of arrays.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_coalescing_segments.cu`](01_coalescing_segments.cu) | One warp of 32 threads runs `gatherKernel` with the five patterns of the slides' calculator: `a[t]`, `a[8 + t]`, `a[2t]`, `a[4t]` and `a[32t]`. Since `a[i] = i`, the values read tell the addresses, and the host counts the 128-byte segments and 32-byte sectors the warp touched and the share of moved bytes it used (100 % down to 12.5 %). (slide: *Coalescing: one request for a whole warp*) | The five lines go from 1 segment and 100 % of the moved bytes used (`a[t]`) to 32 segments and 12.5 % (`a[32t]`). `a[8 + t]` needs a second 128 B segment but still only 4 sectors. |
| 2 | [`02_matmul_column_major.cu`](02_matmul_column_major.cu) | The same matrix product loop with N row-major (`N[k*Width + col]`) and N column-major (`N[col*Width + k]`), on 4×4 matrices. It prints the N addresses of four consecutive threads at k = 0..3: 1 apart (coalesced) or Width apart (one transfer per thread). Both give the same P. (slide: *Code: the same loop, two layouts*) | In the row-major table each line of addresses goes up by 1 from thread to thread; in the column-major table by 4 (Width). The two P matrices are identical. |
| 3 | [`03_corner_turning.cu`](03_corner_turning.cu) | The tiled product with M row-major and N column-major, Width = 8, TILE = 4. The naive load of N would be strided; with corner turning thread (tx, ty) loads row `ph*TILE + tx` of column `bx*TILE + ty`, so consecutive tx read consecutive addresses, and stores it in `Nds[tx][ty]`. The inner loop does not change. (slide: *Code: corner turning in the tiled kernel*) | The naive addresses jump by 8 (Width) from thread to thread; with corner turning they are 0 1 2 3 in phase 0 and 4 5 6 7 in phase 1. P is still M · N. |
| 4 | [`04_float4_vector_add.cu`](04_float4_vector_add.cu) | `vecAdd4` on 18 elements: each thread adds one chunk of four with one 16-byte `float4` load per input and one store, a quarter as many threads, and one extra thread adds the 2 tail elements with scalar accesses. (slide: *Code: vector addition with float4*) | `z = x + y` for all 18 elements, including 16 and 17, which come from the tail thread. The last line: 5 threads, 4 chunks of four and 2 tail elements. |
| 5 | [`05_bank_conflicts_padding.cu`](05_bank_conflicts_padding.cu) | A 32×32 transpose through shared memory that stores `a[threadIdx.x][threadIdx.y]`. Each thread records the bank it wrote: with `a[TILE_DIM][TILE_DIM]` all 32 lanes of warp 0 hit bank 0 (32-way conflict, 32 passes); with one column of padding, `a[TILE_DIM][TILE_DIM + 1]`, they hit 32 different banks. Both transposes are correct. (slide: *Code: padding the corner-turning tile*) | `out` is `in` transposed (rows become columns). Without padding warp 0 stores to bank 0 thirty-two times; with padding to banks 0, 1, 2, ... , no conflict. |
| 6 | [`06_thread_coarsening.cu`](06_thread_coarsening.cu) | The coarsening loop of the slides: `scaleKernel` handles one element per thread with 16 threads, `scaleCoarsenedKernel` four consecutive elements per thread with 4 threads. Each element records which thread computed it, so the output shows thread 0 owning elements 0–3, thread 1 elements 4–7, and so on. (slide: *Code: a coarsening loop*) | Both `out` lines are 10 · in. The `thread` line goes 0, 1, 2, ... with one unit per thread and 0 0 0 0 1 1 1 1 ... with coarsening factor 4. |
| 7 | [`07_loop_unrolling.cu`](07_loop_unrolling.cu) | Loop unrolling: each thread scans its chunk of 4 values in a local array `float x[4]`. With `#pragma unroll` the loops disappear and every `x[c]` gets a constant index, so the array can live in registers; with `#pragma unroll 1` the loops stay. Same result; the difference shows in the compiled code. (slide: *Code: unrolling and coarsening loops*) | Each group of four is a running sum: 1 2 3 1 becomes 1 3 6 7. The unrolled and the not unrolled kernels print the same values. |
| 8 | [`08_double_buffering.cu`](08_double_buffering.cu) | Double buffering: 8 threads repeat `buffer[my] = f(buffer[another])` 3 times. With one shared buffer each iteration needs two barriers (write-after-read and read-after-write); with two buffers that swap places, one barrier is enough. Both give the same values, with 6 and 3 barriers. (slide: *Code: double buffering*) | After 3 iterations thread t holds `in[(t + 3) % 8] + 30`: 33 34 ... 37 30 31 32. Both versions agree, with 6 and 3 barriers. |
| 9 | [`09_aos_vs_soa.cu`](09_aos_vs_soa.cu) | Array of structures versus structure of arrays: `sumXAoS` reads `p[i].x` from `struct Particle {x, y, z, m}`, `sumXSoA` reads `p.x[i]` from separate arrays. The results agree; a warp touches 16 sectors (25 % used) in the AoS layout and 4 sectors (100 % used) in the SoA one. (slide: *Code: array of structures versus structure of arrays*) | Both outputs are 2 · x. The last two lines: 16 sectors and 25 % used for AoS, 4 sectors and 100 % for SoA. |
| 10 | [`10_effective_bandwidth.cu`](10_effective_bandwidth.cu) | Measuring before optimizing: the `vecAdd4` kernel of step 4, timed over several launches with the `Timer` of `cuda_compat.h` (the emulation has no CUDA events), and its effective bandwidth, 3 × N × 4 bytes / time, compared with the H100's 3350 GB/s. N is 16,777,218 floats on the GPU and 4098 on the laptop. (slide: *Code: measuring before optimizing*) | The first 8 values of z = x + y, then the bytes per launch (201,326,616 on the GPU, 49,176 on the laptop), the time per launch and the GB/s. On the laptop the emulation starts one thread per CUDA thread, so its GB/s is tiny and says nothing about a GPU. On a Colab T4, whose peak is 320 GB/s, the share of the H100's peak cannot pass 9.6 %. |
| 11 | [`11_exercise_access_patterns.cu`](11_exercise_access_patterns.cu) | The `foo_kernel` of exercise 14.1 as written, run with 2 blocks of 256 threads and checked against the CPU. For warp 0 the host prints each access's indices for lanes 0..3 (L7 with j = 1), its stride and the 128 B segments the warp touches; for the shared arrays, the banks used and the conflict degree. (slide: *Exercises 14.1–14.5*) | The solution of 14.1: `a[i]`, `b[j*blockDim.x*gridDim.x + i]` and `d[i + 8]` are coalesced (stride 1; `d` touches 2 segments because of the shift by 8), `c[i*4 + j]` and `e[i*8]` are not (stride 4 and 8: 4 and 8 segments). In shared memory only `bc_s[threadIdx.x*4]` conflicts: 8 banks, 4-way. |

The next folder, [`../05_conv`](../05_conv/), applies these tools to the first parallel pattern:
convolution with constant memory and tiles with halos.

## 1. DRAM bursts and coalescing

A DRAM cell is read through long, slow bit lines, so one access takes tens of nanoseconds. To
compensate, each access reads a whole run of consecutive locations at once, a **burst**, and the
caches move data in lines. Using every byte of a burst is far faster than touching scattered
locations.

All threads of a warp execute the same load instruction at the same time. If their addresses are
consecutive (thread 0 reads X, thread 1 reads X + 1, ...), the hardware **coalesces** them into a
few transfers of whole segments. The slides model memory as aligned 128-byte segments (32 floats);
current GPUs move 32-byte sectors inside them. Step 1 counts both for one warp:

| Pattern | 128 B segments | 32 B sectors | Bytes used |
|---------|----------------|--------------|------------|
| `a[t]` | 1 | 4 | 100 % |
| `a[8 + t]` | 2 | 4 | 100 % |
| `a[2t]` | 2 | 8 | 50 % |
| `a[4t]` | 4 | 16 | 25 % |
| `a[32t]` | 32 | 32 | 12.5 % |

A misaligned start costs one more segment; strides waste most of each transfer. Stores count too:
a partial write can force a read-modify-write of the whole unit.

To judge an access, read its index from the warp's point of view: which part is the same for all
threads, and which part changes with `threadIdx.x`. In `N[k*Width + col]` only `col` changes, by
1 per thread: coalesced. In `N[col*Width + k]` consecutive threads are Width elements apart: one
transfer per thread (step 2). `M[row*Width + k]` is the same address for the whole warp, a
broadcast, which is fine.

## 2. Corner turning

When the layout is wrong for the access (N stored column-major, for example the transpose of a
row-major matrix), there are three fixes: remap threads to data, change the layout, or load
coalesced into shared memory and do the awkward access there. **Corner turning** is the third:
in the tiled product the threads swap the roles of `threadIdx.x` and `threadIdx.y` in the load of
N, so consecutive `tx` read down a column, which in column-major order is consecutive in memory
(step 3 prints 0, 1, 2, 3 against 0, 8, 16, 24). The value goes to `Nds[tx][ty]`, so the tile in
shared memory has the usual orientation and the inner loop stays the same. Shared memory is SRAM
and needs no coalescing, but that strided store can cause bank conflicts (section 4).

## 3. Vector loads

Casting a `float*` to `float4*` makes the compiler emit 16-byte loads and stores (step 4): two
vector loads replace eight scalar ones, neighbouring threads still read neighbouring chunks, and a
quarter as many threads keep the same bytes in flight. When N is not a multiple of 4, the last
elements are handled with scalar accesses. The address of a `float4` must be a multiple of 16:
`cudaMalloc` returns aligned memory, an offset like `x + 1` would not be.

## 4. Shared memory banks and padding

Shared memory is split into 32 **banks** of 4 bytes; consecutive words go to consecutive banks,
so word `w` is in bank `w % 32`. Threads of a warp that hit different banks are served in
parallel; several threads on different addresses of the same bank are a **bank conflict** and are
served one at a time. In a 32×32 tile, `a[threadIdx.x][threadIdx.y]` is word `x*32 + y`: the 32
lanes of a warp share `y` and all hit bank `y`, a 32-way conflict. With one unused column,
`a[TILE_DIM][TILE_DIM + 1]`, the word is `x*33 + y` and the bank `(x + y) % 32`: all different
(step 5). The costs are 32 more floats of shared memory per tile and slightly more index
arithmetic (`x*33` is not a shift). The other fix is to remap threads so that a warp walks along a
row.

## 5. Coarsening, unrolling, double buffering

**Thread coarsening** gives each thread several units of work (step 6). The finest granularity
scales transparently, but parallelism has overhead: redundant loads, redundant work, barriers,
index arithmetic. If the hardware would serialize the threads anyway, a coarser thread pays that
overhead fewer times. Its pitfalls: coarsening where there is no overhead to remove, coarsening so
much that the GPU is left partly empty or the last wave is partial, and needing so many registers
or so much shared memory per thread that occupancy drops. The `float4` addition of step 4 is
already coarsening, and later chapters coarsen stencils, histograms, reductions and scans.

**Loop unrolling** replaces a loop by copies of its body (step 7). It removes branch instructions,
which GPUs do not predict, and exposes independent instructions the scheduler can overlap. The
compiler unrolls small loops with constant bounds by itself; `#pragma unroll N` sets the factor
and `#pragma unroll 1` forbids it. For a coarsening loop it matters even more: after unrolling,
`x[c]` becomes `x[0] ... x[3]`, constant indices, so the local array can be promoted from local
memory (in DRAM) to registers. On a GPU, `nvcc -Xptxas -v -arch=native 07_loop_unrolling.cu`
prints the registers and the stack frame each kernel uses.

**Double buffering** (step 8) removes barriers that only guard false dependences. With one
buffer, each iteration reads another thread's slot, waits (write-after-read), writes its own slot
and waits again (read-after-write). With separate input and output buffers that swap after every
iteration, a write can never overwrite a value another thread is still reading in the same
iteration, so only the read-after-write barrier remains. The price is a second buffer.

## 6. Data layout: AoS or SoA

An array of structures (`Particle p[n]`, 16 bytes each) puts the `x` of consecutive particles
16 bytes apart: a warp reading only `x` touches 512 bytes to use 128 (step 9). A structure of
arrays keeps all `x` contiguous: 128 consecutive bytes, all of them used. When each thread uses
every field of its particle, AoS can still be fine, especially with one `float4` load per
particle. The same effect on a CPU is measured in `14_performance/example_64.cpp`.

## 7. How the code differs from the slides

- Kernels are launched with `launch(kernel, grid, block, args...)` instead of `<<<grid, block>>>`;
  every CUDA call is wrapped in `CHECK(...)`.
- Step 1 runs the patterns of the slides' coalescing calculator as a real kernel and counts
  32-byte sectors next to the slides' 128-byte segments.
- In steps 2 and 3 the address formulas are small `__host__ __device__` functions, so the kernel
  and the printed address table use the same code. The naive load of step 3 is printed, not run.
- Step 5 uses a transpose to give the padded tile of the slides a full kernel, and the kernel
  computes the bank of each store from the address.
- The slides' `foo(i)` becomes a scale (step 6) and a per-chunk prefix sum (step 7); `f(v)` of
  step 8 is `v + 10`, and `swap` is written out with a temporary pointer.
- The two AoS and SoA kernels, both called `sumX` on the slide, are `sumXAoS` and `sumXSoA`.
- Step 10 times with the `Timer` of `cuda_compat.h` instead of CUDA events, which the emulation
  does not have. Nsight Compute (`ncu`) needs a real GPU and is not part of the steps.
- Step 11 runs the kernel of exercise 14.1 as written; the table of strides, segments and banks
  is computed by the host from the same index expressions.

## 8. How to run

Laptop (no GPU; any C++17 compiler; clang++ works the same way):

```sh
cd ec2/04_performance
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_coalescing_segments.cu -o 01_coalescing_segments && ./01_coalescing_segments
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_matmul_column_major.cu -o 02_matmul_column_major && ./02_matmul_column_major
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_corner_turning.cu -o 03_corner_turning && ./03_corner_turning
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_float4_vector_add.cu -o 04_float4_vector_add && ./04_float4_vector_add
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_bank_conflicts_padding.cu -o 05_bank_conflicts_padding && ./05_bank_conflicts_padding
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_thread_coarsening.cu -o 06_thread_coarsening && ./06_thread_coarsening
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_loop_unrolling.cu -o 07_loop_unrolling && ./07_loop_unrolling
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 08_double_buffering.cu -o 08_double_buffering && ./08_double_buffering
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 09_aos_vs_soa.cu -o 09_aos_vs_soa && ./09_aos_vs_soa
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 10_effective_bandwidth.cu -o 10_effective_bandwidth && ./10_effective_bandwidth
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 11_exercise_access_patterns.cu -o 11_exercise_access_patterns && ./11_exercise_access_patterns
```

With GCC (not Apple's `g++`, which is clang), `-Wall` warns that it ignores `#pragma unroll` in
step 7; the program is still correct.

> **Easiest way on Colab:** upload [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) from
> this folder to [Google Colab](https://colab.research.google.com), choose *Runtime → Change
> runtime type → GPU*, and *Run all*. The notebook runs the eleven steps in order and writes
> `cuda_compat.h` and every `.cu` file itself, so nothing needs to be cloned or uploaded. The
> commands below are for a copy of the repository.

Google Colab (Runtime → Change runtime type → GPU), in a cell:

```sh
!nvcc -O2 -arch=native -I../common 01_coalescing_segments.cu -o 01_coalescing_segments && ./01_coalescing_segments
!nvcc -O2 -arch=native -I../common 02_matmul_column_major.cu -o 02_matmul_column_major && ./02_matmul_column_major
!nvcc -O2 -arch=native -I../common 03_corner_turning.cu -o 03_corner_turning && ./03_corner_turning
!nvcc -O2 -arch=native -I../common 04_float4_vector_add.cu -o 04_float4_vector_add && ./04_float4_vector_add
!nvcc -O2 -arch=native -I../common 05_bank_conflicts_padding.cu -o 05_bank_conflicts_padding && ./05_bank_conflicts_padding
!nvcc -O2 -arch=native -I../common 06_thread_coarsening.cu -o 06_thread_coarsening && ./06_thread_coarsening
!nvcc -O2 -arch=native -I../common 07_loop_unrolling.cu -o 07_loop_unrolling && ./07_loop_unrolling
!nvcc -O2 -arch=native -I../common 08_double_buffering.cu -o 08_double_buffering && ./08_double_buffering
!nvcc -O2 -arch=native -I../common 09_aos_vs_soa.cu -o 09_aos_vs_soa && ./09_aos_vs_soa
!nvcc -O2 -arch=native -I../common 10_effective_bandwidth.cu -o 10_effective_bandwidth && ./10_effective_bandwidth
!nvcc -O2 -arch=native -I../common 11_exercise_access_patterns.cu -o 11_exercise_access_patterns && ./11_exercise_access_patterns
```

Every step prints its input and its result, then `PASS` or `FAIL`, and returns a non-zero exit
code on `FAIL`.
