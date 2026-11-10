# Memory and data locality: tiling with shared memory (chapter 5)

Session 13 (`13_memory_locality`): session 12 explained how the GPU runs threads; this folder is
about feeding them with data. It measures why the matrix product of session 11 is limited by
memory, puts variables in each of the CUDA memories, and then builds the tiled matrix product of
the slides step by step: first for sizes that are a multiple of the tile, then for any size, and
finally with a tile chosen from what the device offers.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_matmul_naive.cu`](01_matmul_naive.cu) | The matrix product of session 11, `matrixMulKernel`, on 4×4 matrices with 2×2 blocks. Every multiply-add reads one M and one N element from global memory, so the program prints 2 FLOP per 8 B: a compute-to-global-memory-access ratio of 0.25 FLOP/B, far below the 20 FLOP/B an H100 needs to stay busy. (slide: *Matrix multiplication: high potential, low reality*) | Row 0 of P is the dot product of row 0 of M with each column of N (56 = 0·0 + 3·6 + 6·5 + 2·4 for the first entry). The last lines give 8 FLOP and 8 loads (32 B) per thread: 0.25 FLOP/B, and the ratio stays 0.25 for any Width. |
| 2 | [`02_memory_types.cu`](02_memory_types.cu) | The kernel of the slides that declares one variable in each memory: `__constant__ float coeff[]` filled with `cudaMemcpyToSymbol`, `__device__ int negCount` in global memory, a `__shared__` tile per block, and `i`, `acc`, `left` in registers. A second kernel reads `negCount` after the first one ends, because a `__device__` variable lives as long as the application. (slide: *Declaring where a variable lives*) | `out[i] = in[i] - 0.5 · in[i-1]`, except at the first thread of each block of 8 (elements 0 and 8), where `left` is the thread itself. `negCount` is 4, the number of negative outputs, read by a different kernel. |
| 3 | [`03_tiled_matmul.cu`](03_tiled_matmul.cu) | The tiled kernel (`Mds`, `Nds`, `ph`, `Pvalue`) for Width = 8 and TILE_WIDTH = 4: each thread loads one element of each tile per phase, the first `__syncthreads()` guards read-after-write and the second write-after-read. Global loads per thread drop from 16 to 4, by exactly TILE_WIDTH. Only for Width a multiple of TILE_WIDTH. (slide: *The tiled kernel*) | P is the same as a plain triple loop would give. The last lines: 2 phases, 16 loads per thread without tiles and 4 with them, and the ratio rising from 0.25 to 1 FLOP/B (TILE_WIDTH times). |
| 4 | [`04_tiled_matmul_boundary.cu`](04_tiled_matmul_boundary.cu) | Boundary checks: ceil(Width / TILE_WIDTH) phases, a missing M or N element is loaded as 0, and only real P elements are stored. It runs Width 3 and 5 with TILE_WIDTH = 2, the cases of the slides; with Width 5 only 25 of the 36 threads store, but all of them load and reach both barriers. (slides: *The kernel with boundary checks*, *Code: launching the tiled kernel from the host*) | Both sizes give the right P. The grid line counts the threads that store: 9 of 16 for Width 3, 25 of 36 for Width 5, with 2 and 3 phases. |
| 5 | [`05_tile_size_occupancy.cu`](05_tile_size_occupancy.cu) | Shared memory as an occupancy limit: from `cudaGetDeviceProperties`, the blocks per SM allowed by threads and by shared memory for 8×8, 16×16 and 32×32 tiles and two heavier blocks of the slides' calculator. Then the host picks TILE_WIDTH 32 if `sharedMemPerBlock` holds 2 × 32 × 32 floats, else 16, and runs the kernel instantiated for that tile. (slides: *Memory usage limits occupancy too*, *Code: shared memory sized at launch time*) | The columns *by threads* and *by shared* show which limit wins: the tiled blocks are limited by threads, the 38 KB and 48 KB blocks by shared memory (occupancy 25 % and 50 % on the laptop). On a T4 the numbers change; the chosen tile is 32 on both. |
| 6 | [`06_arithmetic_intensity.cu`](06_arithmetic_intensity.cu) | The vector addition `C[i] = A[i] + B[i]` of the slides on 8 elements, and the arithmetic around it: 1 FLOP for 12 B per element (0.083 FLOP/B) against the H100 threshold 66.9 TFLOPS / 3.35 TB/s ≈ 20 FLOP/B. A roofline line for each kernel of the session (vector add, naive matmul, tiled 32 × 32, and the matrix product in principle for N = 1024) gives the TFLOPS it can reach, and the last line is the speed-of-light check of a measured run. (slides: *Two hardware limits: compute and memory bandwidth*, *Speed-of-light analysis: how close to the hardware?*, *Matrix multiplication: high potential, low reality*) | C = A + B. The threshold is 19.97 FLOP/B. Vector add (0.083 FLOP/B) and naive matmul (0.25) are memory-bound at 0.28 and 0.84 TFLOPS, 0.4 % and 1.3 % of the peak; 32 × 32 tiles reach 8 FLOP/B and 26.8 TFLOPS (40 %), still memory-bound; only the matrix product in principle (170.7 FLOP/B) is compute-bound. The last line: 1e9 elements in 4 ms move 12 GB, 3.00 TB/s = 89.6 % of the peak. |

The next folder, [`../04_performance`](../04_performance/), looks at how those global loads
reach DRAM: coalescing, bank conflicts, coarsening and other optimizations.

## 1. Compute-bound or memory-bound

A GPU has two peaks: arithmetic throughput (an H100: 66.9 TFLOPS in single precision) and memory
bandwidth (3.35 TB/s from global memory). Their quotient, about 20 FLOP/B on an H100, is the
threshold: a kernel that does fewer floating-point operations per byte it reads from global memory
is **memory-bound**, its cores wait for data; above it, it is **compute-bound**. The number that
decides is the kernel's **compute-to-global-memory-access ratio** (arithmetic intensity).

The matrix product could be very compute-bound: `2N³` FLOP on `12N²` bytes is `N / 6` FLOP/B,
171 for N = 1024. But the kernel of step 1 does not reuse anything: every iteration of its inner
loop loads `M[row*Width + k]` and `N[k*Width + col]` (8 B) for one multiply and one add (2 FLOP),
0.25 FLOP/B. At 3.35 TB/s that is at most 0.84 TFLOPS, about 1.3 % of the peak. The fix is to
read each element from DRAM fewer times, which is what tiling does.

## 2. Where a variable lives

| Declaration | Memory | Scope | Lifetime |
|-------------|--------|-------|----------|
| automatic scalar (`int i`, `float acc`) | register | thread | grid |
| automatic array (`float w[8]`) | local (in DRAM) | thread | grid |
| `__shared__` | shared | block | grid |
| `__device__` | global | grid | application |
| `__constant__` | constant | grid | application |

Registers and shared memory are on chip: fast, small, and free of DRAM traffic. One register
value serves one thread; one shared value serves every thread of the block, which is why shared
memory is the tool for cooperation. Constant memory is read-only in kernels, cached, and broadcast
to a whole warp when its threads read the same address; it is filled from the host with
`cudaMemcpyToSymbol` (step 2, and again in the convolution chapter). A `__device__` variable is
kept between launches: step 2 reads `negCount` with a second kernel. Use such globals sparingly.

A launch of 800 blocks of 256 threads creates 204 800 copies of an automatic scalar, 800 copies of
a `__shared__` array, and one copy of a `__constant__` one.

## 3. Tiling

The threads of one block read the same elements again and again: in a 4×4 product with 2×2
blocks, both threads of a row of the block read the same row of M, and both threads of a column
read the same column of N. Tiling makes them load each element **once** into shared memory and
reuse it from there. The k loop is cut into phases (strip-mining). In each phase:

1. every thread loads one element of the M tile and one of the N tile;
2. `__syncthreads()`: nobody reads the tile before everyone has written it (**read-after-write**,
   a true dependence);
3. every thread does TILE_WIDTH multiply-adds from shared memory, accumulating in the register
   `Pvalue` (register tiling);
4. `__syncthreads()`: nobody overwrites the tile with the next phase while others still read it
   (**write-after-read**, a false dependence that only the reuse of the same location creates).

Each element loaded from DRAM now serves TILE_WIDTH threads, so global loads drop by TILE_WIDTH
(step 3 prints 16 → 4 per thread) and the ratio rises from 0.25 to `0.25 · TILE_WIDTH` FLOP/B:
8 FLOP/B with 32×32 tiles, 26.8 TFLOPS on an H100. Still below 20 FLOP/B, so still memory-bound,
but 32 times faster than before. Removing either barrier gives wrong results that change from run
to run.

## 4. Boundary checks

When Width is not a multiple of TILE_WIDTH, the last tiles stick out of the matrices. Step 4
applies the rule of the slides: **every memory access tests its own indices**. Three tests: the M
load (`Row < Width && mCol < Width`), the N load (`nRow < Width && Col < Width`) and the P store
(`Row < Width && Col < Width`). A missing element is loaded as 0.0f, so it adds nothing to the dot
product. Threads that compute no P element must still load their share of the tiles and reach both
barriers, because the other threads of the block need those values. The number of phases becomes
ceil(Width / TILE_WIDTH).

## 5. Memory limits occupancy too

Shared memory is a per-SM resource like thread slots and registers (folder 02, step 6). A block
of the tiled kernel uses `2 · TILE_WIDTH² · 4` bytes: 2 KB for 16×16 tiles, 8 KB for 32×32. The
blocks an SM can hold are the minimum of what its threads, its shared memory and its block slots
allow; step 5 prints both limits for each block shape. On an H100 (228 KB and 2048 threads per SM),
full occupancy leaves at most 114 B of shared memory per thread; a block of 256 threads with 38 KB
runs far below it.

The slides size the tile at run time with `extern __shared__` and a third launch parameter. The
laptop emulation cannot reserve dynamic shared memory, so step 5 does the same choice with a
template: the host reads `sharedMemPerBlock` and `maxThreadsPerBlock`, picks 32 or 16, and launches
`matrixMulKernel<32>` or `matrixMulKernel<16>`.

## 6. How the code differs from the slides

- Kernels are launched with `launch(kernel, grid, block, args...)` instead of `<<<grid, block>>>`;
  every CUDA call is wrapped in `CHECK(...)`.
- `TILE_WIDTH` is a `const int` instead of a `#define`, and the kernels take `const float*`
  inputs.
- In step 2 the blocks have 8 threads and the tile 8 elements (the slide assumes 256), and the
  second kernel that reads `negCount` is added so the host can see it.
- Step 5 replaces `extern __shared__` (not available in the laptop emulation) by a kernel
  template, and prints the occupancy table instead of the slide's interactive calculator.
- Step 6 times nothing: it runs the slides' vector addition and prints the slides' arithmetic
  (threshold, roofline, speed of light) with the H100 figures as constants.
- The CPU locality example of the deck (`13_memory_locality/example_63.cpp`) stays in the deck's
  folder.

## 7. How to run

Laptop (no GPU; any C++17 compiler; clang++ works the same way):

```sh
cd ec2/03_memory_tiling
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_matmul_naive.cu -o 01_matmul_naive && ./01_matmul_naive
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_memory_types.cu -o 02_memory_types && ./02_memory_types
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_tiled_matmul.cu -o 03_tiled_matmul && ./03_tiled_matmul
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_tiled_matmul_boundary.cu -o 04_tiled_matmul_boundary && ./04_tiled_matmul_boundary
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_tile_size_occupancy.cu -o 05_tile_size_occupancy && ./05_tile_size_occupancy
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_arithmetic_intensity.cu -o 06_arithmetic_intensity && ./06_arithmetic_intensity
```

> **Easiest way on Colab:** upload [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) from
> this folder to [Google Colab](https://colab.research.google.com), choose *Runtime → Change
> runtime type → GPU*, and *Run all*. The notebook runs the six steps in order and writes
> `cuda_compat.h` and every `.cu` file itself, so nothing needs to be cloned or uploaded. On a
> Colab T4 step 5 shows a real GPU's limits (64 KB of shared memory and 1024 threads per SM). The
> commands below are for a copy of the repository.

Google Colab (Runtime → Change runtime type → GPU), in a cell:

```sh
!nvcc -O2 -arch=native -I../common 01_matmul_naive.cu -o 01_matmul_naive && ./01_matmul_naive
!nvcc -O2 -arch=native -I../common 02_memory_types.cu -o 02_memory_types && ./02_memory_types
!nvcc -O2 -arch=native -I../common 03_tiled_matmul.cu -o 03_tiled_matmul && ./03_tiled_matmul
!nvcc -O2 -arch=native -I../common 04_tiled_matmul_boundary.cu -o 04_tiled_matmul_boundary && ./04_tiled_matmul_boundary
!nvcc -O2 -arch=native -I../common 05_tile_size_occupancy.cu -o 05_tile_size_occupancy && ./05_tile_size_occupancy
!nvcc -O2 -arch=native -I../common 06_arithmetic_intensity.cu -o 06_arithmetic_intensity && ./06_arithmetic_intensity
```

Every step prints its input and its result, then `PASS` or `FAIL`, and returns a non-zero exit
code on `FAIL`.
