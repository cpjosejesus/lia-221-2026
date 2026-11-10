# Convolution on the GPU (chapter 7)

Study guide for the convolution chapter: constant memory, caches, and shared-memory tiles
with halo cells. You already know what a convolution computes, so this guide is about how to
make a GPU compute it fast and why each version is faster than the last.

## 1. The pattern in one paragraph

A convolution replaces every element of an array with a weighted sum of the elements around
it. The weights form a small array called the **filter** (also mask or kernel; here "filter",
so it doesn't get confused with a CUDA kernel). Every output is independent of every other
output, so there is plenty of parallelism: one thread per output is the obvious mapping. The
hard part is memory. Neighbouring outputs need almost the same inputs. A 5×5 filter reads each
input pixel 25 times, once for each output that covers it. A fast kernel makes most of those
reads come from on-chip memory instead of DRAM. Chapter 7 is a case study in doing that, and
it carries over to every pattern that reads a neighbourhood (stencils in chapter 8, image
filters, and the 2D layers of neural networks).

## 2. Notation and the sequential baseline

`N` is the input image (`height × width`, row-major), `F` the filter of radius `r` and size
`m = 2r + 1`, `P` the output. Outputs near the border need inputs outside the image. Those
missing inputs are the **ghost cells**, and here they count as 0.

```cpp
for (int outRow = 0; outRow < height; ++outRow) {
  for (int outCol = 0; outCol < width; ++outCol) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < m; ++fRow) {
      for (int fCol = 0; fCol < m; ++fCol) {
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += F[fRow * m + fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[outRow * width + outCol] = Pvalue;
  }
}
```

Work: `2·m²` FLOP per output (one multiply and one add per tap). For a 4096×4096 image and a
5×5 filter that is 0.84 GFLOP. At best the data has to cross the memory bus once each way:
read N once and write P once. That is 8 bytes per output, which gives the **ideal arithmetic
intensity of convolution**:

```
ideal intensity = 2·m² FLOP / 8 B = m²/4 FLOP/B
3×3 → 2.25    5×5 → 6.25    7×7 → 12.25    9×9 → 20.25    11×11 → 30.25
```

An H100 (SXM) delivers 66.9 TFLOPS FP32 and 3.35 TB/s of DRAM bandwidth, so its **ridge
point** is 66.9 / 3.35 ≈ 20 FLOP/B. A kernel whose intensity is below 20 is memory-bound and
runs at `intensity × 3.35 TB/s` at best. A 3×3 convolution is memory-bound however well it is
written (2.25 < 20). An 11×11 convolution can be compute-bound (30.25 > 20), but only if the
kernel gets close to the ideal reuse.

## 3. Kernel (a): one thread per output

```cpp
__global__ void convolution_2D_basic_kernel(
    const float* N, const float* F, float* P, int r, int width, int height) {
  int outCol = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int outRow = (int)(blockIdx.y * blockDim.y + threadIdx.y);

  if (outRow < height && outCol < width) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
        int inRow = outRow - r + fRow;
        int inCol = outCol - r + fCol;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[outRow * width + outCol] = Pvalue;
  }
}
```

- **Mapping.** A 2D grid of 2D blocks (16×16 in `conv.cu`). Thread `(outRow, outCol)` owns one
  output. The grid is rounded up (`cdiv`), so the outer `if` discards the extra threads of the
  last row and column of blocks.
- **Register accumulator.** `Pvalue` stays in a register and P is written once. Writing
  `P[...] +=` inside the loop would cost a global read and write per tap.
- **Ghost cells and divergence.** The inner `if` skips taps that fall outside the image. Only
  threads within `r` of the border take the false branch, so control divergence is confined to
  the warps on the image edge. For a 4096×4096 image, 5×5 filter and 16×16 blocks, the
  border blocks are 4·256 − 4 = 1020 of 65 536 (1.6 %). Only some warps of those blocks
  diverge, and only for some taps. The cost is negligible on large images and noticeable only
  on small ones.
- **Arithmetic intensity.** Every tap loads one N element and one F element (8 bytes) and
  does 2 FLOP, so the intensity is **0.25 FLOP/B**. On an H100 that bounds the kernel at
  0.25 × 3.35 TB/s ≈ 0.84 TFLOPS, about 1.3 % of peak. (In reality L1/L2 absorb many of these
  loads. The intensity counts the loads the code issues, which is the worst case.)

## 4. Constant memory and the cache hierarchy

```
   per thread   registers                     ─┐
   per block    shared memory (scratchpad)     │  on each SM (H100: 132 SMs)
   per SM       L1 cache                       │
   per SM       constant cache (read-only)    ─┘
   whole GPU    L2 cache, shared by all SMs
   whole GPU    DRAM (HBM): global memory, and the 64 KB constant memory
   (fast, small, close to the cores at the top; slow, large, far away at the bottom)
```

Caches are **transparent**: the program issues ordinary loads, and the hardware decides what
to keep and evicts it when it wants. Shared memory is a **scratchpad**: the program decides
what goes in, when, and for how long, and must synchronise the threads that use it. On H100
the L1 cache and shared memory share 256 KB of SRAM per SM, and the L2 is 50 MB.

The filter has three properties that make it ideal for **constant memory**:

1. **It is small.** 5×5 floats are 100 bytes, and constant memory holds 64 KB.
2. **It does not change while the kernel runs.** So the constant cache never has to handle a
   write or keep copies on different SMs coherent. That is what makes it cheap.
3. **All threads of a warp read the same element at the same time.** Every thread runs the
   same `(fRow, fCol)` loop in the same order. When all lanes of a warp ask for the same
   address, the constant cache serves them with one access (a broadcast). When lanes ask for
   different addresses, the accesses are serialised. That is why the image itself does not
   belong there.

The filter is declared at file scope on the device side, the host fills it once with
`cudaMemcpyToSymbol`, and kernel (b) reads it directly:

```cpp
__constant__ float F_c[FILTER_DIM][FILTER_DIM];
```

```cpp
CHECK(cudaMemcpyToSymbol(F_c, problem.weights.data(), weightBytes));
```

```cpp
Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
```

The kernel no longer receives `F` as a pointer: `F_c` is a global symbol visible to every
kernel in the file. Its loads hit the constant cache after the first warp. DRAM traffic per
tap drops to the 4 bytes of N, so the intensity doubles to **0.5 FLOP/B** (bound ≈ 1.7 TFLOPS
on H100).

**Filter by value.** Kernel arguments are themselves placed in a constant bank by the driver.
A small filter can therefore travel as a struct passed by value, which gets the same broadcast
behaviour without a `__constant__` symbol. This is kernel (a') in `conv.cu`:

```cpp
struct Filter {
  float w[FILTER_DIM][FILTER_DIM];
};

__global__ void convolution_2D_struct_kernel(
    const float* N, Filter f, float* P, int width, int height);
```

The limit is the size of the parameter block. The classic limit is 4 KB, and recent toolkits
allow more. The struct is also copied at every launch, which is fine for a filter that changes
between launches.

## 5. Tiling with halo cells

Constant memory took care of F. The other half of the traffic is N, and every N element is
still loaded up to m² times by different threads. The fix is the usual one from the
matrix-multiplication chapter: a block loads the part of N it needs into shared memory once,
and all its threads read from there.

**Output tile and input tile.** The outputs a block computes form its *output tile*. The
inputs they need form the *input tile*, which is the output tile plus a **halo** of width `r`
on every side.

```
  output tile t×t, filter m×m (m = 2r+1)  →  input tile (t+m−1)×(t+m−1)

  ┌───────────────┐   ← halo, r cells wide (belongs to neighbouring output tiles)
  │ ┌───────────┐ │
  │ │  output   │ │
  │ │   tile    │ │
  │ └───────────┘ │
  └───────────────┘
```

The halo is the overhead. Compare the input tile with the output tile:

| output tile | filter | input tile | input/output |
|---|---|---|---|
| 4×4 = 16 | 5×5 | 8×8 = 64 | 4.0 |
| 16×16 = 256 | 5×5 | 20×20 = 400 | 1.56 |
| 28×28 = 784 | 5×5 | 32×32 = 1024 | 1.31 |
| 22×22 = 484 | 11×11 | 32×32 = 1024 | 2.12 |

Small tiles and large filters spend most of the loads on the halo.

**Two thread organisations.** The input tile is bigger than the output tile, so the block
size has to match one of them:

- *Block = input tile.* Each thread loads exactly one input element, and then only the inner
  `t×t` threads compute. Loading is simple, but some threads sit idle during the computation.
  This is kernel (c).
- *Block = output tile.* Each thread computes exactly one output, and the threads loop to load
  the larger input tile (some load two or more elements). Nobody idles while computing, but
  the loading code is messier. Kernel (d) is a variant of this choice that does not load the
  halo at all.

## 6. Kernel (c): block = input tile

With `FILTER_RADIUS = 2`, the output tile is 32 − 4 = 28 wide:

```cpp
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__global__ void convolution_tiled_2D_const_mem_kernel(const float* N,
                                                      float* P,
                                                      int width,
                                                      int height) {
  int col = (int)(blockIdx.x * OUT_TILE_DIM) + (int)threadIdx.x - FILTER_RADIUS;
  int row = (int)(blockIdx.y * OUT_TILE_DIM) + (int)threadIdx.y - FILTER_RADIUS;

  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int tileCol = (int)threadIdx.x - FILTER_RADIUS;
  int tileRow = (int)threadIdx.y - FILTER_RADIUS;
  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
      float Pvalue = 0.0f;
      for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
          Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
        }
      }
      P[row * width + col] = Pvalue;
    }
  }
}
```

The tiled kernels need tile sizes known at compile time (they size the `__shared__` arrays),
so the filter radius is a compile-time constant too: `FILTER_RADIUS` is 2 (a 5×5 filter, as in
the worked numbers of this guide).

Step by step:

1. **Launch geometry.** Blocks of 32×32 = 1024 threads. One block per 28×28 output tile, so
   the grid is `cdiv(width, 28) × cdiv(height, 28)`. Block `bx` starts its *output* tile at
   column `28·bx`, so its *input* tile starts `r` columns earlier. That is where the
   `− FILTER_RADIUS` comes from, and why `col` and `row` can be negative.
2. **Load phase.** Every thread loads one element. Ghost cells get an explicit 0. This is the
   only place where image borders are handled: the compute loop below has no `if`, which
   removes the divergence of kernel (a).
3. **Barrier.** A thread is about to read elements loaded by up to 24 other threads.
   `__syncthreads()` makes all the loads finish before any read. (Remove it and the CPU
   emulation under ThreadSanitizer reports a data race on `N_s`. Try it.)
4. **Compute phase.** The output position inside the tile is `threadIdx − r`. Threads whose
   `tileCol` or `tileRow` falls outside `[0, 28)` are halo loaders: they are done. The
   `col/row` test discards outputs past the image edge in the last tiles. Since `tileRow + fRow`
   is the input-tile coordinate of the tap, the indexing has no `− r` correction.

**Arithmetic intensity.** For an internal block with output tile `t×t` and filter `m×m`, the
block does `t²·m²·2` FLOP, loads `(t+m−1)²` input elements and stores `t²` outputs:

```
intensity = (t² · m² · 2) / (((t+m−1)² + t²) · 4)  FLOP/B
5×5, t = 28:  (784 · 25 · 2) / ((1024 + 784) · 4) = 39 200 / 7 232 ≈ 5.42   (ideal 6.25)
```

From 0.5 FLOP/B (constant memory alone) to 5.42 is an 11× improvement, already 87 % of the
ideal for this filter. (Counting loads only, as some analyses do, gives 39 200 / 4 096 = 9.57.
That ignores the stores, which every version must pay, and its large-tile limit is m²/2
instead of m²/4.) `conv.cu` prints the formula for several tile and filter sizes and marks
each entry against the H100 ridge of 20 FLOP/B:

```
  t \ m       3x3       5x5       7x7       9x9     11x11     13x13
      8     1.76M     3.85M     6.03M     8.10M     9.98M    11.66M
     16     1.99M     4.88M     8.48M    12.46M    16.62M    20.80C
     28     2.10M     5.42M     9.90M    15.27M    21.29C    27.79C
     32     2.11M     5.52M    10.17M    15.80M    22.22C    29.23C
     64     2.18M     5.87M    11.16M    17.88M    25.89C    35.06C
  ideal     2.25M     6.25M    12.25M    20.25C    30.25C    42.25C
```

Read the table along both directions. Bigger tiles push every column towards its ideal, but
the gain flattens quickly (5×5: 5.42 at t = 28, 5.87 at t = 64). Bigger filters raise the
ceiling. Only filters of 11×11 and up cross the ridge with realistic tiles. A 9×9 filter is
memory-bound for any tile you could actually allocate (exercise 4).

**Inefficiencies of kernel (c).**

- **Idle threads.** 1024 threads load, but only 784 (76.6 %) compute. With r = 5 the output
  tile is 22×22 and only 47 % compute.
- **Awkward output tile width.** Tiles 28 wide are not a power of two. Rows of a block's
  output start at columns that are multiples of 28, so many warps' stores straddle 32-byte
  segments, and partial warps appear.
- **Larger filters shrink the output tile.** For a fixed block of 32×32, the output tile is
  `32 − 2r` wide, so the halo fraction grows with r. This is exactly when you would want the
  most reuse.

## 7. Kernel (d): tiled with cached halo

The halo elements of one block are the interior elements of its neighbours. Those
neighbours load them into their own shared memory, which brings them into L2. Kernel (d)
relies on that: the block is the output tile (32×32 threads, 32×32 outputs), shared memory
holds only the block's own input elements, and halo taps go straight to global memory, where
they very likely hit in L2.

```cpp
int sharedCol = (int)threadIdx.x - FILTER_RADIUS + fCol;
int sharedRow = (int)threadIdx.y - FILTER_RADIUS + fRow;
bool insideTile =
    sharedCol >= 0 && sharedCol < TILE_DIM && sharedRow >= 0 && sharedRow < TILE_DIM;

if (insideTile) {
  Pvalue += F_c[fRow][fCol] * N_s[sharedRow][sharedCol];
} else {
  int inRow = row - FILTER_RADIUS + fRow;
  int inCol = col - FILTER_RADIUS + fCol;
  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
    Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
  }
}
```

`(sharedRow, sharedCol)` is the tap's position inside the block's tile. A tap inside the tile
reads shared memory. A halo tap reads global memory, which usually means L2. A ghost cell
(outside the image) is skipped, so it counts as 0.

What changes:

- Every thread loads and every thread computes. There are no idle threads, and tiles are
  32 wide, so loads and stores are aligned.
- Shared memory is `TILE_DIM²` instead of `(TILE_DIM + 2r)²` elements, and the size does not
  depend on r.
- The price is control divergence inside the loop. Threads near the tile edge take the
  global-memory branch for some taps while their warp-mates take the shared branch. The halo
  loads also have L2 latency instead of shared-memory latency. For TILE_DIM = 32 and r = 2
  the distinct halo elements are 36² − 32² = 272, 21 % of the block's distinct inputs.

Which of (c) and (d) wins depends on the GPU (L2 bandwidth and latency) and on r. Measure it.

## 8. Performance summary (H100, FP32, 5×5 filter)

| kernel | where F lives | intensity (FLOP/B) | bound = intensity × 3.35 TB/s |
|---|---|---|---|
| (a) basic | global | 0.25 | 0.84 TFLOPS |
| (a') struct argument | parameter (constant) bank | 0.5 | 1.7 TFLOPS |
| (b) constant memory | constant cache | 0.5 | 1.7 TFLOPS |
| (c) tiled, t = 28 | constant + shared tile | 5.42 | 18 TFLOPS |
| ideal (read once, write once) | | 6.25 | 21 TFLOPS |

All of them are memory-bound on H100 for 5×5. Tiling gets within 13 % of the best a 5×5
filter can ever do, and the rest of the gap is the halo. These are upper bounds from DRAM
traffic. Measured numbers on a real GPU land below them, and kernels (a) and (b) often beat
their bound because L1/L2 catch the repeated loads. Run `conv` on Colab and compare. In the
CPU emulation the timings measure the cost of starting and synchronising threads, not memory,
so only the PASS column means anything there.

## 9. Pitfalls

- **Forgetting the barrier** between loading the tile and reading it. The results are
  sometimes right and sometimes wrong. ThreadSanitizer on the emulation catches it.
- **Returning early** from threads that only load (`if (!inner) return;`) before a
  `__syncthreads()`. A barrier that some threads never reach is undefined behaviour. Guard the
  compute part with an `if` instead.
- **Leaving ghost slots in shared memory uninitialised.** Kernel (c) has no bounds test in its
  inner loop, so ghost slots must hold 0.
- **Unguarded stores** in partial tiles at the right and bottom edges. These are out-of-bounds
  writes that may corrupt other buffers silently.
- **Mixing up the two tile sizes**: grid from the *output* tile, block from the *input* tile
  (kernel c). Getting one wrong leaves gaps or computes tiles twice.
- **Putting large or per-thread-varying data in constant memory.** Different addresses in a
  warp are serialised, and the 64 KB limit is hard.
- **Unsigned arithmetic.** `blockIdx.x * OUT_TILE_DIM + threadIdx.x - r` is unsigned and wraps
  around for the first block. Cast to `int` before subtracting, as `conv.cu` does.

## 10. Ideas for your project

### The bike-sharing examples in this folder

`bikeshare_demand_heatmap.cu` looks at a city bike-sharing system: a 6 km × 6 km map in metres
with 48 docking stations. It asks **where riders want a bike during each rush hour**, so that the
operator knows where a station is missing and which stations the rebalancing trucks should fill
first.

1. The program generates synthetic ride requests, each one the place where a rider opened the
   app. They come from the file's own SplitMix64 generator with a fixed seed, so every run is
   the same. The requests cluster around homes, condos, offices, a university and a market, and
   15 % fall anywhere on the map. Homes are busy around 08:00, and offices and the university
   around 18:00. The first few requests are printed with their time and position.
2. The requests of one window (07:00–09:00 or 17:00–19:00) are binned into a 125×125 grid of
   48 m cells with a plain host loop (binning on the GPU is chapter 9). Single cells are noisy:
   a cell receives a handful of requests, mostly by chance, and the busiest raw cell is printed
   to show it.
3. The histogram is smoothed with a 13×13 Gaussian (σ = 3 cells ≈ 144 m: wide enough to merge
   the requests of one neighbourhood, narrow enough to keep two clusters apart). Its weights are
   normalised to add up to 1, so the smoothed map keeps the total number of requests. They live
   in **constant memory**, and the smoothing is the **tiled kernel (c)** with 32×32 input tiles
   and 20×20 output tiles. The grid side, 125, is not a multiple of 20, so the partial tiles get
   tested. This filter is large enough for the table above to matter: t = 20, m = 13 gives
   (400·169·2) / ((1024 + 400)·4) ≈ 23.7 FLOP/B, which is **compute-bound** on H100. Here
   tiling is what lets the GPU stop waiting for memory. Ghost cells are 0, which is also the
   honest value here: outside the map there are no riders.
4. The five strongest local maxima, at least 15 cells (720 m) apart, are printed with their
   position in metres, the smoothed requests per cell, the number of requests within 250 m and
   the nearest station. When that station is more than 500 m away, the zone is a candidate for
   a new station. Otherwise the program prints how many of the station's docks hold a bike and
   whether a truck should bring bikes before the rush. The morning ranking is led by the
   residential clusters and the evening ranking by the offices and the university, so the
   trucks' plan should follow the clock.
5. Every smoothed map is compared with the CPU version of the same convolution, with a PASS or
   FAIL per window and `ALL PASS` at the end.

The number of requests comes from `BY_BACKEND`: 2 million on a GPU and 40 000 in the
emulation (`-DCPU_RIDE_REQUESTS=...` changes the second one, for example for a quicker
ThreadSanitizer run).

### Where this pattern may appear in your project

- Counts per map cell are noisy when each cell receives only a few events. Smoothing them with a
  small Gaussian turns a scatter of points into zones you can rank. Which events in your
  project have a position on the map?
- If idle workers (couriers, for example) should wait somewhere, a smoothed map of where demand
  appears is one way to choose the spot. Does the best spot change with the time window, as
  lunch and dinner might?
- A wide filter (radius 5 or more) with a shared-memory tile and the weights in constant memory
  can be compute-bound, while a 3×3 one never is. How wide does your filter need to be for the
  cell size you chose?
- What is the honest value outside your map: zero, the value at the edge, or something else?
  The answer decides how you fill the ghost cells.
- Smoothing is not only for demand. A map of travel times, or of rejected or failed requests per
  cell, can be easier to read after a blur.

## 11. Files and how to run

| file | contents |
|---|---|
| `conv.cu` | kernels (a), (a'), (b), (c), (d), CPU reference, PASS/FAIL, timing table, intensity table |
| `bikeshare_demand_heatmap.cu` | rush-hour demand heatmap of a bike-sharing city with the tiled constant-memory kernel: hot zones for new stations and rebalancing |
| `../common/cuda_compat.h` | runs the same `.cu` on a laptop by emulating CUDA with threads |

Laptop (no GPU; any C++17 compiler; g++ works the same way):

```sh
cd ec2/conv
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common conv.cu -o conv && ./conv
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_demand_heatmap.cu -o bikeshare_demand_heatmap && ./bikeshare_demand_heatmap
# race check of the emulation (smaller sizes are fine)
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common conv.cu -o conv_tsan && ./conv_tsan
```

> **Easiest way on Colab:** upload [`colab.ipynb`](colab.ipynb) from this folder to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. The notebook writes `cuda_compat.h` and every `.cu` file of this folder itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

Google Colab (Runtime → Change runtime type → GPU), in a cell:

```sh
!nvcc -O2 -I../common conv.cu -o conv && ./conv
!nvcc -O2 -I../common bikeshare_demand_heatmap.cu -o bikeshare_demand_heatmap && ./bikeshare_demand_heatmap
```

On a GPU, every kernel is launched once to check it (which also warms it up) and then 20 more
times to time it. The emulation reports the time of the checked launch only, to keep the run
short.

The problem size is chosen with `BY_BACKEND(gpu, cpu)`: a 4096×4096 image on the GPU and
500×500 in the emulation. 500 is not a multiple of 16, 28 or 32 on purpose, so the partial
tiles get tested. Output of `conv` in the emulation (the times are emulation overhead):

```
kernel                                     ms    GFLOP/s      GB/s*  check
CPU reference (1 thread)                1.959       6.38          -  -
(a)  basic, filter in global         1735.470       0.01       0.00  PASS (max rel err 0.0e+00)
(a') basic, filter as struct arg     1005.674       0.01       0.00  PASS (max rel err 0.0e+00)
(b)  constant-memory filter           810.097       0.02       0.00  PASS (max rel err 0.0e+00)
(c)  tiled, block = input tile       2818.185       0.00       0.00  PASS (max rel err 0.0e+00)
(d)  tiled, cached halo              2553.047       0.00       0.00  PASS (max rel err 0.0e+00)
```

## 12. Exercises

1. **Divergence at the border.** A 1024×1024 image is convolved with a 7×7 filter by
   kernel (a) using 16×16 blocks. How many blocks contain at least one thread that meets a
   ghost cell? What fraction of all threads meets at least one ghost cell?
2. **The wrong data in constant memory.** A student puts the image N in constant memory "to
   make it faster" and leaves F in global memory. Give two separate reasons why this fails or
   slows the kernel down.
3. **Kernel (c) with a smaller block.** Take IN_TILE_DIM = 16 and r = 3. What is the output
   tile? What fraction of threads compute? What is the arithmetic intensity of an internal
   block (count loads and stores)?
4. **When does 9×9 become compute-bound?** Using the intensity formula, find the smallest
   output tile `t` for which a 9×9 convolution reaches the H100 ridge of 20 FLOP/B. Is that
   tile realistic?
5. **Halo of kernel (d).** For TILE_DIM = 32 and r = 3, how many distinct halo elements does a
   block read from global memory, and what fraction of its distinct inputs are they? Why do
   these reads usually not reach DRAM?
6. **1D version.** Write a 1D convolution kernel with the "block = input tile" organisation:
   blocks of 256 threads, radius r = 3, filter in constant memory. Give its intensity for an
   internal block.
7. **Separable filters.** A Gaussian filter can be applied as a horizontal 1D pass followed by
   a vertical 1D pass. For the 13×13 Gaussian of `bikeshare_demand_heatmap.cu`, compare the FLOP per
   output of the 2D filter with the two 1D passes. What do you pay in memory traffic, and how
   does the intensity change?
8. **Guards.** In kernel (c), the compute phase tests `col < width && row < height` in
   addition to the tile test. Describe precisely what happens in the last column of blocks if
   that test is removed.

## 13. Solutions

1. The grid is 64×64 = 4096 blocks. Border blocks: 4·64 − 4 = **252 (6.2 %)**. A thread meets
   a ghost cell when it is within 3 of an edge, so the unaffected threads form a square of
   side 1024 − 6 = 1018. The fraction is 1 − (1018/1024)² ≈ **1.2 %** of the threads.
2. (i) Size: constant memory is 64 KB, and a 1024×1024 float image is 4 MB. (ii) Access
   pattern: the threads of a warp read *different* N elements, and the constant cache
   serialises different addresses, so a warp's load becomes 32 sequential accesses. Meanwhile
   F, which *is* broadcast-friendly, would still cost DRAM bandwidth.
3. Output tile 16 − 6 = **10×10**. Computing threads 100/256 = **39 %**. Intensity with
   t = 10, m = 7: (100·49·2) / ((16² + 10²)·4) = 9800 / 1424 ≈ **6.9 FLOP/B**, far from the
   ideal 12.25.
4. Solve 162t² ≥ 20·4·((t+8)² + t²), that is 2t² − 1280t − 5120 ≥ 0, which gives
   **t ≥ 644**. A 652×652 input tile would need 1.7 MB of shared memory and 425 000 threads per
   block. It is not realistic: a 9×9 convolution is memory-bound on H100 in practice. (The
   ideal 20.25 is barely above the ridge, so only the limit crosses it.)
5. The input footprint is (32 + 6)² = 1444 elements, the tile 1024, so the halo is **420
   elements, 29 %**. Each of them is an interior element of a neighbouring block, which loads
   it into its own shared memory at about the same time. The line is then usually in L2 (50 MB
   on H100), and often in L1 for the halo elements shared by several threads of the same block.
6. Sketch: `#define IN 256`, `#define OUT (IN - 2*3)`;
   `int i = blockIdx.x*OUT + threadIdx.x - 3; N_s[threadIdx.x] = (i >= 0 && i < n) ? N[i] : 0.f;`
   then `__syncthreads();`. If `threadIdx.x >= 3 && threadIdx.x < IN - 3 && i < n`, sum
   `F_c[j] * N_s[threadIdx.x - 3 + j]` for `j` in 0..6. Intensity:
   (250·7·2) / ((256 + 250)·4) = 3500 / 2024 ≈ **1.73 FLOP/B** (ideal 7·2/8 = 1.75). In 1D the
   halo is almost free.
7. 2D: 2·169 = **338 FLOP** per output. Two 1D passes: 2·(2·13) = **52 FLOP**, 6.5× less. The
   cost is that the intermediate image is written and read once more, which doubles the
   minimum traffic to 16 B per output (intensity 52/16 = 3.25 FLOP/B, memory-bound). Time per
   output on H100, from the bounds: 2D tiled is compute-bound at 338 / 66.9 TFLOPS ≈ 5.1 ps.
   Separable is memory-bound at 16 B / 3.35 TB/s ≈ 4.8 ps. They are about even. Separable wins
   clearly for larger filters, on GPUs with a lower ridge, or when the two passes are fused in
   one kernel that keeps the intermediate rows in shared memory (traffic back to 8 B per
   output).
8. In the last column of blocks the tile extends past the image. Without the test, inner
   threads with `col ≥ width` compute a value from zero ghost slots and store it at
   `P[row·width + col]`. That is the *next row's* first elements (silently overwriting correct
   outputs that another block wrote). In the last row of the last block it is past the end of
   the buffer, an out-of-bounds write.
