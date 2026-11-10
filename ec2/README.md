# EC2 — CUDA from the first kernel to the parallel patterns

**Never used Colab or CUDA before?** Start with the setup guide
[`../00_cuda_colab_setup/cuda_colab_setup.pdf`](../00_cuda_colab_setup/cuda_colab_setup.pdf): it walks you
through your first Colab notebook, getting a GPU, and compiling and running a CUDA hello world.

Runnable CUDA for the GPU part of the course, following *Programming Massively Parallel
Processors* (Hwu, Kirk, El Hajj) and the slide decks 11 to 18. The numbered folders form one
learning path: start at `01_first_kernels` and go in order. Inside every folder the work goes from
simple to complete:

1. **Steps** `01_*.cu`, `02_*.cu`, …: one new idea per file, on a tiny input that is printed, so you
   see what the algorithm does. Each step is checked against the CPU and prints PASS. The folder
   README's *Learning path* table says what each step adds.
2. **Complete programs** (folders 05–12): every kernel of the chapter on large inputs, compared and
   timed (`conv.cu`, `reduction_book.cu`, …).
3. **Bike-sharing examples** `bikeshare_*.cu` (folders 05–12): the pattern applied to a city
   bike-sharing system (stations, bikes, trips, rebalancing trucks). It is a different domain on
   purpose: they give ideas for the final project without solving any part of it. Each README ends
   with hints on where the pattern may appear in your project.

Every folder has its Google Colab notebooks, self-contained (they write the header and the code):
**`colab_1_step_by_step.ipynb`** for the steps and, in folders 05–12,
**`colab_2_full_programs.ipynb`** for the complete programs and the examples.

| Folder | Deck | What the steps build up to |
|--------|------|----------------------------|
| [`01_first_kernels`](01_first_kernels/) | 11 | threads and blocks, vector addition, 2D grids, grayscale, blur, matrix multiplication |
| [`02_gpu_architecture`](02_gpu_architecture/) | 12 | device properties, block scheduling, barriers, warps, control divergence, occupancy |
| [`03_memory_tiling`](03_memory_tiling/) | 13 | memory types, compute-to-memory ratio, tiled matrix multiplication with boundary checks |
| [`04_performance`](04_performance/) | 14 | coalescing, corner turning, bank conflicts, thread coarsening and other optimizations |

The parallel patterns (chapters 7 to 14):

| Folder | Chapter | Kernels | Bike-sharing examples |
|--------|---------|---------|---------------------------|
| [`05_conv`](05_conv/) | 7, convolution | basic; filter as a struct; constant memory; tiled with the halo in shared memory; tiled with cached halos | `bikeshare_demand_heatmap`: smooth a trip-start demand grid to find hot zones for new stations or rebalancing |
| [`06_stencil`](06_stencil/) | 8, stencil | 1D finite difference and its O(h²) error; 2D 5-point basic and tiled; basic 3D 7-point; shared-memory tiling; thread coarsening in z; register tiling | `bikeshare_station_coverage`: a coverage field around stations, and rental demand diffusing over time |
| [`07_histogram`](07_histogram/) | 9, parallel histogram | the race and the atomic fix; atomics (`atomicAdd` and `cuda::atomic_ref`); privatization in global and in shared memory; contiguous and interleaved coarsening; thread-level privatization; aggregation | `bikeshare_trip_durations`: trip-duration buckets, median and percentiles; `bikeshare_station_hour`: rentals per hour and per station |
| [`08_reduction`](08_reduction/) | 10, reduction | simple; convergent; shared memory; segmented with atomics; coarsened; max; and as the book writes them (`reduction_book`): `warp_reduce` with `__shfl_down_sync`, warp-level and two-stage warp reductions, multi-block with `cuda::atomic_ref`, coarsened; the book's exercises (reversed convergent, coarsened max, any length, traced iterations) and its resource and memory-request counts | `bikeshare_daily_summary`: the day's report with block reductions; `bikeshare_nearest_truck`: the rebalancing truck that reaches each empty station first (argmin); `bikeshare_kpis`: several KPIs in one struct reduction (integer cents versus float); `bikeshare_bike_totals`: distance and revenue per bike, one block each, then an argmax with a deterministic tie-break |
| [`09_scan`](09_scan/) | 11, prefix sum | Kogge-Stone (in place and double-buffered); Brent-Kung; coarsened three-phase; hierarchical; single-pass | `bikeshare_group_by_station`: bucket trips by start station; `bikeshare_rentals_in_progress`: bikes out over time and the peak |
| [`10_filter`](10_filter/) | 12, filtering | global atomic; block privatization; flags + scan + scatter; staged coalesced compaction; single pass | `bikeshare_late_returns`: trips over the included time, in log order; `bikeshare_idle_bikes`: charged docked bikes near each rider request |
| [`11_merge`](11_merge/) | 13, merge | co-rank; basic; tiled; circular buffer; coarsening | `bikeshare_station_logs`: merge per-station dock logs into one time-ordered city log, stably |
| [`12_sorting`](12_sorting/) | 14, sorting | 1-bit radix sort; local sort with coalesced scatter (1, 4 and 8 bits); merge sort | `bikeshare_trip_percentiles`: trip-duration percentiles by radix sort; `bikeshare_nearest_bikes`: the k nearest charged bikes per request |

## One source file, two ways to run it

Every `.cu` file includes [`common/cuda_compat.h`](common/cuda_compat.h):

- **Google Colab or any NVIDIA GPU**: it is plain CUDA. Upload the folder's notebooks to
  Colab, choose *Runtime → Change runtime type → GPU* and *Run all*; or, in a copy of the repository:

  ```sh
  cd ec2/05_conv && nvcc -O2 -arch=native -I../common conv.cu -o conv && ./conv
  ```

- **Any laptop, no GPU**: the header emulates the CUDA features the examples use. Each block's
  threads run as real `std::thread`s, `__syncthreads()` is a barrier, `__shared__` memory is
  shared by the block, and atomics are real atomics:

  ```sh
  cd ec2/05_conv && g++ -std=c++17 -O2 -pthread -x c++ -I../common conv.cu -o conv && ./conv
  ```

  (`clang++` works the same way.) The emulation is for learning and for checking results, not
  for speed. It runs one block at a time, with all of the block's threads alive at once, so
  `__syncthreads()`, `__shared__` memory and atomics behave as on a GPU. A `__shared__`
  variable becomes a `static` one: a single copy, shared by the threads of the block that is
  running. After each block, every thread waits at the block's barrier, so the next block
  starts only when the current one is completely done. Its timings measure thread start-up and
  barriers, not GPU memory behaviour: on a laptop only the PASS column and the counts mean
  something. Problem sizes are smaller in this mode (`BY_BACKEND(gpu, cpu)` picks the first
  value under `nvcc` and the second in the emulation, and `CUDA_BACKEND` names the backend in
  each program's first line).

Kernels are launched with `launch(kernel, grid, block, args...)` instead of
`kernel<<<grid, block>>>(args...)`, because the triple-chevron syntax is not C++; under `nvcc`,
`launch` is exactly that `<<<>>>` call.

The atomic functions (`atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch`,
`atomicCAS`) are real atomics with relaxed memory order, as in CUDA: they guarantee that no
update is lost, not any ordering of other memory accesses. The `float` versions use a
compare-and-swap loop. `__threadfence()` is a full (sequentially consistent) fence.

`cuda::atomic_ref` (from `<cuda/atomic>`) is emulated too, with the operations the examples use
(`load`, `store`, `exchange`, `compare_exchange_strong`, `fetch_add`, `fetch_sub`, `fetch_min`,
`fetch_max`) and the memory order each call asks for. Every scope behaves like the strongest
one on the CPU. The scope only matters on a GPU, where a narrower scope (`thread_scope_block`)
can be cheaper because fewer threads have to agree on the value.

Warp-level primitives are emulated as well: the threads `32w .. 32w+31` of a block form warp `w`
(the last warp of a block whose size is not a multiple of 32 has fewer lanes), with
`warpSize`, `__syncwarp`, `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
`__shfl_xor_sync`, `__ballot_sync`, `__any_sync` and `__all_sync`. Only full-warp masks are
supported: the mask argument is ignored and every lane of the warp must call the primitive
(with mask `0xffffffff` CUDA requires the same). Each warp has its own barrier and one exchange
slot per lane. A shuffle writes the caller's value into its lane's slot, waits for the whole
warp, reads the source lane's slot, and waits again, so that no slot is overwritten before
every lane has read it. A lane whose source falls outside its group of `width` lanes gets its
own value back, as in CUDA. The lanes do not run in lockstep, so divergence has no measurable
cost in the emulation, and each shuffle costs two barriers, so warp-heavy programs use smaller
sizes there.

Device memory is ordinary host memory: `cudaMalloc` is `malloc`, every `cudaMemcpy` kind is a
`memcpy`, `cudaMemcpyToSymbol` copies into the `__constant__` variable, and the error functions
always report success. The helpers in the header work under both backends: `CHECK(call)`
stops the program with the file and line of a failing CUDA call, `Timer` measures wall-clock
milliseconds and synchronizes the device before reading the clock, `cdiv(a, b)` rounds a
division up (grid sizes), and `imin`, `imax`, `fminf_` and `fmaxf_` work in host and device
code.

Not emulated: dynamic shared memory (`extern __shared__`) and streams.

## Running everything on a laptop

```sh
cd ec2 && ./run_all.sh
```

builds every program with `-Wall -Wextra`, runs it and prints PASS or FAIL per program (a few
minutes in total on a laptop).

## How this was checked

- All 27 programs compile without warnings and print PASS in the CPU emulation.
- ThreadSanitizer reports no data race in any of them. Removing a `__syncthreads()` from a tiled
  kernel makes it report one, so the emulation does expose missing barriers: try it.
- Every file was compiled with clang's CUDA front end (device and host passes, `-Wall -Wextra`)
  to catch code that `nvcc` would reject, such as host functions called from kernels.
- **Not yet run on a real GPU.** GPU timings in the READMEs are expectations from the
  hardware's limits, not measurements; collecting your own table on Colab is one of the exercises
  of every chapter.

After editing a `.cu` file, regenerate the notebooks with `python3 ec2/make_colab.py`.
