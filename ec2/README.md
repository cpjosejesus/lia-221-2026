# EC2 — Parallel patterns on the GPU

Study material for the parallel patterns of *Programming Massively Parallel Processors*
(Hwu, Kirk, El Hajj), chapters 7 to 14. Each folder holds:

- **`README.md`**: the theory of the chapter, kernel by kernel, with the performance reasoning,
  pitfalls, ideas for using the pattern in your project, and exercises with solutions.
- **`<pattern>.cu`**: every kernel of the chapter, checked against a CPU reference and timed.
- **`bikeshare_*.cu`**: the pattern applied to a city bike-sharing system (stations, bikes, trips,
  rebalancing trucks), with synthetic data from a fixed seed. It is a different domain on purpose:
  the programs give ideas for the final project without solving any part of it. Each folder README
  ends with hints on where the pattern may appear in your project.
- **`colab.ipynb`**: a self-contained Google Colab notebook that writes and runs all of the above.

| Folder | Chapter | Kernels | Bike-sharing examples |
|--------|---------|---------|---------------------------|
| [`conv`](conv/) | 7, convolution | basic; filter as a struct; constant memory; tiled with the halo in shared memory; tiled with cached halos | `bikeshare_demand_heatmap`: smooth a trip-start demand grid to find hot zones for new stations or rebalancing |
| [`stencil`](stencil/) | 8, stencil | 1D finite difference and its O(h²) error; 2D 5-point basic and tiled; basic 3D 7-point; shared-memory tiling; thread coarsening in z; register tiling | `bikeshare_station_coverage`: a coverage field around stations, and rental demand diffusing over time |
| [`histogram`](histogram/) | 9, parallel histogram | the race and the atomic fix; atomics (`atomicAdd` and `cuda::atomic_ref`); privatization in global and in shared memory; contiguous and interleaved coarsening; thread-level privatization; aggregation | `bikeshare_trip_durations`: trip-duration buckets, median and percentiles; `bikeshare_station_hour`: rentals per hour and per station |
| [`reduction`](reduction/) | 10, reduction | simple; convergent; shared memory; segmented with atomics; coarsened; max; and as the book writes them (`reduction_book`): `warp_reduce` with `__shfl_down_sync`, warp-level and two-stage warp reductions, multi-block with `cuda::atomic_ref`, coarsened; the book's exercises (reversed convergent, coarsened max, any length, traced iterations) and its resource and memory-request counts | `bikeshare_daily_summary`: the day's report with block reductions; `bikeshare_nearest_truck`: the rebalancing truck that reaches each empty station first (argmin); `bikeshare_kpis`: several KPIs in one struct reduction (integer cents versus float); `bikeshare_bike_totals`: distance and revenue per bike, one block each, then an argmax with a deterministic tie-break |
| [`scan`](scan/) | 11, prefix sum | Kogge-Stone (in place and double-buffered); Brent-Kung; coarsened three-phase; hierarchical; single-pass | `bikeshare_group_by_station`: bucket trips by start station; `bikeshare_rentals_in_progress`: bikes out over time and the peak |
| [`filter`](filter/) | 12, filtering | global atomic; block privatization; flags + scan + scatter; staged coalesced compaction; single pass | `bikeshare_late_returns`: trips over the included time, in log order; `bikeshare_idle_bikes`: charged docked bikes near each rider request |
| [`merge`](merge/) | 13, merge | co-rank; basic; tiled; circular buffer; coarsening | `bikeshare_station_logs`: merge per-station dock logs into one time-ordered city log, stably |
| [`sorting`](sorting/) | 14, sorting | 1-bit radix sort; local sort with coalesced scatter (1, 4 and 8 bits); merge sort | `bikeshare_trip_percentiles`: trip-duration percentiles by radix sort; `bikeshare_nearest_bikes`: the k nearest charged bikes per request |

## One source file, two ways to run it

Every `.cu` file includes [`common/cuda_compat.h`](common/cuda_compat.h):

- **Google Colab or any NVIDIA GPU**: it is plain CUDA. Upload the folder's `colab.ipynb` to
  Colab, choose *Runtime → Change runtime type → GPU* and *Run all*; or, in a copy of the repository:

  ```sh
  cd ec2/conv && nvcc -O2 -arch=native -I../common conv.cu -o conv && ./conv
  ```

- **Any laptop, no GPU**: the header emulates the CUDA features the examples use. Each block's
  threads run as real `std::thread`s, `__syncthreads()` is a barrier, `__shared__` memory is
  shared by the block, and atomics are real atomics:

  ```sh
  cd ec2/conv && g++ -std=c++17 -O2 -pthread -x c++ -I../common conv.cu -o conv && ./conv
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
