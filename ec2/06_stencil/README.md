# Stencil on the GPU (chapter 8)

Study guide for the stencil chapter: grid-based solvers, the seven-point stencil, 3D
shared-memory tiles, thread coarsening along z, and register tiling. It goes with session 16
(`16_stencil`).

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_finite_difference_1d.cu`](01_finite_difference_1d.cu) | The 1D central difference FD[i] = (F[i+1] − F[i−1]) / 2h on the 7-point grid of sin(x) over [0, π] (h = π/6): one thread per interior point, the two end points are not computed. The result is printed next to cos(x). (slide: *From a finite difference to a stencil*) | Compare the FD row with the cos row: at x = π/6 the difference gives 0.827 against 0.866, and at x = π/2 both are exactly 0. The two dashes are the end points, which have no left or right neighbour. Try: set `GRID_POINTS = 13` (h halves); the error at π/6 drops from 0.039 to about 0.010, four times smaller, as O(h²) predicts. |
| 2 | [`02_stencil_2d.cu`](02_stencil_2d.cu) | The 2D 5-point stencil, one sweep on a 6×6 plate whose top edge is held at 100: every interior point becomes the average of its 4 neighbours. Boundary cells are never written, so `out` starts as a copy of `in`. (slide: *A stencil sweep, with fixed boundary cells*) | After one sweep only row 1 has warmed up: each of its points has exactly one hot neighbour (100 above), so it becomes 100 / 4 = 25, except its two end cells, which lie on the left and right edges and are never written. Rows 2 to 4 stay at 0, because one sweep moves heat by only one cell. |
| 3 | [`03_jacobi_sweeps.cu`](03_jacobi_sweeps.cu) | Sweep after sweep: the same kernel launched 30 times, swapping the two device pointers after each launch (no copy), so every sweep reads only the previous one (Jacobi iteration). The heat spreads into the plate. (slide: *Sweep after sweep: an iterative solver*) | After 30 sweeps the heat reaches the bottom: row 1 is 45.43 59.43 59.43 45.43 and row 4 already 4.52 7.16 7.16 4.52, symmetric left to right as the plate is. 30 sweeps are close to the steady state: 200 sweeps change only the second decimal (45.43 becomes 45.45). Try: set `SWEEPS = 1` to get the output of step 2. |
| 4 | [`04_stencil_3d_basic.cu`](04_stencil_3d_basic.cu) | The basic 3D 7-point `stencil_kernel` on a 5×5×5 grid, one thread per point in 4×4×4 blocks, with the heat-equation coefficients c0 = 1 − 6s and c1…c6 = s (s = 0.1). A single hot point spreads to its six face neighbours. (slide: *Code: the basic 3D 7-point kernel*) | Planes 0 and 4 stay 0; the hot point in plane 2 keeps 40 (c0 × 100 = 0.4 × 100) and each of its six face neighbours gets 10 (c1 × 100): four in plane 2 and one each in planes 1 and 3. The 60 that left the centre went to the six neighbours, so the total heat stays 100. |
| 5 | [`05_stencil_3d_tiled.cu`](05_stencil_3d_tiled.cu) | `stencil_tiled_kernel`: an 8×8×8 input tile in shared memory, one barrier, and only the inner 6×6×6 threads compute. Out-of-grid slots are not loaded and never read. The 10×10×10 grid makes the second tile on each axis partial. (slide: *Code: shared-memory tiled kernel*) | Rows 0 and 9 and columns 0 and 9 of the right half are copied from the left half: they are boundary points. Inside, every output ends in .8: without the `% 10` of the input pattern, the six neighbours of every point add up to exactly 6 × centre + 8, so out = centre + 0.8 (3.0 becomes 3.8 at j = 3, k = 1); where the `% 10` wraps, the value moves further but the .8 remains. |
| 6 | [`06_stencil_3d_coarse.cu`](06_stencil_3d_coarse.cu) | Thread coarsening in z (`stencil_coarse_kernel`): a 2D block of 8×8 threads marches through 6 planes and keeps only three planes in shared memory (`inPrev_s`, `inCurr_s`, `inNext_s`), with two barriers per plane. The slides use 32×32 blocks; 8×8 keeps the grid tiny. (slide: *Code: thread coarsening in z*) | The plane is identical to step 5. Plane 5 is the last of the 6 planes (0 to 5) marched by the first z block, so its upper neighbour came from `inNext_s` loaded from plane 6; plane 6 starts the second z block. Try: print plane 6 instead (`GRID_SIDE / 2 + 1` in `printPlane`) and it still matches the CPU. |
| 7 | [`07_stencil_3d_register.cu`](07_stencil_3d_register.cu) | Register tiling (`stencil_reg_kernel`): the z neighbours `inPrev` and `inNext`, read only by their own thread, move to registers, and only `inCurr_s` stays in shared memory. Same output as step 6. (slide: *Code: coarsening plus register tiling*) | Again the same plane as in steps 5 and 6: moving `inPrev` and `inNext` to registers changes where the values live, not the result. Each block now keeps one 8×8 plane in shared memory instead of three. |

After the steps come the complete programs. `stencil_basics.cu` repeats steps 1 to 3 on larger
inputs: the finite difference on finer and finer grids, which shows the O(h²) error, and the 2D
plate solved with a basic and a tiled kernel. `stencil.cu` runs the four 3D kernels on a large
grid with timings and double-buffered sweeps, and `bikeshare_station_coverage.cu` is the
bike-sharing example.

The steps use the kernel names of the slides (`stencil_kernel`, `stencil_tiled_kernel`,
`stencil_coarse_kernel`, `stencil_reg_kernel`) and write the coefficients as the constants
`c0` … `c6`. `stencil.cu` calls the same kernels `stencil_basic_kernel`, `stencil_tiled_kernel`,
`stencil_coarsened_kernel` and `stencil_register_tiled_kernel`, and keeps the coefficients in
constant memory (`c_c`), as the filter of chapter 7.

## 1. The pattern in one paragraph

Many physical problems (heat flow, fluid motion, electrostatics, wave propagation) are
partial differential equations. A computer solves them by sampling the unknown on a
**structured grid** and replacing each derivative with differences between neighbouring grid
points. The update of a grid point then becomes a fixed weighted sum of the point and a few
neighbours. That recipe is a **stencil**. It looks like a convolution, but with important
differences. The weights come from the equation, not from a filter someone designed. The grid
is usually **3D**. The neighbourhood is **small and star-shaped** (only along the axes), so
most of a dense m×m×m box is unused. **Boundary points** are fixed by the problem (they are
not computed, and there are no ghost zeros). And the stencil is applied **again and again**:
each sweep's output is the next sweep's input. Because every point is computed from the
previous sweep only, all points of a sweep can be computed in parallel. The challenge, as in
chapter 7, is memory traffic, and in 3D the halo costs much more.

## 2. From derivatives to stencils

On a grid with spacing h, Taylor expansion gives

```
f'(x)  ≈ ( f(x+h) − f(x−h) ) / 2h                    (two neighbours)
f''(x) ≈ ( f(x−h) − 2 f(x) + f(x+h) ) / h²           (three points: a 1D "3-point" stencil)
```

Adding the second derivatives along x, y and z gives the 3D Laplacian. That is the
**seven-point stencil**: the centre plus its six face neighbours.

```
                 (i+1, j, k)
                      │   (i, j−1, k)
                      │  ╱
  (i, j, k−1) ─── (i, j, k) ─── (i, j, k+1)
                   ╱  │
        (i, j+1, k)   │
                 (i−1, j, k)
```

Examples of what the coefficients mean:

- **Heat equation, explicit time step:** `u_new = u + s·(sum of 6 neighbours − 6u)` with
  `s = κΔt/h²`, so `c0 = 1 − 6s` and `c1..c6 = s`. The step is stable for `s ≤ 1/6`.
- **Laplace equation, Jacobi iteration:** `u_new = (sum of 6 neighbours)/6`, so `c0 = 0`.

Higher accuracy needs more points. **13-point:** two neighbours each way along each axis (a
star of radius 2, fourth-order accurate). **19-point:** centre, 6 faces and 12 edge
neighbours. **27-point:** the full 3×3×3 box. The first is still star-shaped. The last two
include diagonal neighbours and look more like a small convolution. The order of the stencil
is the radius of its reach, and it fixes the width of the halo.

**Try it:** `stencil_basics.cu` applies the 1D 3-point stencil `(F[i+1] − F[i−1]) / 2h` to
sin(x) sampled on [0, π] (the 7-point grid with h = π/6 first, then finer and finer grids). The
error against cos(x) drops by about 4 every time h halves: the O(h²) of the finite difference.
Its second part runs a 2D 5-point stencil on a square metal plate whose top edge is held at
100 degrees and the other edges at 0. Each sweep replaces every interior point by the average
of its four neighbours (Jacobi iteration for the heat equation at equilibrium), and the
boundary cells are never written. Two kernels do the same sweeps with two buffers swapped after
each one: `stencil2d_basic`, one thread per point with 5 loads from global memory, and
`stencil2d_tiled`, a 32×32 input tile in shared memory with a 30×30 output tile per block (one
halo cell per side). The tiled one is like tiled convolution, except that the four corners of
the tile are never used. Both are checked against a CPU version. The intensities it prints
count the general 5-point stencil with five coefficients, 5 multiplications + 4 additions =
9 FLOP per point, although the plate's update needs only 4 additions and one multiplication
by 0.25.

## 3. The sequential baseline

`N` points per side, `in` and `out` of `N³` floats, index `i` = z (plane), `j` = y (row),
`k` = x (column), `p = i·N² + j·N + k`. Boundary points (any index 0 or N−1) are not updated.

```cpp
for (int i = 1; i < N - 1; ++i) {
  for (int j = 1; j < N - 1; ++j) {
    for (int k = 1; k < N - 1; ++k) {
      size_t p = (size_t)i * N * N + (size_t)j * N + k;
      out[p] = c[0] * in[p] + c[1] * in[p - 1] + c[2] * in[p + 1] + c[3] * in[p - N] +
               c[4] * in[p + N] + c[5] * in[p - (size_t)N * N] + c[6] * in[p + (size_t)N * N];
    }
  }
}
```

Per point: 7 multiplications + 6 additions = **13 FLOP**. Neighbours along k are adjacent in
memory, along j they are N floats apart, and along i they are N² floats apart (256 KB for
N = 256). Those last two are the reason caches alone don't save this loop.

## 4. Kernel (a): one thread per grid point

```cpp
__global__ void stencil_basic_kernel(const float* in, float* out, int N) {
  int i = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * N * N + j * N + k] =
        c_c[0] * in[i * N * N + j * N + k] + c_c[1] * in[i * N * N + j * N + (k - 1)] +
        c_c[2] * in[i * N * N + j * N + (k + 1)] + c_c[3] * in[i * N * N + (j - 1) * N + k] +
        c_c[4] * in[i * N * N + (j + 1) * N + k] + c_c[5] * in[(i - 1) * N * N + j * N + k] +
        c_c[6] * in[(i + 1) * N * N + j * N + k];
  }
}
```

- 3D grid of 8×8×8 blocks, one thread per point. Threads on the boundary do nothing, and
  their output keeps the value copied into `out` before the sweep.
- Coefficients live in `__constant__ float c_c[STENCIL_POINTS]` (7 of them: centre, x−1,
  x+1, y−1, y+1, z−1, z+1), the same broadcast trick as the filter in chapter 7. The host
  values are all different, so that a kernel that mixes up two neighbours fails the check, and
  they add up to 1 (a weighted average), so repeated sweeps stay bounded. Two `static_assert`s
  in `stencil.cu` enforce both properties.
- **Intensity:** 13 FLOP per 7 loads and 1 store of 4 bytes = 13/32 ≈ **0.41 FLOP/B** (the
  book's number; counting only the loads gives 13/28 ≈ 0.46). On H100 that is at most
  0.41 × 3.35 TB/s ≈ **1.4 TFLOPS**, 2.0 % of the 66.9 TFLOPS peak.

## 5. Kernel (b): 3D shared-memory tiles

The same idea as chapter 7: the block loads an **input tile** of `T³` points into shared
memory, and its inner threads compute an **output tile** of `(T−2)³` points. The halo is one
point thick because the stencil reaches one point along each axis.

```cpp
#define IN_TILE_3D 8
#define OUT_TILE_3D (IN_TILE_3D - 2)

__global__ void stencil_tiled_kernel(const float* in, float* out, int N) {
  int i = (int)(blockIdx.z * OUT_TILE_3D) + (int)threadIdx.z - 1;
  int j = (int)(blockIdx.y * OUT_TILE_3D) + (int)threadIdx.y - 1;
  int k = (int)(blockIdx.x * OUT_TILE_3D) + (int)threadIdx.x - 1;

  __shared__ float in_s[IN_TILE_3D][IN_TILE_3D][IN_TILE_3D];
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  }
  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    int tz = (int)threadIdx.z;
    int ty = (int)threadIdx.y;
    int tx = (int)threadIdx.x;
    if (tz >= 1 && tz < IN_TILE_3D - 1 && ty >= 1 && ty < IN_TILE_3D - 1 && tx >= 1 &&
        tx < IN_TILE_3D - 1) {
      out[i * N * N + j * N + k] = c_c[0] * in_s[tz][ty][tx] + c_c[1] * in_s[tz][ty][tx - 1] +
                                   c_c[2] * in_s[tz][ty][tx + 1] + c_c[3] * in_s[tz][ty - 1][tx] +
                                   c_c[4] * in_s[tz][ty + 1][tx] + c_c[5] * in_s[tz - 1][ty][tx] +
                                   c_c[6] * in_s[tz + 1][ty][tx];
    }
  }
}
```

Only interior grid points are updated, and only by the inner threads of the tile.

Unlike convolution, slots outside the grid are simply not loaded. An inner thread at an
interior point only reads neighbours that are inside the grid, so the unloaded slots are never
read.

**Intensity.** A block loads T³ points (4 B each) and stores (T−2)³ outputs (4 B each), as
the book counts it:

```
intensity = 13 · (T−2)³ / (4 · T³ + 4 · (T−2)³)  =  (13/8) / ( ½ · T³/(T−2)³ + ½ )   FLOP/B
T = 8:   0.96        T = 10: 1.10        T = 32: 1.47        T → ∞:  13/8 = 1.625
```

Even infinitely large tiles stop at 1.625 FLOP/B, far below the H100 ridge of 20. **A
seven-point stencil is memory-bound on any current GPU.** Every optimisation is about getting
closer to 1.625. (Counting only the loads, as some texts do, gives 13·(T−2)³/(4·T³): 1.37 for
T = 8 and a limit of 3.25. The conclusions are the same; the numbers here follow the book.)

**Why 3D tiles must be small.** A block has at most 1024 threads, and with one thread per
input point, `T³ ≤ 1024` means **T = 8** (512 threads) or **T = 10** (1000 threads). T = 32
would need 32 768 threads (and 128 KB of shared memory). Small tiles are mostly halo:

| T | threads | shared memory | outputs | computing threads | intensity |
|---|---|---|---|---|---|
| 8 | 512 | 2 KB | 216 | 42 % | 0.96 |
| 10 | 1000 | 3.9 KB | 512 | 51 % | 1.10 |

With T = 8, 58 % of everything the block loads is halo and more than half of the threads
only load. Memory access is poor too. A warp of 32 threads covers 4 rows of only 8 floats
(32 bytes each), and tiles start at `6·bx − 1`, so rows are not aligned to 32-byte sectors.
DRAM bursts come back partly used. In 2D the same tile shape would be 32×32 with 88 % useful
outputs. The third dimension is what hurts.

## 6. Kernel (c): thread coarsening in z

Make the block 2D (T×T threads, T = 32) and let it **march along z**. Each thread computes one
output in each of `T−2` consecutive planes. At any moment the stencil needs only three planes
of input: the one below, the current one and the one above. These live in shared memory, and
each step loads just one new plane.

```
 step i:   inPrev_s = plane i−1      inCurr_s = plane i      inNext_s = plane i+1 (loaded now)
           compute plane i from the three
 shift:    inPrev_s ← inCurr_s,  inCurr_s ← inNext_s
```

```cpp
for (int i = iStart; i < iStart + OUT_TILE_2D; ++i) {
  if (inPlane && i + 1 < N) {
    inNext_s[ty][tx] = in[(i + 1) * N * N + j * N + k];
  }
  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (ty >= 1 && ty < IN_TILE_2D - 1 && tx >= 1 && tx < IN_TILE_2D - 1) {
      out[i * N * N + j * N + k] = c_c[0] * inCurr_s[ty][tx] + c_c[1] * inCurr_s[ty][tx - 1] +
                                   c_c[2] * inCurr_s[ty][tx + 1] + c_c[3] * inCurr_s[ty - 1][tx] +
                                   c_c[4] * inCurr_s[ty + 1][tx] + c_c[5] * inPrev_s[ty][tx] +
                                   c_c[6] * inNext_s[ty][tx];
    }
  }
  __syncthreads();

  inPrev_s[ty][tx] = inCurr_s[ty][tx];
  inCurr_s[ty][tx] = inNext_s[ty][tx];
}
```

The two `if`s select an interior grid point and an inner thread of the tile. Both barriers
matter. The first one makes sure that the three planes are complete (`inNext_s` was just
loaded) before anyone reads them. The second one stops a fast thread from overwriting `inCurr_s` while a slow neighbour is
still reading its x- or y-neighbours. Each thread copies only its own slot, so no barrier is
needed between the shift and the next load.

**What it buys.**

- **Tile size.** 1024 threads now make a 32×32 tile with 30×30 = 900 outputs per plane
  (88 % of the threads compute).
- **Shared memory:** 3 planes × 32² × 4 B = **12 KB** per block. A 32³ tile would need 128 KB.
- **Intensity.** In steady state each step loads 32² points and stores 30² outputs:
  13·900 / (4·1024 + 4·900) = **1.52 FLOP/B** (the book's number, against 0.96 for the 8³
  tile). Over a whole block (30 output planes, 32 loaded planes including the two extra halo
  planes in z) it is 13·27 000 / (4·32 768 + 4·27 000) = **1.47 FLOP/B**, exactly a 32³
  tile. The marching block gets the intensity of a 3D tile that could never be launched.
  Marching deeper raises it towards 1.52. For example a block that covers all 256 planes gets
  13·254·900 / (4·256·1024 + 4·254·900) ≈ 1.51.
- **Coalescing.** Every warp loads one full row of 32 consecutive floats (128 bytes).

## 7. Kernel (d): register tiling

Look at who reads what in kernel (c). The x- and y-neighbours of a point are read by
*other* threads, so `inCurr_s` must be shared. But `inPrev_s[ty][tx]` and `inNext_s[ty][tx]`
are read only by thread `(ty, tx)` itself. They can be **registers**:

```cpp
float inPrev = 0.0f;
float inCurr = 0.0f;
float inNext = 0.0f;
__shared__ float inCurr_s[IN_TILE_2D][IN_TILE_2D];
...
for (int i = iStart; i < iStart + OUT_TILE_2D; ++i) {
  if (inPlane && i + 1 < N) {
    inNext = in[(i + 1) * N * N + j * N + k];
  }
  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (ty >= 1 && ty < IN_TILE_2D - 1 && tx >= 1 && tx < IN_TILE_2D - 1) {
      out[i * N * N + j * N + k] = c_c[0] * inCurr + c_c[1] * inCurr_s[ty][tx - 1] +
                                   c_c[2] * inCurr_s[ty][tx + 1] + c_c[3] * inCurr_s[ty - 1][tx] +
                                   c_c[4] * inCurr_s[ty + 1][tx] + c_c[5] * inPrev +
                                   c_c[6] * inNext;
    }
  }
  __syncthreads();

  inPrev = inCurr;
  inCurr = inNext;
  inCurr_s[ty][tx] = inNext;
}
```

`inPrev`, `inCurr` and `inNext` are local variables, so they live in registers. The shift at the
end of the loop is now two register moves and one shared-memory store.

The book's listing of this kernel ends the loop with `inCurr_s[...] = inNext_s;`, a leftover of
the previous kernel: that array no longer exists, and the value to store is the register
`inNext`, as above.

Effects, per thread and per step:

| | kernel (c) | kernel (d) |
|---|---|---|
| shared memory per block | 12 KB (3 planes) | **4 KB** (1 plane) |
| shared stores | 3 (load of next plane, two shifts) | **1** |
| shared loads | 9 (7 for the stencil, 2 for the shifts) | **4** (the x/y neighbours) |
| extra registers | 0 | 3 |
| DRAM intensity | 1.47 FLOP/B | 1.47 FLOP/B (unchanged) |

DRAM traffic doesn't change, because the same planes are loaded once each. What improves is
pressure on shared memory, both capacity (more blocks fit per SM, or the space is freed for
other uses) and bandwidth. Registers are the fastest storage on the chip. The trade-off is 3
more registers per thread, which can lower occupancy if a kernel is already near the register
limit. The general lesson: **data that only one thread uses belongs in registers, and data
that several threads share belongs in shared memory.**

## 8. Performance summary (H100, FP32, seven-point)

| kernel | intensity (FLOP/B) | DRAM bound (× 3.35 TB/s) | % of 66.9 TFLOPS |
|---|---|---|---|
| (a) basic | 0.41 | 1.4 TFLOPS | 2.0 % |
| (b) 3D tile, T = 8 | 0.96 | 3.2 TFLOPS | 4.8 % |
| (b) 3D tile, T = 10 | 1.10 | 3.7 TFLOPS | 5.5 % |
| (c)/(d) coarsened, T = 32, 30 planes | 1.47 | 4.9 TFLOPS | 7.4 % |
| limit (T → ∞) | 1.625 | 5.4 TFLOPS | 8.1 % |

`stencil.cu` prints the per-block numbers (threads, shared bytes, outputs, intensity) and
checks every kernel against the CPU. Like in chapter 7, real L1/L2 hits make kernel (a)
better than its bound (neighbouring threads share rows). The ranking holds, though, and so
does the main conclusion: a stencil is a bandwidth problem. Further gains come from reducing
bytes: half precision, compression, or doing several sweeps per pass over memory
("temporal blocking").

## 9. Iterative solvers and double buffering

A solver applies the stencil hundreds or thousands of times. Two buffers are enough:

```cpp
float* src = device.in;
float* dst = device.out;
CHECK(cudaMemcpy(src, input.data(), device.bytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(dst, src, device.bytes, cudaMemcpyDeviceToDevice));

Timer timer;
for (int s = 0; s < ITERATIVE_SWEEPS; ++s) {
  launch(stencil_register_tiled_kernel,
         shapes.marching2D.grid,
         shapes.marching2D.block,
         (const float*)src,
         dst,
         GRID_SIDE);
  std::swap(src, dst);
}
```

Both buffers start as copies of the input, so both hold the boundary values. After each
sweep the swap makes this sweep's output the next sweep's input, and after the loop the result
is in `src`.

- The kernel launch boundary is the global barrier between sweeps: all of sweep s finishes
  before sweep s+1 starts.
- Updating in place (`in == out`) would be a data race. Some neighbours would already hold
  the new value and others the old one, depending on timing. The result would be neither
  Jacobi nor a well-defined Gauss–Seidel, and it would change from run to run.
- Boundary values must be present in **both** buffers, because the kernel never writes them.
- Convergence checks (the largest change between two sweeps) need a reduction (chapter 10).
  Doing it every K sweeps keeps the cost small.

## 10. Pitfalls

- **Missing the second barrier** in kernels (c)/(d): the shift overwrites `inCurr_s` while
  neighbours still read it. It is rare and timing-dependent, and ThreadSanitizer on the
  emulation finds it.
- **Early `return`** for halo threads before the barriers inside the z loop. Every thread
  must reach every `__syncthreads()`, so guard the computation with `if`.
- **Wrong grid for coarsened kernels.** `gridDim.z = cdiv(N, OUT_TILE_2D)`, because each block
  covers `OUT_TILE_2D` planes and not one.
- **Forgetting the boundary copy.** If `out` does not start as a copy of `in` (or of the
  boundary), boundary points hold garbage after the first swap.
- **Symmetric test coefficients.** If all six neighbour coefficients are equal, a kernel that
  swaps `k−1` and `k+1` passes the check. `stencil.cu` uses six different values.
- **`int` overflow** of `i·N·N` for large grids (N³ exceeds 2³¹ from N = 1291). Use 64-bit
  indices beyond about 1000³.

## 11. Ideas for your project

### The bike-sharing examples in this folder

`bikeshare_station_coverage.cu` runs two 2D five-point stencils over the map of a city
bike-sharing system (6 km × 6 km in metres, 48 docking stations placed by the file's own
SplitMix64 generator with a fixed seed). Both parts use one tiled shared-memory kernel
(16×16 input tiles, 14×14 output tiles), two buffers, and a mask of fixed cells.

**Part A: station coverage.** Station cells are fixed sources (u = 1), the map edge is fixed
at 0, and every other cell becomes `λ · (average of its 4 neighbours)`. Iterating to steady
state (Jacobi, double buffered) gives a smooth field that is high near stations and fades with
distance. With λ = 1 it is Laplace's equation, which needs O(grid²) sweeps and spreads coverage
over the whole map. With λ = 4 / (4 + (h/L)²), coverage decays over a distance L = 250 m, about
a three-minute walk to the nearest dock. The program prints an ASCII coverage map (S marks a
station, north at the top), the mean coverage of 16 zones (A–D north to south, 1–4 west to
east), and the **worst covered zone**. The sweeps stop when no cell changes by more than 10⁻⁵
between two consecutive sweeps (checked every 25 sweeps in the emulation and every 200 on a
GPU); the distance to the true steady state is then about change / (1 − λ), small enough to
rank zones. In the emulation run it reports D4 overall (a corner, penalised by the fixed u = 0
edge, which is a modelling choice rather than a fact about service) and C3 among the inner
zones, and it suggests a new docking station at the centre of C3, (3750 m, 2250 m).

**Part B: rental demand diffusing over time.** Every simulated minute from 17:30 to 18:00, the
new ride requests add +1 at the cell where the rider opened the app (an `atomicAdd` kernel,
because two requests can land on the same cell). Between minutes the field diffuses with
explicit heat-equation steps `u' = d·(u + α·(sum of 4 neighbours − 4u))`, which are stable for
α ≤ 1/4 (a `static_assert` checks α = 0.2), and d < 1 makes old demand fade, by 9 % per minute.
In the stencil form of the kernel that is `cC = d·(1 − 4α)` and `cN = d·α`. The physics is
fixed per simulated minute (a spread of about 160 m per minute along each axis and the 9 % fading), so a finer
grid needs more, smaller steps: 3 per minute on the 40-cell grid, 3·(G/40)² on a G-cell grid
(another `static_assert` requires G to be a multiple of 40). At 17:30 most requests come from
the university in the west, and by 18:00 most come from the offices in the centre. The result
is a heatmap with memory: the hottest cell moves from the university (17:40) to the offices
(17:50), and the hottest zone follows it from C1 to C3 by 18:00.

Both parts check the GPU against a CPU reference that runs exactly the same sweeps, print PASS
or FAIL, and end with `ALL PASS`. The grid side comes from `BY_BACKEND` (160 on a GPU, 40 in
the emulation, `-DCPU_GRID_SIDE=...` to change the second).

### Where this pattern may appear in your project

- A field that is held fixed at some cells (the sources) and relaxes everywhere else gives a
  smooth "how well served is this place" map. The lowest cells are where a new source would help
  most. What could the sources be in your project: restaurants, waiting points, something else?
- If a value should fade with distance instead of spreading over the whole map, a factor λ < 1
  in the update gives a decay length that you choose in metres.
- Demand that appears at points and then spreads and fades over time is a time-stepping stencil
  plus an injection step. How often does new demand arrive in your simulation, and how fast
  should old demand be forgotten?
- A relaxation that moves information one cell per sweep can approximate a "distance to the
  nearest X" field for the whole map at once. Could such a field pre-filter candidates (for
  example couriers near a pickup) before an exact search?
- Iterating to convergence needs a stopping test. How often would you check it, and what does
  the check itself cost (a reduction, chapter 10)?

## 12. Files and how to run

| file | contents |
|---|---|
| `01_finite_difference_1d.cu` … `07_stencil_3d_register.cu` | the learning path: one small kernel per step on a tiny grid, printed and checked against the CPU |
| `stencil.cu` | kernels (a)–(d), CPU reference, PASS/FAIL, timing table, double-buffered sweeps, resource/intensity table |
| `stencil_basics.cu` | 1D finite difference of sin(x) and its O(h²) error; 2D 5-point heat plate, basic vs tiled |
| `bikeshare_station_coverage.cu` | station coverage field (steady state) and rental demand diffusion (time stepping) over a bike-sharing city |
| `../common/cuda_compat.h` | runs the same `.cu` on a laptop by emulating CUDA with threads |

Laptop (no GPU; g++ works the same way):

```sh
cd ec2/06_stencil
# the steps, one at a time
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_finite_difference_1d.cu -o 01_finite_difference_1d && ./01_finite_difference_1d
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_stencil_2d.cu -o 02_stencil_2d && ./02_stencil_2d
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_jacobi_sweeps.cu -o 03_jacobi_sweeps && ./03_jacobi_sweeps
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_stencil_3d_basic.cu -o 04_stencil_3d_basic && ./04_stencil_3d_basic
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_stencil_3d_tiled.cu -o 05_stencil_3d_tiled && ./05_stencil_3d_tiled
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_stencil_3d_coarse.cu -o 06_stencil_3d_coarse && ./06_stencil_3d_coarse
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_stencil_3d_register.cu -o 07_stencil_3d_register && ./07_stencil_3d_register
# the complete programs
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common stencil.cu -o stencil && ./stencil
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common stencil_basics.cu -o stencil_basics && ./stencil_basics
clang++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common bikeshare_station_coverage.cu -o bikeshare_station_coverage && ./bikeshare_station_coverage
clang++ -std=c++17 -O1 -g -fsanitize=thread -pthread -x c++ -I../common stencil.cu -o stencil_tsan && ./stencil_tsan
```

> **Easiest way on Colab:** this folder has two notebooks. [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) runs the learning path, one step after another, and [`colab_2_full_programs.ipynb`](colab_2_full_programs.ipynb) runs `stencil_basics.cu`, `stencil.cu` and the bike-sharing example. Upload one to [Google Colab](https://colab.research.google.com), choose *Runtime → Change runtime type → GPU*, and *Run all*. Each notebook writes `cuda_compat.h` and its `.cu` files itself, so nothing needs to be cloned or uploaded. The commands below are for a copy of the repository.

Google Colab (GPU runtime), in a cell (the other steps the same way):

```sh
!nvcc -O2 -I../common 01_finite_difference_1d.cu -o 01_finite_difference_1d && ./01_finite_difference_1d
!nvcc -O2 -I../common stencil.cu -o stencil && ./stencil
!nvcc -O2 -I../common stencil_basics.cu -o stencil_basics && ./stencil_basics
!nvcc -O2 -I../common bikeshare_station_coverage.cu -o bikeshare_station_coverage && ./bikeshare_station_coverage
```

On a GPU, every kernel of `stencil` is launched once to check it (which also warms it up) and
then 20 more times to time it. The emulation reports the time of the checked launch only, to
keep the run short.

Sizes come from `BY_BACKEND(gpu, cpu)`: a 256³ grid on the GPU and 40³ in the emulation. At
40³ the z-coarsened kernels use two blocks along each axis, so the partial tiles get tested.
The coverage grid of `bikeshare_station_coverage.cu` is 160×160 cells (37.5 m) on a GPU and
40×40 (150 m) in the emulation; ThreadSanitizer runs it at that size in about 25 seconds.
Output of `stencil` in the emulation (the times are thread-scheduling overhead):

```
kernel (one sweep)                       ms    GFLOP/s      GB/s*  check
CPU reference (1 thread)              0.017      41.75          -  -
(a) basic                            83.573       0.01       0.01  PASS (max rel err 0.0e+00)
(b) shared-memory tiles 8^3         542.120       0.00       0.00  PASS (max rel err 0.0e+00)
(c) coarsened in z, 32^2            918.468       0.00       0.00  PASS (max rel err 0.0e+00)
(d) register tiling, 32^2           963.218       0.00       0.00  PASS (max rel err 0.0e+00)
3 sweeps of (d) with double buffering: 2952.701 ms (984.234 ms/sweep)  PASS (max rel err 0.0e+00)
```

## 13. Exercises

1. **2D five-point stencil.** Count the FLOP and loads of the basic 2D five-point kernel and
   give its intensity. Then do the same for a 2D shared-memory tile of 32×32 threads (output
   30×30). What is the limit for large tiles?
2. **Choosing T in 3D.** For kernel (b) with T = 6, 8 and 10, give the threads per block, the
   shared memory, the fraction of computing threads, and the intensity. Why is T = 11 not
   allowed?
3. **Depth of the march.** Write the intensity of kernel (c) as a function of T and of the
   number Z of output planes per block. Evaluate it for T = 32 with Z = 6, 30 and 254. What is
   the price of a very large Z?
4. **Shared-memory traffic.** Count the shared-memory loads and stores per thread per step in
   kernels (c) and (d). Which of the counts does register tiling change, and which stays the
   same?
5. **13-point stencil.** For the radius-2 star (two neighbours each way along each axis), how
   many planes must the z-marching kernel keep, how many of them can be registers, and what is
   the steady-state intensity with T = 32 threads per side?
6. **In-place update.** A student removes the second buffer to save memory and launches the
   basic kernel with `in == out`. Explain what goes wrong, and why the result can differ
   between two runs on the same GPU.
7. **Sweeps for the coverage field.** In `bikeshare_station_coverage.cu`, why do the sweeps
   needed in Part A grow like (L/h)²? Estimate the sweeps for L = 250 m on the 150 m grid and
   on the 37.5 m grid, for a change tolerance of 10⁻⁵.

## 14. Solutions

1. Basic: 5 multiplications + 4 additions = 9 FLOP per 5 loads and 1 store of 4 B, so
   9/24 = **0.375 FLOP/B**. Tiled: 9·900 / (4·1024 + 4·900) = **1.05 FLOP/B**. Limit:
   9/8 = **1.125 FLOP/B** (the book's ideal for the 2D 5-point stencil). 2D tiles get close
   to the limit because the halo is a thin ring, not a shell.
2. T = 6: 216 threads, 864 B, 64/216 = 30 %, 13·64/(4·216 + 4·64) = 0.74 FLOP/B. T = 8: 512,
   2 KB, 42 %, 0.96. T = 10: 1000, 4000 B, 51 %, 1.10. T = 11 needs 1331 threads, more than
   the 1024 a block may have.
3. intensity = 13·Z·(T−2)² / (4·(Z+2)·T² + 4·Z·(T−2)²). For T = 32: Z = 6 → 1.29,
   Z = 30 → 1.47, Z = 254 → 1.51 (limit 1.52). Large Z means few blocks: for N = 256 and T = 32, Z = 254
   gives only 9·9 = 81 blocks, fewer than the 132 SMs of an H100. Parallelism and the sweep's
   latency hiding suffer.
4. (c): 3 stores (next plane, two shifts) and 9 loads (7 stencil reads, 2 shift reads).
   (d): 1 store and 4 loads. Register tiling removes the reads of the z-neighbours and the
   shift traffic. The x/y-neighbour reads stay in shared memory because other threads need
   them. DRAM traffic is the same in both.
5. Five planes (i−2 … i+2). Four of them (i−2, i−1, i+1, i+2) are only read at the thread's
   own (y, x), so they can be registers. Only plane i is shared. The halo is 2 wide, so the
   output tile is (T−4)² = 28². FLOP: 13 multiplications + 12 additions = 25.
   Intensity per plane = 25·784 / (4·1024 + 4·784) ≈ **2.71 FLOP/B**. That is higher than the seven-point
   stencil (more work per loaded point) but still memory-bound on H100.
6. With one array, a thread may read a neighbour that another thread has already updated in
   this sweep, or one that it hasn't. Which of the two depends on block scheduling and warp
   timing. That is a data race (undefined in CUDA's memory model). Numerically you get a mix
   of Jacobi and Gauss–Seidel that is different on every run, so results can't be reproduced
   or verified.
7. The error of Jacobi with λ < 1 shrinks by about λ per sweep, so the sweeps needed are
   ≈ ln(1/tol)/(1 − λ). From λ = 4/(4 + (h/L)²), 1 − λ ≈ (h/L)²/4, so the sweeps are
   ≈ 4·ln(1/tol)·(L/h)². For tol = 10⁻⁵, ln(10⁵) ≈ 11.5. With h = 150 m (L/h = 1.67) the
   estimate is ≈ 4·11.5·2.78 ≈ 130 (the exact 1 − λ = 0.083 gives about 140). The program stops
   at 100 because its criterion is the *change* per sweep, which falls below 10⁻⁵ before the
   error does. With h = 37.5 m (L/h = 6.67) it is ≈ 4·11.5·44.4 ≈ 2000 sweeps. Refining the
   grid 4× costs 16× more sweeps of 16× more cells. That is why real solvers use multigrid.
