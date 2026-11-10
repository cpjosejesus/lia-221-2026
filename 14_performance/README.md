# 14 — Performance considerations

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). Every diagram is animated or interactive. The deck is generated from
> [`src/`](src/) with `python3 14_performance/src/build.py`; edit `src/body.html` (slides)
> and `src/anim.js` (animations), not `slides.html`.

Based on chapter 6 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj).
Sessions 12 and 13 explained how the GPU runs threads and which memories it has; this session is
about matching a kernel's demands to the hardware's limits.

## What this session covers

- **DRAM:** why each access is slow, bursts, cache lines.
- **Global memory coalescing:** when a warp's accesses combine into one request, alignment,
  row-major versus column-major access, corner turning, coalesced stores, the carpool analogy.
- **Hiding memory latency:** channels, buses and banks, bus bandwidth and HBM, how banks
  overlap latency, interleaved distribution, and why occupancy also feeds DRAM.
- **Vector loads and stores** with `float4`.
- **Shared memory bank conflicts** and padding.
- **Thread coarsening**, its three pitfalls and the wave count.
- **Loop unrolling** and instruction scheduling, register promotion of local arrays.
- **Double buffering** to remove barriers that guard false dependences.
- **A checklist of optimizations** and an **optimization strategy** driven by measurement.

## Diagrams in the deck

| Slide | Diagram |
|-------|---------|
| 3 | Why DRAM is slow: decoder, cell, long bit line, sense amplifier |
| 4 | DRAM bursts: consecutive versus scattered requests, bytes moved and used |
| 5 | Coalescing calculator: segments touched by one warp load for any stride and alignment |
| 7 | Row-major placement of a 2D array |
| 8 | The matrix product loop: coalesced (row-major N) versus uncoalesced (column-major N) |
| 10 | Corner turning: loading a column-major tile with and without it |
| 12 | Coalescing as carpooling: traffic, carpools, similar schedules |
| 13 | Channels, buses and banks |
| 14 | Bus bandwidth and number of buses calculator (DDR, HBM) |
| 15 | Bank timelines: latency overlap and channel utilization for any R and number of banks |
| 16 | Interleaved distribution of an array over channels and banks |
| 17 | Channels used by each phase of the tiled matrix product |
| 18 | Scalar versus float4 loads: instructions per element |
| 20 | Shared memory bank conflict calculator for any stride |
| 21 | Padding a tile to remove the column-access conflict |
| 23 | Thread coarsening on small and large hardware |
| 25 | Coarsening factor and waves calculator (tail effect) |
| 26 | Loop unrolling and instruction scheduling timeline |
| 28 | Double buffering: barriers and waiting with one and two buffers |
| 30 | Interactive checklist of optimizations by category |
| 32 | Optimization cycle and recommendations per bottleneck |

## Code examples in the deck

| Slide | Code |
|-------|------|
| 9 | Matrix product loop with N row-major and column-major |
| 11 | Tiled kernel with corner turning for a column-major N |
| 19 | float4 vector addition with the tail handled |
| 22 | Padding a shared tile: 32-way conflict versus none |
| 24 | A coarsening loop with factor 4 |
| 27 | #pragma unroll, and unrolling that promotes a local array to registers |
| 29 | One buffer with two barriers versus two buffers with one |
| 33 | Timing with CUDA events, effective bandwidth, Nsight Compute |
| 34 | AoS versus SoA kernels |
| 35 | Stride order and AoS/SoA on the CPU, with measured times |

## Files

| File | What it demonstrates |
|------|----------------------|
| `example_64.cpp` | The CPU twin of coalescing: summing the same 128 MB in stride order (1 to 64) and one field of an array of structures versus a structure of arrays. Prints the times. |

Measured on an Apple M1 Pro with `g++ -O2` (best of 5 runs): stride 1, 2 and 4 take about
17 ms, stride 16 (one value per 64-byte line) 2.3 times as long, strides 32 and 64 about 6
times; the array of structures is about 4 times slower than the structure of arrays.

## Exercises

Slides 37 and 38 hold exercises 14.1 to 14.10 (coalescing in a given kernel, corner
turning, coalescing of the tiled kernel, float4 boundaries, bank conflicts by stride, bus and bank
bandwidth, coarsening waves, AoS, double buffering); slides 39 and 40 hold the solutions.

## Compiling

```sh
g++ -std=c++17 -O2 example_64.cpp -o example_64 && ./example_64
```
