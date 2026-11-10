# 13 — Memory architecture and data locality

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). Every diagram is animated or interactive. The deck is generated from
> [`src/`](src/) with `python3 13_memory_locality/src/build.py`; edit `src/body.html` (slides)
> and `src/anim.js` / `src/anim_mem.js` (animations), not `slides.html`.

Based on chapter 5 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj).
Session 12 explained how the GPU runs threads; this session is about feeding them with data.

## What this session covers

- **Bandwidth as a limit:** peak compute throughput versus peak memory bandwidth, the
  compute-to-global-memory-access ratio (FLOP/B), compute-bound versus memory-bound, the
  roofline model and speed-of-light analysis.
- **The memory wall:** why global memory (off-chip DRAM) is slow, and how congestion leaves
  the cores idle.
- **CUDA memory architecture:** registers, local, shared, global and constant memory; who can
  read and write each one (host, thread, block), the hierarchy and its sizes on an H100, the
  von Neumann view of registers versus memory, distributed shared memory in Hopper clusters,
  and how to choose where a variable lives.
- **Scope and lifetime:** how many copies a launch creates and which variables survive between
  kernel launches.
- **Tiling:** threads of a block load a tile into shared memory together, wait at a barrier and
  reuse it; global traffic drops by TILE_WIDTH.
- **The tiled matrix multiplication kernel:** index arithmetic, the two barriers
  (read-after-write and write-after-read), strip-mining, register tiling.
- **Boundary checks** for sizes that are not multiples of the tile.
- **Memory and occupancy:** shared memory per SM as a limit, and shared memory sized at launch
  time with `extern __shared__`.
- **Locality on the CPU:** the same ideas with caches, measured in C++.

## Diagrams in the deck

| Slide | Diagram (book figure) |
|-------|-----------------------|
| 3 | The memory wall: long DRAM latency, limited bandwidth, congestion (§5 intro) |
| 5 | Compute-bound versus memory-bound: busy and idle time of memory and cores (5.1) |
| 6 | Roofline model with five kernels and your own (sidebar) |
| 7 | Speed-of-light calculator for vector addition |
| 9 | CUDA memory types, fastest to slowest (5.1) |
| 10 | Who can read and write each memory: host, threads, blocks (5.1) |
| 11 | Memory hierarchy of an H100: capacity, scope, latency |
| 12 | Von Neumann processor: fetch, decode, operands in registers, loads (5.2) |
| 13 | Operand in a register versus in global memory (5.2) |
| 14 | Three reasons for registers (latency, instructions, energy) and the occupancy price |
| 15 | Shared memory versus registers in a processing block (5.3) |
| 16 | Distributed shared memory in a Hopper thread block cluster (5.2) |
| 18 | Lifetime: grid versus application, across two kernel launches (5.4) |
| 19 | How many copies of each kind of variable a launch creates |
| 20 | Local memory: arrays with constant versus run-time indices |
| 21 | Constant memory: broadcast to a warp versus serialised reads |
| 22 | Decision tree: where should a variable live? |
| 24 | Threads of a block reading the same M and N elements (5.5, 5.6) |
| 25 | Traffic saved by tiling, for any Width and TILE_WIDTH |
| 26 | Tiled matrix multiplication phase by phase, with real numbers (5.7, 5.8) |
| 28 | Index arithmetic: which element each thread loads in each phase (5.10) |
| 29 | What breaks without each of the two barriers |
| 30 | Strip-mining the k loop into phases with barriers |
| 32 | CPU cache, GPU cache shared by many warps, GPU shared memory |
| 34 | Boundary checks for Width 3 and 5, every block and phase (5.11, 5.12) |
| 37 | Occupancy limited by shared memory |

## Code examples in the deck

| Slide | Code |
|-------|------|
| 4 | Vector addition: FLOP per byte |
| 8 | The naive matrix product: FLOP per byte |
| 17 | Declaring constant, global, shared and register variables in one kernel |
| 27 | The tiled kernel (Fig. 5.9), line by line |
| 33 | Naive, row-order and tiled matrix products on the CPU |
| 35 | The kernel with boundary checks (Fig. 5.13) |
| 36 | Host code: allocation, copies, grid and block, launch, error check |
| 38 | Shared memory sized at launch time (Fig. 5.14) |

## Files

| File | What it demonstrates |
|------|----------------------|
| `example_63.cpp` | Locality on the CPU: a naive (i, j, k) matrix product, the same product in row order (i, k, j), a tiled version, and tiled strips on several threads. Prints times, GFLOP/s and checks that all results agree. |

The tiled CUDA kernels on the slides were checked by emulating their launch on the CPU (every
block, phase and barrier) for widths 3 to 1000 and tiles 2 to 32: the results match a plain
triple loop, and global loads drop by exactly TILE_WIDTH when Width is a multiple of it.

## Exercises

Slides 40 and 41 hold exercises 13.1 to 13.10 (shared memory and reuse, traffic with tiles,
missing barriers, variable copies, roofline classification, shared-memory occupancy, boundary
checks); slides 42 and 43 hold the solutions.

## Compiling

```sh
g++ -std=c++17 -pthread -O2 example_63.cpp -o example_63 && ./example_63
```
