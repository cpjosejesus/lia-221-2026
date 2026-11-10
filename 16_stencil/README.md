# 16 — Stencil computation on the GPU

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). Every diagram is animated or interactive. The deck is generated from
> [`src/`](src/) with `python3 16_stencil/src/build.py`; edit `src/body.html` (slides)
> and `src/anim.js` (animations), not `slides.html`.

Based on chapter 8 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj). The
students already know tiling, halos and arithmetic intensity from the convolution session, so
the deck frames stencils as "what is different from convolution, and why it needs new
optimizations". The runnable code (all four 3D kernels with CPU checks, the 1D and 2D basics,
and a bike-sharing example that gives ideas for the project) is in [`../ec2/stencil`](../ec2/stencil).

## What this session covers

- Background: discretizing sin(x) on a grid, the central difference f′(x) ≈ (f(x+h) − f(x−h))/2h
  with error O(h²), stencil order, 1D/2D/3D stencil shapes, stencil sweeps with fixed boundary
  cells, and iterative solvers that swap two buffers between sweeps.
- The basic 3D 7-point kernel: 13 FLOP per point, 13/32 = 0.41 FLOP/B (0.46 if the store is not
  counted) against the ideal 13/8 = 1.625.
- Why shared-memory tiling buys less than for convolution: sparse stars (2D 5-point 1.125 vs
  3 × 3 convolution 2.25, 3D order 3 4.625 vs 85.75), 3D tiles limited to t = 8 by the 1024
  threads per block (0.96 FLOP/B), 58 % halo, and poor coalescing of 8 × 8 × 8 blocks.
- Thread coarsening in z: a 32 × 32 block marching through the grid with three planes in shared
  memory (12 KB), 1.52 FLOP/B per step.
- Register tiling: the z neighbours in registers, one shared plane (4 KB), 3072 more registers
  per block, unchanged global traffic.

## Diagrams in the deck

| Slide | Diagram |
|-------|---------|
| 3 | sin(x) on a grid of any size, the central difference and its error |
| 5 | Stencil shapes by dimension and order, against the convolution box |
| 6 | A 2D sweep: the 5-point stencil, parallel outputs, fixed boundary, buffer swap |
| 7 | Jacobi sweeps spreading heat from a fixed boundary |
| 9 | Where the 7 neighbours live in memory, and the basic kernel's FLOP/B |
| 10 | Ideal intensity of stencils against convolutions, by dimension and order |
| 12 | Tiled and coarsened intensity against tile size, with the 1024-thread limits |
| 13 | Halo fraction of 2D and 3D tiles for any tile width |
| 14 | Warp access and DRAM sectors of 8 × 8 × 8 and 32 × 8 × 4 blocks |
| 15 | The z-march: inPrev_s, inCurr_s, inNext_s, loads and outputs counted |
| 17 | Register tiling: which slots are shared and which belong to one thread |
| 19 | Threads, shared memory, registers, outputs and FLOP/B per block for any order and tile |

## Code examples in the deck

| Slide | Code |
|-------|------|
| 7 | Host loop of an iterative solver with swapped buffers |
| 8 | The basic 3D 7-point kernel |
| 11 | The shared-memory tiled kernel |
| 16 | Thread coarsening in z |
| 18 | Thread coarsening plus register tiling (with the book's `inNext_s` typo corrected) |

## Exercises

Slide 22 holds exercises 16.1 to 16.6 (the book's exercises 1 and 2 on a 120³ grid and on
32 × 32 blocks with coarsening factor 16, an order-2 3D stencil, the finite-difference error of
sin, the 32 × 8 × 4 tiled block, and Jacobi for Laplace's equation); slide 23 holds the
solutions.
