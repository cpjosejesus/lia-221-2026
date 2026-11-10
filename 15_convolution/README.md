# 15 — Convolution on the GPU

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). Every diagram is animated or interactive. The deck is generated from
> [`src/`](src/) with `python3 15_convolution/src/build.py`; edit `src/body.html` (slides)
> and `src/anim.js` (animations), not `slides.html`.

Based on chapter 7 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj). The
students already know what a convolution computes, so the deck covers only the CUDA side. The
runnable code (all four kernels, checks, timings and a bike-sharing example that gives ideas for the project) is in
[`../ec2/05_conv`](../ec2/05_conv).

## What this session covers

- One thread per output element; the input patch of each thread; ghost cells and the control
  divergence they cause near the edges.
- Arithmetic intensity: the ideal m²/4 FLOP/B, the basic kernel's 0.25, constant memory's 0.5,
  and the tiled kernel's formula, against the H100 ridge of 20 FLOP/B.
- Constant memory: why the filter fits it, `__constant__`, `cudaMemcpyToSymbol`, passing the
  filter as a struct by value; the cache hierarchy (L1, L2, constant cache) versus shared memory.
- Tiling: input tiles larger than output tiles, halo cells, the two thread organizations, the
  tiled kernel with the halo in shared memory, its costs, and the kernel that leaves the halo to
  the L2 cache.

## Diagrams in the deck

| Slide | Diagram |
|-------|---------|
| 3 | Threads, output elements and input patches, with ghost cells |
| 5 | Ghost cells and divergence for any image size and radius |
| 6 | Arithmetic intensity of the four kernels for any filter and tile size |
| 7 | The constant cache broadcasting the filter to a warp |
| 9 | The cache hierarchy and where a load is served from |
| 10 | Input tile versus output tile for any tile size and radius |
| 11 | Block = input tile versus block = output tile |
| 12 | The tiled kernel step by step: load, barrier, compute, idle ring |
| 14 | Intensity table of the tiled kernel, recoloured for any ridge |
| 15 | Idle threads, tile width and blocks for any tile and radius |
| 16 | The cached-halo kernel: shared memory, L2 and ghost cells per thread |

## Code examples in the deck

| Slide | Code |
|-------|------|
| 4 | The basic 2D kernel with ghost-cell handling |
| 8 | Constant memory: declaration, `cudaMemcpyToSymbol` and the kernel |
| 13 | The tiled kernel with the halo in shared memory |
| 17 | The tiled kernel with cached halos |

## Exercises

Slide 20 holds exercises 15.1 to 15.8 (1D example, ghost cells and multiplications in 1D and
2D, launch configuration and shared memory of both tiled kernels, idle threads, constant versus
global memory, threads reading the halo, achievable throughput); slide 21 holds the solutions.
