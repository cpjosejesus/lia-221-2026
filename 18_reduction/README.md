# 18 — Parallel reduction

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). Every diagram is animated or interactive. The deck is generated from
> [`src/`](src/) with `python3 18_reduction/src/build.py`; edit `src/body.html` (slides)
> and `src/anim.js` (animations), not `slides.html`.

Based on chapter 10 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), in the
edition that writes the multi-block atomic with `cuda::atomic_ref`. The students already know
reduce, associativity and the CPU tree (sessions 09–10), warps, divergence, coalescing and
coarsening (sessions 12–14) and `cuda::atomic_ref` (session 17), so the background is short and
the deck spends its time on the CUDA kernels and their analysis. The runnable code (every kernel
of the chapter checked against the CPU, the exercises' maximum and arbitrary-N versions, and
bike-sharing examples that give ideas for the project) is in [`../ec2/08_reduction`](../ec2/08_reduction).

## What this session covers

- Background: operator and identity, the sequential and general forms, `std::accumulate` versus
  `std::reduce`, Thrust and CUB.
- Reduction trees for max and sum; associativity and commutativity, with the fine print on
  integer overflow, NaN and floating point; work N − 1, span log₂ N, average versus peak
  parallelism (102.3 versus 512 for N = 1024); a knockout tournament as a max reduction.
- The simple kernel (stride doubling, owner computes, `__syncthreads()` as barrier and memory
  fence) and its control divergence: (4·6 + 2 + 1)·32 = 864 lane slots for 255 adds, 0.30.
- The convergent kernel: (4 + 2 + 1 + 5·1)·32 = 384, 66 %; global memory requests 141 versus 36
  for N = 256.
- Shared memory: N + 1 global accesses, N/32 + 1 = 9 requests, input left unchanged.
- Warp-level primitives: `__syncwarp`, `__shfl_down_sync` and its parameters, `laneIdx` /
  `warpIdx`, `warp_reduce`, the warp-level kernel, the `width` parameter, the other shuffle, vote,
  match and reduce functions.
- Two-stage warp-wide reduction: one barrier and almost no shared memory, at the price of more
  idle lanes, worth it because the kernel is latency-bound.
- Arbitrary length: one segment per block and a device-scope `atomic_ref<float>::fetch_add`;
  alternatives with a second kernel or the host.
- Thread coarsening: two serialized blocks (8 steps, 2 fully used) versus one coarsened block
  (6 steps, 3 fully used), and how to choose the factor.

## Diagrams in the deck

| Slide | Diagram |
|-------|---------|
| 3 | Max and sum reduction trees step by step, with sequential and tree parenthesizations |
| 4 | Eight float32 values summed in four orders (sequential, reversed, simple tree, convergent tree): 6, 8, 6, 0 |
| 5 | Work, span, peak and average parallelism for any N, with an optional limit on execution units |
| 6 | An eight-team knockout bracket as a max reduction, with 4 or 3 stadiums (span 3 or 4) |
| 8 | Simple versus convergent kernel on 16 elements, with a warp view counting issued warps, idle lanes and efficiency |
| 10 | Warps × iterations grid of active lanes, lane slots consumed and efficiency for any N, both kernels |
| 11 | Global memory requests of four warps per iteration, segment by segment: 141, 36 and 9 |
| 13 | The shared-memory kernel on 16 elements: global versus shared accesses counted |
| 15 | `__shfl_down_sync` on 8 and 32 lanes, with `width = 4`, and `__shfl_xor_sync` as a butterfly |
| 19 | Barriers, shared accesses and shuffles per block: shared-memory, warp-level and two-stage kernels |
| 20 | Four blocks reduce their segments; atomic adds arrive in a different order each run, or partial sums go to a second kernel or the host |
| 22 | Coarsening timelines per SM: fully used and underutilized steps, with threads, blocks, factor and SMs adjustable |
| 26 | The reversed access pattern of exercise 18.3 |
| 27 | Exercise 18.6 traced for the simple, convergent and reversed kernels |

## Code examples in the deck

| Slide | Code |
|-------|------|
| 2 | The sequential sum and the general form with an operator and its identity |
| 7 | The simple kernel |
| 9 | The convergent kernel |
| 12 | Partial sums in shared memory |
| 14 | `warpIdx`, `laneIdx`, the `__shfl_down_sync` signature and `warp_reduce` |
| 16 | The warp-level kernel |
| 18 | The two-stage warp-wide kernel |
| 21 | The multi-block kernel with `cuda::atomic_ref<float, cuda::thread_scope_device>` |
| 23 | The coarsened kernel, completed after the book's `...` with the two-stage, multi-block tail |
| 28–29 | Solutions: the reversed kernel, the coarsened maximum and the coarsened kernel for any N |

## Exercises

Slide 26 holds exercises 18.1 to 18.6, the book's 1 to 6: divergent warps in the 5th iteration of
the simple and the convergent kernel for 1024 elements, the reversed access pattern, a maximum and
an arbitrary-length version of the coarsened kernel, and the array {6, 2, 7, 4, 5, 8, 3, 1} traced
through both kernels. Slide 27 animates exercise 6; slides 28 and 29 hold the solutions.
