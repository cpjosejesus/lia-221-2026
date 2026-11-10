# 17 — Parallel histogram: atomics and privatization

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). Every diagram is animated or interactive. The deck is generated from
> [`src/`](src/) with `python3 17_histogram/src/build.py`; edit `src/body.html` (slides)
> and `src/anim.js` (animations), not `slides.html`.

Based on chapter 9 of *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), in the
edition that writes atomics with `cuda::atomic_ref`. The students already know histograms, races,
`std::atomic` and CAS (session 08) and the "accumulate privately, merge rarely" histogram on the
CPU (session 10), so the deck covers only the CUDA side. The runnable code (every kernel checked
against the CPU, a race demo, the text histogram with aggregation and two bike-sharing
examples that give ideas for the project) is in [`../ec2/histogram`](../ec2/histogram).

## What this session covers

- Output interference: one thread per pixel, any thread may update any bin; the read-modify-write
  race and its timing tables.
- Atomics in CUDA C++: `#include <cuda/atomic>`, `cuda::atomic_ref<T, scope>`,
  `fetch_add(1, cuda::memory_order_relaxed)`, why relaxed is enough, and the thread scopes
  (thread, block, device, system); the older `atomicAdd` / `atomicAdd_block`.
- Latency and throughput of atomics: serialization on one location, the 128 GB/s versus
  2.5 M atomics/s example, 256 bins → 640 M/s, skewed data, atomics in the last-level cache.
- Privatization: a private copy per block in global memory (`bins_pool`), in shared memory, per
  cluster in distributed shared memory, per thread in registers; the init and commit cost.
- Thread coarsening: contiguous versus interleaved partitioning, and why only interleaved
  coalesces on a GPU.
- Thread-level privatization: register counters without atomics, the unrolled compare-and-add
  that keeps them out of local memory, keeping only the frequent bins, and aggregation of runs.

## Diagrams in the deck

| Slide | Diagram |
|-------|---------|
| 3 | The 8 × 8 tree image and its 4-bin histogram, scanned sequentially or by one thread per pixel |
| 4 | Two threads' read-modify-write step by step: no overlap, lost update, atomic |
| 7 | Thread scopes: thread, block, device and system, on a two-GPU system |
| 9 | A warp of atomics queueing on 8 bins, for any hot-bin share |
| 10 | Atomics on one location on a time axis: DRAM, L2 and shared memory |
| 11 | Atomic throughput calculator against memory bandwidth (channels, clock, latencies, bins, skew) |
| 12 | Private copies per block: count, barrier, commit |
| 14 | Where the private copy lives, with competing threads, number of copies and latency |
| 16 | Init and commit overhead versus coarsening factor, blocks and waves |
| 17 | Contiguous versus interleaved partitioning, iteration by iteration, with transactions counted |
| 19 | Thread-level privatization: register counters, commit to shared, commit to global |
| 21 | Per-thread copy size and the share of updates caught by a few hot bins (Zipf skew) |
| 22 | Privatizing the most recent bin: one atomic per run of equal bins |

## Code examples in the deck

| Slide | Code |
|-------|------|
| 2 | The sequential histogram |
| 5 | `cuda::atomic_ref`, `fetch_add` and the C-style `atomicAdd` |
| 6 | The basic kernel with a device-scope `atomic_ref` |
| 8 | The load / atomic dependence that makes `memory_order_relaxed` enough |
| 13 | Private copies per block in global memory |
| 15 | Private copies per block in shared memory |
| 18 | Coarsening: contiguous and interleaved differ in one line |
| 20 | Thread-level privatization with register counters |
| 23 | The book's kernel that privatizes the most recent bin (`bin_r`) |

## Exercises

Slide 26 holds exercises 17.1 to 17.8: the book's exercises 1 to 5 (atomic throughput in DRAM,
with L2 hits, floating-point throughput limited by atomics, with shared-memory privatization, and
global atomics of the basic, privatized and coarsened kernels), then the size and scopes of
`bins_pool`, coalescing of both partitionings, and the atomics of the most-recent-bin kernel on a
run of bins. Slide 27 holds the solutions.
