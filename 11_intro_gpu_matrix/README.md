# 11 — Intro to the GPU: grids, images and matrices

> **Teaching:** open [`slides.html`](slides.html) in a browser (arrow keys or space to move,
> `p` to print). The diagrams are animated and interactive; each has its own Play/Pause and
> Step buttons.

## What this session covers

Session 10 cut a picture into strips, one per CPU core. On a GPU every pixel or matrix
element gets its own thread. This session is about how those threads are organised and how
each one finds its data:

- **The thread hierarchy:** a kernel launch creates a *grid* of *blocks* of *threads*, each
  level up to three-dimensional, described with `dim3`.
- **Launch configuration:** `kernel<<<grid, block>>>`, the built-ins `gridDim`, `blockDim`,
  `blockIdx`, `threadIdx`, and the hardware limits (1024 threads per block).
- **Mapping threads to 2D data:** `col = blockIdx.x * blockDim.x + threadIdx.x`,
  `row = blockIdx.y * blockDim.y + threadIdx.y`, and the guard that stops the extra threads.
- **Flattening:** multidimensional arrays live in memory as one line, row after row
  (row-major); kernels compute `row * width + col` themselves.
- **Three kernels:** colour to grayscale (a map), blur (neighbours and borders), and matrix
  multiplication (one dot product per thread).

## Diagrams in the deck

| Slide | Diagram |
|-------|---------|
| 3 | Grid → blocks → threads: walks all 64 threads of a 2 × 2 grid of 4 × 2 × 2 blocks, with their coordinates. |
| 5 | Launch configurator: type any grid and block size; it reports threads per block, total threads and whether the launch is allowed. |
| 6 | A 2D grid covering a picture: idle threads, the four kinds of blocks, hover any cell for its block, thread, row and column. Block size and picture size can be changed. |
| 8 | Row-major flattening of a 4 × 4 matrix, animated; switchable to column-major. |
| 10 | Colour to grayscale: which three bytes each thread reads and which byte it writes. |
| 12 | Blur: a 3 × 3 patch over the picture, including corners and edges where cells are skipped. |
| 14 | Matrix multiplication: the k-loop of one thread, step by step; click any cell of P. |
| 15 | Tiling P: which block and thread compute each element. |

## Exercises

Slides 19 and 20 hold the exercises (11.1 to 11.5): a row and a column per thread, a
matrix-vector product, counting threads in a launch, and flattening indices. Solutions follow
on slides 22 to 24.

## How the code was checked

There is no CUDA compiler on the machine that produced this deck, so every kernel and host
function shown on the slides was run through a small C++ emulation of a CUDA launch: it
defines `dim3`, the built-in index variables and the memory API, and runs the kernel once for
every block and thread. Each kernel was compared with a plain CPU result, under
AddressSanitizer. The slide code is taken from that same tested file.

To run the kernels on a real GPU, Google Colab's free tier offers an NVIDIA GPU with `nvcc`
installed.

Reading: Kirk & Hwu, *Programming Massively Parallel Processors*, chapter 3.
