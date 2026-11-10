# First kernels: threads, grids, images and matrices (chapter 3)

Session 11 (`11_intro_gpu_matrix`): a GPU gives every element its own thread. This folder writes
the first kernels of the course, from a kernel that only says hello to the matrix product with one
thread per output element. Everything here is index arithmetic: each thread turns its coordinates
(`blockIdx`, `blockDim`, `threadIdx`) into the position of the data it owns, and checks that the
position exists.

## Learning path

| Step | File | What it adds | What to look at in the output |
|------|------|--------------|-------------------------------|
| 1 | [`01_hello_threads.cu`](01_hello_threads.cu) | The first kernel: a grid of 2 blocks of 4 threads, each thread prints its `blockIdx.x`, `threadIdx.x` and global index `blockIdx.x * blockDim.x + threadIdx.x` with `printf`, and writes that index to device memory so the host can check it. The order of the printed lines is not fixed. | Each line names a block, a thread and the global index `block * 4 + thread`: block 1, thread 3 has index 7. The lines may come out in any order: on the GPU the four lines of one block (one warp) usually stay together, but which block prints first can change between runs. Still, the array copied back is always 0 1 2 3 4 5 6 7, because each thread writes to its own slot. Try: set `BLOCKS = 3` and the indices run to 11. |
| 2 | [`02_vector_add.cu`](02_vector_add.cu) | Vector addition with the host code of the slides: `cudaMalloc` three arrays, `cudaMemcpy` A and B to the device, launch one block of 8 threads (thread i adds element i), copy C back, `cudaFree`. | Read the three lines column by column: column i is the work of thread i, for example C[3] = 3 + 40 = 43. All 8 threads are in one block and no thread uses another thread's element. |
| 3 | [`03_vector_add_any_n.cu`](03_vector_add_any_n.cu) | `vecAddKernel` for any n: several blocks, the global index, the guard `if (i < n)` and the grid size by ceiling division `(n + 3) / 4`. With n = 10 and blocks of 4, 12 threads run and 2 stop at the guard. (slide: *`dim3`: describing the grid and the blocks*) | The first two lines are the grid arithmetic: (10 + 3) / 4 = 3 blocks, 12 threads, and threads 10 and 11 (the last two of block 2) stop at `i < n`, so C still has exactly 10 values, ending in 9 + 100 = 109. Try: set `N = 13`; you get 4 blocks, 16 threads and 3 stop at the guard. |
| 4 | [`04_grid_2d.cu`](04_grid_2d.cu) | A 2D grid over a 6×5 picture: `dim3 block(4, 4)`, a 2×2 grid, `col` from x and `row` from y, the 2D guard and the row-major index `row * width + col`. Prints which block owns each pixel: 64 threads for 30 pixels. (slides: *Launching a 2D grid*, *Flattening by hand*) | The first map shows the four blocks as 4×4 squares: block 0 covers columns 0 to 3 of rows 0 to 3, while blocks 1, 2 and 3 are cut by the picture's edge (block 3 owns only pixels (4, 4) and (4, 5)). The second map is the row-major index: pixel (row 2, col 3) is 2 × 6 + 3 = 15. 34 of the 64 threads fail the 2D guard. |
| 5 | [`05_color_to_gray.cu`](05_color_to_gray.cu) | `colorToGray` on a 4×3 RGB picture: each thread reads 3 bytes at `rgbOffset = grayOffset * CHANNELS` and writes 1 gray byte. A pure map: no thread touches another thread's pixel. (slides: *Launching a 2D grid*, *The grayscale kernel*) | Each gray value is 0.299 r + 0.587 g + 0.114 b, truncated: pure red gives 76, pure green 149 and pure blue only 29, because the eye is most sensitive to green. White stays 255 and black 0. |
| 6 | [`06_blur.cu`](06_blur.cu) | `blurKernel` with `BLUR_SIZE = 1` on a 6×6 picture: every output averages its 3×3 patch, skipping neighbours outside the picture, so corners divide by 4, edges by 6 and the inside by 9. (slides: *The blur kernel*, *Session 10 on the CPU, session 11 on the GPU*) | The bright 2×2 square becomes a soft spot: its pixels drop to 50 (4 × 100 + 5 × 10 = 450, divided by 9) and the ring around it gets 20 or 30. The bright corner pixel at the top right becomes 32 (130 / 4, only 4 pixels in its patch) and its neighbour on the left 25 (150 / 6). Try: set `BLUR_SIZE` to 2 for a 5×5 patch and a wider, flatter spot. |
| 7 | [`07_matmul.cu`](07_matmul.cu) | `matMulKernel` on 4×4 matrices with `BLOCK_WIDTH = 2` (4 blocks of 4 threads): thread (row, col) computes the dot product of row `row` of M and column `col` of N. The grid replaces the two outer loops of the CPU version. (slide: *The matrix multiplication kernel and its host code*) | Check one entry by hand: P[0][0] = 1·0 + 2·1 + 3·2 + 4·0 = 8, row 0 of M times column 0 of N. Columns 0 and 3 of P are equal because columns 0 and 3 of N are equal (0, 1, 2, 0). Each of the 16 threads computed one of these entries. |
| 8 | [`08_error_checking.cu`](08_error_checking.cu) | Error checking. Steps 1 to 7 call `cudaMalloc`, `cudaMemcpy` and the kernel as the slides do, ignoring the result. CUDA never stops a program on an error: every call returns a `cudaError_t`, and a kernel launch returns nothing, so its error must be asked for with `cudaGetLastError()`. This step launches the vector addition with 2048 threads per block (the limit is 1024), first without checking, then asking for the error, and introduces `CHECK(call)`, which prints the error with file and line and stops. From here on every program wraps its CUDA calls in `CHECK`. | Part 1: C still holds the marker -1.0 in every position, so the kernel never ran, and no message appeared. Part 2: `cudaGetLastError()` returns 9, "invalid configuration argument", the same error a real GPU gives. Part 3: the valid launch (1 block of 256 threads) computes C = A + B. Try: set `THREADS_PER_BLOCK` to 1025 and run again: now `CHECK(cudaGetLastError())` in part 3 prints the error with the file and line, and the program stops. |
| 9 | [`09_flatten_3d.cu`](09_flatten_3d.cu) | A 3D grid over a 2 × 3 × 4 array: `plane`, `row` and `col` from z, y and x, the 3D guard and the flat index `(plane * height + row) * width + col`. Each thread writes its own coordinates, so the array copied back shows the memory order. The same index function answers exercises 11.4 and 11.5. (slides: *Flattening by hand*, *Exercises (2/2): count the threads, find the index*) | The memory line reads 000 001 002 003 010 … 123: the column changes fastest, then the row, then the plane, and plane 1 starts at index 12 = 1 × 3 × 4. 32 threads are launched for 24 elements and 8 stop at the guard: they have row 3, which does not exist. The last lines are the exercise answers: 4540 row-major, 8015 column-major and 184540 in 3D. |
| 10 | [`10_count_threads.cu`](10_count_threads.cu) | The `scaleKernel` of exercise 11.3 on its 200 × 120 matrix with blocks of 32 × 8 and the grid `((cols - 1) / 32 + 1, (rows - 1) / 8 + 1)`. Two `atomicAdd` counters count the threads that start and the threads that reach line A. (slide: *Exercises (2/2): count the threads, find the index*) | The four answers of 11.3, each worked out and then counted by the kernel: 256 threads per block, 4 × 25 = 100 blocks, 25600 threads, and 24000 of them on line A, one per element; the other 1600 fail the guard. The corner shows out = in × 1.5. |
| 11 | [`11_matmul_row_col.cu`](11_matmul_row_col.cu) | The two kernels of exercise 11.1: `matMulRowKernel`, where each thread computes a whole row of P, and `matMulColKernel`, a whole column, both launched with `<<<(width + 255) / 256, 256>>>` on the 4 × 4 matrices of step 7. (slide: *Solution 11.1 · a row or a column per thread*) | Both P matrices equal the one of step 7 (P[0][0] = 8), but only 4 of the 256 threads do any work, and each one computes 4 elements instead of 1. For width 1000 that is 1000 busy threads against 1000000 in step 7. |
| 12 | [`12_matvec.cu`](12_matvec.cu) | The matrix-vector product of exercise 11.2: `matVecKernel`, one thread per output element, and the host function `matVec(a, B, c, N)` that allocates, copies B and c, launches, copies a back and frees. (slide: *Solution 11.2 · matrix-vector product*) | B is lower triangular with ones, so row i of B picks the first i + 1 values of c and a holds the running sums 1, 3, 6, 10, 15. Thread 2 walks row 2 of B, which starts at B[2 × 5]. |

Session 12 continues in [`../02_gpu_architecture`](../02_gpu_architecture/): where these blocks run,
warps, divergence and occupancy.

## 1. Host, device and the first kernel

A CUDA program has two sides. The **host** is the CPU and its memory, where `main` runs. The
**device** is the GPU and its own memory. A function marked `__global__` is a **kernel**: the host
launches it, and it runs on the device once per thread. Step 1 shows the smallest possible kernel:
every thread prints who it is. All threads run the same code; the only thing that tells them apart
is their built-in coordinates.

The device cannot read host arrays, so data has to travel. Step 2 is the pattern that every later
program repeats: allocate device memory with `cudaMalloc`, copy the inputs with
`cudaMemcpy(..., cudaMemcpyHostToDevice)`, launch, copy the result back with
`cudaMemcpyDeviceToHost`, and free with `cudaFree`. The copy back waits for the kernel to finish.
The convention of the book, also used here, is a `_d` suffix for device pointers.

## 2. Grids, blocks and threads

A launch creates a **grid** of **blocks**, and every block has the same number of **threads**.
Both sizes are `dim3` values with up to three dimensions; unused ones default to 1, so
`dim3 block(256)` is 256 × 1 × 1. Inside the kernel the sizes are `gridDim` and `blockDim` and the
positions are `blockIdx` and `threadIdx`. The global index of a thread in a 1D grid is

```
i = blockIdx.x * blockDim.x + threadIdx.x
```

The grid follows the data. n elements need `ceil(n / blockSize)` blocks, which in integers is
`(n + blockSize - 1) / blockSize`. The last block is usually not full, so the extra threads must
stop at a guard, `if (i < n)` (step 3). Limits on current GPUs: at most 1024 threads per block,
`blockDim` up to 1024 × 1024 × 64, `gridDim.x` up to 2³¹ − 1 and `gridDim.y`, `gridDim.z` up to
65 535.

## 3. 2D grids and row-major memory

For pictures and matrices a 2D grid is more natural: x is the horizontal direction (columns) and y
the vertical one (rows).

```
col = blockIdx.x * blockDim.x + threadIdx.x
row = blockIdx.y * blockDim.y + threadIdx.y
if (col < width && row < height) { ... }
```

Memory is one long line, so a 2D array is stored **row-major**, row after row: element (row, col)
is at `row * width + col`. `row * width` skips the rows before, `+ col` moves along the row. A 3D
array adds one more term: `(plane * height + row) * width + col`. Sizes known only at run time
mean the kernel flattens the index itself, as every kernel here does. Step 4 prints the block that
owns each pixel and the flat index of each pixel, so you can see both mappings at once.

## 4. Three kernels on that mapping

- **Grayscale** (step 5): `gray = 0.299 r + 0.587 g + 0.114 b`, one thread per pixel. The input
  has 3 bytes per pixel, so the input offset is the output offset times `CHANNELS`. The GPU
  compiler may fuse the multiply-adds, which rounds slightly differently, so after the cast to
  `unsigned char` a gray value can differ from the CPU's by 1; the check accepts that.
- **Blur** (step 6): each output is the average of a (2·BLUR_SIZE + 1)² patch. Neighbours outside
  the picture are skipped, and `pixels` counts the ones really added: dividing by 9 everywhere
  would darken the borders. Neighbouring threads read overlapping inputs, which later chapters
  exploit with shared memory.
- **Matrix multiplication** (step 7): `P[row][col]` is the dot product of row `row` of M and column
  `col` of N. Along a row of M the index grows by 1 (consecutive memory); down a column of N it
  grows by `width`. Each thread does `width` multiply-adds and the grid of threads replaces the
  two outer loops.

## 5. How the code differs from the slides

- Kernels are launched with `launch(kernel, grid, block, args...)` instead of
  `kernel<<<grid, block>>>(args...)`. Under `nvcc` it is exactly that launch; on a laptop it runs
  every thread of every block with `std::thread`.
- Steps 1 to 7 call the CUDA functions as the slides do, without looking at the error code they
  return. Step 8 shows why that is risky and introduces `CHECK(call)`, a macro from
  `cuda_compat.h`: if the call does not return `cudaSuccess`, it prints the error with the file and
  line and stops the program. A kernel launch returns no error code, so `CHECK(cudaGetLastError())`
  right after the launch is the way to find out that it failed. Steps 9 to 12 and every later
  folder use `CHECK`.
- `cudaMalloc(&A_d, bytes)` uses the typed overload of the runtime, so no `(void**)` cast is
  needed.

## 6. How to run

Laptop (no GPU; any C++17 compiler; clang++ works the same way):

```sh
cd ec2/01_first_kernels
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 01_hello_threads.cu -o 01_hello_threads && ./01_hello_threads
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 02_vector_add.cu -o 02_vector_add && ./02_vector_add
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 03_vector_add_any_n.cu -o 03_vector_add_any_n && ./03_vector_add_any_n
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 04_grid_2d.cu -o 04_grid_2d && ./04_grid_2d
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 05_color_to_gray.cu -o 05_color_to_gray && ./05_color_to_gray
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 06_blur.cu -o 06_blur && ./06_blur
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 07_matmul.cu -o 07_matmul && ./07_matmul
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 08_error_checking.cu -o 08_error_checking && ./08_error_checking
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 09_flatten_3d.cu -o 09_flatten_3d && ./09_flatten_3d
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 10_count_threads.cu -o 10_count_threads && ./10_count_threads
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 11_matmul_row_col.cu -o 11_matmul_row_col && ./11_matmul_row_col
g++ -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common 12_matvec.cu -o 12_matvec && ./12_matvec
```

> **Easiest way on Colab:** upload [`colab_1_step_by_step.ipynb`](colab_1_step_by_step.ipynb) from
> this folder to [Google Colab](https://colab.research.google.com), choose *Runtime → Change
> runtime type → GPU*, and *Run all*. The notebook runs the twelve steps in order and writes
> `cuda_compat.h` and every `.cu` file itself, so nothing needs to be cloned or uploaded. The
> commands below are for a copy of the repository.

Google Colab (Runtime → Change runtime type → GPU), in a cell:

```sh
!nvcc -O2 -arch=native -I../common 01_hello_threads.cu -o 01_hello_threads && ./01_hello_threads
!nvcc -O2 -arch=native -I../common 02_vector_add.cu -o 02_vector_add && ./02_vector_add
!nvcc -O2 -arch=native -I../common 03_vector_add_any_n.cu -o 03_vector_add_any_n && ./03_vector_add_any_n
!nvcc -O2 -arch=native -I../common 04_grid_2d.cu -o 04_grid_2d && ./04_grid_2d
!nvcc -O2 -arch=native -I../common 05_color_to_gray.cu -o 05_color_to_gray && ./05_color_to_gray
!nvcc -O2 -arch=native -I../common 06_blur.cu -o 06_blur && ./06_blur
!nvcc -O2 -arch=native -I../common 07_matmul.cu -o 07_matmul && ./07_matmul
!nvcc -O2 -arch=native -I../common 08_error_checking.cu -o 08_error_checking && ./08_error_checking
!nvcc -O2 -arch=native -I../common 09_flatten_3d.cu -o 09_flatten_3d && ./09_flatten_3d
!nvcc -O2 -arch=native -I../common 10_count_threads.cu -o 10_count_threads && ./10_count_threads
!nvcc -O2 -arch=native -I../common 11_matmul_row_col.cu -o 11_matmul_row_col && ./11_matmul_row_col
!nvcc -O2 -arch=native -I../common 12_matvec.cu -o 12_matvec && ./12_matvec
```

Every step prints its input and its result, then `PASS` or `FAIL`, and returns a non-zero exit
code on `FAIL`.
