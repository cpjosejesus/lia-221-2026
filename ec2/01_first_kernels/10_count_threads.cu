#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const unsigned ROWS = 200;
const unsigned COLS = 120;
const unsigned ELEMENTS = ROWS * COLS;
const unsigned BLOCK_X = 32;
const unsigned BLOCK_Y = 8;
const unsigned CORNER = 3;

struct ThreadCounts {
  unsigned launched;
  unsigned ranLineA;
};

__global__ void scaleKernel(
    const float* in, float* out, unsigned rows, unsigned cols, ThreadCounts* counts) {
  atomicAdd(&counts->launched, 1u);
  unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    out[row * cols + col] = in[row * cols + col] * 1.5f;
    atomicAdd(&counts->ranLineA, 1u);
  }
}

static ThreadCounts scale(const float* in_d, float* out_d, dim3 grid, dim3 block) {
  ThreadCounts counts{0, 0};
  ThreadCounts* counts_d;
  CHECK(cudaMalloc(&counts_d, sizeof(ThreadCounts)));
  CHECK(cudaMemcpy(counts_d, &counts, sizeof(ThreadCounts), cudaMemcpyHostToDevice));

  launch(scaleKernel, grid, block, in_d, out_d, ROWS, COLS, counts_d);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(&counts, counts_d, sizeof(ThreadCounts), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(counts_d));
  return counts;
}

static std::vector<float> scaleOnCpu(const std::vector<float>& in) {
  std::vector<float> out(ELEMENTS);
  for (unsigned i = 0; i < ELEMENTS; ++i) {
    out[i] = in[i] * 1.5f;
  }
  return out;
}

static void printCorner(const char* title, const std::vector<float>& matrix) {
  std::printf("%s\n", title);
  for (unsigned row = 0; row < CORNER; ++row) {
    for (unsigned col = 0; col < CORNER; ++col) {
      std::printf(" %5.1f", matrix[row * COLS + col]);
    }
    std::printf("  ...\n");
  }
}

int main() {
  std::vector<float> in(ELEMENTS);
  for (unsigned i = 0; i < ELEMENTS; ++i) {
    in[i] = static_cast<float>(i % 10);
  }

  const size_t bytes = ELEMENTS * sizeof(float);
  float* in_d;
  float* out_d;
  CHECK(cudaMalloc(&in_d, bytes));
  CHECK(cudaMalloc(&out_d, bytes));
  CHECK(cudaMemcpy(in_d, in.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((COLS - 1) / BLOCK_X + 1, (ROWS - 1) / BLOCK_Y + 1);
  const ThreadCounts counts = scale(in_d, out_d, grid, block);

  std::vector<float> out(ELEMENTS);
  CHECK(cudaMemcpy(out.data(), out_d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(in_d));
  CHECK(cudaFree(out_d));

  const unsigned threadsPerBlock = block.x * block.y;
  const unsigned blocks = grid.x * grid.y;
  std::printf("rows = %u, cols = %u, block (%u, %u), grid ((%u - 1) / %u + 1, (%u - 1) / %u + 1)\n",
              ROWS,
              COLS,
              BLOCK_X,
              BLOCK_Y,
              COLS,
              BLOCK_X,
              ROWS,
              BLOCK_Y);
  std::printf("(a) threads per block:     %u x %u = %u\n", block.x, block.y, threadsPerBlock);
  std::printf("(b) blocks in the grid:    %u x %u = %u\n", grid.x, grid.y, blocks);
  std::printf("(c) threads in the grid:   %u x %u = %u, counted by the kernel: %u\n",
              blocks,
              threadsPerBlock,
              blocks * threadsPerBlock,
              counts.launched);
  std::printf("(d) threads on line A:     %u x %u = %u, counted by the kernel: %u\n\n",
              ROWS,
              COLS,
              ELEMENTS,
              counts.ranLineA);
  printCorner("in, top-left corner:", in);
  printCorner("out = in * 1.5, top-left corner:", out);

  const bool ok = out == scaleOnCpu(in) && counts.launched == blocks * threadsPerBlock &&
                  counts.ranLineA == ELEMENTS;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
