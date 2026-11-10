#include "cuda_compat.h"

#include <cstdio>

constexpr unsigned int WIDTH = 8;
constexpr unsigned int HEIGHT = 8;
constexpr unsigned int NUM_PIXELS = WIDTH * HEIGHT;
constexpr unsigned int NUM_INTENSITIES = 256;
constexpr unsigned int NUM_BINS = 4;
constexpr unsigned int INTENSITIES_PER_BIN = NUM_INTENSITIES / NUM_BINS;
constexpr unsigned int THREADS_PER_BLOCK = 8;
constexpr unsigned int COARSE_FACTOR = 4;

const unsigned char TREE_IMAGE[HEIGHT][WIDTH] = {
    {231, 236, 242, 247, 250, 245, 239, 233},
    {228, 234, 240, 96, 102, 238, 232, 226},
    {225, 230, 88, 110, 115, 92, 229, 224},
    {222, 227, 84, 105, 118, 90, 226, 221},
    {220, 224, 80, 35, 42, 86, 223, 219},
    {218, 222, 216, 33, 40, 215, 220, 217},
    {150, 156, 162, 30, 38, 165, 158, 152},
    {145, 160, 170, 175, 168, 172, 155, 148},
};

void histogram_sequential(const unsigned char* image,
                          unsigned int* bins,
                          unsigned int width,
                          unsigned int height) {
  for (unsigned int i = 0; i < width * height; ++i) {
    unsigned char b = image[i] / INTENSITIES_PER_BIN;
    ++bins[b];
  }
}

__global__ void histogram_coarse_contig(const unsigned char* image,
                                        unsigned int* bins,
                                        unsigned int width,
                                        unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS];
  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    bins_s[b] = 0;
  }
  __syncthreads();

  unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
  for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
    unsigned int i = segment + threadIdx.x * COARSE_FACTOR + c;
    if (i < width * height) {
      unsigned char b = image[i] / INTENSITIES_PER_BIN;
      cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
      bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
  }
  __syncthreads();

  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (bins_s[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
      bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
    }
  }
}

__global__ void histogram_coarse_interl(const unsigned char* image,
                                        unsigned int* bins,
                                        unsigned int width,
                                        unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS];
  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    bins_s[b] = 0;
  }
  __syncthreads();

  unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
  for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
    unsigned int i = segment + c * blockDim.x + threadIdx.x;
    if (i < width * height) {
      unsigned char b = image[i] / INTENSITIES_PER_BIN;
      cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
      bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
  }
  __syncthreads();

  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (bins_s[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
      bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
    }
  }
}

static void printImage() {
  std::printf("image (%u x %u intensities):\n", WIDTH, HEIGHT);
  for (unsigned int row = 0; row < HEIGHT; ++row) {
    for (unsigned int col = 0; col < WIDTH; ++col) {
      std::printf(" %3u", TREE_IMAGE[row][col]);
    }
    std::printf("\n");
  }
}

static void printBinHeader() {
  std::printf("%-12s", "bin");
  for (unsigned int b = 0; b < NUM_BINS; ++b) {
    std::printf(" [%3u-%3u]", b * INTENSITIES_PER_BIN, (b + 1) * INTENSITIES_PER_BIN - 1);
  }
  std::printf("\n");
}

static void printBins(const char* label, const unsigned int* bins) {
  std::printf("%-12s", label);
  for (unsigned int b = 0; b < NUM_BINS; ++b) {
    std::printf(" %9u", bins[b]);
  }
  std::printf("\n");
}

static bool sameBins(const unsigned int* bins, const unsigned int* expected) {
  for (unsigned int b = 0; b < NUM_BINS; ++b) {
    if (bins[b] != expected[b]) {
      return false;
    }
  }
  return true;
}

static unsigned int contiguousPixel(unsigned int thread, unsigned int c) {
  return thread * COARSE_FACTOR + c;
}

static unsigned int interleavedPixel(unsigned int thread, unsigned int c) {
  return c * THREADS_PER_BLOCK + thread;
}

static void printPixelsReadByBlock0(const char* name,
                                    unsigned int (*pixelOf)(unsigned int, unsigned int)) {
  std::printf("%s: pixels read by threads 0..%u of block 0\n", name, THREADS_PER_BLOCK - 1);
  for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
    std::printf("  c = %u:", c);
    for (unsigned int thread = 0; thread < THREADS_PER_BLOCK; ++thread) {
      std::printf(" %3u", pixelOf(thread, c));
    }
    std::printf("\n");
  }
}

template <class Kernel>
static void runKernel(Kernel kernel, const unsigned char* image_d, unsigned int* bins) {
  unsigned int* bins_d = nullptr;
  CHECK(cudaMalloc(&bins_d, NUM_BINS * sizeof(unsigned int)));
  CHECK(cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int)));

  const unsigned int numBlocks = cdiv(NUM_PIXELS, COARSE_FACTOR * THREADS_PER_BLOCK);
  launch(kernel, dim3(numBlocks), dim3(THREADS_PER_BLOCK), image_d, bins_d, WIDTH, HEIGHT);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(bins_d));
}

int main() {
  unsigned char* image_d = nullptr;
  CHECK(cudaMalloc(&image_d, NUM_PIXELS));
  CHECK(cudaMemcpy(image_d, &TREE_IMAGE[0][0], NUM_PIXELS, cudaMemcpyHostToDevice));

  unsigned int contiguous[NUM_BINS];
  unsigned int interleaved[NUM_BINS];
  runKernel(histogram_coarse_contig, image_d, contiguous);
  runKernel(histogram_coarse_interl, image_d, interleaved);
  CHECK(cudaFree(image_d));

  unsigned int expected[NUM_BINS] = {0, 0, 0, 0};
  histogram_sequential(&TREE_IMAGE[0][0], expected, WIDTH, HEIGHT);

  printImage();
  std::printf("\n%u blocks of %u threads, COARSE_FACTOR = %u pixels per thread\n",
              cdiv(NUM_PIXELS, COARSE_FACTOR * THREADS_PER_BLOCK),
              THREADS_PER_BLOCK,
              COARSE_FACTOR);
  printPixelsReadByBlock0("contiguous", contiguousPixel);
  printPixelsReadByBlock0("interleaved", interleavedPixel);
  std::printf("\n");
  printBinHeader();
  printBins("contiguous", contiguous);
  printBins("interleaved", interleaved);
  printBins("sequential", expected);

  const bool ok = sameBins(contiguous, expected) && sameBins(interleaved, expected);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
