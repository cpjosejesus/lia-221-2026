#include "cuda_compat.h"

#include <cstdio>

constexpr unsigned int WIDTH = 8;
constexpr unsigned int HEIGHT = 8;
constexpr unsigned int NUM_PIXELS = WIDTH * HEIGHT;
constexpr unsigned int NUM_INTENSITIES = 256;
constexpr unsigned int NUM_BINS = 4;
constexpr unsigned int INTENSITIES_PER_BIN = NUM_INTENSITIES / NUM_BINS;
constexpr unsigned int THREADS_PER_BLOCK = 64;

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

__global__ void histogram_racy(const unsigned char* image,
                               unsigned int* bins,
                               unsigned int width,
                               unsigned int height) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i] / INTENSITIES_PER_BIN;
    ++bins[b];
  }
}

__global__ void histogram_racy_all_read_first(const unsigned char* image,
                                              unsigned int* bins,
                                              unsigned int width,
                                              unsigned int height) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool hasPixel = i < width * height;
  unsigned char b = hasPixel ? image[i] / INTENSITIES_PER_BIN : 0;

  unsigned int oldValue = hasPixel ? bins[b] : 0;
  __syncthreads();

  if (hasPixel) {
    bins[b] = oldValue + 1;
  }
}

__global__ void histogram_basic(const unsigned char* image,
                                unsigned int* bins,
                                unsigned int width,
                                unsigned int height) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i] / INTENSITIES_PER_BIN;
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
    bins_ref.fetch_add(1, cuda::memory_order_relaxed);
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

static void printBins(const char* label, const unsigned int* bins) {
  unsigned int counted = 0;
  std::printf("%-22s", label);
  for (unsigned int b = 0; b < NUM_BINS; ++b) {
    std::printf(" %9u", bins[b]);
    counted += bins[b];
  }
  std::printf(" %9u\n", NUM_PIXELS - counted);
}

template <class Kernel>
static void runKernel(Kernel kernel, const unsigned char* image_d, unsigned int* bins) {
  unsigned int* bins_d = nullptr;
  CHECK(cudaMalloc(&bins_d, NUM_BINS * sizeof(unsigned int)));
  CHECK(cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int)));

  launch(kernel,
         dim3(cdiv(NUM_PIXELS, THREADS_PER_BLOCK)),
         dim3(THREADS_PER_BLOCK),
         image_d,
         bins_d,
         WIDTH,
         HEIGHT);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(bins_d));
}

int main() {
  unsigned char* image_d = nullptr;
  CHECK(cudaMalloc(&image_d, NUM_PIXELS));
  CHECK(cudaMemcpy(image_d, &TREE_IMAGE[0][0], NUM_PIXELS, cudaMemcpyHostToDevice));

  unsigned int expected[NUM_BINS] = {0, 0, 0, 0};
  unsigned int racy[NUM_BINS];
  unsigned int allReadFirst[NUM_BINS];
  unsigned int atomic[NUM_BINS];
  histogram_sequential(&TREE_IMAGE[0][0], expected, WIDTH, HEIGHT);
  runKernel(histogram_racy, image_d, racy);
  runKernel(histogram_racy_all_read_first, image_d, allReadFirst);
  runKernel(histogram_basic, image_d, atomic);
  CHECK(cudaFree(image_d));

  printImage();
  std::printf("\n%u threads, one per pixel\n", NUM_PIXELS);
  std::printf("%-22s %9s %9s %9s %9s %9s\n",
              "kernel",
              "[0-63]",
              "[64-127]",
              "[128-191]",
              "[192-255]",
              "lost");
  printBins("sequential", expected);
  printBins("++bins[b]", racy);
  printBins("all read, then write", allReadFirst);
  printBins("atomic_ref fetch_add", atomic);

  bool ok = true;
  for (unsigned int b = 0; b < NUM_BINS; ++b) {
    ok = ok && atomic[b] == expected[b];
  }
  std::printf("%s\n", ok ? "PASS (atomic kernel exact)" : "FAIL (atomic kernel wrong)");
  return ok ? 0 : 1;
}
