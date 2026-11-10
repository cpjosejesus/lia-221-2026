#include "cuda_compat.h"

#include <cstdio>
#include <string>
#include <vector>

constexpr unsigned NUM_BINS = 256;
constexpr unsigned THREADS_PER_BLOCK = 256;
const unsigned NUM_PIXELS = BY_BACKEND(1u << 24, 1u << 16);

constexpr unsigned char TRUNK_INTENSITY = 20;
constexpr unsigned char LEAVES_INTENSITY = 100;
constexpr unsigned char GRASS_INTENSITY = 160;
constexpr unsigned char SKY_INTENSITY = 240;
constexpr unsigned NUM_INTENSITIES_IN_IMAGE = 4;
constexpr unsigned char IMAGE_INTENSITIES[NUM_INTENSITIES_IN_IMAGE] = {
    TRUNK_INTENSITY,
    LEAVES_INTENSITY,
    GRASS_INTENSITY,
    SKY_INTENSITY,
};

constexpr unsigned PIXEL_SCATTER_MULTIPLIER = 7;
constexpr unsigned PIXEL_SCATTER_DIVISOR = 3;

__device__ unsigned int readFromMemory(volatile unsigned int* bins, unsigned char b) {
  return bins[b];
}

__device__ void writeToMemory(volatile unsigned int* bins, unsigned char b, unsigned int value) {
  bins[b] = value;
}

__global__ void racy_kernel(const unsigned char* image, unsigned int* bins, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned char b = image[i];

    unsigned int oldValue = readFromMemory(bins, b);
    unsigned int newValue = oldValue + 1;
    writeToMemory(bins, b, newValue);
  }
}

__global__ void atomic_kernel(const unsigned char* image, unsigned int* bins, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned char b = image[i];
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
    bins_ref.fetch_add(1, cuda::memory_order_relaxed);
  }
}

enum class KernelVersion { Racy, Atomic };

using Histogram = std::vector<unsigned>;

static unsigned char intensityOfPixel(unsigned pixel) {
  unsigned scattered = pixel * PIXEL_SCATTER_MULTIPLIER + pixel / PIXEL_SCATTER_DIVISOR;
  return IMAGE_INTENSITIES[scattered % NUM_INTENSITIES_IN_IMAGE];
}

static std::vector<unsigned char> makeImage() {
  std::vector<unsigned char> image(NUM_PIXELS);
  for (unsigned pixel = 0; pixel < NUM_PIXELS; ++pixel) {
    image[pixel] = intensityOfPixel(pixel);
  }
  return image;
}

static Histogram sequentialHistogram(const std::vector<unsigned char>& image) {
  Histogram bins(NUM_BINS, 0);
  for (unsigned char intensity : image) {
    ++bins[intensity];
  }
  return bins;
}

static void launchKernel(KernelVersion version, const unsigned char* d_image, unsigned* d_bins) {
  const dim3 grid(cdiv(NUM_PIXELS, THREADS_PER_BLOCK));
  const dim3 block(THREADS_PER_BLOCK);
  if (version == KernelVersion::Racy) {
    launch(racy_kernel, grid, block, d_image, d_bins, NUM_PIXELS);
  } else {
    launch(atomic_kernel, grid, block, d_image, d_bins, NUM_PIXELS);
  }
}

static Histogram runOnDevice(KernelVersion version,
                             const unsigned char* d_image,
                             unsigned* d_bins) {
  CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned)));
  launchKernel(version, d_image, d_bins);
  CHECK(cudaGetLastError());

  Histogram bins(NUM_BINS);
  CHECK(cudaMemcpy(bins.data(), d_bins, NUM_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return bins;
}

static unsigned long long lostUpdates(const Histogram& bins) {
  unsigned long long counted = 0;
  for (unsigned count : bins) {
    counted += count;
  }
  return static_cast<unsigned long long>(NUM_PIXELS) - counted;
}

static void printTableHeader() {
  std::printf("%-8s", "kernel");
  for (unsigned char intensity : IMAGE_INTENSITIES) {
    std::string column = "bin " + std::to_string(intensity);
    std::printf(" %12s", column.c_str());
  }
  std::printf(" %12s\n", "lost");
}

static void printBinsOfImage(const char* kernelName, const Histogram& bins) {
  std::printf("%-8s", kernelName);
  for (unsigned char intensity : IMAGE_INTENSITIES) {
    std::printf(" %12u", bins[intensity]);
  }
}

static void printReferenceRow(const Histogram& reference) {
  printBinsOfImage("correct", reference);
  std::printf(" %12s\n", "-");
}

static void printMeasuredRow(const char* kernelName, const Histogram& bins) {
  printBinsOfImage(kernelName, bins);
  std::printf(" %12llu\n", lostUpdates(bins));
}

int main() {
  std::printf("Read-modify-write race on a histogram (chapter 9), backend: %s\n", CUDA_BACKEND);

  const std::vector<unsigned char> image = makeImage();
  const Histogram reference = sequentialHistogram(image);

  unsigned char* d_image;
  unsigned* d_bins;
  CHECK(cudaMalloc(&d_image, NUM_PIXELS));
  CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned)));
  CHECK(cudaMemcpy(d_image, image.data(), NUM_PIXELS, cudaMemcpyHostToDevice));

  std::printf("%u pixels, %u distinct values, %u threads per block\n\n",
              NUM_PIXELS,
              NUM_INTENSITIES_IN_IMAGE,
              THREADS_PER_BLOCK);
  printTableHeader();
  printReferenceRow(reference);

  const Histogram racyBins = runOnDevice(KernelVersion::Racy, d_image, d_bins);
  printMeasuredRow("racy", racyBins);

  const Histogram atomicBins = runOnDevice(KernelVersion::Atomic, d_image, d_bins);
  printMeasuredRow("atomic", atomicBins);

  CHECK(cudaFree(d_image));
  CHECK(cudaFree(d_bins));

  const bool atomicIsExact = atomicBins == reference;
  std::printf(
      "\nThe racy count is whatever the interleaving produced; the atomic one must be exact.\n");
  std::printf("%s\n", atomicIsExact ? "PASS (atomic kernel exact)" : "FAIL (atomic kernel wrong)");
  return atomicIsExact ? 0 : 1;
}
