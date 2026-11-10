#include "cuda_compat.h"

#include <cstdint>
#include <cstdio>
#include <vector>

constexpr unsigned NUM_INTENSITIES = 256;
constexpr unsigned NUM_BINS = NUM_INTENSITIES;
constexpr unsigned NUM_BINS_SMALL = 16;
constexpr unsigned INTENSITIES_PER_SMALL_BIN = NUM_INTENSITIES / NUM_BINS_SMALL;
constexpr unsigned COARSE_FACTOR = 32;
constexpr unsigned THREADS_PER_BLOCK = 256;

const unsigned WIDTH = BY_BACKEND(4096u, 512u);
const unsigned HEIGHT = BY_BACKEND(4096u, 512u);
const int TIMED_RUNS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);
const bool RUNNING_ON_CPU_EMULATION = BY_BACKEND(false, true);

void histogram_sequential(const unsigned char* image,
                          unsigned int* bins,
                          unsigned int width,
                          unsigned int height) {
  for (unsigned int i = 0; i < width * height; ++i) {
    unsigned char b = image[i];
    ++bins[b];
  }
}

__global__ void histogram_basic(const unsigned char* image,
                                unsigned int* bins,
                                unsigned int width,
                                unsigned int height) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i];
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
    bins_ref.fetch_add(1, cuda::memory_order_relaxed);
  }
}

__global__ void histogram_private_global(const unsigned char* image,
                                         unsigned int* bins,
                                         unsigned int* bins_pool,
                                         unsigned int width,
                                         unsigned int height) {
  unsigned int* bins_priv = &bins_pool[blockIdx.x * NUM_BINS];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i];
    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_priv_ref(bins_priv[b]);
    bins_priv_ref.fetch_add(1, cuda::memory_order_relaxed);
  }
  __syncthreads();

  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (bins_priv[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
      bins_ref.fetch_add(bins_priv[b], cuda::memory_order_relaxed);
    }
  }
}

__global__ void histogram_private_shared(const unsigned char* image,
                                         unsigned int* bins,
                                         unsigned int width,
                                         unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS];
  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    bins_s[b] = 0;
  }
  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    unsigned char b = image[i];
    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
    bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
  }
  __syncthreads();

  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (bins_s[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
      bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
    }
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
      unsigned char b = image[i];
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
      unsigned char b = image[i];
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

__global__ void histogram_thread_private(const unsigned char* image,
                                         unsigned int* bins,
                                         unsigned int width,
                                         unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS_SMALL];
  for (unsigned int b = threadIdx.x; b < NUM_BINS_SMALL; b += blockDim.x) {
    bins_s[b] = 0;
  }
  __syncthreads();

  unsigned int bins_thread[NUM_BINS_SMALL];
  for (unsigned int b = 0; b < NUM_BINS_SMALL; ++b) {
    bins_thread[b] = 0;
  }

  unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
  for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
    unsigned int i = segment + c * blockDim.x + threadIdx.x;
    if (i < width * height) {
      ++bins_thread[image[i] / INTENSITIES_PER_SMALL_BIN];
    }
  }

  for (unsigned int b = 0; b < NUM_BINS_SMALL; ++b) {
    if (bins_thread[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
      bins_s_ref.fetch_add(bins_thread[b], cuda::memory_order_relaxed);
    }
  }
  __syncthreads();

  for (unsigned int b = threadIdx.x; b < NUM_BINS_SMALL; b += blockDim.x) {
    if (bins_s[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
      bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
    }
  }
}

__global__ void histogram_thread_run(const unsigned char* image,
                                     unsigned int* bins,
                                     unsigned int width,
                                     unsigned int height) {
  __shared__ unsigned int bins_s[NUM_BINS];
  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    bins_s[b] = 0;
  }
  __syncthreads();

  unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
  if (segment + threadIdx.x < width * height) {
    unsigned int b = image[segment + threadIdx.x];
    unsigned int bin_r = 1;
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
      unsigned int i = segment + c * blockDim.x + threadIdx.x;
      if (i < width * height) {
        unsigned char bNext = image[i];
        if (bNext == b) {
          ++bin_r;
        } else {
          cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
          bins_s_ref.fetch_add(bin_r, cuda::memory_order_relaxed);
          b = bNext;
          bin_r = 1;
        }
      }
    }
    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
    bins_s_ref.fetch_add(bin_r, cuda::memory_order_relaxed);
  }
  __syncthreads();

  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (bins_s[b] > 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
      bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
    }
  }
}

class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {
  }

  uint64_t next() {
    state_ += GOLDEN_RATIO_INCREMENT;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * FIRST_MIX_MULTIPLIER;
    z = (z ^ (z >> 27)) * SECOND_MIX_MULTIPLIER;
    return z ^ (z >> 31);
  }

  unsigned below(unsigned bound) {
    return static_cast<unsigned>(next() % bound);
  }

 private:
  static constexpr uint64_t GOLDEN_RATIO_INCREMENT = 0x9E3779B97F4A7C15ull;
  static constexpr uint64_t FIRST_MIX_MULTIPLIER = 0xBF58476D1CE4E5B9ull;
  static constexpr uint64_t SECOND_MIX_MULTIPLIER = 0x94D049BB133111EBull;

  uint64_t state_;
};

using Image = std::vector<unsigned char>;
using Histogram = std::vector<unsigned>;

constexpr uint64_t NOISE_SEED = 11;
constexpr uint64_t PHOTO_SEED = 7;

static Image makeNoise(unsigned width, unsigned height) {
  SplitMix64 rng(NOISE_SEED);
  Image image(static_cast<size_t>(width) * height);
  for (unsigned char& pixel : image) {
    pixel = static_cast<unsigned char>(rng.below(NUM_INTENSITIES));
  }
  return image;
}

enum class Region { Grass, Leaves, Trunk, Sky };

struct IntensityRange {
  unsigned lowest;
  unsigned count;
};

constexpr IntensityRange GRASS_INTENSITIES = {150, 30};
constexpr IntensityRange LEAVES_INTENSITIES = {90, 30};
constexpr IntensityRange TRUNK_INTENSITIES = {0, 40};
constexpr IntensityRange SKY_GLARE_INTENSITIES = {225, 31};
constexpr unsigned char PLAIN_SKY_INTENSITY = 240;
constexpr unsigned ONE_GLARE_PIXEL_EVERY = 10;

constexpr unsigned FIFTHS_OF_HEIGHT = 5;
constexpr unsigned GRASS_TOP_FIFTHS_DOWN = 4;
constexpr unsigned CROWN_CENTER_FIFTHS_DOWN = 2;
constexpr unsigned CROWN_RADIUS_DIVISOR = 5;
constexpr unsigned TRUNK_HALF_WIDTH_DIVISOR = 40;

static unsigned fifthsDown(unsigned height, unsigned fifths) {
  return height * fifths / FIFTHS_OF_HEIGHT;
}

static Region regionOf(unsigned x, unsigned y, unsigned width, unsigned height) {
  const unsigned crownCenterX = width / 2;
  const unsigned crownCenterY = fifthsDown(height, CROWN_CENTER_FIFTHS_DOWN);
  const int dx = static_cast<int>(x) - static_cast<int>(crownCenterX);
  const int dy = static_cast<int>(y) - static_cast<int>(crownCenterY);
  const int crownRadiusSquared =
      static_cast<int>(width * width / (CROWN_RADIUS_DIVISOR * CROWN_RADIUS_DIVISOR));
  const int trunkHalfWidth = static_cast<int>(width / TRUNK_HALF_WIDTH_DIVISOR);

  if (y > fifthsDown(height, GRASS_TOP_FIFTHS_DOWN)) {
    return Region::Grass;
  }
  if (dx * dx + dy * dy < crownRadiusSquared) {
    return Region::Leaves;
  }
  const bool belowCrownCenter = y > crownCenterY;
  const int distanceFromTrunkAxis = dx < 0 ? -dx : dx;
  if (belowCrownCenter && distanceFromTrunkAxis < trunkHalfWidth) {
    return Region::Trunk;
  }
  return Region::Sky;
}

static unsigned char randomIntensity(IntensityRange range, SplitMix64& rng) {
  return static_cast<unsigned char>(range.lowest + rng.below(range.count));
}

static unsigned char intensityOf(Region region, SplitMix64& rng) {
  switch (region) {
    case Region::Grass:
      return randomIntensity(GRASS_INTENSITIES, rng);
    case Region::Leaves:
      return randomIntensity(LEAVES_INTENSITIES, rng);
    case Region::Trunk:
      return randomIntensity(TRUNK_INTENSITIES, rng);
    case Region::Sky:
      break;
  }
  const bool isGlare = rng.below(ONE_GLARE_PIXEL_EVERY) == 0;
  return isGlare ? randomIntensity(SKY_GLARE_INTENSITIES, rng) : PLAIN_SKY_INTENSITY;
}

static Image makePhoto(unsigned width, unsigned height) {
  SplitMix64 rng(PHOTO_SEED);
  Image image(static_cast<size_t>(width) * height);
  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x) {
      image[static_cast<size_t>(y) * width + x] = intensityOf(regionOf(x, y, width, height), rng);
    }
  }
  return image;
}

constexpr unsigned NUM_QUARTERS = 4;
constexpr unsigned BINS_PER_QUARTER = NUM_BINS / NUM_QUARTERS;
constexpr int BAR_WIDTH = 50;
const char* const QUARTER_LABELS[NUM_QUARTERS] = {
    "[0-63]   ",
    "[64-127] ",
    "[128-191]",
    "[192-255]",
};

static void printBar(unsigned long long count, unsigned long long total) {
  const int width = static_cast<int>(BAR_WIDTH * static_cast<double>(count) / total + 0.5);
  for (int j = 0; j < width; ++j) {
    std::putchar('#');
  }
  std::putchar('\n');
}

static void printQuarters(const Histogram& reference) {
  unsigned long long quarterCounts[NUM_QUARTERS] = {0, 0, 0, 0};
  unsigned long long total = 0;
  for (unsigned b = 0; b < NUM_BINS; ++b) {
    quarterCounts[b / BINS_PER_QUARTER] += reference[b];
    total += reference[b];
  }

  for (unsigned quarter = 0; quarter < NUM_QUARTERS; ++quarter) {
    std::printf("  %s %10llu  ", QUARTER_LABELS[quarter], quarterCounts[quarter]);
    printBar(quarterCounts[quarter], total);
  }
}

static unsigned fullestBin(const Histogram& histo) {
  unsigned fullest = 0;
  for (unsigned b = 1; b < histo.size(); ++b) {
    if (histo[b] > histo[fullest]) {
      fullest = b;
    }
  }
  return fullest;
}

static Histogram mergeIntoSmallBins(const Histogram& histo) {
  Histogram small(NUM_BINS_SMALL, 0);
  for (unsigned b = 0; b < NUM_BINS; ++b) {
    small[b / INTENSITIES_PER_SMALL_BIN] += histo[b];
  }
  return small;
}

enum class ImageKernel {
  Basic,
  PrivateGlobal,
  PrivateShared,
  CoarseContiguous,
  CoarseInterleaved,
  ThreadPrivate,
  ThreadRun,
};

enum class Coarsening { OnePixelPerThread, CoarseFactorPixelsPerThread };

struct KernelVariant {
  const char* name;
  ImageKernel kernel;
  Coarsening coarsening;
};

const KernelVariant VARIANTS[] = {
    {"1 basic (device atomics)", ImageKernel::Basic, Coarsening::OnePixelPerThread},
    {"2 private copies, global", ImageKernel::PrivateGlobal, Coarsening::OnePixelPerThread},
    {"3 private copy, shared", ImageKernel::PrivateShared, Coarsening::OnePixelPerThread},
    {"4 + coarsening contiguous",
     ImageKernel::CoarseContiguous,
     Coarsening::CoarseFactorPixelsPerThread},
    {"5 + coarsening interleaved",
     ImageKernel::CoarseInterleaved,
     Coarsening::CoarseFactorPixelsPerThread},
    {"6 thread-private, 16 bins",
     ImageKernel::ThreadPrivate,
     Coarsening::CoarseFactorPixelsPerThread},
    {"7 most recent bin private", ImageKernel::ThreadRun, Coarsening::CoarseFactorPixelsPerThread},
};

static unsigned numPixels() {
  return WIDTH * HEIGHT;
}

static unsigned blocksFor(Coarsening coarsening) {
  if (coarsening == Coarsening::OnePixelPerThread) {
    return cdiv(numPixels(), THREADS_PER_BLOCK);
  }
  return cdiv(numPixels(), THREADS_PER_BLOCK * COARSE_FACTOR);
}

struct DeviceBuffers {
  unsigned char* image;
  unsigned* bins;
  unsigned* bins_pool;
};

static size_t poolBytes(unsigned blocks) {
  return static_cast<size_t>(blocks) * NUM_BINS * sizeof(unsigned);
}

static DeviceBuffers copyImageToDevice(const Image& image) {
  DeviceBuffers device;
  CHECK(cudaMalloc(&device.image, numPixels()));
  CHECK(cudaMalloc(&device.bins, NUM_BINS * sizeof(unsigned)));
  CHECK(cudaMalloc(&device.bins_pool, poolBytes(blocksFor(Coarsening::OnePixelPerThread))));
  CHECK(cudaMemcpy(device.image, image.data(), numPixels(), cudaMemcpyHostToDevice));
  return device;
}

static void freeDeviceBuffers(const DeviceBuffers& device) {
  CHECK(cudaFree(device.image));
  CHECK(cudaFree(device.bins));
  CHECK(cudaFree(device.bins_pool));
}

static void clearAndLaunch(ImageKernel kernel, unsigned blocks, const DeviceBuffers& device) {
  const unsigned char* image = device.image;
  const dim3 grid(blocks);
  const dim3 block(THREADS_PER_BLOCK);

  CHECK(cudaMemset(device.bins, 0, NUM_BINS * sizeof(unsigned)));
  switch (kernel) {
    case ImageKernel::Basic:
      launch(histogram_basic, grid, block, image, device.bins, WIDTH, HEIGHT);
      break;
    case ImageKernel::PrivateGlobal:
      CHECK(cudaMemset(device.bins_pool, 0, poolBytes(blocks)));
      launch(histogram_private_global,
             grid,
             block,
             image,
             device.bins,
             device.bins_pool,
             WIDTH,
             HEIGHT);
      break;
    case ImageKernel::PrivateShared:
      launch(histogram_private_shared, grid, block, image, device.bins, WIDTH, HEIGHT);
      break;
    case ImageKernel::CoarseContiguous:
      launch(histogram_coarse_contig, grid, block, image, device.bins, WIDTH, HEIGHT);
      break;
    case ImageKernel::CoarseInterleaved:
      launch(histogram_coarse_interl, grid, block, image, device.bins, WIDTH, HEIGHT);
      break;
    case ImageKernel::ThreadPrivate:
      launch(histogram_thread_private, grid, block, image, device.bins, WIDTH, HEIGHT);
      break;
    case ImageKernel::ThreadRun:
      launch(histogram_thread_run, grid, block, image, device.bins, WIDTH, HEIGHT);
      break;
  }
}

static double timeKernel(ImageKernel kernel, unsigned blocks, const DeviceBuffers& device) {
  if (WARM_UP_BEFORE_TIMING) {
    clearAndLaunch(kernel, blocks, device);
  }
  CHECK(cudaGetLastError());

  Timer timer;
  for (int run = 0; run < TIMED_RUNS; ++run) {
    clearAndLaunch(kernel, blocks, device);
  }
  const double ms = timer.ms() / TIMED_RUNS;
  CHECK(cudaGetLastError());
  return ms;
}

static Histogram copyBinsToHost(const DeviceBuffers& device) {
  Histogram bins(NUM_BINS);
  CHECK(cudaMemcpy(bins.data(), device.bins, NUM_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return bins;
}

static bool firstBinsMatch(const Histogram& got, const Histogram& expected) {
  for (unsigned b = 0; b < expected.size(); ++b) {
    if (got[b] != expected[b]) {
      return false;
    }
  }
  return true;
}

struct References {
  Histogram bins;
  Histogram smallBins;
  double cpuMs;
};

static References sequentialReferences(const Image& image) {
  References references;
  references.bins.assign(NUM_BINS, 0);
  Timer timer;
  histogram_sequential(image.data(), references.bins.data(), WIDTH, HEIGHT);
  references.cpuMs = timer.ms();
  references.smallBins = mergeIntoSmallBins(references.bins);
  return references;
}

static bool isCorrect(ImageKernel kernel, const Histogram& got, const References& references) {
  if (kernel == ImageKernel::ThreadPrivate) {
    return firstBinsMatch(got, references.smallBins);
  }
  return got == references.bins;
}

static void printImageSummary(const char* label, const Histogram& reference) {
  const unsigned fullest = fullestBin(reference);
  std::printf("\n=== %s: %ux%u pixels; fullest bin %u holds %.1f%% of the pixels ===\n",
              label,
              WIDTH,
              HEIGHT,
              fullest,
              100.0 * reference[fullest] / numPixels());
  printQuarters(reference);
}

static void printTableHeader(double cpuMs) {
  std::printf("%-28s %7s %9s  %s\n", "kernel", "blocks", "ms", "check");
  std::printf("%-28s %7s %9.3f  %s\n", "0 CPU sequential", "-", cpuMs, "reference");
}

static bool runImage(const char* label, const Image& image) {
  const References references = sequentialReferences(image);
  printImageSummary(label, references.bins);

  const DeviceBuffers device = copyImageToDevice(image);
  printTableHeader(references.cpuMs);

  bool allCorrect = true;
  for (const KernelVariant& variant : VARIANTS) {
    const unsigned blocks = blocksFor(variant.coarsening);
    const double ms = timeKernel(variant.kernel, blocks, device);
    const bool correct = isCorrect(variant.kernel, copyBinsToHost(device), references);
    allCorrect = allCorrect && correct;
    std::printf("%-28s %7u %9.3f  %s\n", variant.name, blocks, ms, correct ? "PASS" : "FAIL");
  }

  freeDeviceBuffers(device);
  return allCorrect;
}

int main() {
  std::printf("Image histogram with cuda::atomic_ref (chapter 9), backend: %s\n", CUDA_BACKEND);
  std::printf("%u bins, %u threads per block, coarsening factor %u, %d timed run(s)\n",
              NUM_BINS,
              THREADS_PER_BLOCK,
              COARSE_FACTOR,
              TIMED_RUNS);

  bool allPass = runImage("noise (uniform intensities)", makeNoise(WIDTH, HEIGHT));
  allPass = runImage("photo (tree under a bright sky)", makePhoto(WIDTH, HEIGHT)) && allPass;

  std::printf("\n%s\n", allPass ? "ALL PASS" : "SOME FAILED");
  if (RUNNING_ON_CPU_EMULATION) {
    std::printf("(CPU emulation: the times measure thread start-up and barriers, not the GPU.)\n");
  }
  return allPass ? 0 : 1;
}
