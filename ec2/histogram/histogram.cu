#include "cuda_compat.h"

#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int ALPHABET_SIZE = 26;
constexpr int LETTERS_PER_BIN = 4;
constexpr unsigned NUM_BINS = (ALPHABET_SIZE + LETTERS_PER_BIN - 1) / LETTERS_PER_BIN;

constexpr int NOT_A_LETTER = -1;
constexpr unsigned THREADS_PER_BLOCK = 256;
constexpr unsigned CFACTOR = 32;

#ifndef CPU_CHARS
#define CPU_CHARS (1u << 18)
#endif
const unsigned TEXT_LENGTH = BY_BACKEND(1u << 26, CPU_CHARS);
const int TIMED_RUNS = BY_BACKEND(10, 1);
const bool WARM_UP_BEFORE_TIMING = BY_BACKEND(true, false);

__host__ __device__ inline int binOf(char c) {
  int alphabet_position = c - 'a';
  if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) {
    return alphabet_position / LETTERS_PER_BIN;
  }
  return NOT_A_LETTER;
}

__global__ void histo_basic(const char* data, unsigned length, unsigned* histo) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int b = binOf(data[i]);
    if (b >= 0) {
      atomicAdd(&histo[b], 1u);
    }
  }
}

__global__ void histo_private_global(const char* data, unsigned length, unsigned* histo) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int b = binOf(data[i]);
    if (b >= 0) {
      atomicAdd(&histo[blockIdx.x * NUM_BINS + b], 1u);
    }
  }
  __syncthreads();

  if (blockIdx.x > 0) {
    for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
      unsigned v = histo[blockIdx.x * NUM_BINS + bin];
      if (v > 0) {
        atomicAdd(&histo[bin], v);
      }
    }
  }
}

__global__ void histo_private_shared(const char* data, unsigned length, unsigned* histo) {
  __shared__ unsigned histo_s[NUM_BINS];
  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int b = binOf(data[i]);
    if (b >= 0) {
      atomicAdd(&histo_s[b], 1u);
    }
  }
  __syncthreads();

  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned v = histo_s[bin];
    if (v > 0) {
      atomicAdd(&histo[bin], v);
    }
  }
}

__global__ void histo_coarse_contig(const char* data, unsigned length, unsigned* histo) {
  __shared__ unsigned histo_s[NUM_BINS];
  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned begin = tid * CFACTOR;
  unsigned end = begin + CFACTOR;
  if (end > length) {
    end = length;
  }
  for (unsigned i = begin; i < end; ++i) {
    int b = binOf(data[i]);
    if (b >= 0) {
      atomicAdd(&histo_s[b], 1u);
    }
  }
  __syncthreads();

  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned v = histo_s[bin];
    if (v > 0) {
      atomicAdd(&histo[bin], v);
    }
  }
}

__global__ void histo_coarse_interl(const char* data, unsigned length, unsigned* histo) {
  __shared__ unsigned histo_s[NUM_BINS];
  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = tid; i < length; i += blockDim.x * gridDim.x) {
    int b = binOf(data[i]);
    if (b >= 0) {
      atomicAdd(&histo_s[b], 1u);
    }
  }
  __syncthreads();

  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned v = histo_s[bin];
    if (v > 0) {
      atomicAdd(&histo[bin], v);
    }
  }
}

__global__ void histo_aggregation(const char* data, unsigned length, unsigned* histo) {
  __shared__ unsigned histo_s[NUM_BINS];
  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }
  __syncthreads();

  unsigned accumulator = 0;
  int prevBin = -1;
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = tid; i < length; i += blockDim.x * gridDim.x) {
    int b = binOf(data[i]);
    if (b >= 0) {
      if (b == prevBin) {
        ++accumulator;
      } else {
        if (accumulator > 0) {
          atomicAdd(&histo_s[prevBin], accumulator);
        }
        accumulator = 1;
        prevBin = b;
      }
    }
  }
  if (accumulator > 0) {
    atomicAdd(&histo_s[prevBin], accumulator);
  }
  __syncthreads();

  for (unsigned bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned v = histo_s[bin];
    if (v > 0) {
      atomicAdd(&histo[bin], v);
    }
  }
}

static void histogram_sequential(const char* data, unsigned length, unsigned* histo) {
  for (unsigned b = 0; b < NUM_BINS; ++b) {
    histo[b] = 0;
  }

  for (unsigned i = 0; i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) {
      histo[alphabet_position / LETTERS_PER_BIN]++;
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

constexpr unsigned ENGLISH_LETTER_FREQUENCY_PER_10000[ALPHABET_SIZE] = {
    817, 149, 278, 425, 1270, 223, 202, 609, 697, 15,  77, 403, 241,
    675, 751, 193, 10,  599,  633, 906, 276, 98,  236, 15, 197, 7,
};

constexpr unsigned sumOfLetterFrequencies() {
  unsigned total = 0;
  for (unsigned frequency : ENGLISH_LETTER_FREQUENCY_PER_10000) {
    total += frequency;
  }
  return total;
}

constexpr unsigned LETTER_FREQUENCY_TOTAL = sumOfLetterFrequencies();
static_assert(LETTER_FREQUENCY_TOTAL == 10002, "the rounded frequencies add up to 10002");

constexpr char MOST_FREQUENT_LETTER = 'e';
constexpr unsigned MAX_WORD_LENGTH = 10;
constexpr unsigned ONE_SENTENCE_END_EVERY_N_WORDS = 12;
constexpr char SKY_LETTER = 'a';
constexpr unsigned SKY_PERCENT = 95;
constexpr unsigned ONE_HUNDRED_PERCENT = 100;
constexpr uint64_t ENGLISH_TEXT_SEED = 2024;
constexpr uint64_t SKY_TEXT_SEED = 7;

static char randomEnglishLetter(SplitMix64& rng) {
  unsigned draw = rng.below(LETTER_FREQUENCY_TOTAL);
  unsigned cumulativeFrequency = 0;
  for (int letter = 0; letter < ALPHABET_SIZE; ++letter) {
    cumulativeFrequency += ENGLISH_LETTER_FREQUENCY_PER_10000[letter];
    if (draw < cumulativeFrequency) {
      return static_cast<char>('a' + letter);
    }
  }
  return MOST_FREQUENT_LETTER;
}

static std::vector<char> makeEnglishLikeText(unsigned length, uint64_t seed) {
  SplitMix64 rng(seed);
  std::vector<char> text;
  text.reserve(length);
  auto hasRoom = [&] {
    return text.size() < length;
  };

  while (hasRoom()) {
    unsigned wordLength = 1 + rng.below(MAX_WORD_LENGTH);
    for (unsigned k = 0; k < wordLength && hasRoom(); ++k) {
      text.push_back(randomEnglishLetter(rng));
    }
    if (hasRoom() && rng.below(ONE_SENTENCE_END_EVERY_N_WORDS) == 0) {
      text.push_back('.');
    }
    if (hasRoom()) {
      text.push_back(' ');
    }
  }
  return text;
}

static std::vector<char> makeSkyText(unsigned length, uint64_t seed) {
  SplitMix64 rng(seed);
  std::vector<char> text(length);
  for (char& c : text) {
    bool isSky = rng.below(ONE_HUNDRED_PERCENT) < SKY_PERCENT;
    c = isSky ? SKY_LETTER : randomEnglishLetter(rng);
  }
  return text;
}

using Histogram = std::vector<unsigned>;

static Histogram sequentialReference(const std::vector<char>& text, double& elapsedMs) {
  Histogram reference(NUM_BINS);
  Timer timer;
  histogram_sequential(text.data(), static_cast<unsigned>(text.size()), reference.data());
  elapsedMs = timer.ms();
  return reference;
}

static unsigned long long totalCount(const Histogram& histo) {
  unsigned long long total = 0;
  for (unsigned count : histo) {
    total += count;
  }
  return total;
}

enum class FirstCopyIsFinal { No, Yes };

template <class BlockOfCharacter>
static unsigned long long countCommitAtomics(const std::vector<char>& text,
                                             unsigned numBlocks,
                                             BlockOfCharacter blockOfCharacter,
                                             FirstCopyIsFinal firstCopyIsFinal) {
  std::vector<bool> binIsUsed(static_cast<size_t>(numBlocks) * NUM_BINS, false);
  for (unsigned i = 0; i < text.size(); ++i) {
    int b = binOf(text[i]);
    if (b >= 0) {
      binIsUsed[static_cast<size_t>(blockOfCharacter(i)) * NUM_BINS + b] = true;
    }
  }

  size_t firstCommittedCounter = firstCopyIsFinal == FirstCopyIsFinal::Yes ? NUM_BINS : 0;
  unsigned long long commits = 0;
  for (size_t counter = firstCommittedCounter; counter < binIsUsed.size(); ++counter) {
    if (binIsUsed[counter]) {
      ++commits;
    }
  }
  return commits;
}

static unsigned long long countAggregatedAtomics(const std::vector<char>& text,
                                                 unsigned threadCount) {
  std::vector<int> previousBinOfThread(threadCount, NOT_A_LETTER);
  unsigned long long runs = 0;
  for (unsigned i = 0; i < text.size(); ++i) {
    int b = binOf(text[i]);
    int& previousBin = previousBinOfThread[i % threadCount];
    if (b >= 0 && b != previousBin) {
      ++runs;
      previousBin = b;
    }
  }
  return runs;
}

constexpr int BAR_WIDTH = 50;
const char* const BIN_LABELS[NUM_BINS] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};

static void printBar(unsigned count, unsigned long long total) {
  int width = total ? static_cast<int>(BAR_WIDTH * static_cast<double>(count) / total + 0.5) : 0;
  for (int j = 0; j < width; ++j) {
    std::putchar('#');
  }
  std::putchar('\n');
}

static void printLetterHistogram(const Histogram& histo) {
  const unsigned long long total = totalCount(histo);
  for (unsigned b = 0; b < NUM_BINS; ++b) {
    std::printf("  %s %10u  ", BIN_LABELS[b], histo[b]);
    printBar(histo[b], total);
  }
}

using HistoKernel = void (*)(const char*, unsigned, unsigned*);

struct KernelVariant {
  const char* name;
  HistoKernel kernel;
  unsigned blocks;
  size_t countersToClear;
  unsigned long long globalAtomics;
  unsigned long long sharedAtomics;
};

struct LaunchShape {
  unsigned fineBlocks;
  unsigned coarseBlocks;
  unsigned coarseThreads;
};

static LaunchShape launchShapeFor(unsigned length) {
  LaunchShape shape;
  shape.fineBlocks = cdiv(length, THREADS_PER_BLOCK);
  shape.coarseBlocks = cdiv(length, THREADS_PER_BLOCK * CFACTOR);
  shape.coarseThreads = shape.coarseBlocks * THREADS_PER_BLOCK;
  return shape;
}

static std::vector<KernelVariant> kernelVariants(const std::vector<char>& text,
                                                 const LaunchShape& shape,
                                                 unsigned long long letters) {
  const unsigned coarseThreads = shape.coarseThreads;
  auto blockOfOneCharacterPerThread = [](unsigned i) {
    return i / THREADS_PER_BLOCK;
  };
  auto blockOfContiguousChunk = [](unsigned i) {
    return (i / CFACTOR) / THREADS_PER_BLOCK;
  };
  auto blockOfInterleavedCharacter = [=](unsigned i) {
    return (i % coarseThreads) / THREADS_PER_BLOCK;
  };

  const unsigned long long fineCommits = countCommitAtomics(
      text, shape.fineBlocks, blockOfOneCharacterPerThread, FirstCopyIsFinal::No);
  const unsigned long long fineCommitsIntoFirstCopy = countCommitAtomics(
      text, shape.fineBlocks, blockOfOneCharacterPerThread, FirstCopyIsFinal::Yes);
  const unsigned long long contiguousCommits =
      countCommitAtomics(text, shape.coarseBlocks, blockOfContiguousChunk, FirstCopyIsFinal::No);
  const unsigned long long interleavedCommits = countCommitAtomics(
      text, shape.coarseBlocks, blockOfInterleavedCharacter, FirstCopyIsFinal::No);
  const unsigned long long aggregatedAtomics = countAggregatedAtomics(text, coarseThreads);
  const size_t oneCopyPerBlock = static_cast<size_t>(shape.fineBlocks) * NUM_BINS;

  return {
      {"1 basic (global atomics)", histo_basic, shape.fineBlocks, NUM_BINS, letters, 0},
      {"2 private copies, global",
       histo_private_global,
       shape.fineBlocks,
       oneCopyPerBlock,
       letters + fineCommitsIntoFirstCopy,
       0},
      {"3 private copy, shared",
       histo_private_shared,
       shape.fineBlocks,
       NUM_BINS,
       fineCommits,
       letters},
      {"4 + coarsening contiguous",
       histo_coarse_contig,
       shape.coarseBlocks,
       NUM_BINS,
       contiguousCommits,
       letters},
      {"5 + coarsening interleaved",
       histo_coarse_interl,
       shape.coarseBlocks,
       NUM_BINS,
       interleavedCommits,
       letters},
      {"6 + aggregation",
       histo_aggregation,
       shape.coarseBlocks,
       NUM_BINS,
       interleavedCommits,
       aggregatedAtomics},
  };
}

struct DeviceBuffers {
  char* text;
  unsigned* histo;
  unsigned length;
};

static DeviceBuffers copyTextToDevice(const std::vector<char>& text, size_t histoCounters) {
  DeviceBuffers device;
  device.length = static_cast<unsigned>(text.size());

  CHECK(cudaMalloc(&device.text, device.length));
  CHECK(cudaMemcpy(device.text, text.data(), device.length, cudaMemcpyHostToDevice));

  CHECK(cudaMalloc(&device.histo, histoCounters * sizeof(unsigned)));
  return device;
}

static void freeDeviceBuffers(const DeviceBuffers& device) {
  CHECK(cudaFree(device.text));
  CHECK(cudaFree(device.histo));
}

static double timeKernel(const KernelVariant& variant, const DeviceBuffers& device) {
  auto clearAndLaunch = [&] {
    CHECK(cudaMemset(device.histo, 0, variant.countersToClear * sizeof(unsigned)));
    launch(variant.kernel,
           dim3(variant.blocks),
           dim3(THREADS_PER_BLOCK),
           device.text,
           device.length,
           device.histo);
  };

  if (WARM_UP_BEFORE_TIMING) {
    clearAndLaunch();
  }
  CHECK(cudaGetLastError());

  Timer timer;
  for (int run = 0; run < TIMED_RUNS; ++run) {
    clearAndLaunch();
  }
  double ms = timer.ms() / TIMED_RUNS;
  CHECK(cudaGetLastError());
  return ms;
}

static Histogram copyHistogramToHost(const DeviceBuffers& device) {
  Histogram histo(NUM_BINS);
  CHECK(
      cudaMemcpy(histo.data(), device.histo, NUM_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return histo;
}

static void printTableHeader(double cpuMs) {
  std::printf("%-28s %8s %9s %15s %15s  %s\n",
              "kernel",
              "blocks",
              "ms",
              "global atomics",
              "shared atomics",
              "check");
  std::printf(
      "%-28s %8s %9.3f %15s %15s  %s\n", "CPU sequential", "-", cpuMs, "-", "-", "reference");
}

static void printVariantRow(const KernelVariant& variant, double ms, bool correct) {
  std::printf("%-28s %8u %9.3f %15llu %15llu  %s\n",
              variant.name,
              variant.blocks,
              ms,
              variant.globalAtomics,
              variant.sharedAtomics,
              correct ? "PASS" : "FAIL");
}

static bool runInput(const char* label, const std::vector<char>& text) {
  double cpuMs;
  const Histogram reference = sequentialReference(text, cpuMs);
  const unsigned long long letters = totalCount(reference);

  std::printf("\n=== input: %s, %zu characters, %llu letters ===\n", label, text.size(), letters);
  printLetterHistogram(reference);

  const LaunchShape shape = launchShapeFor(static_cast<unsigned>(text.size()));
  const std::vector<KernelVariant> variants = kernelVariants(text, shape, letters);
  const DeviceBuffers device =
      copyTextToDevice(text, static_cast<size_t>(shape.fineBlocks) * NUM_BINS);

  printTableHeader(cpuMs);
  bool allCorrect = true;
  for (const KernelVariant& variant : variants) {
    const double ms = timeKernel(variant, device);
    const bool correct = copyHistogramToHost(device) == reference;
    allCorrect = allCorrect && correct;
    printVariantRow(variant, ms, correct);
  }

  freeDeviceBuffers(device);
  return allCorrect;
}

int main() {
  std::printf("Parallel histogram (chapter 9), backend: %s\n", CUDA_BACKEND);
  std::printf(
      "%u bins of %d letters, %u threads per block, coarsening factor %u, %d timed run(s)\n",
      NUM_BINS,
      LETTERS_PER_BIN,
      THREADS_PER_BLOCK,
      CFACTOR,
      TIMED_RUNS);

  bool allPass = runInput("English-like text", makeEnglishLikeText(TEXT_LENGTH, ENGLISH_TEXT_SEED));
  allPass = runInput("sky (95% in one bin)", makeSkyText(TEXT_LENGTH, SKY_TEXT_SEED)) && allPass;

  std::printf("\n%s\n", allPass ? "ALL PASS" : "SOME FAILED");
  return allPass ? 0 : 1;
}
