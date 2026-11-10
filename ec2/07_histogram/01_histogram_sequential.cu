#include "cuda_compat.h"

#include <cstdio>

constexpr unsigned int WIDTH = 8;
constexpr unsigned int HEIGHT = 8;
constexpr unsigned int NUM_INTENSITIES = 256;
constexpr unsigned int NUM_BINS = 4;
constexpr unsigned int INTENSITIES_PER_BIN = NUM_INTENSITIES / NUM_BINS;

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

const unsigned int EXPECTED_BINS[NUM_BINS] = {6, 12, 14, 32};

void histogram_sequential(const unsigned char* image,
                          unsigned int* bins,
                          unsigned int width,
                          unsigned int height) {
  for (unsigned int i = 0; i < width * height; ++i) {
    unsigned char b = image[i] / INTENSITIES_PER_BIN;
    ++bins[b];
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

int main() {
  printImage();
  std::printf("\nbin = intensity / %u; the loop visits the pixels row by row\n",
              INTENSITIES_PER_BIN);
  printBinHeader();

  unsigned int bins[NUM_BINS] = {0, 0, 0, 0};
  for (unsigned int row = 0; row < HEIGHT; ++row) {
    histogram_sequential(TREE_IMAGE[row], bins, WIDTH, 1);
    char label[16];
    std::snprintf(label, sizeof(label), "after row %u", row);
    printBins(label, bins);
  }
  printBins("expected", EXPECTED_BINS);

  const bool ok = sameBins(bins, EXPECTED_BINS);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
