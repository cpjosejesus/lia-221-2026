#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "cuda_compat.h"

constexpr int THRESHOLD = 100;
constexpr int F_THREADS = 4;
constexpr int F_COARSEN = 4;
constexpr int F_SECTION = F_THREADS * F_COARSEN;

__host__ __device__ inline bool keep(int v, int threshold) {
  return v < threshold;
}

__host__ __device__ inline int exclusive_from_inclusive(const int* inclusive, int i) {
  return (i > 0) ? inclusive[i - 1] : 0;
}

__device__ void load_section_coalesced(const int* in, int n, int base, int* vals) {
  for (int k = 0; k < F_COARSEN; ++k) {
    const int j = k * F_THREADS + threadIdx.x;
    vals[j] = (base + j < n) ? in[base + j] : 0;
  }
}

__device__ bool is_real_and_kept(const int* vals, int j, int base, int n, int threshold) {
  return base + j < n && keep(vals[j], threshold);
}

__device__ void kogge_stone_in_shared(int* part) {
  const int t = threadIdx.x;
  for (int stride = 1; stride < F_THREADS; stride *= 2) {
    __syncthreads();
    int temp = part[t];
    if (t >= stride) {
      temp += part[t - stride];
    }
    __syncthreads();
    part[t] = temp;
  }
}

__device__ void copy_out_coalesced(const int* stage, int count, int* out) {
  for (int k = threadIdx.x; k < count; k += F_THREADS) {
    out[k] = stage[k];
  }
}

__device__ int block_compact(
    const int* in, int n, int base, int threshold, int* vals, int* stage, int* part) {
  const int t = threadIdx.x;
  const int first = t * F_COARSEN;

  load_section_coalesced(in, n, base, vals);
  __syncthreads();

  int count = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    if (is_real_and_kept(vals, first + k, base, n, threshold)) {
      ++count;
    }
  }
  part[t] = count;

  kogge_stone_in_shared(part);
  __syncthreads();

  int pos = exclusive_from_inclusive(part, t);
  for (int k = 0; k < F_COARSEN; ++k) {
    if (is_real_and_kept(vals, first + k, base, n, threshold)) {
      stage[pos++] = vals[first + k];
    }
  }
  __syncthreads();
  return part[F_THREADS - 1];
}

__global__ void count_kept(const int* in, int* counts, int n, int threshold) {
  __shared__ int blockCount;
  if (threadIdx.x == 0) {
    blockCount = 0;
  }
  __syncthreads();

  const int base = blockIdx.x * F_SECTION;
  int c = 0;
  for (int k = 0; k < F_COARSEN; ++k) {
    const int j = base + k * F_THREADS + threadIdx.x;
    if (j < n && keep(in[j], threshold)) {
      ++c;
    }
  }
  atomicAdd(&blockCount, c);
  __syncthreads();

  if (threadIdx.x == 0) {
    counts[blockIdx.x] = blockCount;
  }
}

__global__ void scan_counts(const int* counts, int* scannedCounts, int sections) {
  __shared__ int part[F_THREADS];
  const int t = threadIdx.x;
  part[t] = (t < sections) ? counts[t] : 0;
  kogge_stone_in_shared(part);
  if (t < sections) {
    scannedCounts[t] = part[t];
  }
}

__global__ void compact_staged(
    const int* in, int* out, const int* scannedCounts, int n, int threshold) {
  __shared__ int vals[F_SECTION];
  __shared__ int stage[F_SECTION];
  __shared__ int part[F_THREADS];
  const int count = block_compact(in, n, blockIdx.x * F_SECTION, threshold, vals, stage, part);
  const int start = exclusive_from_inclusive(scannedCounts, blockIdx.x);
  if (threadIdx.x == 0) {
    printf("  section %d: stage holds %d, goes to out[%d..%d)\n",
           blockIdx.x,
           count,
           start,
           start + count);
  }
  copy_out_coalesced(stage, count, out + start);
}

void printRows(const char* label, const std::vector<int>& values) {
  for (size_t first = 0; first < values.size(); first += F_SECTION) {
    std::printf("  %-7s", first == 0 ? label : "");
    for (size_t j = first; j < values.size() && j < first + F_SECTION; ++j) {
      std::printf("%5d", values[j]);
    }
    std::printf("\n");
  }
}

int* deviceCopyOf(const std::vector<int>& host) {
  int* device = nullptr;
  CHECK(cudaMalloc(&device, host.size() * sizeof(int)));
  CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
  return device;
}

std::vector<int> hostCopyOf(const int* device, int count) {
  std::vector<int> host(count);
  CHECK(cudaMemcpy(host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost));
  return host;
}

std::vector<int> cpuFilter(const std::vector<int>& in) {
  std::vector<int> out;
  std::copy_if(in.begin(), in.end(), std::back_inserter(out), [](int v) {
    return keep(v, THRESHOLD);
  });
  return out;
}

int main() {
  const std::vector<int> in = {812, 45,  30,  17,  999, 60,  431, 8,   77,  650, 230, 905, 140, 91,
                               3,   520, 5,   310, 66,  720, 12,  480, 99,  250, 870, 41,  150, 600,
                               38,  7,   940, 210, 88,  360, 515, 29,  700, 64,  130, 905};
  const int n = static_cast<int>(in.size());
  const int sections = static_cast<int>(cdiv(n, F_SECTION));

  int* d_in = deviceCopyOf(in);
  int* d_counts = deviceCopyOf(std::vector<int>(sections));
  int* d_scannedCounts = deviceCopyOf(std::vector<int>(sections));
  int* d_out = deviceCopyOf(std::vector<int>(n, -1));

  std::printf("keep v < %d, %d sections of %d elements\n", THRESHOLD, sections, F_SECTION);
  printRows("in", in);
  launch(count_kept, dim3(sections), dim3(F_THREADS), d_in, d_counts, n, THRESHOLD);
  launch(scan_counts, dim3(1), dim3(F_THREADS), d_counts, d_scannedCounts, sections);
  printRows("counts", hostCopyOf(d_counts, sections));
  printRows("scanned", hostCopyOf(d_scannedCounts, sections));
  launch(
      compact_staged, dim3(sections), dim3(F_THREADS), d_in, d_out, d_scannedCounts, n, THRESHOLD);
  CHECK(cudaGetLastError());

  const int count = hostCopyOf(d_scannedCounts, sections)[sections - 1];
  const std::vector<int> out = hostCopyOf(d_out, count);
  printRows("out", out);

  const bool countsFitOneBlock = sections <= F_THREADS;
  const bool pass = countsFitOneBlock && (out == cpuFilter(in));
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_counts));
  CHECK(cudaFree(d_scannedCounts));
  CHECK(cudaFree(d_out));
  return pass ? 0 : 1;
}
