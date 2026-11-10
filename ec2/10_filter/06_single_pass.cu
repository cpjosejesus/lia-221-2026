#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "cuda_compat.h"

constexpr int THRESHOLD = 100;
constexpr int F_THREADS = 4;
constexpr int F_COARSEN = 4;
constexpr int F_SECTION = F_THREADS * F_COARSEN;
constexpr int FLAG_READY = 1;

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

__device__ int next_ticket(int* ticket) {
  return atomicAdd(ticket, 1);
}

__device__ int atomic_read(int* address) {
  return atomicAdd(address, 0);
}

__device__ void wait_until_ready(int* flag) {
  int state = atomic_read(flag);
  while (state != FLAG_READY) {
    state = atomic_read(flag);
  }
}

__device__ int wait_for_sum_of_previous_sections(int* flags, int* prefix, int bid) {
  if (bid == 0) {
    return 0;
  }
  wait_until_ready(&flags[bid]);
  return atomic_read(&prefix[bid]);
}

__device__ void publish_sum_of_sections_up_to(int bid, int sum, int* flags, int* prefix) {
  atomicExch(&prefix[bid + 1], sum);
  __threadfence();
  atomicExch(&flags[bid + 1], FLAG_READY);
}

__global__ void compact_single_pass(
    const int* in, int* out, int* flags, int* prefix, int* ticket, int n, int threshold) {
  __shared__ int vals[F_SECTION];
  __shared__ int stage[F_SECTION];
  __shared__ int part[F_THREADS];
  __shared__ int bid_s;
  __shared__ int start_s;

  if (threadIdx.x == 0) {
    bid_s = next_ticket(ticket);
  }
  __syncthreads();
  const int bid = bid_s;

  const int count = block_compact(in, n, bid * F_SECTION, threshold, vals, stage, part);

  if (threadIdx.x == 0) {
    const int before = wait_for_sum_of_previous_sections(flags, prefix, bid);
    publish_sum_of_sections_up_to(bid, before + count, flags, prefix);
    printf("  ticket %d: stage holds %d, sum before = %d, goes to out[%d..%d)\n",
           bid,
           count,
           before,
           before,
           before + count);
    start_s = before;
  }
  __syncthreads();

  copy_out_coalesced(stage, count, out + start_s);
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
  int* d_out = deviceCopyOf(std::vector<int>(n, -1));
  int* d_flags = deviceCopyOf(std::vector<int>(sections + 1, 0));
  int* d_prefix = deviceCopyOf(std::vector<int>(sections + 1, 0));
  int* d_ticket = deviceCopyOf({0});

  std::printf("keep v < %d, %d sections of %d elements\n", THRESHOLD, sections, F_SECTION);
  printRows("in", in);
  launch(compact_single_pass,
         dim3(sections),
         dim3(F_THREADS),
         d_in,
         d_out,
         d_flags,
         d_prefix,
         d_ticket,
         n,
         THRESHOLD);
  CHECK(cudaGetLastError());

  const int count = hostCopyOf(d_prefix, sections + 1)[sections];
  const std::vector<int> out = hostCopyOf(d_out, count);
  printRows("out", out);

  const bool pass = (out == cpuFilter(in));
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_flags));
  CHECK(cudaFree(d_prefix));
  CHECK(cudaFree(d_ticket));
  return pass ? 0 : 1;
}
