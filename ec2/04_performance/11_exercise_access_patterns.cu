#include "cuda_compat.h"

#include <cstdio>
#include <set>
#include <vector>

const unsigned BLOCK_SIZE = 256;
const unsigned BLOCK_COUNT = 2;
const unsigned THREAD_COUNT = BLOCK_SIZE * BLOCK_COUNT;
const unsigned CHUNKS = 4;
const unsigned D_OFFSET = 8;
const unsigned E_STRIDE = 8;
const unsigned WARP_SIZE = 32;
const unsigned SEGMENT_BYTES = 128;
const unsigned BANK_COUNT = 32;
const unsigned LANES_SHOWN = 4;
const unsigned J_SHOWN = 1;
const unsigned VALUES_SHOWN = 6;

__global__ void foo_kernel(const float* a, const float* b, const float* c, float* d, float* e) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float a_s[BLOCK_SIZE];
  __shared__ float bc_s[CHUNKS * BLOCK_SIZE];
  a_s[threadIdx.x] = a[i];
  for (unsigned j = 0; j < CHUNKS; ++j) {
    bc_s[j * BLOCK_SIZE + threadIdx.x] = b[j * blockDim.x * gridDim.x + i] + c[i * CHUNKS + j];
  }
  __syncthreads();
  d[i + D_OFFSET] = a_s[threadIdx.x];
  e[i * E_STRIDE] = bc_s[threadIdx.x * CHUNKS];
}

struct Access {
  const char* line;
  const char* expression;
  unsigned (*indexOfLane)(unsigned lane);
};

const Access GLOBAL_ACCESSES[] = {
    {"L5 ",
     "a[i]",
     [](unsigned i) {
       return i;
     }},
    {"L7 ",
     "b[j*blockDim.x*gridDim.x + i]",
     [](unsigned i) {
       return J_SHOWN * THREAD_COUNT + i;
     }},
    {"L7 ",
     "c[i*4 + j]",
     [](unsigned i) {
       return i * CHUNKS + J_SHOWN;
     }},
    {"L10",
     "d[i + 8]",
     [](unsigned i) {
       return i + D_OFFSET;
     }},
    {"L11",
     "e[i*8]",
     [](unsigned i) {
       return i * E_STRIDE;
     }},
};

const Access SHARED_ACCESSES[] = {
    {"L5 ",
     "a_s[threadIdx.x]",
     [](unsigned t) {
       return t;
     }},
    {"L7 ",
     "bc_s[j*256 + threadIdx.x]",
     [](unsigned t) {
       return J_SHOWN * BLOCK_SIZE + t;
     }},
    {"L10",
     "a_s[threadIdx.x]",
     [](unsigned t) {
       return t;
     }},
    {"L11",
     "bc_s[threadIdx.x*4]",
     [](unsigned t) {
       return t * CHUNKS;
     }},
};

static void printLaneIndices(const Access& access) {
  std::printf("  %s %-30s lanes 0..%u:", access.line, access.expression, LANES_SHOWN - 1);
  for (unsigned lane = 0; lane < LANES_SHOWN; ++lane) {
    std::printf("%5u", access.indexOfLane(lane));
  }
}

static void printGlobalAccess(const Access& access) {
  std::set<unsigned> segments;
  for (unsigned lane = 0; lane < WARP_SIZE; ++lane) {
    segments.insert(access.indexOfLane(lane) * sizeof(float) / SEGMENT_BYTES);
  }
  const unsigned stride = access.indexOfLane(1) - access.indexOfLane(0);
  printLaneIndices(access);
  std::printf("  stride %u, 128 B segments: %2zu, %s",
              stride,
              segments.size(),
              stride == 1 ? "coalesced\n" : "uncoalesced\n");
}

static void printSharedAccess(const Access& access) {
  std::vector<unsigned> lanesPerBank(BANK_COUNT, 0);
  for (unsigned lane = 0; lane < WARP_SIZE; ++lane) {
    ++lanesPerBank[access.indexOfLane(lane) % BANK_COUNT];
  }
  unsigned banksUsed = 0;
  unsigned ways = 0;
  for (unsigned lanes : lanesPerBank) {
    banksUsed += lanes > 0 ? 1 : 0;
    ways = lanes > ways ? lanes : ways;
  }
  printLaneIndices(access);
  std::printf("  %2u banks, ", banksUsed);
  if (ways == 1) {
    std::printf("no conflict\n");
  } else {
    std::printf("%u-way conflict\n", ways);
  }
}

static void runFooOnCpu(const std::vector<float>& a,
                        const std::vector<float>& b,
                        const std::vector<float>& c,
                        std::vector<float>& d,
                        std::vector<float>& e) {
  for (unsigned block = 0; block < BLOCK_COUNT; ++block) {
    std::vector<float> bc_s(CHUNKS * BLOCK_SIZE);
    for (unsigned t = 0; t < BLOCK_SIZE; ++t) {
      const unsigned i = block * BLOCK_SIZE + t;
      for (unsigned j = 0; j < CHUNKS; ++j) {
        bc_s[j * BLOCK_SIZE + t] = b[j * THREAD_COUNT + i] + c[i * CHUNKS + j];
      }
    }
    for (unsigned t = 0; t < BLOCK_SIZE; ++t) {
      const unsigned i = block * BLOCK_SIZE + t;
      d[i + D_OFFSET] = a[i];
      e[i * E_STRIDE] = bc_s[t * CHUNKS];
    }
  }
}

static float* copyToDevice(const std::vector<float>& values) {
  float* values_d;
  CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
  CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
  return values_d;
}

static void copyToHost(std::vector<float>& values, float* values_d) {
  CHECK(cudaMemcpy(values.data(), values_d, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(values_d));
}

static void printFirstValues(const char* name, const std::vector<float>& values, unsigned step) {
  std::printf("%-12s", name);
  for (unsigned k = 0; k < VALUES_SHOWN; ++k) {
    std::printf("%7.1f", values[k * step]);
  }
  std::printf("\n");
}

int main() {
  std::vector<float> a(THREAD_COUNT);
  std::vector<float> b(CHUNKS * THREAD_COUNT);
  std::vector<float> c(THREAD_COUNT * CHUNKS);
  for (unsigned k = 0; k < a.size(); ++k) {
    a[k] = static_cast<float>(k % 50);
  }
  for (unsigned k = 0; k < b.size(); ++k) {
    b[k] = static_cast<float>(k % 7);
    c[k] = 0.5f * static_cast<float>(k % 4);
  }

  std::vector<float> d(THREAD_COUNT + D_OFFSET, 0.0f);
  std::vector<float> e(THREAD_COUNT * E_STRIDE, 0.0f);
  float* a_d = copyToDevice(a);
  float* b_d = copyToDevice(b);
  float* c_d = copyToDevice(c);
  float* d_d = copyToDevice(d);
  float* e_d = copyToDevice(e);

  launch(foo_kernel, dim3(BLOCK_COUNT), dim3(BLOCK_SIZE), a_d, b_d, c_d, d_d, e_d);
  CHECK(cudaGetLastError());

  copyToHost(d, d_d);
  copyToHost(e, e_d);
  CHECK(cudaFree(a_d));
  CHECK(cudaFree(b_d));
  CHECK(cudaFree(c_d));

  std::printf("%u blocks of %u threads; warp 0 = threads 0..31, j = %u for L7\n",
              BLOCK_COUNT,
              BLOCK_SIZE,
              J_SHOWN);
  printFirstValues("a[0..]", a, 1);
  printFirstValues("b[0..]", b, 1);
  printFirstValues("c[0..]", c, 1);
  printFirstValues("d[8..]", std::vector<float>(d.begin() + D_OFFSET, d.end()), 1);
  printFirstValues("e[0, 8, ..]", e, E_STRIDE);

  std::printf("global memory, one warp:\n");
  for (const Access& access : GLOBAL_ACCESSES) {
    printGlobalAccess(access);
  }
  std::printf("shared memory (coalescing does not apply), one warp:\n");
  for (const Access& access : SHARED_ACCESSES) {
    printSharedAccess(access);
  }

  std::vector<float> expectedD(d.size(), 0.0f);
  std::vector<float> expectedE(e.size(), 0.0f);
  runFooOnCpu(a, b, c, expectedD, expectedE);
  const bool pass = d == expectedD && e == expectedE;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
