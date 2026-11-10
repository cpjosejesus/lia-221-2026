#include "cuda_compat.h"

#include <cstdio>
#include <vector>

const int PARTICLE_COUNT = 8;
const int BLOCK_SIZE = 8;
const int WARP_SIZE = 32;
const int SECTOR_BYTES = 32;

struct Particle {
  float x;
  float y;
  float z;
  float m;
};

struct Particles {
  float* x;
  float* y;
  float* z;
  float* m;
};

__global__ void sumXAoS(const Particle* p, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = p[i].x * 2.0f;
  }
}

__global__ void sumXSoA(Particles p, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = p.x[i] * 2.0f;
  }
}

static std::vector<Particle> makeParticles() {
  std::vector<Particle> particles(PARTICLE_COUNT);
  for (int i = 0; i < PARTICLE_COUNT; ++i) {
    const float base = static_cast<float>(i);
    particles[i] = {base, base + 0.25f, base + 0.5f, 1.0f};
  }
  return particles;
}

template <class Layout>
static std::vector<float> sumXOnDevice(void (*kernel)(Layout, float*, int), Layout particles_d) {
  std::vector<float> out(PARTICLE_COUNT);
  float* out_d;
  CHECK(cudaMalloc(&out_d, PARTICLE_COUNT * sizeof(float)));

  launch(kernel,
         dim3(cdiv(PARTICLE_COUNT, BLOCK_SIZE)),
         dim3(BLOCK_SIZE),
         particles_d,
         out_d,
         PARTICLE_COUNT);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(out.data(), out_d, PARTICLE_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(out_d));
  return out;
}

static std::vector<float> runAoS(const std::vector<Particle>& particles) {
  Particle* particles_d;
  const size_t bytes = PARTICLE_COUNT * sizeof(Particle);
  CHECK(cudaMalloc(&particles_d, bytes));
  CHECK(cudaMemcpy(particles_d, particles.data(), bytes, cudaMemcpyHostToDevice));

  const std::vector<float> out = sumXOnDevice<const Particle*>(sumXAoS, particles_d);
  CHECK(cudaFree(particles_d));
  return out;
}

static float* copyFieldToDevice(const std::vector<Particle>& particles, float Particle::* field) {
  std::vector<float> values(PARTICLE_COUNT);
  for (int i = 0; i < PARTICLE_COUNT; ++i) {
    values[i] = particles[i].*field;
  }

  float* values_d;
  CHECK(cudaMalloc(&values_d, PARTICLE_COUNT * sizeof(float)));
  CHECK(
      cudaMemcpy(values_d, values.data(), PARTICLE_COUNT * sizeof(float), cudaMemcpyHostToDevice));
  return values_d;
}

static std::vector<float> runSoA(const std::vector<Particle>& particles) {
  Particles particles_d;
  particles_d.x = copyFieldToDevice(particles, &Particle::x);
  particles_d.y = copyFieldToDevice(particles, &Particle::y);
  particles_d.z = copyFieldToDevice(particles, &Particle::z);
  particles_d.m = copyFieldToDevice(particles, &Particle::m);

  const std::vector<float> out = sumXOnDevice<Particles>(sumXSoA, particles_d);
  CHECK(cudaFree(particles_d.x));
  CHECK(cudaFree(particles_d.y));
  CHECK(cudaFree(particles_d.z));
  CHECK(cudaFree(particles_d.m));
  return out;
}

static void printWarpTraffic(const char* layout, int bytesBetweenThreads) {
  const int usefulBytes = WARP_SIZE * static_cast<int>(sizeof(float));
  const int spannedBytes = (WARP_SIZE - 1) * bytesBetweenThreads + static_cast<int>(sizeof(float));
  const int sectors = cdiv(spannedBytes, SECTOR_BYTES);
  const float usedPercent = 100.0f * usefulBytes / (sectors * SECTOR_BYTES);

  std::printf("%s: threads %2d B apart, a warp touches %2d sectors of 32 B, %5.1f %% used\n",
              layout,
              bytesBetweenThreads,
              sectors,
              usedPercent);
}

static void printValues(const char* name, const std::vector<float>& values) {
  std::printf("%-10s =", name);
  for (float value : values) {
    std::printf("%6.2f", value);
  }
  std::printf("\n");
}

int main() {
  const std::vector<Particle> particles = makeParticles();
  const std::vector<float> fromAoS = runAoS(particles);
  const std::vector<float> fromSoA = runSoA(particles);

  std::vector<float> x(PARTICLE_COUNT);
  std::vector<float> expected(PARTICLE_COUNT);
  for (int i = 0; i < PARTICLE_COUNT; ++i) {
    x[i] = particles[i].x;
    expected[i] = particles[i].x * 2.0f;
  }

  printValues("p.x", x);
  printValues("AoS out", fromAoS);
  printValues("SoA out", fromSoA);
  printWarpTraffic("AoS", static_cast<int>(sizeof(Particle)));
  printWarpTraffic("SoA", static_cast<int>(sizeof(float)));

  const bool pass = fromAoS == expected && fromSoA == expected;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
