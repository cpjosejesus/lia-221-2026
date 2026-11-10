// Coalescing has a CPU twin: memory is moved in whole cache lines (64 B), just as DRAM delivers
// whole bursts to a GPU. Reading consecutive elements uses every byte of each line; reading with
// a stride, or reading one field of an array of structures, pays for bytes that are thrown away.
// Part 1 sums the same 128 MB in a different order each time: consecutive (stride 1) or jumping
// `stride` values ahead and coming back for the rest in later passes. Part 2 compares an array of
// structures with a structure of arrays.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

volatile uint64_t sink;  // every result is stored here, so no loop can be removed

template <class F>
double ms(F f) {  // best of 5 runs, in milliseconds
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    sink = f();
    double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (t < best) best = t;
  }
  return best;
}

struct Particle {
  uint32_t x, y, z, m;
};  // 16 B: an array of these is an array of structures

int main() {
  const size_t n = size_t(1) << 25;  // 32 Mi values of 4 B = 128 MB, far larger than the caches
  std::vector<uint32_t> a(n, 1);     // integers: the compiler can vectorise the sum freely
  ms([&] { uint64_t w = 0; for (size_t i = 0; i < n; ++i) w += a[i]; return w; });                       // warm-up

  std::printf("part 1: sum all %zu values, visited in stride order (16 values = one 64 B line)\n", n);
  std::printf("%8s %12s %16s\n", "stride", "time (ms)", "vs stride 1");
  double t1 = 0;
  for (size_t stride : {1, 2, 4, 8, 16, 32, 64}) {
    double t = ms([&] {
      uint32_t s = 0;
      for (size_t off = 0; off < stride; ++off)              // `stride` passes over the array ...
        for (size_t i = off; i < n; i += stride) s += a[i];  // ... each reading every stride-th value
      return s;
    });
    if (stride == 1) t1 = t;
    std::printf("%8zu %12.1f %15.1fx\n", stride, t, t / t1);
  }

  const size_t p = size_t(1) << 23;  // 8 Mi particles
  std::vector<Particle> aos(p, Particle{1, 2, 3, 4});
  std::vector<uint32_t> x(p, 1), y(p, 2), z(p, 3), m(p, 4);  // structure of arrays
  // same arithmetic in both loops: only the layout differs
  double tAos = ms([&] { uint32_t s = 0; for (size_t i = 0; i < p; ++i) s += aos[i].x; return s; });
  double tSoa = ms([&] { uint32_t s = 0; for (size_t i = 0; i < p; ++i) s += x[i]; return s; });
  std::printf("\npart 2: sum of x over %zu particles\n", p);
  std::printf("array of structures: %7.1f ms  (reads 16 B to use 4 B)\n", tAos);
  std::printf("structure of arrays: %7.1f ms  (every byte read is used)\n", tSoa);
  std::printf("ratio: %.1f x\n", tAos / tSoa);
}
