// Parallel map: compute an ETA for every vehicle into an output vector.
// Each task writes only its own range of the output, so there is no data race
// and no lock. Output layout decides whether parallel map is safe.
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct Vehicle { int id; double distanceKm; double speedKmh; };

double etaMinutes(const Vehicle& v) {
  double eta = v.distanceKm / v.speedKmh * 60.0;
  for (int i = 0; i < 150; ++i) eta = std::sqrt(eta * eta + 1e-6);
  return eta;
}

template <class In, class Out, class Fn>
void parallelMap(const std::vector<In>& in, std::vector<Out>& out, Fn fn, int nTasks) {
  out.resize(in.size());                     // allocate ONCE, before the tasks start: no push_back races
  size_t chunk = (in.size() + nTasks - 1) / nTasks;
  std::vector<std::future<void>> tasks;
  for (int t = 0; t < nTasks; ++t) {
    size_t b = t * chunk, e = std::min(in.size(), b + chunk);
    tasks.emplace_back(std::async(std::launch::async, [&, b, e] {
      for (size_t i = b; i < e; ++i) out[i] = fn(in[i]);   // disjoint ranges: out[b..e) belongs to this task
    }));
  }
  for (auto& t : tasks) t.get();
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 500'000; ++i) fleet.push_back({i, 1.0 + i % 30, 20.0 + i % 40});
  std::vector<double> etas;

  for (int n : {1, 2, 4, (int)std::thread::hardware_concurrency()}) {
    auto t0 = std::chrono::steady_clock::now();
    parallelMap(fleet, etas, etaMinutes, n);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n << " task(s): " << ms << " ms, eta[0] = " << etas[0] << " min, eta[last] = " << etas.back() << " min\n";
  }
  std::cout << "\nWhy no lock is needed: task t writes out[b_t..e_t) and nobody else touches those slots.\n"
               "Why push_back would break it: two tasks growing the same vector is a data race.\n";
  return 0;
}
