// Load balancing: static vs dynamic chunking. When work per element is uneven
// (some vehicles need a long inspection, most a short one), equal-size static chunks
// leave some tasks idle while one task does all the heavy elements. Dynamic
// scheduling hands out small chunks from an atomic counter: whoever is free takes the next.
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct Vehicle { int id; int workUnits; };   // uneven: a few vehicles are 200x more work

double inspect(const Vehicle& v) {
  double x = 1.0;
  for (int i = 0; i < v.workUnits * 50; ++i) x = std::sqrt(x + 1.0);
  return x;
}

double staticChunks(const std::vector<Vehicle>& fleet, int nTasks) {
  std::vector<std::future<double>> parts;
  size_t chunk = (fleet.size() + nTasks - 1) / nTasks;
  for (int t = 0; t < nTasks; ++t) {
    size_t b = t * chunk, e = std::min(fleet.size(), b + chunk);
    parts.emplace_back(std::async(std::launch::async, [&, b, e] { double s = 0; for (size_t i = b; i < e; ++i) s += inspect(fleet[i]); return s; }));
  }
  double s = 0; for (auto& p : parts) s += p.get(); return s;
}

double dynamicChunks(const std::vector<Vehicle>& fleet, int nTasks, size_t grain) {
  std::atomic<size_t> next{0};                                    // the shared "next chunk" counter
  std::vector<std::future<double>> parts;
  for (int t = 0; t < nTasks; ++t) {
    parts.emplace_back(std::async(std::launch::async, [&] {
      double s = 0;
      while (true) {
        size_t b = next.fetch_add(grain);
        if (b >= fleet.size()) break;
        size_t e = std::min(fleet.size(), b + grain);
        for (size_t i = b; i < e; ++i) s += inspect(fleet[i]);
      }
      return s;
    }));
  }
  double s = 0; for (auto& p : parts) s += p.get(); return s;
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 200'000; ++i) fleet.push_back({i, i < 20'000 ? 200 : 1});   // the heavy ones are all at the front
  int n = std::thread::hardware_concurrency();

  auto t0 = std::chrono::steady_clock::now();
  double s1 = staticChunks(fleet, n);
  double staticMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "static  chunks, " << n << " tasks: " << staticMs << " ms (sum " << s1 << ")\n";

  for (size_t grain : {1000ul, 100ul, 10ul, 1ul}) {
    t0 = std::chrono::steady_clock::now();
    double s2 = dynamicChunks(fleet, n, grain);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "dynamic chunks, grain " << grain << ": " << ms << " ms (sum " << s2 << ")\n";
  }
  std::cout << "\nStatic: task 0 gets all 20 000 heavy vehicles and finishes last while the others idle.\n"
               "Dynamic: the heavy vehicles spread over whoever is free. Too small a grain and the atomic counter becomes the bottleneck.\n";
  return 0;
}
