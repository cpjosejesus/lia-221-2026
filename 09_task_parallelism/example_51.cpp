// Speedup table and Amdahl's law. A job has a serial part (loading the fleet: one
// thread, cannot be split) and a parallel part (processing the vehicles). We measure
// the speedup for 1, 2, 4, 8, 16 tasks and compare it with Amdahl's prediction.
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct Vehicle { int id; double load; };

double process(const Vehicle& v) {          // the parallel part: independent per vehicle
  double x = v.load;
  for (int i = 0; i < 300; ++i) x = std::sqrt(x + 1.0);
  return x;
}

std::vector<Vehicle> loadFleet(int n) {     // the serial part: one thread, sequential by nature
  std::vector<Vehicle> fleet;
  fleet.reserve(n);
  double seed = 1.0;
  for (int i = 0; i < n; ++i) {
    seed = std::fmod(seed * 1.000001 + 0.5, 100.0);   // each value depends on the previous one
    fleet.push_back({i, seed});
  }
  return fleet;
}

double job(int nTasks, double& serialMs, double& parallelMs) {
  auto t0 = std::chrono::steady_clock::now();
  std::vector<Vehicle> fleet = loadFleet(3'000'000);
  auto t1 = std::chrono::steady_clock::now();

  std::vector<std::future<double>> parts;
  size_t chunk = (fleet.size() + nTasks - 1) / nTasks;
  for (int t = 0; t < nTasks; ++t) {
    size_t b = t * chunk, e = std::min(fleet.size(), b + chunk);
    parts.emplace_back(std::async(std::launch::async, [&fleet, b, e] {
      double s = 0; for (size_t i = b; i < e; ++i) s += process(fleet[i]); return s;
    }));
  }
  double total = 0;
  for (auto& p : parts) total += p.get();
  auto t2 = std::chrono::steady_clock::now();

  serialMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  parallelMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
  return total;
}

int main() {
  std::cout << "hardware_concurrency = " << std::thread::hardware_concurrency() << "\n\n";
  double s1, p1;
  job(1, s1, p1);
  double T1 = s1 + p1, f = s1 / T1;   // f = serial fraction
  std::cout << "1 task: serial " << s1 << " ms + parallel " << p1 << " ms = " << T1 << " ms, serial fraction f = " << f << "\n\n";
  std::cout << "tasks   time(ms)   speedup   Amdahl: 1/(f + (1-f)/n)\n";
  for (int n : {1, 2, 4, 8, 16}) {
    double s, p;
    job(n, s, p);
    double T = s + p;
    std::cout << n << "\t" << (int)T << "\t   " << T1 / T << "\t     " << 1.0 / (f + (1 - f) / n) << "\n";
  }
  std::cout << "\nNo matter how many tasks: speedup <= 1/f = " << 1 / f << "x\n";
  return 0;
}
