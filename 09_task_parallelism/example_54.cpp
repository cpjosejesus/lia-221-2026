// Parallel prefix sum (scan). Input: vehicles entering each street segment per minute.
// Output: cumulative vehicles up to each minute. Sequential scan is inherently serial
// (each value depends on the previous one), yet it parallelizes in three phases.
#include <chrono>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// Phase 1: each task scans its own chunk and reports the chunk total.
// Phase 2: (sequential, tiny) prefix-sum the chunk totals -> offset of every chunk.
// Phase 3: each task adds its offset to its chunk.
std::vector<long> parallelScan(const std::vector<int>& in, int nTasks) {
  std::vector<long> out(in.size());
  size_t chunk = (in.size() + nTasks - 1) / nTasks;
  std::vector<long> chunkTotal(nTasks, 0);

  std::vector<std::future<void>> phase1;
  for (int t = 0; t < nTasks; ++t) {
    size_t b = t * chunk, e = std::min(in.size(), b + chunk);
    phase1.emplace_back(std::async(std::launch::async, [&, t, b, e] {
      long run = 0;
      for (size_t i = b; i < e; ++i) { run += in[i]; out[i] = run; }
      chunkTotal[t] = run;                         // distinct slot per task
    }));
  }
  for (auto& f : phase1) f.get();

  std::vector<long> offset(nTasks, 0);             // phase 2: exclusive scan of the totals
  for (int t = 1; t < nTasks; ++t) offset[t] = offset[t - 1] + chunkTotal[t - 1];

  std::vector<std::future<void>> phase3;
  for (int t = 1; t < nTasks; ++t) {              // chunk 0 needs no offset
    size_t b = t * chunk, e = std::min(in.size(), b + chunk);
    phase3.emplace_back(std::async(std::launch::async, [&, t, b, e] {
      for (size_t i = b; i < e; ++i) out[i] += offset[t];
    }));
  }
  for (auto& f : phase3) f.get();
  return out;
}

int main() {
  std::vector<int> arrivalsPerMinute(20'000'000);
  for (size_t i = 0; i < arrivalsPerMinute.size(); ++i) arrivalsPerMinute[i] = (int)(i % 5);

  auto t0 = std::chrono::steady_clock::now();
  std::vector<long> seq(arrivalsPerMinute.size());
  std::partial_sum(arrivalsPerMinute.begin(), arrivalsPerMinute.end(), seq.begin());
  double seqMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  int n = std::thread::hardware_concurrency();
  t0 = std::chrono::steady_clock::now();
  std::vector<long> par = parallelScan(arrivalsPerMinute, n);
  double parMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::cout << "cumulative vehicles after minute 10: " << par[10] << ", after last minute: " << par.back() << "\n";
  std::cout << "sequential scan " << seqMs << " ms, parallel scan (" << n << " tasks, 3 phases) " << parMs << " ms\n";
  std::cout << (seq == par ? "results identical" : "MISMATCH") << "\n";
  std::cout << "Note: the parallel version does ~2x the work (two passes). It pays off only with enough cores and data.\n";
  return 0;
}
