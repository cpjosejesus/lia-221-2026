// A thread pool: the folder-07 monitor queue (mutex + condition variable), N worker
// threads created once, and std::packaged_task so every submitted job returns a future.
// Creating a thread per job (std::async) costs ~50-100 us each; a pool reuses them.
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(size_t nWorkers) {
    for (size_t i = 0; i < nWorkers; ++i)
      _workers.emplace_back([this] { workerLoop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _stopping = true;                 // shutdown flag, checked inside the predicate (folder 07 lesson)
    }
    _cond.notify_all();
    for (auto& w : _workers) w.join();  // every thread gets a deliberate fate
  }

  // Submit any callable; get a future for its result.
  template <class Fn>
  auto submit(Fn fn) -> std::future<decltype(fn())> {
    using R = decltype(fn());
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(fn));  // packaged_task is move-only; share it with the queue lambda
    std::future<R> result = task->get_future();
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _jobs.push([task] { (*task)(); });
    }
    _cond.notify_one();
    return result;
  }

 private:
  void workerLoop() {
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _cond.wait(lock, [this] { return _stopping || !_jobs.empty(); });
        if (_stopping && _jobs.empty()) return;
        job = std::move(_jobs.front());
        _jobs.pop();
      }
      job();                            // run OUTSIDE the lock
    }
  }

  std::vector<std::thread> _workers;
  std::queue<std::function<void()>> _jobs;
  std::mutex _mutex;
  std::condition_variable _cond;
  bool _stopping = false;
};

struct Vehicle { int id; double km; };

double inspect(const Vehicle& v) {
  double x = v.km;
  for (int i = 0; i < 2000; ++i) x = std::sqrt(x + 1);
  return x;
}

int main() {
  std::vector<Vehicle> fleet;
  for (int i = 0; i < 20'000; ++i) fleet.push_back({i, 1.0 + i % 50});

  auto t0 = std::chrono::steady_clock::now();
  {
    std::vector<std::future<double>> results;
    for (auto& v : fleet) results.push_back(std::async(std::launch::async, inspect, v));   // one thread per job
    double sum = 0; for (auto& r : results) sum += r.get();
    std::cout << "std::async, one thread per job: sum " << sum;
  }
  std::cout << " in " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << " ms\n";

  t0 = std::chrono::steady_clock::now();
  {
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<double>> results;
    for (auto& v : fleet) results.push_back(pool.submit([v] { return inspect(v); }));
    double sum = 0; for (auto& r : results) sum += r.get();
    std::cout << "thread pool, " << std::thread::hardware_concurrency() << " workers: sum " << sum;
  }   // pool destructor: stop flag, notify_all, join
  std::cout << " in " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << " ms\n";
  return 0;
}
