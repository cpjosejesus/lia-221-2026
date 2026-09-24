// Data parallelism on a grid: a box blur over a traffic-density image of the city.
// Every output pixel is the mean of a (2r+1)x(2r+1) window of the input, and every
// output pixel is independent of the others: the image is split into horizontal
// strips, one task per strip. Reading the input is shared; writing is disjoint.
#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct Image {
  int w, h;
  std::vector<float> px;                       // row-major: px[y * w + x]
  float& at(int x, int y) { return px[y * w + x]; }
  float at(int x, int y) const { return px[y * w + x]; }
};

float boxBlurKernel(const Image& src, int x, int y, int r) {
  float sum = 0; int count = 0;
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx) {
      int xx = std::clamp(x + dx, 0, src.w - 1), yy = std::clamp(y + dy, 0, src.h - 1);
      sum += src.at(xx, yy); ++count;
    }
  return sum / count;
}

void blurRows(const Image& src, Image& dst, int r, int fromRow, int toRow) {
  for (int y = fromRow; y < toRow; ++y)
    for (int x = 0; x < src.w; ++x) dst.at(x, y) = boxBlurKernel(src, x, y, r);
}

void blurParallel(const Image& src, Image& dst, int r, int nTasks) {
  int rowsPerTask = (src.h + nTasks - 1) / nTasks;
  std::vector<std::future<void>> tasks;
  for (int t = 0; t < nTasks; ++t) {
    int from = t * rowsPerTask, to = std::min(src.h, from + rowsPerTask);
    if (from >= to) break;
    tasks.emplace_back(std::async(std::launch::async, blurRows, std::cref(src), std::ref(dst), r, from, to));
  }
  for (auto& t : tasks) t.get();
}

int main() {
  Image density{1600, 1200, {}};
  density.px.resize(density.w * density.h);
  for (int y = 0; y < density.h; ++y)                       // a few busy intersections: bright spots
    for (int x = 0; x < density.w; ++x) density.at(x, y) = ((x / 200 + y / 200) % 2) ? 1.0f : 0.0f;
  Image out{density.w, density.h, std::vector<float>(density.px.size())};
  const int radius = 4;

  for (int n : {1, 2, 4, (int)std::thread::hardware_concurrency()}) {
    auto t0 = std::chrono::steady_clock::now();
    blurParallel(density, out, radius, n);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << n << " strip(s): " << ms << " ms, sample out(800,600) = " << out.at(800, 600) << "\n";
  }
  std::cout << "\nWhy strips and not single pixels: a task per pixel is ~2M tasks; the work per task must dwarf the cost of a task.\n"
               "Why the input can be shared without a lock: nobody writes it while the blur runs.\n";
  return 0;
}
