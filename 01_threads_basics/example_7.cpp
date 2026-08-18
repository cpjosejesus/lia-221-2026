// detach() severs the handle: the worker becomes a daemon nobody waits for.
// Whether its output appears at all is a race against main returning.
#include <iostream>
#include <thread>

void publishTick() {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "worker: tick published\n";
}

int main() {
  std::thread publisher(publishTick);

  publisher.detach();  // we can never join() this thread again

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "main: done\n";

  return 0;
}
