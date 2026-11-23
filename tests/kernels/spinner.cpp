// A co-runner for load tests: N threads spinning at default QoS until killed.   spinner [threads]
#include <cstdlib>
#include <thread>
#include <vector>
int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 2;
  std::vector<std::thread> th;
  for (int i = 0; i < n; i++)
    th.emplace_back([] {
      volatile unsigned long x = 0;
      for (;;) x = x + 1;
    });
  for (auto& t : th) t.join();
}
