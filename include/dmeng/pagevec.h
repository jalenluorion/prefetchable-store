// Large buffers taken straight from the VM system, given back to it when freed. The system allocator keeps a freed large
// block for reuse and it stays in the process's footprint (measured on macOS: 420 MB of vectors freed, the footprint
// unchanged, malloc_zone_pressure_relief not returning them; a vector grown in steps leaves every old buffer charged), so
// a prompt's scratch or a dropped checkpoint would stay charged to the process after its release. Blocks under 1 MB go
// through malloc as before.
#pragma once
#include <sys/mman.h>

#include <cstdlib>
#include <new>
#include <vector>

namespace dm {

template <class T>
struct PageAlloc {
  using value_type = T;
  static constexpr size_t MIN = size_t(1) << 20;
  PageAlloc() = default;
  template <class U>
  PageAlloc(const PageAlloc<U>&) {}
  T* allocate(size_t n) {
    const size_t b = n * sizeof(T);
    if (b < MIN) {
      void* p = std::malloc(b ? b : 1);
      if (!p) throw std::bad_alloc();
      return (T*)p;
    }
    void* p = mmap(nullptr, b, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    return (T*)p;
  }
  void deallocate(T* p, size_t n) {
    const size_t b = n * sizeof(T);
    if (b < MIN) std::free(p);
    else munmap(p, b);
  }
  template <class U>
  bool operator==(const PageAlloc<U>&) const { return true; }
  template <class U>
  bool operator!=(const PageAlloc<U>&) const { return false; }
};

template <class T>
using PageVec = std::vector<T, PageAlloc<T>>;

}  // namespace dm
