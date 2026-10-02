#include <silt/silt.hpp>
#include <cuda_runtime.h>
#include <silt/core/error.hpp>
#include <silt/core/memory.hpp>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>

namespace silt {

namespace {

//! Allocation ledger; never destroyed, so frees during static teardown stay valid.
struct ledger_t {
  struct entry_t {
    size_t bytes;
    bool gpu;
  };
  std::mutex mutex;
  std::unordered_map<const void*, entry_t> live;
  memory_stats stats;
};

ledger_t& ledger() {
  static ledger_t* instance = [] {
    auto* l = new ledger_t();
    l->stats.id = reinterpret_cast<size_t>(l);
    return l;
  }();
  return *instance;
}

void record_alloc(const void* ptr, const size_t bytes, const bool gpu) {
  if (ptr == nullptr)
    return;
  ledger_t& l = ledger();
  std::lock_guard<std::mutex> lock(l.mutex);
  l.live[ptr] = {bytes, gpu};
  size_t& live = gpu ? l.stats.gpu_bytes : l.stats.cpu_bytes;
  size_t& peak = gpu ? l.stats.gpu_peak : l.stats.cpu_peak;
  (gpu ? l.stats.gpu_allocations : l.stats.cpu_allocations)++;
  live += bytes;
  peak = live > peak ? live : peak;
}

// Pointers not in the ledger (allocated elsewhere) are ignored.
void record_free(const void* ptr) {
  if (ptr == nullptr)
    return;
  ledger_t& l = ledger();
  std::lock_guard<std::mutex> lock(l.mutex);
  const auto it = l.live.find(ptr);
  if (it == l.live.end())
    return;
  const bool gpu = it->second.gpu;
  (gpu ? l.stats.gpu_bytes : l.stats.cpu_bytes) -= it->second.bytes;
  (gpu ? l.stats.gpu_allocations : l.stats.cpu_allocations)--;
  l.live.erase(it);
}

} // namespace

void* host_alloc(size_t bytes) {
  void* p = std::malloc(bytes);
  if (p == nullptr)
    throw std::bad_alloc();
  record_alloc(p, bytes, false);
  return p;
}

void host_free(void* p) {
  record_free(p);
  std::free(p);
}

void* device_alloc(size_t bytes) {
  void* p;
  gpuErrchk(cudaMalloc(&p, bytes));
  record_alloc(p, bytes, true);
  return p;
}

void device_free(void* p) {
  record_free(p);
  gpuErrchk(cudaFree(p));
}

memory_stats memory_usage() {
  ledger_t& l = ledger();
  std::lock_guard<std::mutex> lock(l.mutex);
  return l.stats;
}

void memory_reset_peak() {
  ledger_t& l = ledger();
  std::lock_guard<std::mutex> lock(l.mutex);
  l.stats.cpu_peak = l.stats.cpu_bytes;
  l.stats.gpu_peak = l.stats.gpu_bytes;
}

void device_memory_info(size_t& free, size_t& total) {
  gpuErrchk(cudaMemGetInfo(&free, &total));
}

void device_copy(void* d, const void* s, size_t n, copy_t k) {
  if (n == 0)
    return;
  if (k == copy_t::HOST_TO_HOST) {
    std::memcpy(d, s, n);
    return;
  }
  gpuErrchk(cudaMemcpy(d, s, n, cudaMemcpyKind(k)));
}

void synchronize() {
  gpuErrchk(cudaDeviceSynchronize());
}

} // namespace silt
