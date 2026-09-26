#include <silt/silt.hpp>
#include <cuda_runtime.h>
#include <silt/core/error.hpp>
#include <silt/core/memory.hpp>

namespace silt {

void* device_alloc(size_t bytes) {
  void* p;
  gpuErrchk(cudaMalloc(&p, bytes));
  return p;
}

void device_free(void* p) {
  gpuErrchk(cudaFree(p));
}

void device_copy(void* d, const void* s, size_t n, copy_t k) {
  gpuErrchk(cudaMemcpy(d, s, n, cudaMemcpyKind(k)));
}

void synchronize() {
  gpuErrchk(cudaDeviceSynchronize());
}

} // namespace silt
