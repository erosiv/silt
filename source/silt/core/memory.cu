#ifndef SILT_MEMORY_CU
#define SILT_MEMORY_CU

#include <silt/silt.hpp>
#include <silt/core/memory.hpp>
#include <cuda_runtime.h>

namespace silt {

void* device_alloc(size_t bytes) { 
  void* p;
  cudaMalloc(&p, bytes);
  return p;
}

void device_free(void* p) { 
  cudaFree(p);
}

void device_copy(void* d, const void* s, size_t n, copy_t k) {
  cudaMemcpy(d, s, n, cudaMemcpyKind(k));
}

}

#endif