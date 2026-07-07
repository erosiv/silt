#ifndef SILT_MEMORY
#define SILT_MEMORY

#include <silt/silt.hpp>

// This file contains an interface for cuda runtime memory 
// management functions to simplify linkage of dependencies.

namespace silt {

//! copy_t mirrors the cudaMemcpyKind enumerator
enum copy_t {
  HOST_TO_HOST = 0,
  HOST_TO_DEVICE = 1,
  DEVICE_TO_HOST = 2,
  DEVICE_TO_DEVICE = 3,
  DEFAULT = 4
};

// Runtime Memory Interface
 
EXPORT_SHARED void* device_alloc(size_t bytes);
EXPORT_SHARED void  device_free(void* ptr);
EXPORT_SHARED void  device_copy(void* dst, const void* src, size_t bytes, copy_t copy);

}

#endif