#pragma once

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

//! Bytes of tensor storage currently held through silt, per host.
//!
//! The ledger lives inside silt_lib and is reached only through exported
//! functions, so every consumer that links the same silt_lib shares one count.
//! Only allocations made via host_alloc / device_alloc are counted; library
//! internals (e.g. thrust's temporary buffers) and the CUDA context are not.
struct memory_stats {
  size_t cpu_bytes = 0;       //!< Live CPU bytes
  size_t gpu_bytes = 0;       //!< Live GPU bytes
  size_t cpu_peak = 0;        //!< High-water mark of cpu_bytes since start or memory_reset_peak
  size_t gpu_peak = 0;        //!< High-water mark of gpu_bytes since start or memory_reset_peak
  size_t cpu_allocations = 0; //!< Live CPU allocations
  size_t gpu_allocations = 0; //!< Live GPU allocations
  size_t id = 0;              //!< Identifies the ledger; differs between separate silt_lib copies
};

// Runtime Memory Interface

EXPORT_SHARED void* host_alloc(size_t bytes);
EXPORT_SHARED void host_free(void* ptr);
EXPORT_SHARED void* device_alloc(size_t bytes);
EXPORT_SHARED void device_free(void* ptr);

//! Snapshot of the allocation ledger.
EXPORT_SHARED memory_stats memory_usage();

//! Reset the peak counters to the current usage.
EXPORT_SHARED void memory_reset_peak();

//! Free and total GPU memory of the current device, as reported by the driver.
//! Unlike memory_usage, this covers every allocation on the device.
EXPORT_SHARED void device_memory_info(size_t& free, size_t& total);
EXPORT_SHARED void device_copy(void* dst, const void* src, size_t bytes, copy_t copy);

//! Block until all queued GPU work completes, and raise silt::error::cuda_error
//! if any of it (including a kernel launched earlier) failed.
EXPORT_SHARED void synchronize();

} // namespace silt
