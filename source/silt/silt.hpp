#pragma once

#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

//
// Macro Definitions
//

// Suppress Unnecessary Warnings

#if defined(HAS_CUDA) && defined(__CUDACC__)
#pragma nv_diag_suppress 177    // variable declared but never referenced (unused template params)
#pragma nv_diag_suppress 445    // constant not used in declaring parameter types (used only for template dispatch, e.g. []<host_t S>())
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20012  // __host__ ignored on defaulted special member (fires from glm's mat headers)
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

// Enable Host+Device Code Macro

#ifndef GPU_ENABLE
#define GPU_ENABLE
#ifdef HAS_CUDA
#undef GPU_ENABLE
#define GPU_ENABLE __host__ __device__
#endif
#endif

// Exported Symbols from the Shared DLL Macro

#if defined(_WIN32)
#if defined(SILT_SHARED_BUILD)
#define EXPORT_SHARED __declspec(dllexport)
#else
#define EXPORT_SHARED __declspec(dllimport)
#endif
#else
#if defined(SILT_SHARED_BUILD)
#define EXPORT_SHARED __attribute__((visibility("default")))
#else
#define EXPORT_SHARED
#endif
#endif

namespace silt {

}; // namespace silt
