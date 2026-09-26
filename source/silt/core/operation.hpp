#pragma once

#include <silt/silt.hpp>
#include <silt/core/error.hpp>
#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>

// curand_kernel.h relies on silt.hpp's diagnostics-suppression state,
// which only applies to code parsed after it -- see types.hpp.
#include <curand_kernel.h>

// This file contains generic template operations for tensors.
// These are written to function both on the GPU and the CPU,
// and are intended to simplify the code structure for common
// operation types between tensors.

namespace silt {

namespace {

// elem is int64_t (a tensor can exceed 2^31 elements); the block count
// itself stays well within range for any realistic launch.
inline int64_t block(const int64_t elem, const int thread) {
  return (elem + thread - 1) / thread;
}

} // namespace

namespace op {

// Kernels behind the public entry points below. Not part of the public
// API surface: a double-underscore name is reserved to the implementation
// at any scope per [lex.name], namespace or not, so these are named plainly
// and kept in `detail` instead, rather than just being renamed in place.
namespace detail {

// Templated In-Place Unary Operations

template<typename T, typename F>
__global__ void uniop_inplace_gpu(T lhs, F func) {
  // 64-bit index: blockIdx.x * blockDim.x alone can already exceed
  // UINT32_MAX for a large enough grid, wrapping silently in 32-bit
  // arithmetic.
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < lhs.elem()) {
    lhs[n] = func(lhs[n]);
  }
}

template<typename T, typename F>
__host__ void uniop_inplace_cpu(T lhs, F func) {
  for (size_t n = 0; n < lhs.elem(); ++n) {
    lhs[n] = func(lhs[n]);
  }
}

// Templated In-Place Binary Operations

template<typename T, typename F>
__global__ void binop_inplace_gpu(T lhs, const T rhs, F func) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < lhs.elem()) {
    lhs[n] = func(lhs[n], rhs[n]);
  }
}

template<typename T, typename F>
__host__ void binop_inplace_cpu(T lhs, const T rhs, F func) {
  for (size_t n = 0; n < lhs.elem(); ++n) {
    lhs[n] = func(lhs[n], rhs[n]);
  }
}

// In-Place Indexed Unary Operation

template<typename T, typename F>
__global__ void uniop_inplace_indexed_gpu(tensor_t<T> lhs, const tensor_t<int> ind, F func) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < ind.elem()) {
    // ind stores plain `int` indices -- unrelated to this widening pass;
    // deferred to the generic-indexed-operations redesign (see B5).
    const int i = ind[n];
    if (i < lhs.elem()) {
      lhs[i] = func(lhs[i]);
    }
  }
}

template<typename T, typename F>
__host__ void uniop_inplace_indexed_cpu(tensor_t<T> lhs, const tensor_t<int> ind, F func) {
  for (size_t n = 0; n < ind.elem(); ++n) {
    const int i = ind[n];
    if (i < lhs.elem()) {
      lhs[i] = func(lhs[i]);
    }
  }
}

} // namespace detail

template<typename T, typename F>
void uniop_inplace(T lhs, F func) {

  if (lhs.host() == silt::host_t::CPU) {
    detail::uniop_inplace_cpu(lhs, func);
  }

  else if (lhs.host() == silt::host_t::GPU) {
    detail::uniop_inplace_gpu<<<block(lhs.elem(), 512), 512>>>(lhs, func);
    gpuErrchk(cudaGetLastError());
  }
}

template<typename T, typename F>
void binop_inplace(T lhs, const T rhs, F func) {

  if (lhs.host() != rhs.host())
    throw silt::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.elem() != rhs.elem())
    throw silt::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() == silt::host_t::CPU) {
    detail::binop_inplace_cpu(lhs, rhs, func);
  }

  else if (lhs.host() == silt::host_t::GPU) {
    detail::binop_inplace_gpu<<<block(lhs.elem(), 512), 512>>>(lhs, rhs, func);
    gpuErrchk(cudaGetLastError());
  }
}

template<typename T, typename F>
void uniop_inplace_indexed(tensor_t<T> lhs, const tensor_t<int> ind, F func) {

  if (lhs.host() == silt::host_t::CPU) {
    detail::uniop_inplace_indexed_cpu(lhs, ind, func);
  }

  else if (lhs.host() == silt::host_t::GPU) {
    detail::uniop_inplace_indexed_gpu<<<block(ind.elem(), 512), 512>>>(lhs, ind, func);
    gpuErrchk(cudaGetLastError());
  }
}

} // namespace op
} // namespace silt
