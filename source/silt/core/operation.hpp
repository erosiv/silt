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
//
// `ind` holds the flat tensor indices to visit.

template<typename T, typename F>
__global__ void indexed_apply_gpu(tensor_t<T> lhs, const tensor_t<int64_t> ind, F func) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < ind.elem()) {
    const int64_t i = ind[n];
    if (i >= 0 && i < (int64_t)lhs.elem()) {
      lhs[i] = func(lhs[i]);
    }
  }
}

template<typename T, typename F>
__host__ void indexed_apply_cpu(tensor_t<T> lhs, const tensor_t<int64_t> ind, F func) {
  for (int64_t n = 0; n < (int64_t)ind.elem(); ++n) {
    const int64_t i = ind[n];
    if (i >= 0 && i < (int64_t)lhs.elem()) {
      lhs[i] = func(lhs[i]);
    }
  }
}

// In-Place Indexed Binary Operation
//
// rhs is read at the same flat index i as lhs.

template<typename T, typename F>
__global__ void indexed_binop_apply_gpu(tensor_t<T> lhs, const tensor_t<T> rhs, const tensor_t<int64_t> ind, F func) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < ind.elem()) {
    const int64_t i = ind[n];
    if (i >= 0 && i < (int64_t)lhs.elem()) {
      lhs[i] = func(lhs[i], rhs[i]);
    }
  }
}

template<typename T, typename F>
__host__ void indexed_binop_apply_cpu(tensor_t<T> lhs, const tensor_t<T> rhs, const tensor_t<int64_t> ind, F func) {
  for (int64_t n = 0; n < (int64_t)ind.elem(); ++n) {
    const int64_t i = ind[n];
    if (i >= 0 && i < (int64_t)lhs.elem()) {
      lhs[i] = func(lhs[i], rhs[i]);
    }
  }
}

// Gather: out[n] = src[ind[n]]
//
// `C` is any container with a flat subscript (tensor_t, view_t), so `ind`
// addresses its logical linear space. Out-of-range indices gather T(0).

template<typename C, typename T>
__global__ void gather_gpu(tensor_t<T> out, const C src, const tensor_t<int64_t> ind) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < (int64_t)ind.elem()) {
    const int64_t i = ind[n];
    out[n] = (i >= 0 && i < (int64_t)src.elem()) ? src[i] : T(0);
  }
}

template<typename C, typename T>
__host__ void gather_cpu(tensor_t<T> out, const C src, const tensor_t<int64_t> ind) {
  for (int64_t n = 0; n < (int64_t)ind.elem(); ++n) {
    const int64_t i = ind[n];
    out[n] = (i >= 0 && i < (int64_t)src.elem()) ? src[i] : T(0);
  }
}

// Scatter: dst[ind[n]] = src[n]
//
// Out-of-range indices are skipped. Duplicate indices race.

template<typename C, typename T>
__global__ void scatter_gpu(C dst, const tensor_t<T> src, const tensor_t<int64_t> ind) {
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n < (int64_t)ind.elem()) {
    const int64_t i = ind[n];
    if (i >= 0 && i < (int64_t)dst.elem()) {
      dst[i] = src[n];
    }
  }
}

template<typename C, typename T>
__host__ void scatter_cpu(C dst, const tensor_t<T> src, const tensor_t<int64_t> ind) {
  for (int64_t n = 0; n < (int64_t)ind.elem(); ++n) {
    const int64_t i = ind[n];
    if (i >= 0 && i < (int64_t)dst.elem()) {
      dst[i] = src[n];
    }
  }
}

} // namespace detail

template<typename C>
tensor_t<typename C::val_t> gather(const C src, const tensor_t<int64_t> ind) {

  using T = typename C::val_t;

  if (src.host() != ind.host())
    throw silt::error::mismatch_host(src.host(), ind.host());

  tensor_t<T> out(silt::shape((int)ind.elem()), src.host());
  if (ind.elem() == 0)
    return out;

  if (src.host() == silt::host_t::CPU) {
    detail::gather_cpu(out, src, ind);
  }

  else if (src.host() == silt::host_t::GPU) {
    detail::gather_gpu<<<block(ind.elem(), 512), 512>>>(out, src, ind);
    gpuErrchk(cudaGetLastError());
  }

  return out;
}

template<typename C>
void scatter(C dst, const tensor_t<typename C::val_t> src, const tensor_t<int64_t> ind) {

  if (dst.host() != src.host())
    throw silt::error::mismatch_host(dst.host(), src.host());

  if (dst.host() != ind.host())
    throw silt::error::mismatch_host(dst.host(), ind.host());

  if (src.elem() != ind.elem())
    throw silt::error::mismatch_size(ind.elem(), src.elem());

  if (ind.elem() == 0)
    return;

  if (dst.host() == silt::host_t::CPU) {
    detail::scatter_cpu(dst, src, ind);
  }

  else if (dst.host() == silt::host_t::GPU) {
    detail::scatter_gpu<<<block(ind.elem(), 512), 512>>>(dst, src, ind);
    gpuErrchk(cudaGetLastError());
  }
}

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
void indexed_apply(tensor_t<T> lhs, const tensor_t<int64_t> ind, F func) {

  if (lhs.host() != ind.host())
    throw silt::error::mismatch_host(lhs.host(), ind.host());

  if (lhs.host() == silt::host_t::CPU) {
    detail::indexed_apply_cpu(lhs, ind, func);
  }

  else if (lhs.host() == silt::host_t::GPU) {
    detail::indexed_apply_gpu<<<block(ind.elem(), 512), 512>>>(lhs, ind, func);
    gpuErrchk(cudaGetLastError());
  }
}

template<typename T, typename F>
void indexed_binop_apply(tensor_t<T> lhs, const tensor_t<T> rhs, const tensor_t<int64_t> ind, F func) {

  if (lhs.host() != rhs.host())
    throw silt::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.elem() != rhs.elem())
    throw silt::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() == silt::host_t::CPU) {
    detail::indexed_binop_apply_cpu(lhs, rhs, ind, func);
  }

  else if (lhs.host() == silt::host_t::GPU) {
    detail::indexed_binop_apply_gpu<<<block(ind.elem(), 512), 512>>>(lhs, rhs, ind, func);
    gpuErrchk(cudaGetLastError());
  }
}

} // namespace op
} // namespace silt
