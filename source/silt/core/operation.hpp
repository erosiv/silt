#ifndef SILT_OPERATION
#define SILT_OPERATION

#include <silt/core/tensor.hpp>
#include <silt/core/view.hpp>
#include <silt/core/error.hpp>
#include <curand_kernel.h>

// This file contains generic template operations for tensors.
// These are written to function both on the GPU and the CPU,
// and are intended to simplify the code structure for common
// operation types between tensors.

namespace silt {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

namespace op {

// Templated In-Place Unary Operations

template<typename T, typename F>
__global__ void __uniop_inplace_gpu(T lhs, F func){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    lhs[n] = func(lhs[n]);
  }
}

template<typename T, typename F>
__host__ void __uniop_inplace_cpu(T lhs, F func){
  for(unsigned int n = 0; n < lhs.elem(); ++n){
    lhs[n] = func(lhs[n]);
  }
}

template<typename T, typename F>
void uniop_inplace(T lhs, F func) {

  if(lhs.host() == silt::host_t::CPU) {
    __uniop_inplace_cpu(lhs, func);
  }

  else if(lhs.host() == silt::host_t::GPU) {
    __uniop_inplace_gpu<<<block(lhs.elem(), 512), 512>>>(lhs, func);
  }

}

// Templated In-Place Binary Operations

template<typename T, typename F>
__global__ void __binop_inplace_gpu(T lhs, const T rhs, F func) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    lhs[n] = func(lhs[n], rhs[n]);
  }
}

template<typename T, typename F>
__host__ void __binop_inplace_cpu(T lhs, const T rhs, F func){
  for(unsigned int n = 0; n < lhs.elem(); ++n){
    lhs[n] = func(lhs[n], rhs[n]);
  }
}

template<typename T, typename F>
void binop_inplace(T lhs, const T rhs, F func) {

  if(lhs.host() == silt::host_t::CPU){
    __binop_inplace_cpu(lhs, rhs, func);
  }

  else if(lhs.host() == silt::host_t::GPU){
    __binop_inplace_gpu<<<block(lhs.elem(), 512), 512>>>(lhs, rhs, func);
  }

}

// In-Place Indexed Unary Operation

template<typename T, typename F>
__global__ void __uniop_inplace_indexed_gpu(tensor_t<T> lhs, const tensor_t<int> ind, F func){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < ind.elem()){
    const int i = ind[n];
    if(i < lhs.elem()){
      lhs[i] = func(lhs[i]);
    }
  }
}

template<typename T, typename F>
__host__ void __uniop_inplace_indexed_cpu(tensor_t<T> lhs, const tensor_t<int> ind, F func){
  for(unsigned int n = 0; n < ind.elem(); ++n) {
    const int i = ind[n];
    if(i < lhs.elem()) {
      lhs[i] = func(lhs[i]);
    }
  }
}

template<typename T, typename F>
void uniop_inplace_indexed(tensor_t<T> lhs, const tensor_t<int> ind, F func) {

  if(lhs.host() == silt::host_t::CPU) {
    __uniop_inplace_indexed_cpu(lhs, ind, func);
  }

  else if(lhs.host() == silt::host_t::GPU){
    __uniop_inplace_indexed_gpu<<<block(ind.elem(), 512), 512>>>(lhs, ind, func);
  }

}

}
}

#endif