#pragma once

#include <limits>
#include <silt/core/tensor.hpp>

namespace silt {

//
// Common Operation Declarations
//

// Unary Operations

template<typename T>
void set(tensor_t<T> lhs, const T value);

template<typename T>
void set(view_t<T> lhs, const T value);

template<typename T>
void add(tensor_t<T> lhs, const T value);

template<typename T>
void add(view_t<T> lhs, const T value);

template<typename T>
void multiply(tensor_t<T> lhs, const T value);

template<typename T>
void multiply(view_t<T> lhs, const T value);

template<typename T>
void divide(tensor_t<T> lhs, const T value);

template<typename T>
void divide(view_t<T> lhs, const T value);

template<typename T>
void clamp(tensor_t<T> lhs, const T min, const T max);

template<typename T>
void clamp(view_t<T> lhs, const T min, const T max);

template<typename T>
void minimum(tensor_t<T> lhs, const T value);

template<typename T>
void minimum(view_t<T> lhs, const T value);

template<typename T>
void maximum(tensor_t<T> lhs, const T value);

template<typename T>
void maximum(view_t<T> lhs, const T value);

// Binary Operations

template<typename T>
void set(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void add(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void multiply(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void divide(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void mix(tensor_t<T> lhs, const tensor_t<T> rhs, const float w);

template<typename T>
void minimum(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void maximum(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename To, typename From>
silt::tensor_t<To> cast(const silt::tensor_t<From>& tensor) {
  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::mismatch_host(silt::host_t::CPU, tensor.host());

  tensor_t<To> tensor_to(tensor.shape(), silt::host_t::CPU);
  for (size_t i = 0; i < tensor.elem(); ++i) {
    tensor_to[i] = (To)tensor[i];
  }
  return tensor_to;
}

//
// Reductions
//

template<typename T>
T min(const silt::tensor_t<T>& tensor) {

  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::mismatch_host(silt::host_t::CPU, tensor.host());

  T val = std::numeric_limits<T>::max();
  for (size_t i = 0; i < tensor.elem(); ++i) {
    const T b = tensor[i];
    if (!std::isnan(b)) {
      val = std::min(val, b);
    }
  }
  return val;
}

template<typename T>
T max(const silt::tensor_t<T>& tensor) {

  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::mismatch_host(silt::host_t::CPU, tensor.host());

  T val = std::numeric_limits<T>::lowest();
  for (size_t i = 0; i < tensor.elem(); ++i) {
    const T b = tensor[i];
    if (!std::isnan(b)) {
      val = std::max(val, b);
    }
  }
  return val;
}

} // end of namespace silt
