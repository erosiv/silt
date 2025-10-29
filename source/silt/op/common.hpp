#ifndef SILT_OP_COMMON
#define SILT_OP_COMMON

#include <limits>
#include <silt/core/tensor.hpp>
#include <silt/core/operation.hpp>

namespace silt {

//
// Common Operation Declarations
//

template<typename T>
void set(tensor_t<T> lhs, const T value);

template<typename T>
void add(tensor_t<T> lhs, const T value);

template<typename T>
void multiply(tensor_t<T> lhs, const T value);

template<typename T>
void set(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void add(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void multiply(tensor_t<T> lhs, const tensor_t<T> rhs);

template<typename T>
void mix(tensor_t<T> lhs, const tensor_t<T> rhs, const float w);

template<typename T>
void clamp(tensor_t<T> lhs, const T min, const T max);

template<typename T>
tensor_t<T> resize(const tensor_t<T> rhs, const shape shape);

template<typename T>
void resample(
  tensor_t<T> target,       //!< Target Buffer
  const tensor_t<T> source, //!< Source Buffer
  const vec3 t_scale,       //!< Target World-Space Scale (incl. z)
  const vec3 s_scale,       //!< Source World-Space Scale (incl. z)
  const vec2 posdiff        //!< World-Space Positional Difference
);

//
// RNG Functions
//

//! Seed a Random Number Generator Tensor
void seed(tensor_t<rng>& buf, const size_t seed, const size_t offset);

//! Generate Uniform Samples from a Random Number Generator Tensor
tensor_t<float> sample_uniform(tensor_t<rng>& buf);
tensor_t<float> sample_uniform(tensor_t<rng>& buf, const float min, const float max);

//! Generate Normal Distributed Samples from a Random Number Generator Tensor
tensor_t<float> sample_normal(tensor_t<rng>& buf);
tensor_t<float> sample_normal(tensor_t<rng>& buf, const float mean, const float std);


//
// Legacy Functions
//! \todo get rid of this...

template<typename To, typename From>
void copy(silt::tensor_t<To> &out, const silt::tensor_t<From> &in, vec2 gmin, vec2 gmax, vec2 gscale, vec2 wmin, vec2 wmax, vec2 wscale, float pscale) {

  const ivec2 pmin = ivec2(pscale * (gmin - wmin) / wscale);
  const ivec2 pmax = ivec2(pscale * (gmax - wmin) / wscale);
  const ivec2 pext = ivec2(pscale * (wmax - wmin) / wscale);
  const ivec2 gext = ivec2((gmax - gmin) / gscale);

  for (int x = pmin[1]; x < pmax[1]; ++x) {
    for (int y = pmin[0]; y < pmax[0]; ++y) {

      const int ind_out = y + pext[0] * (pext[1] - x - 1);

      const size_t px = size_t((pmax[1] - x - 1) / pscale);
      const size_t py = size_t((y - pmin[0]) / pscale);
      const size_t ind_in = py + px * gext[0];

      out[ind_out] = To(From(pscale) * in[ind_in]);
    }
  }
}

//
// Other Operations that need cleaning / deprecation...
//

//
// Casting
//

template<typename To, typename From>
silt::tensor_t<To> cast(const silt::tensor_t<From> &tensor) {
  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::mismatch_host(silt::host_t::CPU, tensor.host());

  tensor_t<To> tensor_to(tensor.shape(), silt::host_t::CPU);
  for(int i = 0; i < tensor.elem(); ++i){
    tensor_to[i] = (To)tensor[i];
  }
  return tensor_to;
}

//
// Set Buffer from Value and Buffer
//

template<typename T>
void set_impl(silt::tensor_t<T> tensor, const T val, size_t start, size_t stop, size_t step);

template<typename T>
void set(silt::tensor_t<T> tensor, const T val, size_t start, size_t stop, size_t step) {

  if (tensor.host() == silt::host_t::CPU) {
    for (int i = start; i < stop; i += step)
      tensor[i] = val;
  }

  else if (tensor.host() == silt::host_t::GPU) {
    set_impl(tensor, val, start, stop, step);
  }
}

//
// Reductions
//

template<typename T>
T min(const silt::tensor_t<T> &tensor) {

  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::mismatch_host(silt::host_t::CPU, tensor.host());

  T val = std::numeric_limits<T>::max();
  for(int i = 0; i < tensor.elem(); ++i){
    const T b = tensor[i];
    if (!std::isnan(b)) {
      val = std::min(val, b);
    }
  }
  return val;
}

template<typename T>
T max(const silt::tensor_t<T> &tensor) {

  if (tensor.host() != silt::host_t::CPU)
    throw silt::error::mismatch_host(silt::host_t::CPU, tensor.host());

  T val = std::numeric_limits<T>::min();
  for(int i = 0; i < tensor.elem(); ++i){
    const T b = tensor[i];
    if (!std::isnan(b)) {
      val = std::max(val, b);
    }
  }
  return val;
}


} // end of namespace silt

#endif